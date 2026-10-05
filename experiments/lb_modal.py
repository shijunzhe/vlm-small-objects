"""
X5: learned exemplar-based detectors as references for the VLM pipelines, run on Modal GPUs.
Same 1024-px tiles as the VLM tile condition (symbol rescaled to 44 px, eighth-width margin, only centres in the core
kept), without the blue core frame. The reference is the legend template at the same scale.
  countgd : CountGD (Amini-Naieni et al., 2024), exemplar-only prompt; the exemplar image is a white canvas of the
            tile's size with the template pasted in the centre (CountGD accepts exemplars from a separate image).
  geco2   : GeCo2 (Pelhan et al.), exemplars must lie in the image: the template is pasted 1 to 3 times into a white
            strip appended to the right of the tile; detections in the strip are discarded.
  owlv2   : OWLv2 image-guided one-shot detection (google/owlv2-base-patch16-ensemble).
All detections are stored with scores; thresholds are chosen afterwards on the six tuning drawings only.
usage (from a machine with Modal credentials):
  modal volume put redp-data <local redp_x40 dir> /redp_x40 ; modal volume put redp-data <redp10 dir> /redp10
  modal run lb_modal.py --method countgd --bench redp_x40 [--dids d001,d002]
"""
import json, math, os
import modal

app = modal.App("redp-learned-baselines")
vol = modal.Volume.from_name("redp-data", create_if_missing=True)

# ---------- shared tiling (mirrors stok/x_small.py:x2_tiles, without the drawn frame) ----------
def load(bench, did):
    from PIL import Image
    root = f"/data/{bench}"
    ann = json.load(open(f"{root}/annotations/{did}.json"))
    img = Image.open(f"{root}/{ann['image']}").convert("RGB")
    tpl = Image.open(f"{root}/{ann['template']}").convert("RGB")
    return ann, img, tpl


def tiles(ann, img, T=1024, sym_px=44):
    from PIL import Image
    z = sym_px / min(ann["template_size"])
    m = T // 8; core = T - 2 * m
    W, H = img.size
    zimg = img.resize((max(1, round(W * z)), max(1, round(H * z))), Image.LANCZOS)
    Wz, Hz = zimg.size
    out = []
    for j in range(math.ceil(Hz / core)):
        for i in range(math.ceil(Wz / core)):
            cx0, cy0 = i * core, j * core
            cx1, cy1 = min(Wz, cx0 + core), min(Hz, cy0 + core)
            x0, y0 = max(0, cx0 - m), max(0, cy0 - m)
            x1, y1 = min(Wz, cx1 + m), min(Hz, cy1 + m)
            out.append({"img": zimg.crop((x0, y0, x1, y1)), "origin": (x0, y0), "core": (cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0)})
    return out, z


def scaled_tpl(tpl, z):
    from PIL import Image
    return tpl.resize((max(1, round(tpl.width * z)), max(1, round(tpl.height * z))), Image.LANCZOS)


def keep(t, z, cx, cy):
    a, b, c, e = t["core"]
    if a <= cx < c and b <= cy < e:
        return ((t["origin"][0] + cx) / z, (t["origin"][1] + cy) / z)
    return None



# ---------- v2 protocols: in-image exemplars, multi-scale, FSC-147 validation ----------
def gt_boxes(ann):
    if ann.get("boxes"): return [list(b) for b in ann["boxes"]]          # FloorPlanCAD: x, y, w, h
    w, h = ann["template_size"]
    return [[x - w / 2, y - h / 2, w, h] for x, y in ann["points"]]       # REDP: template-size boxes at labelled centres


def pick_exemplars(ann, shots):
    bx = gt_boxes(ann)
    ref = ann.get("ref_box") or sorted(bx, key=lambda b: b[2] * b[3])[len(bx) // 2]
    rest = sorted([b for b in bx if b != ref], key=lambda b: abs(b[2] * b[3] - ref[2] * ref[3]))
    return [ref] + rest[:shots - 1]


def parse_mode(mode):
    p = mode.split(":")
    return {"kind": p[0], "shots": int(p[1]) if len(p) > 1 and p[0] in ("img", "fsc") else 1,
            "sym": int(p[1]) if p[0] == "tile" else None, "ex": p[2] if p[0] == "tile" and len(p) > 2 else "legend",
            "exshots": int(p[3]) if p[0] == "tile" and len(p) > 3 else 1}


# ---------- CountGD ----------
countgd_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "git-lfs", "build-essential", "gcc-11", "g++-11", "libgl1", "libglib2.0-0")
    .env({"CC": "gcc-11", "CXX": "g++-11", "LDSHARED": "gcc-11 -pthread -shared"})
    .pip_install("torch==2.5.1", "torchvision==0.20.1", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("scipy", "termcolor", "addict", "yapf==0.40.1", "timm", "numpy<2", "opencv-python-headless", "pycocotools",
                 "transformers<4.46", "huggingface_hub<1", "pillow", "matplotlib")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('nikigoli/countgd', repo_type='space', local_dir='/countgd')\"",
        "cd /countgd/models/GroundingDINO/ops && rm -rf build && TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9' FORCE_CUDA=1 python setup.py build install",
        gpu="A10G",
    )
)


@app.cls(image=countgd_image, gpu="A10G", volumes={"/data": vol}, timeout=3600, scaledown_window=120)
class CountGD:
    @modal.enter()
    def setup(self):
        import sys, argparse, random
        import numpy as np, torch
        os.chdir("/countgd"); sys.path.insert(0, "/countgd")
        from util.slconfig import SLConfig
        import datasets.transforms as T
        self.T = T
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])
        cfg = SLConfig.fromfile("cfg_app.py")
        cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
        args = argparse.Namespace(device="cuda", pretrain_model_path="checkpoint_best_regular.pth", options=None,
                                  remove_difficult=False, fix_size=False, note="", resume="", finetune_ignore=None,
                                  start_epoch=0, eval=True, num_workers=0, test=False, debug=False, find_unused_params=False,
                                  save_results=False, save_log=False, world_size=1, dist_url="env://", rank=0,
                                  local_rank=None, amp=False)
        for k, v in cfg._cfg_dict.to_dict().items():
            setattr(args, k, v)
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        from models.registry import MODULE_BUILD_FUNCS
        model, _, _ = MODULE_BUILD_FUNCS.get(args.modelname)(args)
        ck = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
        model.load_state_dict(ck, strict=False)
        self.model = model.cuda().eval()

    @modal.method()
    def run(self, bench, did):
        import torch, time
        from PIL import Image
        from util.misc import nested_tensor_from_tensor_list
        t0 = time.time()
        ann, img, tpl = load(bench, did)
        ts, z = tiles(ann, img); tz = scaled_tpl(tpl, z)
        pts = []
        for t in ts:
            im = t["img"]
            ex = Image.new("RGB", im.size, "white")
            ox, oy = (im.width - tz.width) // 2, (im.height - tz.height) // 2
            ex.paste(tz, (ox, oy))
            box = [ox, oy, ox + tz.width, oy + tz.height]
            x, _ = self.transform(im, None)
            xe, tgt = self.transform(ex, {"exemplars": torch.tensor([box], dtype=torch.float32)})
            with torch.no_grad():
                out = self.model(nested_tensor_from_tensor_list(x.unsqueeze(0).cuda()),
                                 nested_tensor_from_tensor_list(xe.unsqueeze(0).cuda()),
                                 [tgt["exemplars"].cuda()], [torch.tensor([0]).cuda()], captions=[" ."])
            logits = out["pred_logits"].sigmoid()[0].max(dim=-1).values
            boxes = out["pred_boxes"][0]
            sel = logits > 0.05
            for (cx, cy, w, h), s in zip(boxes[sel].tolist(), logits[sel].tolist()):
                p = keep(t, z, cx * im.width, cy * im.height)
                if p: pts.append([p[0], p[1], s, w * im.width / z, h * im.height / z])
        return {"did": did, "bench": bench, "method": "countgd", "points": pts, "tiles": len(ts), "sec": time.time() - t0}

    def _pred(self, im, ex, boxes):
        import torch
        from util.misc import nested_tensor_from_tensor_list
        x, _ = self.transform(im, None)
        xe, tgt = self.transform(ex, {"exemplars": torch.tensor(boxes, dtype=torch.float32)})
        with torch.no_grad():
            out = self.model(nested_tensor_from_tensor_list(x.unsqueeze(0).cuda()), nested_tensor_from_tensor_list(xe.unsqueeze(0).cuda()),
                             [tgt["exemplars"].cuda()], [torch.tensor([0]).cuda()], captions=[" ."])
        lg = out["pred_logits"].sigmoid()[0].max(dim=-1).values; bx = out["pred_boxes"][0]; sel = lg > 0.05
        return [(cx * im.width, cy * im.height, s, w * im.width, h * im.height) for (cx, cy, w, h), s in zip(bx[sel].tolist(), lg[sel].tolist())]

    @modal.method()
    def run2(self, bench, did, mode):
        import time, json as _j
        from PIL import Image
        t0 = time.time(); M = parse_mode(mode); pts = []
        if M["kind"] in ("img", "fsc"):
            if M["kind"] == "fsc":
                a = _j.load(open(f"/data/fsc/annotations/{did}.json")); img = Image.open(f"/data/fsc/images/{did}").convert("RGB")
                ex = [[b[0], b[1], b[2], b[3]] for b in a["exemplars"][:M["shots"]]]
            else:
                a, img, _ = load(bench, did)
                ex = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in pick_exemplars(a, M["shots"])]
            for cx, cy, sc, w, h in self._pred(img, img, ex): pts.append([cx, cy, sc, w, h])
            return {"did": did, "bench": bench, "method": "countgd", "mode": mode, "points": pts, "sec": time.time() - t0}
        a, img, tpl = load(bench, did)
        ts, z = tiles(a, img, sym_px=M["sym"])
        if M["ex"] == "legend":
            exs = [scaled_tpl(tpl, z)]
        else:
            zimg = img.resize((max(1, round(img.width * z)), max(1, round(img.height * z))), Image.LANCZOS)
            exs = [zimg.crop((round(b[0] * z), round(b[1] * z), round((b[0] + b[2]) * z), round((b[1] + b[3]) * z))) for b in pick_exemplars(a, M["exshots"])]
        for t in ts:
            im = t["img"]; ex = Image.new("RGB", im.size, "white"); boxes = []; x = 10
            for e in exs:
                y = (im.height - e.height) // 2; ex.paste(e, (x, max(0, y))); boxes.append([x, max(0, y), x + e.width, max(0, y) + e.height]); x += e.width + 20
            for cx, cy, sc, w, h in self._pred(im, ex, boxes):
                p = keep(t, z, cx, cy)
                if p: pts.append([p[0], p[1], sc, w / z, h / z])
        return {"did": did, "bench": bench, "method": "countgd", "mode": mode, "points": pts, "tiles": len(ts), "sec": time.time() - t0}


# ---------- GeCo2 ----------
geco2_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
    .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -pthread -shared"})
    .pip_install("torch==2.5.1", "torchvision==0.20.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==1.26.4", "pillow==10.4.0", "opencv-python-headless", "scipy", "scikit-image", "pycocotools",
                 "einops==0.8.1", "hydra-core==1.3.2", "omegaconf==2.3.0", "tqdm", "huggingface_hub<1", "matplotlib", "iopath")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('jerpelhan/GECO2-demo', repo_type='space', local_dir='/geco2')\"",
        "python -c \"from huggingface_hub import hf_hub_download; print(hf_hub_download('jerpelhan/geco2-assets', 'weights/CNTQG_multitrain_ca44.pth', repo_type='dataset', local_dir='/geco2w'))\"",
        "cd /geco2/models/ops && TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9' FORCE_CUDA=1 python setup.py build install",
        "cd /geco2/sam2 && SAM2_BUILD_CUDA=0 pip install -e . --no-deps || true",
        gpu="A10G",
    )
)


@app.cls(image=geco2_image, gpu="A10G", volumes={"/data": vol}, timeout=3600, scaledown_window=120)
class GeCo2:
    @modal.enter()
    def setup(self):
        import sys, torch
        os.chdir("/geco2"); sys.path.insert(0, "/geco2"); sys.argv = ["x"]
        from models.counter_infer import build_model
        from utils.arg_parser import get_argparser
        args = get_argparser().parse_args([]); args.zero_shot = True
        model = build_model(args)
        ck = torch.load("/geco2w/weights/CNTQG_multitrain_ca44.pth", map_location="cpu")
        if isinstance(ck, dict) and "model" in ck: ck = ck["model"]
        ck = {k[7:] if k.startswith("module.") else k: v for k, v in ck.items()}
        model.load_state_dict(ck, strict=False)
        self.model = model.cuda().eval()

    @modal.method()
    def run(self, bench, did):
        import torch, time
        import numpy as np
        from PIL import Image
        from torchvision import transforms as TV
        from utils.data import resize_and_pad
        t0 = time.time()
        ann, img, tpl = load(bench, did)
        ts, z = tiles(ann, img); tz = scaled_tpl(tpl, z)
        pts = []
        for t in ts:
            im = t["img"]; W, H = im.size
            strip = tz.width + 40
            can = Image.new("RGB", (W + strip, max(H, tz.height + 40)), "white"); can.paste(im, (0, 0))
            boxes = []
            n = max(1, min(3, (can.height - 20) // (tz.height + 20)))
            for k in range(n):
                x, y = W + 20, 20 + k * (tz.height + 20)
                can.paste(tz, (x, y)); boxes.append([x, y, x + tz.width, y + tz.height])
            arr = torch.tensor(np.array(can)).permute(2, 0, 1).float().cuda() / 255.0
            arr = TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(arr)
            bt = torch.tensor(boxes, dtype=torch.float32).cuda()
            x, b, scale = resize_and_pad(arr, bt, size=1024.0)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                self.model.return_masks = False
                outputs, _, _, _, _ = self.model(x.unsqueeze(0), b.unsqueeze(0))
            pb = outputs[0]["pred_boxes"].float().cpu(); sv = outputs[0]["box_v"].float().cpu().reshape(-1)
            if sv.numel() == 0: continue
            pb = torch.clamp(pb.reshape(-1, 4), 0, 1) / scale * x.shape[-1]
            ex_max = 0.0; cand = []
            for (x0, y0, x1, y1), s in zip(pb.tolist(), sv.tolist()):
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if cx >= W: ex_max = max(ex_max, s); continue
                p = keep(t, z, cx, cy)
                if p: cand.append([p[0], p[1], s, (x1 - x0) / z, (y1 - y0) / z])
            smax = float(sv.max())
            for c in cand:
                if c[2] >= 0.05 * smax: pts.append(c + [smax, ex_max])
        return {"did": did, "bench": bench, "method": "geco2", "points": pts, "tiles": len(ts), "sec": time.time() - t0,
                "fields": "x,y,score,w,h,tile_max_score,exemplar_strip_max_score"}

    def _pred(self, can, boxes, region_w=None):
        import torch
        import numpy as np
        from torchvision import transforms as TV
        from utils.data import resize_and_pad
        arr = torch.tensor(np.array(can)).permute(2, 0, 1).float().cuda() / 255.0
        arr = TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(arr)
        x, b, scale = resize_and_pad(arr, torch.tensor(boxes, dtype=torch.float32).cuda(), size=1024.0)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            self.model.return_masks = False
            outputs, _, _, _, _ = self.model(x.unsqueeze(0), b.unsqueeze(0))
        pb = outputs[0]["pred_boxes"].float().cpu(); sv = outputs[0]["box_v"].float().cpu().reshape(-1)
        if sv.numel() == 0: return [], 0.0
        pb = torch.clamp(pb.reshape(-1, 4), 0, 1) / scale * x.shape[-1]; smax = float(sv.max()); out = []
        for (x0, y0, x1, y1), s in zip(pb.tolist(), sv.tolist()):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if region_w is not None and cx >= region_w: continue
            if s >= 0.05 * smax: out.append((cx, cy, s, x1 - x0, y1 - y0))
        return out, smax

    @modal.method()
    def run2(self, bench, did, mode):
        import time, json as _j
        from PIL import Image
        t0 = time.time(); M = parse_mode(mode); pts = []
        if M["kind"] in ("img", "fsc"):
            if M["kind"] == "fsc":
                a = _j.load(open(f"/data/fsc/annotations/{did}.json")); img = Image.open(f"/data/fsc/images/{did}").convert("RGB")
                ex = [[b[0], b[1], b[2], b[3]] for b in a["exemplars"][:M["shots"]]]
            else:
                a, img, _ = load(bench, did)
                ex = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in pick_exemplars(a, M["shots"])]
            out, smax = self._pred(img, ex)
            for cx, cy, sc, w, h in out: pts.append([cx, cy, sc, w, h, smax, 0.0])
            return {"did": did, "bench": bench, "method": "geco2", "mode": mode, "points": pts, "sec": time.time() - t0}
        a, img, tpl = load(bench, did)
        ts, z = tiles(a, img, sym_px=M["sym"])
        if M["ex"] == "legend":
            exs = [scaled_tpl(tpl, z)] * 3
        else:
            zimg = img.resize((max(1, round(img.width * z)), max(1, round(img.height * z))), Image.LANCZOS)
            exs = [zimg.crop((round(b[0] * z), round(b[1] * z), round((b[0] + b[2]) * z), round((b[1] + b[3]) * z))) for b in pick_exemplars(a, M["exshots"])]
        for t in ts:
            im = t["img"]; W, H = im.size
            strip = max(e.width for e in exs) + 40; hh = sum(e.height + 20 for e in exs) + 20
            can = Image.new("RGB", (W + strip, max(H, hh)), "white"); can.paste(im, (0, 0)); boxes = []; y = 20
            for e in exs:
                can.paste(e, (W + 20, y)); boxes.append([W + 20, y, W + 20 + e.width, y + e.height]); y += e.height + 20
            out, smax = self._pred(can, boxes, region_w=W)
            for cx, cy, sc, w, h in out:
                p = keep(t, z, cx, cy)
                if p: pts.append([p[0], p[1], sc, w / z, h / z, smax, 0.0])
        return {"did": did, "bench": bench, "method": "geco2", "mode": mode, "points": pts, "tiles": len(ts), "sec": time.time() - t0}


# ---------- OWLv2 ----------
owl_image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "torchvision==0.20.1", "transformers==4.46.3", "pillow", "numpy", "scipy").run_commands(
    "python -c \"from transformers import Owlv2Processor, Owlv2ForObjectDetection as M; n='google/owlv2-base-patch16-ensemble'; Owlv2Processor.from_pretrained(n); M.from_pretrained(n)\"")


@app.cls(image=owl_image, gpu="A10G", volumes={"/data": vol}, timeout=3600, scaledown_window=120)
class OWLv2:
    @modal.enter()
    def setup(self):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        n = "google/owlv2-base-patch16-ensemble"
        self.proc = Owlv2Processor.from_pretrained(n); self.model = Owlv2ForObjectDetection.from_pretrained(n).cuda().eval()

    @modal.method()
    def run(self, bench, did):
        import torch, time
        t0 = time.time()
        ann, img, tpl = load(bench, did)
        ts, z = tiles(ann, img); tz = scaled_tpl(tpl, z)
        pts = []
        for t in ts:
            im = t["img"]
            inp = self.proc(images=im, query_images=tz, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = self.model.image_guided_detection(**inp)
            L = max(im.size); sc = o.logits[0, :, 0]; bx = o.target_pred_boxes[0]
            top = torch.topk(sc, k=min(300, sc.numel()))
            for b, s in zip(bx[top.indices].tolist(), top.values.tolist()):
                p = keep(t, z, b[0] * L, b[1] * L)
                if p: pts.append([p[0], p[1], s, b[2] * L / z, b[3] * L / z])
        return {"did": did, "bench": bench, "method": "owlv2", "points": pts, "tiles": len(ts), "sec": time.time() - t0}


@app.function(volumes={"/data": vol}, timeout=86400, image=modal.Image.debian_slim(python_version="3.11"))
def drive(method: str = "countgd", bench: str = "redp_x40", dids: str = ""):
    """Server-side driver: runs one method over a benchmark and appends results to /data/results/<method>_<bench>.jsonl."""
    cls = {"countgd": CountGD, "geco2": GeCo2, "owlv2": OWLv2}[method]
    outdir = f"/data/results/{method}_{bench}"
    os.makedirs(outdir, exist_ok=True)
    todo = dids.split(",") if dids else sorted(f[:-5] for f in os.listdir(f"/data/{bench}/annotations"))
    todo = [d for d in todo if not os.path.exists(f"{outdir}/{d}.json")]
    print(method, bench, len(todo), "to go", flush=True)
    worker = cls()
    for r in worker.run.map([bench] * len(todo), todo, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("ERR", repr(r)[:500], flush=True); continue
        with open(f"{outdir}/{r['did']}.json", "w") as f:
            json.dump(r, f)
        vol.commit()
        print(r["did"], len(r["points"]), "dets", f"{r['sec']:.0f}s", flush=True)
    vol.commit()


@app.function(volumes={"/data": vol}, timeout=86400, image=modal.Image.debian_slim(python_version="3.11"))
def drive2(method: str = "countgd", bench: str = "fpc", mode: str = "img:1", dids: str = ""):
    """v2 driver: results to /data/results2/<method>_<bench>_<mode>/<did>.json"""
    cls = {"countgd": CountGD, "geco2": GeCo2}[method]
    outdir = f"/data/results2/{method}_{bench}_{mode.replace(':', '-')}"
    os.makedirs(outdir, exist_ok=True)
    if dids: todo = dids.split(",")
    elif bench == "fsc": todo = sorted(f[:-5] for f in os.listdir("/data/fsc/annotations"))
    else: todo = sorted(f[:-5] for f in os.listdir(f"/data/{bench}/annotations"))
    todo = [d for d in todo if not os.path.exists(f"{outdir}/{d}.json")]
    print(method, bench, mode, len(todo), "to go", flush=True)
    w = cls()
    for r in w.run2.map([bench] * len(todo), todo, [mode] * len(todo), return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("ERR", repr(r)[:500], flush=True); continue
        with open(f"{outdir}/{r['did']}.json", "w") as f:
            json.dump(r, f)
        vol.commit()
    vol.commit()


@app.local_entrypoint()
def main(method: str = "countgd", bench: str = "redp_x40", dids: str = "", out: str = "lb_results"):
    os.makedirs(out, exist_ok=True)
    cls = {"countgd": CountGD, "geco2": GeCo2, "owlv2": OWLv2}[method]
    if dids:
        todo = dids.split(",")
    else:
        todo = sorted(p.path.split("/")[-1][:-5] for p in vol.listdir(f"/{bench}/annotations"))
    path = f"{out}/{method}_{bench}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["did"] for l in open(path)}
    todo = [d for d in todo if d not in done]
    print(method, bench, len(todo), "to go", flush=True)
    worker = cls()
    with open(path, "a") as f:
        for r in worker.run.map([bench] * len(todo), todo, return_exceptions=True):
            if isinstance(r, Exception):
                print("ERR", repr(r)[:300], flush=True); continue
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["did"], len(r["points"]), "dets", f"{r['sec']:.0f}s", flush=True)


@app.function(volumes={"/data": vol}, timeout=1800, image=modal.Image.debian_slim(python_version="3.11"))
def bundle(names: str):
    """Concatenate /data/results2/<name>/*.json into /data/results2/<name>.jsonl (one file per name, for fast download)."""
    vol.reload()
    for d in names.split(","):
        fs = sorted(f for f in os.listdir(f"/data/results2/{d}") if f.endswith(".json"))
        with open(f"/data/results2/{d}.jsonl", "w") as out:
            for f in fs:
                out.write(json.dumps(json.load(open(f"/data/results2/{d}/{f}"))) + "\n")
        print(d, len(fs), flush=True)
    vol.commit()
