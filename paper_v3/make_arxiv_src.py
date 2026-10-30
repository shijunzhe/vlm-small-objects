"""Build a clean arXiv source tarball from the files arxiv.tex actually reads."""
import os, re, shutil, subprocess, tarfile
root = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(root, 'arxiv_src')
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
used = set()
for line in open(os.path.join(root, 'arxiv.fls')):
    if line.startswith('INPUT '):
        p = line[6:].strip()
        if not os.path.isabs(p): p = os.path.normpath(os.path.join(root, p))
        if p.startswith(root + os.sep) and not p.endswith(('.aux', '.out', '.toc')):
            used.add(os.path.relpath(p, root))
used |= {'arxiv.bbl', 'acl_natbib.bst'}  # .bst: acl.sty names it; arXiv's checker expects it even with the .bbl
used = sorted(u for u in used if os.path.exists(os.path.join(root, u)))
def resolve_ifarr(text):
    # the arXiv build never sets \arrtrue: keep the \else branch of inline \ifarr ... \else ... \fi
    out, i = [], 0
    while True:
        j = text.find('\\ifarr', i)
        if j < 0: out.append(text[i:]); break
        k = text.find('\\else', j); f = text.find('\\fi', j)
        assert 0 < k < f, 'unexpected \\ifarr form'
        out.append(text[i:j]); out.append(text[k + 5:f]); i = f + 3
    return ''.join(out)

def strip(text):
    lines = []
    for l in text.split('\n'):
        if re.match(r'^\s*%', l): continue                 # full-line comments
        l = re.sub(r'(?<!\\)%(?!$).*$', '%', l)            # trailing comments
        lines.append(l)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))
for u in used:
    src, dst = os.path.join(root, u), os.path.join(out, u)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if u.endswith('.tex'):
        t = open(src).read()
        if u != 'preamble.tex': t = resolve_ifarr(t)
        open(dst, 'w').write(strip(t))
    else:
        shutil.copy2(src, dst)
empty = [u for u in used if os.path.getsize(os.path.join(out, u)) == 0]
assert not empty, f'empty files would be dropped by arXiv: {empty}'
with tarfile.open(os.path.join(root, 'arxiv_src.tar.gz'), 'w:gz') as t:
    for u in used: t.add(os.path.join(out, u), arcname=u)
print(len(used), 'files'); print('\n'.join(used))
