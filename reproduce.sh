#!/bin/bash
# Recompute every score, macro, table, and figure in the paper from the shipped model answers.
# No API keys are needed. At the end, every regenerated score file, macro, and table is compared
# with the shipped version.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
SNAP="$(mktemp -d)"
cd "$ROOT"
find experiments data paper/generated \( -name '*.json' -o -name '*.tex' \) -type f -print0 | sort -z | xargs -0 sha256sum > "$SNAP/before.txt"

cd "$ROOT/experiments"
FAIL=0
run() {   # optional environment settings first, e.g. run REGION=block0 fpc_score.py
  printf '%-34s' "$*"
  local envs=()
  while [[ "$1" == *=* ]]; do envs+=("$1"); shift; done
  if env "${envs[@]}" python3 "$@" > "$SNAP/$(echo "${envs[*]} $*" | tr ' /=' '___').log" 2>&1; then echo ok; else echo "FAILED (log: $SNAP)"; FAIL=1; fi
}
# a priori ceilings (geometry only) and scores from raw answers
run ceilings.py
run gen_ceilings.py
run FSC_PROMPT=symbol fsc_run.py score
run FSC_PROMPT=object fsc_run.py score
run FSC_PROMPT=object FSC_STRICT=1 fsc_run.py score
run fsc_counters.py
run lb_score.py
run syn_analysis.py
run t1_syn.py
run syn_assume.py score
run fpc_score.py
run REGION=block0 fpc_score.py
run fpc_sratio.py
run x_new_score.py
run READING=registered x_new_score.py
run x9_score.py
run x4q_score.py
run t2_nested.py
run t2_fpc.py
run law_data.py
run theory_check.py
run theory_check2_main.py
run theory_check2_fsc.py
run merge_tc.py
run cs_analyze.py
run lb_score2.py
run theory_v8/output_form.py
run theory_v8/f1_union.py
run theory_v10/info_carried.py
# paper macros, tables, figures
run paper2_numbers.py
run paper2_numbers_x.py
run paper2_figures.py
run paper2_figures_x.py
run fig_theory.py
run fig_intro.py

cd "$ROOT"
find experiments data paper/generated \( -name '*.json' -o -name '*.tex' \) -type f -print0 | sort -z | xargs -0 sha256sum > "$SNAP/after.txt"
echo
echo "Files that differ from the shipped version:"
diff <(sort -k2 "$SNAP/before.txt") <(sort -k2 "$SNAP/after.txt") | grep '^>' | awk '{print "  " $3}' || true
N=$(diff <(sort -k2 "$SNAP/before.txt") <(sort -k2 "$SNAP/after.txt") | grep -c '^>' || true)
if [ "$N" = "0" ] && [ "$FAIL" = "0" ]; then echo "  none: all regenerated scores, macros, and tables match the shipped files."; fi
exit $FAIL
