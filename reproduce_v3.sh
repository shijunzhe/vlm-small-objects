#!/bin/bash
# Regenerate every table and macro of version 3 of the paper (main text and appendices) from the shipped
# model answers, without any model call, and compare them with the shipped versions in paper_v3/generated/.
# Run reproduce.sh first to recompute the upstream scores; this script reads them.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
SNAP="$(mktemp -d)"
cd "$ROOT"
TARGETS="paper_v3/generated experiments/results_ls/ls_summary.json experiments/results_x/agent_err_check.json"
find $TARGETS -type f \( -name '*.json' -o -name '*.tex' \) -print0 | sort -z | xargs -0 sha256sum > "$SNAP/before.txt"
cd "$ROOT/experiments"
FAIL=0
run() { printf '%-44s' "$*"; if python3 "$@" > "$SNAP/$(basename "$1").log" 2>&1; then echo ok; else echo "FAILED (log: $SNAP)"; FAIL=1; fi; }
run loadsweep/ls_analyze.py
run ../paper_v3/scripts/tab1.py
run ../paper_v3/scripts/gain_vs_sw.py
run ../paper_v3/scripts/rule_checks.py
run ../paper_v3/scripts/gen_supp_tables.py
run ../paper_v3/scripts/agent_err_check.py
run ../paper_v3/scripts/tabL_load.py
run ../paper_v3/scripts/geom_recipes.py ..
run ../paper_v3/scripts/gen_macros.py vspec
cd "$ROOT"
find $TARGETS -type f \( -name '*.json' -o -name '*.tex' \) -print0 | sort -z | xargs -0 sha256sum > "$SNAP/after.txt"
echo; echo "Files that differ from the shipped version:"
diff <(sort -k2 "$SNAP/before.txt") <(sort -k2 "$SNAP/after.txt") | grep '^>' | awk '{print "  " $3}' || true
N=$(diff <(sort -k2 "$SNAP/before.txt") <(sort -k2 "$SNAP/after.txt") | grep -c '^>' || true)
if [ "$N" = "0" ] && [ "$FAIL" = "0" ]; then echo "  none: all regenerated tables and macros match the shipped files."; fi
exit $FAIL
