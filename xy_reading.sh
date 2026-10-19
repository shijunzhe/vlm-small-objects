#!/bin/bash
# Optional reading (paper v2, Appendix E): rescore every experiment with Gemini 3.8's points written with x/y keys
# read as well, in a temporary copy of this repository, and write paper/generated/macros_xy.tex with the values that
# change. The shipped files are not touched by the rescoring; reproduce.sh keeps the registered reading.
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
cp -a "$ROOT/." "$TMP/"
( cd "$TMP" && POINT_XY=1 bash reproduce.sh > "$TMP/xy_reproduce.log" 2>&1 ) || true
cd "$ROOT/experiments" && python3 xy_macros.py "$TMP"
echo "rescored copy: $TMP (log: $TMP/xy_reproduce.log)"
