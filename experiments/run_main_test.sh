#!/bin/bash
# one chain per model (limits memory and per-vendor concurrency); every step is resumable
cd .
chain() { m=$1; shift; for step in "$@"; do c=${step%%:*}; r=${step#*:}; python3 main_test.py $c $m $r >> logs_test/$m.log 2>&1; done; echo "CHAIN DONE $m" >> logs_test/$m.log; }
mkdir -p logs_test
chain gpt     A1:0,1 B:0,1 H:0 D:0,1 Dmask:0 &
chain claude  A1:0,1 H:0 D:0,1 Dmask:0 &
chain gemini  A1:0,1 B:0,1 H:0 C:0,1 D:0,1 &
chain gemma4  A1:0,1 H:0 C:0,1 D:0,1 &
chain qwen3vl A1:0,1 H:0 D:0,1 &
wait
