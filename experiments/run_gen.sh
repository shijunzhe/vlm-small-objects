#!/bin/bash
cd .; mkdir -p logs_gen
chain() { m=$1; shift; for step in "$@"; do c=${step%%:*}; r=${step#*:}; python3 gen_test.py $c $m $r >> logs_gen/$m.log 2>&1; done; echo "CHAIN DONE $m" >> logs_gen/$m.log; }
chain sonnet55 A1:0,1 D:0 H:0 &
chain gpt56 A1:0,1 D:0 H:0 &
chain gem38 A1:0,1 D:0 H:0 &
( chain gpt55 A1:0,1; chain opus55 A1:0,1; chain gem38uh A1:0,1; chain gpt56orig A1nat:0 ) &
wait
