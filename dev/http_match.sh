#!/bin/bash
# Play agent.py vs a dev opponent through the official judge over HTTP, one half at a time
# with fresh servers (like the tournament pods). Usage: dev/http_match.sh <opponent.py> <seed> [deals]
PY=${PY:-python}
OPP=$1; SEED=$2; DEALS=${3:-1000}
OUT=${OUT:-/tmp}
for HALF in 1 2; do
  PORT=5108 $PY agent.py > $OUT/agent_h$HALF.log 2>&1 &
  A=$!
  PORT=5109 $PY dev/serve_bot.py $OPP > $OUT/opp_h$HALF.log 2>&1 &
  B=$!
  $PY judge_engine.py --agent1 5108 --agent2 5109 --half $HALF --seed $SEED --deals $DEALS --json $OUT/h$HALF.json > $OUT/judge_h$HALF.txt 2>&1
  kill $A $B 2>/dev/null; wait $A $B 2>/dev/null
done
$PY judge_engine.py combine $OUT/h1.json $OUT/h2.json | grep -E "agent1:|interval|thinking|RESULT|First error" 
