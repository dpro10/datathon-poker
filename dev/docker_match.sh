#!/bin/bash
# Official judge vs our bot running in its tournament-like container (fresh container per half).
# Usage: dev/docker_match.sh <opponent.py> <seed> [deals]
PY=${PY:-python}; OPP=$1; SEED=$2; DEALS=${3:-1000}; OUT=${OUT:-/tmp}
for HALF in 1 2; do
  docker rm -f pk >/dev/null 2>&1
  docker run --detach --name pk -p 5208:5008 --read-only --tmpfs /tmp:rw,nosuid,nodev,size=512m \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --cpus 1 --memory 4g \
    --env PORT=5008 poker-agent-check:latest >/dev/null
  PORT=5209 $PY dev/serve_bot.py $OPP > $OUT/dm_opp.log 2>&1 &
  B=$!
  $PY judge_engine.py --agent1 5208 --agent2 5209 --half $HALF --seed $SEED --deals $DEALS --json $OUT/d$HALF.json > $OUT/dm_judge$HALF.txt 2>&1
  kill $B 2>/dev/null; wait $B 2>/dev/null
done
docker rm -f pk >/dev/null 2>&1
$PY judge_engine.py combine $OUT/d1.json $OUT/d2.json | grep -E "agent1:|interval|agent1 half|RESULT|First error"
