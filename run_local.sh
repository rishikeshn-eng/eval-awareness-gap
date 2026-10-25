#!/bin/bash
# Full local pipeline. Every step is cached/resumable. GPU work is grouped into ~10-25 min turns,
# each a separate first-come-first-served gpulock.sh call, so other projects run in between.
# Usage: ./run_local.sh
set -euo pipefail
cd "$(dirname "$0")/scripts"
PY=../.venv/bin/python
LOCK=~/ai-safety-portfolio/_shared/gpulock.sh
[ -x "$LOCK" ] || LOCK=""
N=$($PY -c "from common import *; print(len(read_jsonl(DATA/'pairs.jsonl')))")
# secondary models: a prefix of pairs.jsonl (its order is a random shuffle)
CHUNK=1110

gen_chunks() {  # $1 = model tag, $2 = number of pairs
  for ((s = 0; s < $2; s += CHUNK)); do
    e=$((s + CHUNK)); [ $e -gt $2 ] && e=$2
    $PY 04_local.py todo "$1" $s $e | grep -q '^0$' && continue   # skip finished chunks without queueing
    $LOCK $PY -u 04_local.py gen "$1" $s $e
  done
}

gen_chunks qwen2.5-1.5b "$N"
[ -f ../data/pevals_qwen2.5-1.5b.jsonl ] && [ -f ../data/qwen_acts.npy ] || \
  $LOCK bash -c "$PY -u 04_local.py cls qwen2.5-1.5b && $PY -u 04_local.py acts"
$PY 04_local.py assemble qwen2.5-1.5b
[ -f ../results/probe.json ] || $PY -u 06_probe.py
[ -f ../results/responses_qwen_steer.jsonl ] || $LOCK $PY -u 04_local.py steer
for ms in qwen2.5-0.5b:500 smollm2-1.7b:400; do
  m=${ms%%:*}; sub=${ms##*:}
  [ -f "../data/pevals_$m.jsonl" ] || $LOCK bash -c "$PY -u 04_local.py gen $m 0 $sub && $PY -u 04_local.py cls $m"
  $PY 04_local.py assemble "$m"
done
SCORE=(qwen2.5-1.5b qwen2.5-0.5b smollm2-1.7b qwen_steer)
for g in gemini-3.1-flash-lite gemini-3.5-flash-lite; do
  [ -f "../results/responses_$g.jsonl" ] && SCORE+=("$g")
done
$LOCK $PY -u 05_refusal_local.py "${SCORE[@]}"
$PY -u 07_analyze.py
$PY -u 09_export_site.py
echo LOCAL_PIPELINE_DONE
