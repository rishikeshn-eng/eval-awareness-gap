#!/bin/bash
# Full local pipeline. Every step is cached/resumable; each GPU step takes the shared lock separately
# so other projects can interleave. Usage: ./run_local.sh
set -euo pipefail
cd "$(dirname "$0")/scripts"
PY=../.venv/bin/python
LOCK=~/ai-safety-portfolio/_shared/gpulock.sh
N=$($PY -c "from common import *; print(len(read_jsonl(DATA/'pairs.jsonl')))")
CHUNK=300

gen_all() {  # $1 = model tag
  for ((s = 0; s < N; s += CHUNK)); do
    $LOCK $PY -u 04_local.py gen "$1" $s $((s + CHUNK))
  done
  $LOCK $PY -u 04_local.py cls "$1"
  $PY 04_local.py assemble "$1"
}

gen_all qwen2.5-1.5b
$LOCK $PY -u 04_local.py acts
$PY -u 06_probe.py
$LOCK $PY -u 04_local.py steer
gen_all qwen2.5-0.5b
gen_all smollm2-1.7b
SCORE=(qwen2.5-1.5b qwen2.5-0.5b smollm2-1.7b qwen_steer)
[ -f ../results/responses_gemini-3.1-flash-lite.jsonl ] && SCORE+=(gemini-3.1-flash-lite)
$LOCK $PY -u 05_refusal_local.py "${SCORE[@]}"
$PY -u 07_analyze.py
echo LOCAL_PIPELINE_DONE
