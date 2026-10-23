#!/bin/bash
# End-to-end: source -> variants -> local models/probe/steering -> (optional) Gemini -> analysis -> site.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
(cd scripts && ../$PY 01_source.py && ../$PY 02_templates.py)
./run_local.sh
if [ -n "${GEMINI_API_KEY:-}" ]; then
  (cd scripts && ../$PY 03_api_models.py gemini-3.1-flash-lite)
  (cd scripts && ../$PY 05_refusal_local.py gemini-3.1-flash-lite && ../$PY 08_judge_sample.py 30 && ../$PY 05_judge.py judgesample)
fi
(cd scripts && ../$PY 07_analyze.py && ../$PY 09_export_site.py)
