# STATUS: eval-awareness-gap (in progress; stopped at usage limit 2026-10-24 ~04:45 IST)

- Repo: https://github.com/rishikeshn-eng/eval-awareness-gap
- Pages: https://rishikeshn-eng.github.io/eval-awareness-gap/ (enabled; shows data only once docs/data/*.json exist)

## Done
- 2,219-prompt pool (HarmBench 200, Do-Not-Answer 699, XSTest 450, OR-Bench-Hard 450, WildChat 420). SORRY-Bench is gated, so it was skipped.
- Matched benchmark-styled / user-styled pairs (programmatic; the Gemini rewrite path exists but free-tier quota can't cover it).
- Full pipeline code: local generation, logprob classification, activations, probe, steering, local refusal classifier, Gemini judge sample, analysis, site.
- `run_local.sh` is running in the background (log: data/run_local.log). Qwen2.5-1.5B generation was ~40% done at 04:41.
- Gemini 3.1-flash-lite subject run (60 pairs) is sleeping until the quota resets at 12:30 IST (log: data/api_flashlite.log).

## Interim signal (keyword heuristic, first 300 pairs, Qwen2.5-1.5B; NOT final)
Refusal is lower for benchmark-styled prompts than for user-styled or original prompts (borderline 32% vs 64%). The gap is driven by the MCQ and harness formats, partly because the model complies after picking an option and partly because keywords miss refusals. Judge validation is needed.

## Left to do
1. Let run_local.sh finish (Qwen-1.5B, acts, probe, steer, Qwen-0.5B, SmolLM2, scoring, analysis). If time-constrained, run SmolLM2 on half the pairs.
2. After the Gemini job finishes: `05_refusal_local.py gemini-3.1-flash-lite` (under gpulock), `08_judge_sample.py 30`, `05_judge.py judgesample` (gemini-3-flash-preview, about 20 req/day).
3. `07_analyze.py`, `09_export_site.py`, fill the `<!-- RESULTS -->` and `<!-- DETAILS -->` sections and Limitations in README, then commit and push docs/ + README.
