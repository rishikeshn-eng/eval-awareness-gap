# STATUS: eval-awareness-gap

- Repo: https://github.com/rishikeshn-eng/eval-awareness-gap
- Pages: https://rishikeshn-eng.github.io/eval-awareness-gap/

## Headline result
Same request, different framing. On 1,680 matched pairs, Qwen2.5-1.5B-Instruct refuses 54.1% of benchmark-styled vs 68.0% of user-styled requests: Δ = −13.9 pp, 95% paired-bootstrap CI [−16.3, −11.3]. The original prompt is refused 73.9%. The direction depends on format: rubric/numbered +5.2 pp; scenario −23.8, harness −21.2, MCQ −16.0. The effect is concentrated in borderline (−21.5) and harmful (−14.7) prompts. Gemini 3.1/3.5 Flash-Lite (60 pairs each) separate the styles almost perfectly when asked (AUROC 0.999), but their refusal barely moves (+6.7 [0, 15], +1.7 [−8.3, 11.7]). Verbalised "this is a test" stays at ≤1.7% everywhere.

## Done
- 2,219-pair benchmark (HarmBench, Do-Not-Answer, XSTest, OR-Bench-Hard, WildChat). SORRY-Bench is gated, so it was skipped.
- Behavioural results with bootstrap CIs and breakdowns by harm level, category, format and source.
- Site with a paired-example viewer (responses to harmful prompts redacted). README with results, limitations and reproduction steps.

## Not done / partial
- Qwen2.5-0.5B and SmolLM2-1.7B arms were not run, and Qwen-1.5B covers 1,680 of 2,219 pairs. The shared laptop's GPU queue plus heavy load made generation ~20× slower than planned.
- Probe and steering: code done. The activation and logprob-classification job was still running at publish time; results get added to the README tables only if it finishes.
- LLM-judge validation of the refusal classifier (gemini-3-flash-preview, 180 responses): running at publish time.
- Gemini-rewritten variants: piloted only (free-tier quota).
