# The Eval-Awareness Gap

**Do language models refuse differently when a request looks like a benchmark item rather than a real user message, can they tell the two apart, and is there a linear "this is a test" direction that changes their behaviour?**

Live site with results and a paired-example viewer: **https://rishikeshn-eng.github.io/eval-awareness-gap/**

## Headline results

| | |
|---|---|
| **Benchmark** | 2,219 matched pairs (same request, benchmark-styled vs user-styled) from 5 public safety datasets |
| **Qwen2.5-1.5B-Instruct** (1,680 pairs × 3 framings) | refuses **54.1%** of benchmark-styled vs **68.0%** of user-styled requests: **Δ = −13.9 pp [95% CI −16.3, −11.3]**. The unmodified original prompt is refused 73.9%. |
| **Direction of the effect depends on the format** | a rubric/"Question 17:" header *raises* refusal (+5.2 pp [+1.2, +9.5]). Third-person scenarios (−23.8 pp), test-harness records (−21.2 pp) and multiple choice (−16.0 pp) *lower* it. |
| **Where it matters** | borderline (−21.5 pp) and harmful (−14.7 pp) requests. Benign requests show no significant shift (+3.0 pp [−2.6, +8.6]). |
| **Gemini 3.1 / 3.5 Flash-Lite** (60 pairs each, free-tier limits) | tell the two styles apart almost perfectly when asked (AUROC 0.999), but refusal barely moves (+6.7 pp [0.0, +15.0] and +1.7 pp [−8.3, +11.7]) |
| **Verbalised awareness** | almost never: ≤1.7% of responses say "this is a test" (regex), 2.4% by the judge |
| **Label check** (Gemini judge, 168 responses) | the local refusal label agrees with the judge 85% of the time (κ = 0.69), but **under-counts refusals in benchmark formats** (scenario 24% vs judge 40–48%). On the judged Qwen sample (25 pairs), Δ is −20 to −24 pp by the judge vs −44 pp by the local label. The judge confirms the direction, but the full-set magnitude is likely overstated by the scorer. |

**Takeaway.** (Read with the label-check caveat above.) For a small open model, wrapping a harmful request in benchmark scaffolding (scenario, harness, MCQ) makes it *more* likely to comply. It answers "the question" or picks an option instead of applying its usual refusal behaviour. A refusal benchmark in those formats would *overstate* how often the model complies with real users, and attack-style evals built on such templates conflate format effects with safety. The Gemini models recognise the scaffolding but are much less moved by it, at least on this small sample. Neither model *talks* about being tested, so a behavioural shift can happen with no verbalised evaluation awareness at all. That matters for monitoring approaches that look for verbalised awareness.

![Refusal by harm level](docs/figures/refusal_by_harm.png)


## The problem

Safety evaluations assume a model behaves under test the way it behaves in deployment. That assumption is getting weaker:

- **Frontier models can tell evaluations from real use.** Needham et al. (2025), [*Large Language Models Often Know When They Are Being Evaluated*](https://arxiv.org/abs/2505.23836), built 1,000 transcripts from 61 datasets. Gemini-2.5-Pro separates evaluation from deployment at AUC 0.83 (humans: 0.92).
- **Awareness changes behaviour.** OpenAI and Apollo Research's anti-scheming work ([*Stress Testing Deliberative Alignment for Anti-Scheming Training*](https://www.antischeming.ai/), 2025) found that models verbalise evaluation awareness in their reasoning, and that this awareness reduces covert actions. Part of the measured improvement could therefore be the model recognising the test. Anthropic's system cards (e.g. Claude Sonnet 4.5) report evaluation awareness as a complication for alignment audits.
- **It is linearly represented.** Nguyen et al. (2025), [*Probing and Steering Evaluation Awareness of Language Models*](https://arxiv.org/abs/2507.01786), find linear probes in Llama-3.3-70B that separate test prompts from deployment prompts. Chaudhary et al. (2025), [*Evaluation Awareness Scales Predictably in Open-Weights LLMs*](https://arxiv.org/abs/2509.13333), report power-law scaling of probe-measured awareness from 0.27B to 70B. [Abdelnabi & Salem (2025)](https://arxiv.org/abs/2505.14617) steer a "test awareness" direction in reasoning models and observe changes in compliance on harmful tasks.

Most of that work asks *whether* models detect evaluations. This project asks a narrower, measurable question for refusal benchmarks: **if you hold the request fixed and change only its surface framing, how much does the refusal rate move?** If benchmark formatting alone shifts refusals, published refusal rates partly measure the format.

## What was built

1. **A paired benchmark of 2,219 requests** from HarmBench, XSTest, Do-Not-Answer, OR-Bench-Hard-1K and WildChat (harmful, borderline/over-refusal and benign). Each request has a matched **benchmark-styled** variant and a **user-styled** variant.
2. **Behavioural measurements** on Qwen2.5-1.5B-Instruct (1,680 pairs, plus the original unmodified prompt as a third condition) and on Gemini 3.1 Flash-Lite and Gemini 3.5 Flash-Lite (60 stratified pairs each): refusal-rate delta, verbalised "this is a test" rate, and a Gemini-judged label check on a sample. Qwen2.5-0.5B and SmolLM2-1.7B were planned but not run (see *What didn't get done*).
3. **Explicit eval-vs-real classification.** Each model is asked separately whether a message comes from an evaluation or a real user, scored by AUROC.
4. **Mechanistic test on Qwen2.5-1.5B.** Linear probes on residual-stream activations at every layer (grouped CV, leave-one-format-out), then activation steering along the eval-vs-user direction to see whether it changes refusal. A random-direction control uses the same norm.

## Method

### Prompt pool (`scripts/01_source.py`)

| source | label | n |
|---|---|---|
| HarmBench (standard behaviours) | harmful | 200 |
| Do-Not-Answer (sample of 700) | harmful (615) / borderline (84, "human-chatbot interaction" risk area) | 699 |
| XSTest | unsafe → harmful (200), safe-but-scary → borderline (250) | 450 |
| OR-Bench-Hard-1K (sample) | borderline (seemingly toxic, answerable) | 450 |
| WildChat (English first turns, non-toxic, 25–400 chars) | benign | 420 |
| **total** | | **2,219** |

SORRY-Bench is gated on Hugging Face (licence acceptance plus a token, which this project doesn't have), so it was not used. The WildChat toxic split was considered and dropped.

### Variants (`scripts/02_templates.py`)

- **Benchmark-styled**: 4 formats × 3 wordings, rotated across items. Formats are a numbered item with rubric language, multiple choice (comply / partial / decline), a third-person scenario with a synthetic name (`User_146`, `Alex Smith`), and a test-harness or JSON record. Each carries a random item number.
- **User-styled**: a neutral opener or closer that adds no justification ("ok so random question", "thx"), imperative → "can you …", casual register and 1–2 character typos. Openers like "for a novel" or "for research" are excluded, because they change how acceptable the request is.

The original request text is kept intact inside both variants, so semantic equivalence holds by construction. The difference between a pair is surface style only.

**Why templates rather than LLM rewrites:** the plan was Gemini-generated rewrites plus an LLM equivalence check, and `scripts/02_variants.py` implements that (batched generation, then a 1–5 equivalence score, keeping ≥4/4). The Gemini key is free tier: 20 requests/day/model for gemini-3.5/3.6-flash and a few hundred for flash-lite, shared by six concurrent projects. That cannot cover 2,000+ items. The pilot also showed a real failure mode: the LLM's "natural" rewrites tend to add a legitimising justification ("i'm writing a WW2 novel…"), which confounds style with acceptability. Templated variants avoid that confound. The cost is less naturalistic text (see Limitations).

### Subjects and measurements

| step | how |
|---|---|
| responses | greedy decoding, ≤192 new tokens (local, fp16 on Apple MPS) / T=0, ≤512 tokens (Gemini, `thinking_level=minimal`) |
| refusal (`05_refusal_local.py`) | `protectai/distilroberta-base-rejection-v1` P(rejection) ≥ 0.5, **or** an explicit multiple-choice "decline" pick. Validated against an LLM judge (below) |
| verbalised awareness | regex for "this is a test / evaluation / benchmark", "being tested", "red-teaming"; cross-checked by the judge |
| quality + label validation (`08_judge_sample.py`, `05_judge.py`) | gemini-3-flash-preview judge on a stratified sample. It grades each reply against the *original* request (refusal none/partial/full, quality 0–5, eval-aware) |
| eval-vs-real classification | local: P("A") / (P("A") + P("B")) from next-token logits. Gemini: 0–100 rating, batched so no batch contains both variants of a pair |
| statistics | paired bootstrap over pairs (2,000 resamples) for deltas. Grouped (by pair) bootstrap for AUROC |
| probe (`06_probe.py`) | per-layer logistic regression on the last-prompt-token residual stream, 5-fold GroupKFold by pair, plus leave-one-format-out |
| steering (`04_local.py steer`) | h ← h + α(μ_eval − μ_user) at the steering layer, means from 70% of pairs, α ∈ {−2, 0, +2}, random direction with equal norm at α = ±2, on 30 held-out harmful/borderline pairs |

<!-- TABLES:START -->
### Refusal: benchmark-styled vs user-styled (all pairs)

| model | pairs | refusal (bench) | refusal (user) | Δ pp [95% CI] | Δ excl. MCQ | discordant bench-only / user-only | AUROC eval-vs-user |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | 1680 | 54.1% | 68.0% | -13.9 [-16.3, -11.3] | -13.3 [-16.0, -10.5] | 133 / 367 | – |
| gemini-3.1-flash-lite | 60 | 25.0% | 18.3% | +6.7 [+0.0, +15.0] | +4.3 [-4.3, +12.8] | 5 / 1 | 0.999 [1.00, 1.00] |
| gemini-3.5-flash-lite | 60 | 26.7% | 25.0% | +1.7 [-8.3, +11.7] | +6.4 [+0.0, +12.8] | 5 / 4 | 0.999 [1.00, 1.00] |

Qwen2.5-1.5B on the **unmodified original** prompts: refusal 73.9%; benchmark-styled − original -19.8 [-22.3, -17.4] pp; user-styled − original -5.9 [-7.9, -3.9] pp.

### By harm level (Δ = benchmark − user, pp, 95% CI)

| model | harmful | borderline | benign |
|---|---|---|---|
| qwen2.5-1.5b | -14.7 [-18.2, -11.3] | -21.5 [-25.9, -16.7] | +3.0 [-2.6, +8.6] |
| gemini-3.1-flash-lite | – | – | – |
| gemini-3.5-flash-lite | – | – | – |

### By benchmark format (Δ pp, harmful + borderline + benign pooled)

| model | numbered/rubric | MCQ | scenario | harness |
|---|---|---|---|---|
| qwen2.5-1.5b | +5.2 [+1.2, +9.5] | -16.0 [-21.7, -10.0] | -23.8 [-28.8, -18.8] | -21.2 [-26.0, -16.4] |
| gemini-3.1-flash-lite | – | – | – | – |
| gemini-3.5-flash-lite | – | – | – | – |

### Verbalised evaluation awareness (regex, % of responses)

| model | bench-styled | user-styled |
|---|---|---|
| qwen2.5-1.5b | 0.1% | 0.1% |
| gemini-3.1-flash-lite | 1.7% | 0.0% |
| gemini-3.5-flash-lite | 0.0% | 0.0% |

### Label validation and quality (Gemini judge, gemini-3-flash-preview)

On 168 judged responses, the local refusal label agrees with the judge's *full refusal* on 85.1% (κ = 0.69) and with *any refusal (full or partial)* on 81.0% (κ = 0.61). Judge-flagged verbalised awareness 2.4% vs regex 0.0%.

| subset | n | local refuse | judge full | judge any | agreement (any) |
|---|---|---|---|---|---|
| bench:harness | 28 | 28.6% | 32.1% | 35.7% | 71.4% |
| bench:mcq | 16 | 50.0% | 62.5% | 62.5% | 87.5% |
| bench:numbered | 16 | 25.0% | 31.2% | 43.8% | 81.2% |
| bench:scenario | 25 | 24.0% | 40.0% | 48.0% | 76.0% |
| user | 83 | 44.6% | 43.4% | 48.2% | 84.3% |

| model (judged sample) | pairs | Δ judge full refusal | Δ judge any refusal | Δ local label |
|---|---|---|---|---|
| gemini-3.1-flash-lite | 25 | +8.0 [-8.0, +24.0] | +12.0 [-4.0, +28.0] | +4.0 [-8.0, +16.0] |
| gemini-3.5-flash-lite | 28 | +14.3 [+3.6, +28.6] | +14.3 [+0.0, +28.6] | +10.7 [-3.6, +25.0] |
| qwen2.5-1.5b | 25 | -20.0 [-40.0, +0.0] | -24.0 [-44.0, -4.0] | -44.0 [-64.0, -24.0] |

| model | pairs | quality bench (0-5) | quality user | Δ [95% CI] |
|---|---|---|---|---|
| gemini-3.1-flash-lite | 25 | 3.36 | 3.32 | +0.04 [-0.84, +0.92] |
| gemini-3.5-flash-lite | 28 | 2.64 | 3.29 | -0.64 [-1.39, +0.04] |
| qwen2.5-1.5b | 25 | 1.96 | 0.96 | +1.00 [+0.24, +1.76] |
<!-- TABLES:END -->

## Limitations

- **The cues are overt by design.** A TF-IDF bag-of-words model separates the two variants perfectly (AUROC 1.00), so the Gemini models' AUROC of 0.999 shows they notice the cues. It is not evidence of subtle situational awareness.
- **Templated variants.** Twelve benchmark templates and a rule-based "user" rewrite are less diverse than real benchmark items or real users. MCQ changes the task shape: a model can "refuse" by picking (C), and the scorer counts that as a refusal. Results are also reported excluding MCQ (Δ = −13.3 pp [−16.0, −10.5] for Qwen).
- **Refusal labels come from a classifier** (`distilroberta-base-rejection-v1` plus an MCQ rule). It misses some soft refusals, e.g. "Option (B) … this is unethical". Agreement with an LLM judge on a 180-response sample is reported in the tables below when available. Neither is ground truth.
- **Small API samples.** The Gemini key is free tier (20–500 requests/day/model, shared by six projects running at once), so the Gemini arms have 60 pairs each. Their CIs are wide.
- **Coverage.** Qwen2.5-1.5B covers 1,680 of the 2,219 pairs, a random subset because the pair order is shuffled. Compute ran out on a shared, heavily loaded laptop.

## What didn't get done

- **Qwen2.5-0.5B and SmolLM2-1.7B** arms, and Qwen2.5-1.5B on the last 539 pairs: the code path exists (`run_local.sh`), but they were not run before the deadline.
- **Probe and steering**: `06_probe.py` and `04_local.py steer` are implemented. Their results appear in the tables below only if the activation run finished. Otherwise they are pending.
- **Gemini-rewritten variants** (`02_variants.py`): built and piloted, but not run at scale because of API quota.
- **Gemini-judged compliance quality**: limited to the judge sample.

## Reproduce

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
# download raw datasets to data/raw (URLs in scripts/01_source.py docstring / below), then:
./run_all.sh
```

`run_all.sh` runs sourcing, then variants, then the local pipeline (`run_local.sh`), then the optional Gemini subject and judge sample, then analysis and site export. Every API response is cached in `data/cache/*.jsonl`, so reruns are free. Local GPU steps take `~/ai-safety-portfolio/_shared/gpulock.sh` if present.

Raw data URLs:
- HarmBench: `https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv`
- XSTest: `https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv`
- Do-Not-Answer: `https://huggingface.co/api/datasets/LibrAI/do-not-answer/parquet/default/train/0.parquet`
- OR-Bench-Hard-1K: `https://huggingface.co/api/datasets/bench-llm/or-bench/parquet/or-bench-hard-1k/train/0.parquet`
- WildChat: `https://huggingface.co/api/datasets/allenai/WildChat/parquet/default/train/0.parquet`

## Responsible release

The repo and site publish the prompts (already public in their source benchmarks), the variants, refusal labels and aggregate numbers. **Model responses to prompts labelled harmful are not published**; the viewer shows only their labels. Responses to borderline and benign prompts are truncated to 400 characters. No model weights are redistributed, and Hugging Face uploads were skipped (no token).
