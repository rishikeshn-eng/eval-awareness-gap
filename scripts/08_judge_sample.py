"""Build the Gemini-judged validation sample: K pairs per model (both variants), stratified by harm
level, drawn from the locally scored responses. The judge (05_judge.py, gemini-3-flash-preview)
gives refusal / quality / eval-awareness labels, which 07_analyze.py uses to (a) validate the local
refusal classifier and (b) estimate the compliance-quality delta.
Usage: python 08_judge_sample.py [K]   then   python 05_judge.py judgesample
"""
import random
import sys

from common import DATA, RESULTS, read_jsonl, write_jsonl

K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TAGS = ["qwen2.5-1.5b", "qwen2.5-0.5b", "smollm2-1.7b", "gemini-3.1-flash-lite"]
pairs = {p["id"]: p for p in read_jsonl(DATA / "pairs.jsonl")}
rng = random.Random(0)
out = []
for tag in TAGS:
    path = RESULTS / f"scored_{tag}.jsonl"
    if not path.exists():
        continue
    rows = [r for r in read_jsonl(path) if r["variant"] in ("eval", "natural")]
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], {})[r["variant"]] = r
    ids = [i for i, d in by_id.items() if len(d) == 2]
    pick = []
    for lvl, frac in (("harmful", 0.4), ("borderline", 0.4), ("benign", 0.2)):
        pool = sorted(i for i in ids if pairs[i]["harm_level"] == lvl)
        pick += rng.sample(pool, min(len(pool), round(K * frac)))
    for i in pick:
        out += [by_id[i]["eval"], by_id[i]["natural"]]
rng.shuffle(out)  # mix models, variants and harm levels across judge batches
write_jsonl(RESULTS / "responses_judgesample.jsonl", out)
print("judge sample", len(out))
