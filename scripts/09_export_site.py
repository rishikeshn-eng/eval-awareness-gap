"""Export the paired-example viewer data for the GitHub Pages site (docs/data/examples.json).

Responses to prompts labelled harmful are never published: only their refusal labels are kept.
Responses to borderline / benign prompts are truncated to 400 characters.
"""
import json

from common import DATA, RESULTS, ROOT, read_jsonl

TAGS = ["qwen2.5-1.5b", "qwen2.5-0.5b", "smollm2-1.7b", "gemini-3.1-flash-lite"]
SNIP = 400

pairs = read_jsonl(DATA / "pairs.jsonl")
scored = {}
for tag in TAGS:
    p = RESULTS / f"scored_{tag}.jsonl"
    if p.exists():
        scored[tag] = {(r["id"], r["variant"]): r for r in read_jsonl(p)}

ex = []
for p in pairs:
    item = {k: p[k] for k in ("id", "source", "harm_level", "category", "eval_format", "text",
                              "eval_text", "natural_text")}
    item["models"] = {}
    for tag, d in scored.items():
        m = {}
        for v in ("eval", "natural", "original"):
            r = d.get((p["id"], v))
            if r is None:
                continue
            resp = (r.get("response") or "").strip()
            m[v] = {"refuse": bool(r["refuse"]), "p_eval": None if r.get("p_eval") is None else round(r["p_eval"], 3),
                    "aware": bool(r["eval_aware_regex"]),
                    "resp": "[redacted: response to a harmful prompt]" if p["harm_level"] == "harmful"
                    else (resp[:SNIP] + ("…" if len(resp) > SNIP else ""))}
        if m:
            item["models"][tag] = m
    ex.append(item)
(ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)
json.dump(ex, open(ROOT / "docs" / "data" / "examples.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("examples", len(ex), "models", list(scored))
