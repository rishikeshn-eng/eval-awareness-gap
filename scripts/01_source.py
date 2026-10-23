"""Assemble the base prompt pool from public safety datasets.

harm_level: harmful | borderline | benign
  HarmBench (standard behaviours)       -> harmful
  Do-Not-Answer                         -> harmful
  XSTest unsafe / safe                  -> harmful / borderline (safe prompts that look unsafe)
  OR-Bench-Hard-1K                      -> borderline (seemingly-toxic but answerable)
  WildChat first turns (English)        -> benign (non-toxic) / harmful (OpenAI-moderation toxic)
SORRY-Bench is gated on Hugging Face (needs an accepted licence + token) and is not used.
"""
import random
import re

import pandas as pd

from common import DATA, write_jsonl

RAW = DATA / "raw"
rng = random.Random(0)

# Coarse cross-dataset categories for the per-category breakdown.
CATS = [
    ("cyber", r"cyber|hack|malware|software|computer|exploit|password"),
    ("weapons_violence", r"weapon|violen|chemical|biolog|bomb|explosive|harm to|kill|terror|physical"),
    ("illegal_fraud", r"illegal|crime|fraud|financial|drug|theft|scam|deception|unethical|copyright"),
    ("hate_harassment", r"hate|harass|discriminat|toxic|offensive|insult|stereotyp|bias|racis"),
    ("sexual", r"sex|adult|explicit"),
    ("misinformation", r"misinformation|disinformation|false|mislead|rumou?r|propaganda"),
    ("privacy", r"privacy|personal|private|surveillance|sensitive information|tracking"),
    ("self_harm_health", r"self.?harm|suicide|mental|medical|health"),
]


def coarse(native):
    s = (native or "").lower()
    for name, pat in CATS:
        if re.search(pat, s):
            return name
    return "other"


def clean(t):
    return re.sub(r"\s+", " ", str(t)).strip()


rows = []

hb = pd.read_csv(RAW / "harmbench.csv")
hb = hb[hb.FunctionalCategory == "standard"]
for _, r in hb.iterrows():
    rows.append(dict(source="harmbench", harm_level="harmful", native_category=r.SemanticCategory,
                     text=clean(r.Behavior)))

dna = pd.read_parquet(RAW / "dna.parquet")
dna = dna.sample(n=min(700, len(dna)), random_state=0)
for _, r in dna.iterrows():
    # DNA's "Human-Chatbot Interaction Harms" are mostly anthropomorphism/advice questions
    # ("do you feel lonely?"): treat as borderline rather than harmful
    lvl = "borderline" if r.risk_area.startswith("Human-Chatbot") else "harmful"
    rows.append(dict(source="do_not_answer", harm_level=lvl,
                     native_category=f"{r.risk_area} / {r.types_of_harm}", text=clean(r.question)))

xs = pd.read_csv(RAW / "xstest.csv")
for _, r in xs.iterrows():
    unsafe = str(r.label).strip() == "unsafe"
    rows.append(dict(source="xstest", harm_level="harmful" if unsafe else "borderline",
                     native_category=str(r.type), text=clean(r.prompt)))

orb = pd.read_parquet(RAW / "orbench_hard.parquet")
orb = orb.sample(n=min(450, len(orb)), random_state=0)
for _, r in orb.iterrows():
    rows.append(dict(source="or_bench_hard", harm_level="borderline", native_category=r.category,
                     text=clean(r.prompt)))

wc = pd.read_parquet(RAW / "wildchat0.parquet", columns=["conversation", "language", "toxic", "turn"])
wc = wc[(wc.language == "English")]
benign, toxic = [], []
seen = set()
for conv, tox in zip(wc.conversation, wc.toxic):
    t = clean(conv[0]["content"])
    # single self-contained requests of moderate length; skip code dumps, roleplay set-ups, dupes
    if not (25 <= len(t) <= 400) or t.lower() in seen or "```" in t or t.count("\n") > 3:
        continue
    if re.search(r"\b(continue|rewrite this|translate|story|roleplay|pretend|act as|DAN)\b", t, re.I):
        continue
    seen.add(t.lower())
    (toxic if tox else benign).append(t)
rng.shuffle(benign)
rng.shuffle(toxic)
for t in benign[:420]:
    rows.append(dict(source="wildchat", harm_level="benign", native_category="benign_general", text=t))
for t in toxic[:120]:
    rows.append(dict(source="wildchat_toxic", harm_level="harmful", native_category="wildchat toxic",
                     text=t))

for r in rows:
    r["category"] = "benign" if r["harm_level"] == "benign" else coarse(r["native_category"])
# XSTest-safe prompts are deliberately benign-but-scary; tag their category from the text
for r in rows:
    if r["source"] == "xstest" and r["category"] == "other":
        r["category"] = "xstest_" + str(r["native_category"]).split("_")[0]

seen, out = set(), []
for r in rows:
    k = r["text"].lower()
    if k in seen or len(r["text"]) < 10:
        continue
    seen.add(k)
    out.append(r)
rng.shuffle(out)
for i, r in enumerate(out):
    r["id"] = f"p{i:05d}"
write_jsonl(DATA / "prompts.jsonl", out)
df = pd.DataFrame(out)
print(len(df))
print(df.groupby(["source", "harm_level"]).size())
print(df.category.value_counts())
