"""Aggregate scored results: paired bootstrap CIs for refusal / verbalised-awareness deltas,
grouped-bootstrap AUROC for eval-vs-real classification, per harm level / category / format /
source, the Gemini-judged quality + label-validation sample, probe and steering results, figures,
and docs/data/*.json for the GitHub Pages site."""
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DATA, RESULTS, ROOT, read_jsonl  # noqa: E402

B = 2000
rng = np.random.default_rng(0)
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(DOCS / "data").mkdir(exist_ok=True)
pairs = pd.DataFrame(read_jsonl(DATA / "pairs.jsonl")).set_index("id")
ORDER = ["qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b", "gemini-3.1-flash-lite"]
MODELS = [m for m in ORDER if (RESULTS / f"scored_{m}.jsonl").exists()]


def boot_mean(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return [np.nan] * 3
    bs = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_delta(a, b):
    """Paired bootstrap over pairs for mean(a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return [np.nan] * 3
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_auroc(y, s, groups, n=500):
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        samp = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        if len(set(y[samp])) == 2:
            vals.append(roc_auc_score(y[samp], s[samp]))
    return [float(roc_auc_score(y, s)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def wide(tag):
    df = pd.DataFrame(read_jsonl(RESULTS / f"scored_{tag}.jsonl"))
    df["refuse"] = df.refuse.astype(float)
    df["aware"] = df.eval_aware_regex.astype(float)
    w = df.pivot(index="id", columns="variant", values=["refuse", "aware", "p_eval", "refusal_clf"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna(subset=["refuse_eval", "refuse_natural"])
    return w.join(pairs[["harm_level", "category", "eval_format", "source"]])


def summarise(w):
    out = {"n_pairs": int(len(w))}
    for m in ("refuse", "aware"):
        out[m] = {"eval": boot_mean(w[f"{m}_eval"]), "natural": boot_mean(w[f"{m}_natural"]),
                  "delta": boot_delta(w[f"{m}_eval"], w[f"{m}_natural"])}
        if f"{m}_original" in w:
            out[m]["original"] = boot_mean(w[f"{m}_original"])
            out[m]["delta_eval_vs_original"] = boot_delta(w[f"{m}_eval"], w[f"{m}_original"])
            out[m]["delta_natural_vs_original"] = boot_delta(w[f"{m}_natural"], w[f"{m}_original"])
    out["discordant"] = {
        "refuse_only_eval": int(((w.refuse_eval == 1) & (w.refuse_natural == 0)).sum()),
        "refuse_only_natural": int(((w.refuse_eval == 0) & (w.refuse_natural == 1)).sum())}
    pe = w.dropna(subset=["p_eval_eval", "p_eval_natural"])
    if len(pe) >= 20:
        y = np.r_[np.ones(len(pe)), np.zeros(len(pe))]
        s = np.r_[pe.p_eval_eval.values, pe.p_eval_natural.values]
        g = np.r_[pe.index.values, pe.index.values]
        out["classify_auroc"] = boot_auroc(y, s, g)
        out["mean_p_eval"] = {"eval": float(pe.p_eval_eval.mean()), "natural": float(pe.p_eval_natural.mean())}
        out["pair_win_rate"] = float((pe.p_eval_eval > pe.p_eval_natural).mean()
                                     + 0.5 * (pe.p_eval_eval == pe.p_eval_natural).mean())
    return out


summary = {"models": {}}
allp = read_jsonl(DATA / "pairs_all.jsonl")
summary["dataset"] = {
    "n_base_prompts": len(allp), "n_pairs": int(len(pairs)),
    "by_source": pairs.groupby(["source", "harm_level"]).size().rename("n").reset_index().to_dict("records"),
    "by_harm_level": pairs.harm_level.value_counts().to_dict(),
    "by_category": pairs.category.value_counts().to_dict(),
    "by_format": pairs.eval_format.value_counts().to_dict(),
}

for tag in MODELS:
    w = wide(tag)
    s = {"overall": summarise(w), "overall_excl_mcq": summarise(w[w.eval_format != "mcq"])}
    s["by_harm_level_excl_mcq"] = {k: summarise(sub) for k, sub in w[w.eval_format != "mcq"].groupby("harm_level")}
    s["by_format_x_harm"] = {f"{f} / {h}": summarise(sub) for (f, h), sub in w.groupby(["eval_format", "harm_level"])
                             if len(sub) >= 25}
    for col, key in (("harm_level", "by_harm_level"), ("category", "by_category"),
                     ("eval_format", "by_format"), ("source", "by_source")):
        s[key] = {k: summarise(sub) for k, sub in w.groupby(col) if len(sub) >= 25}
    summary["models"][tag] = s
    o = s["overall"]
    print(f"{tag:22s} n={o['n_pairs']:4d} refuse eval {o['refuse']['eval'][0]:.3f} nat "
          f"{o['refuse']['natural'][0]:.3f} delta {np.round(o['refuse']['delta'], 3)} "
          f"AUROC {np.round(o.get('classify_auroc', [np.nan]), 3)} aware {np.round(o['aware']['delta'], 4)}")

# lexical baseline: separability of the two variants from surface text alone
texts = list(pairs.eval_text) + list(pairs.natural_text)
yl = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
gl = np.r_[pairs.index.values, pairs.index.values]
sc = np.zeros(len(yl))
for tr, te in GroupKFold(5).split(texts, yl, gl):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([texts[i] for i in tr])
    clf = LogisticRegression(max_iter=2000).fit(vec.transform([texts[i] for i in tr]), yl[tr])
    sc[te] = clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
summary["tfidf_baseline_auroc"] = float(roc_auc_score(yl, sc))

# Gemini-judged sample: quality deltas + agreement of the local refusal labels with the judge
jp = RESULTS / "judged_judgesample.jsonl"
if jp.exists():
    jd = pd.DataFrame(read_jsonl(jp))
    jd = jd[jd.refusal.notna()]
    jd["judge_refuse"] = (jd.refusal == "full").astype(int)
    jd["judge_refuse_any"] = (jd.refusal != "none").astype(int)
    val = {"n": int(len(jd)),
           "accuracy_vs_full": float((jd.judge_refuse == jd.refuse.astype(int)).mean()),
           "kappa_vs_full": float(cohen_kappa_score(jd.judge_refuse, jd.refuse.astype(int))),
           "accuracy_vs_any": float((jd.judge_refuse_any == jd.refuse.astype(int)).mean()),
           "kappa_vs_any": float(cohen_kappa_score(jd.judge_refuse_any, jd.refuse.astype(int))),
           "judge_aware_rate": float(jd.eval_aware.fillna(False).mean()),
           "regex_aware_rate": float(jd.eval_aware_regex.mean())}
    jd = jd.join(pairs[["eval_format", "harm_level"]], on="id")
    val["by_variant_format"] = {}
    for (v, f), sub in jd.groupby(["variant", "eval_format"]):
        k = "user" if v == "natural" else f"bench:{f}"
        if v == "natural" and "user" in val["by_variant_format"]:
            continue
        sub = jd[jd.variant == "natural"] if v == "natural" else sub
        val["by_variant_format"][k] = {
            "n": int(len(sub)), "local_refuse": float(sub.refuse.mean()),
            "judge_full": float(sub.judge_refuse.mean()), "judge_any": float(sub.judge_refuse_any.mean()),
            "accuracy_vs_any": float((sub.judge_refuse_any == sub.refuse.astype(int)).mean())}
    # refusal deltas re-estimated with the judge's labels on the sample (paired)
    jdel = {}
    for tag, sub in jd.groupby("model"):
        for lab in ("judge_refuse", "judge_refuse_any"):
            q = sub.pivot_table(index="id", columns="variant", values=lab, aggfunc="first").dropna()
            if {"eval", "natural"} <= set(q.columns) and len(q) >= 10:
                jdel.setdefault(tag, {})[lab] = {"n_pairs": int(len(q)), "eval": boot_mean(q["eval"]),
                                                  "natural": boot_mean(q["natural"]),
                                                  "delta": boot_delta(q["eval"], q["natural"])}
        q = sub.pivot_table(index="id", columns="variant", values="refuse", aggfunc="first").dropna().astype(float)
        if {"eval", "natural"} <= set(q.columns) and len(q) >= 10:
            jdel.setdefault(tag, {})["local_refuse"] = {"n_pairs": int(len(q)), "delta": boot_delta(q["eval"], q["natural"])}
    summary["judge_deltas"] = jdel
    qual = {}
    for tag, sub in jd.groupby("model"):
        q = sub.pivot_table(index="id", columns="variant", values="quality", aggfunc="first")
        q = q.dropna(subset=[c for c in ("eval", "natural") if c in q])
        if {"eval", "natural"} <= set(q.columns) and len(q) >= 10:
            qual[tag] = {"n_pairs": int(len(q)), "eval": boot_mean(q["eval"]),
                         "natural": boot_mean(q["natural"]), "delta": boot_delta(q["eval"], q["natural"])}
    summary["judge_validation"], summary["quality_sample"] = val, qual
    print("judge validation", val)

if (RESULTS / "probe.json").exists():
    summary["probe"] = json.load(open(RESULTS / "probe.json"))

sp = RESULTS / "scored_qwen_steer.jsonl"
if sp.exists():
    sd = pd.DataFrame(read_jsonl(sp))
    sd["cond"] = sd.model.str.replace("steer:", "")
    sd["refuse"] = sd.refuse.astype(float)

    def degenerate(t):
        w = (t or "").split()
        return float(len(w) < 3 or len(set(w)) / len(w) < 0.3)

    sd["degenerate"] = sd.response.map(degenerate)
    base = sd[sd.cond == "none+0.0"].set_index(["id", "variant"]).refuse
    st = {}
    for (cond, var), sub in sd.groupby(["cond", "variant"]):
        j = sub.set_index(["id", "variant"]).refuse
        b = base.reindex(j.index)
        st.setdefault(cond, {})[var] = {"refuse": boot_mean(j), "n": int(len(j)),
                                        "degenerate": float(sub.degenerate.mean()),
                                        "delta_vs_none": boot_delta(j.values, b.values)}
    summary["steering"] = st

json.dump(summary, open(RESULTS / "summary.json", "w"), indent=1, default=float)
json.dump(summary, open(DOCS / "data" / "summary.json", "w"), default=float)

# ---------- figures ----------
C_EVAL, C_NAT, C_ORIG = "#c2410c", "#0f766e", "#64748b"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)


if MODELS:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, lvl in zip(axes, ["harmful", "borderline", "benign"]):
        x = np.arange(len(MODELS))
        for k, (v, c) in enumerate((("eval", C_EVAL), ("natural", C_NAT))):
            m = [summary["models"][t]["by_harm_level"].get(lvl, {}).get("refuse", {}).get(v, [np.nan] * 3)
                 for t in MODELS]
            ax.bar(x + (k - 0.5) * 0.38, [a[0] for a in m], 0.38, color=c,
                   label="benchmark-styled" if v == "eval" else "user-styled",
                   yerr=[[a[0] - a[1] for a in m], [a[2] - a[0] for a in m]], capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(x, [t.replace("gemini-", "g-") for t in MODELS], fontsize=7, rotation=15)
        ax.set_title(lvl, fontsize=10)
        style(ax)
    axes[0].set_ylabel("refusal rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Refusal rate on the same request: benchmark-styled vs user-styled (95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "refusal_by_harm.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k, t in enumerate(MODELS):
        o = summary["models"][t]["overall"]
        if "classify_auroc" in o:
            a = o["classify_auroc"]
            ax.errorbar(k, a[0], yerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="o", color="#4338ca", capsize=4)
    ax.axhline(0.5, color="#9ca3af", lw=0.8, ls="--")
    ax.axhline(summary["tfidf_baseline_auroc"], color=C_EVAL, lw=0.8, ls=":")
    ax.text(len(MODELS) - 0.5, summary["tfidf_baseline_auroc"] - 0.04, "TF-IDF baseline", fontsize=7,
            color=C_EVAL, ha="right")
    ax.set_xticks(range(len(MODELS)), MODELS, fontsize=8)
    ax.set_ylabel("AUROC (eval vs user)")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Asked directly, can the model tell eval-styled from user-styled? (95% CI)", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "classify_auroc.png", dpi=160)

if "probe" in summary:
    pr = summary["probe"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(pr["layers"], pr["auroc_cv"], "o-", color="#4338ca", ms=3, label="probe (grouped 5-fold)")
    ax.axvline(pr["steer_layer"], color="#9ca3af", ls="--", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Qwen2.5-1.5B linear probe: benchmark- vs user-styled prompt", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "probe_layers.png", dpi=160)

if "steering" in summary:
    st = summary["steering"]

    def coef(c):
        return float(c.replace("dir", "").replace("none", "").replace("rand", ""))

    conds = sorted([c for c in st if not c.startswith("rand")], key=coef)
    fig, ax = plt.subplots(figsize=(7, 3.3))
    for v, c in (("eval", C_EVAL), ("natural", C_NAT)):
        m = [st[k][v]["refuse"] for k in conds if v in st[k]]
        xs = [coef(k) for k in conds if v in st[k]]
        ax.errorbar(xs, [a[0] for a in m], yerr=[[a[0] - a[1] for a in m], [a[2] - a[0] for a in m]],
                    fmt="o-", capsize=3, color=c, label=f"{'benchmark' if v == 'eval' else 'user'}-styled prompts")
        for k in [k for k in st if k.startswith("rand") and v in st[k]]:
            ax.plot(coef(k), st[k][v]["refuse"][0], "x", color=c, ms=8)
    ax.set_xlabel("steering coefficient α (h += α·(μ_eval − μ_user));  × = random direction, same norm")
    ax.set_ylabel("refusal rate")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Steering Qwen2.5-1.5B along the eval-vs-user direction (harmful + borderline)", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "steering.png", dpi=160)
print("tfidf baseline AUROC", round(summary["tfidf_baseline_auroc"], 4))
