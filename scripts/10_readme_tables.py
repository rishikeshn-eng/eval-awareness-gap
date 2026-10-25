"""Render results tables from results/summary.json into README.md between
<!-- TABLES:START --> and <!-- TABLES:END --> (so README numbers always match the data)."""
import json
import re

from common import RESULTS, ROOT

S = json.load(open(RESULTS / "summary.json"))


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}%"


def pp(d):
    if not d or d[0] != d[0]:
        return "–"
    return f"{100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]"


def auc(a):
    return "–" if not a else f"{a[0]:.3f} [{a[1]:.2f}, {a[2]:.2f}]"


out = ["### Refusal: benchmark-styled vs user-styled (all pairs)", "",
       "| model | pairs | refusal (bench) | refusal (user) | Δ pp [95% CI] | Δ excl. MCQ | discordant bench-only / user-only | AUROC eval-vs-user |",
       "|---|---|---|---|---|---|---|---|"]
for m, s in S["models"].items():
    o, x = s["overall"], s["overall_excl_mcq"]
    out.append(f"| {m} | {o['n_pairs']} | {pct(o['refuse']['eval'][0])} | {pct(o['refuse']['natural'][0])} | "
               f"{pp(o['refuse']['delta'])} | {pp(x['refuse']['delta'])} | {o['discordant']['refuse_only_eval']} / "
               f"{o['discordant']['refuse_only_natural']} | {auc(o.get('classify_auroc'))} |")
main = S["models"].get("qwen2.5-1.5b")
if main and "original" in main["overall"]["refuse"]:
    r = main["overall"]["refuse"]
    out += ["", f"Qwen2.5-1.5B on the **unmodified original** prompts: refusal {pct(r['original'][0])}; "
                f"benchmark-styled − original {pp(r['delta_eval_vs_original'])} pp; user-styled − original "
                f"{pp(r['delta_natural_vs_original'])} pp."]

out += ["", "### By harm level (Δ = benchmark − user, pp, 95% CI)", "",
        "| model | harmful | borderline | benign |", "|---|---|---|---|"]
for m, s in S["models"].items():
    cells = [pp(s["by_harm_level"].get(h, {}).get("refuse", {}).get("delta")) for h in ("harmful", "borderline", "benign")]
    out.append(f"| {m} | " + " | ".join(cells) + " |")

out += ["", "### By benchmark format (Δ pp, harmful + borderline + benign pooled)", "",
        "| model | numbered/rubric | MCQ | scenario | harness |", "|---|---|---|---|---|"]
for m, s in S["models"].items():
    cells = [pp(s["by_format"].get(f, {}).get("refuse", {}).get("delta")) for f in ("numbered", "mcq", "scenario", "harness")]
    out.append(f"| {m} | " + " | ".join(cells) + " |")

out += ["", "### Verbalised evaluation awareness (regex, % of responses)", "",
        "| model | bench-styled | user-styled |", "|---|---|---|"]
for m, s in S["models"].items():
    a = s["overall"]["aware"]
    out.append(f"| {m} | {pct(a['eval'][0])} | {pct(a['natural'][0])} |")

if "judge_validation" in S:
    v = S["judge_validation"]
    out += ["", "### Label validation and quality (Gemini judge, gemini-3-flash-preview)", "",
            f"On {v['n']} judged responses, the local refusal label agrees with the judge's *full refusal* on "
            f"{pct(v['accuracy_vs_full'])} (κ = {v['kappa_vs_full']:.2f}) and with *any refusal (full or partial)* on "
            f"{pct(v['accuracy_vs_any'])} (κ = {v['kappa_vs_any']:.2f}). Judge-flagged verbalised awareness "
            f"{pct(v['judge_aware_rate'])} vs regex {pct(v['regex_aware_rate'])}.", "",
            "| subset | n | local refuse | judge full | judge any | agreement (any) |", "|---|---|---|---|---|---|"]
    for k, x in v.get("by_variant_format", {}).items():
        out.append(f"| {k} | {x['n']} | {pct(x['local_refuse'])} | {pct(x['judge_full'])} | {pct(x['judge_any'])} | {pct(x['accuracy_vs_any'])} |")
    if S.get("judge_deltas"):
        out += ["", "| model (judged sample) | pairs | Δ judge full refusal | Δ judge any refusal | Δ local label |", "|---|---|---|---|---|"]
        for m, d in S["judge_deltas"].items():
            n = (d.get("judge_refuse") or d.get("local_refuse") or {}).get("n_pairs", "–")
            out.append(f"| {m} | {n} | {pp(d.get('judge_refuse', {}).get('delta'))} | "
                       f"{pp(d.get('judge_refuse_any', {}).get('delta'))} | {pp(d.get('local_refuse', {}).get('delta'))} |")
    if S.get("quality_sample"):
        out += ["", "| model | pairs | quality bench (0-5) | quality user | Δ [95% CI] |", "|---|---|---|---|---|"]
        for m, q in S["quality_sample"].items():
            out.append(f"| {m} | {q['n_pairs']} | {q['eval'][0]:.2f} | {q['natural'][0]:.2f} | "
                       f"{q['delta'][0]:+.2f} [{q['delta'][1]:+.2f}, {q['delta'][2]:+.2f}] |")

if "probe" in S:
    p = S["probe"]
    best = max(p["auroc_cv"])
    out += ["", "### Probe and steering (Qwen2.5-1.5B)", "",
            f"Linear probe AUROC (grouped 5-fold): best {best:.3f} at layer {p['best_layer']}; steering layer "
            f"{p['steer_layer']} (difference-of-means held-out AUROC {p['diffmeans_test_auroc']:.3f}). "
            f"Leave-one-format-out AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in p["lofo"].items()) +
            f". TF-IDF text baseline: {S['tfidf_baseline_auroc']:.3f}. ‖μ_eval − μ_user‖ = {p['direction_norm']:.1f} "
            f"vs mean ‖h‖ = {p['mean_resid_norm']:.1f}."]
if "steering" in S:
    st = S["steering"]

    def key(c):
        return (c.startswith("rand"), float(re.sub(r"[a-z]+", "", c)))

    out += ["", "| condition | refusal bench | Δ vs α=0 | refusal user | Δ vs α=0 | degenerate outputs |", "|---|---|---|---|---|---|"]
    for c in sorted(st, key=key):
        e, n = st[c].get("eval", {}), st[c].get("natural", {})
        label = c.replace("dir", "direction α=").replace("rand", "random α=").replace("none+0.0", "none (α=0)")
        deg = (e.get("degenerate", 0) + n.get("degenerate", 0)) / 2
        out.append(f"| {label} | {pct(e.get('refuse', [None])[0])} | {pp(e.get('delta_vs_none'))} | "
                   f"{pct(n.get('refuse', [None])[0])} | {pp(n.get('delta_vs_none'))} | {pct(deg)} |")

text = "\n".join(out)
readme = (ROOT / "README.md").read_text()
block = f"<!-- TABLES:START -->\n{text}\n<!-- TABLES:END -->"
if "<!-- TABLES:START -->" in readme:
    readme = re.sub(r"<!-- TABLES:START -->.*<!-- TABLES:END -->", lambda _: block, readme, flags=re.S)
else:
    readme = readme.replace("<!-- DETAILS -->", block)
(ROOT / "README.md").write_text(readme)
print(text)
