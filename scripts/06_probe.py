"""Linear probe for eval-styled vs deployment-styled prompts on Qwen residual activations.

Splits are by PAIR (both variants of a base prompt land in the same fold), so the probe can't win by
memorising content. Reports per-layer AUROC (5-fold grouped CV), and a leave-one-format-out test
(train on 3 eval formats, test on the 4th) to check the direction is not just one template.
Saves the difference-of-means direction at the best layer, fit on a train split, for steering.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, RESULTS, read_jsonl

X = np.load(DATA / "qwen_acts.npy").astype(np.float32)  # [N, L+1, D]
meta = read_jsonl(DATA / "qwen_acts_meta.jsonl")
pairs = {p["id"]: p for p in read_jsonl(DATA / "pairs.jsonl")}
y = np.array([m["variant"] == "eval" for m in meta], int)
groups = np.array([m["id"] for m in meta])
fmt = np.array([pairs[m["id"]]["eval_format"] for m in meta])
n_layers = X.shape[1]


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


res = {"layers": [], "auroc_cv": [], "acc_cv": [], "lofo": {}}
gkf = GroupKFold(n_splits=5)
for L in range(1, n_layers):
    scores = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = probe().fit(X[tr, L], y[tr])
        scores[te] = m.predict_proba(X[te, L])[:, 1]
    res["layers"].append(L)
    res["auroc_cv"].append(float(roc_auc_score(y, scores)))
    res["acc_cv"].append(float(((scores > 0.5) == y).mean()))
    print(f"layer {L:2d} AUROC {res['auroc_cv'][-1]:.3f}", flush=True)

best = int(res["layers"][int(np.argmax(res["auroc_cv"]))])
# pick a mid-depth layer for steering: best AUROC among the middle third (steering at the last
# layers mostly changes surface tokens)
mid = [l for l in res["layers"] if n_layers // 3 <= l <= 2 * n_layers // 3]
steer_layer = max(mid, key=lambda l: res["auroc_cv"][l - 1])
res["best_layer"], res["steer_layer"] = best, steer_layer

# leave-one-eval-format-out: train on pairs of 3 formats, test on pairs of the held-out format
for f in sorted(set(fmt)):
    tr, te = fmt != f, fmt == f
    m = probe().fit(X[tr, steer_layer], y[tr])
    res["lofo"][f] = float(roc_auc_score(y[te], m.predict_proba(X[te, steer_layer])[:, 1]))
print("LOFO", res["lofo"])

# steering direction: difference of means on a 70% pair split; held-out 30% used for steering eval
ids = np.array(sorted(set(groups)))
rng = np.random.default_rng(0)
test_ids = set(rng.choice(ids, size=int(0.3 * len(ids)), replace=False))
tr = np.array([g not in test_ids for g in groups])
Xl = X[:, steer_layer]
mu_e, mu_n = Xl[tr & (y == 1)].mean(0), Xl[tr & (y == 0)].mean(0)
direction = mu_e - mu_n
te = ~tr
proj = Xl[te] @ (direction / np.linalg.norm(direction))
res["diffmeans_test_auroc"] = float(roc_auc_score(y[te], proj))
res["direction_norm"] = float(np.linalg.norm(direction))
res["mean_resid_norm"] = float(np.linalg.norm(Xl, axis=1).mean())
np.savez(DATA / "steer_direction.npz", layer=steer_layer, direction=direction.astype(np.float32),
         test_ids=np.array(sorted(test_ids)))
json.dump(res, open(RESULTS / "probe.json", "w"), indent=1)
print("best", best, "steer", steer_layer, "diffmeans test AUROC", res["diffmeans_test_auroc"],
      "|dir|", res["direction_norm"], "mean |h|", res["mean_resid_norm"])
