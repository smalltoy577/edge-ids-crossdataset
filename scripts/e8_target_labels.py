"""E8: does a small number of labelled target flows restore detection?

For every target dataset and label budget n (100, 1,000, 10,000 labelled flows
from the target's training split) we compare three regimes:
  target-only    : train on the n target flows alone
  pooled         : 100,000 flows of one source dataset + the n target flows
  pooled-weighted: as pooled, but the target flows get the same total weight
                   as the source flows
plus the source-only reference (n = 0, as in E2). Five repeats with new samples
and model seeds. Test data: the same 60,000-flow test samples as E2 and E5.
Writes results_e8_target_labels.csv.

Paper: "Accurate but Not Reliably Transferable" (Sections 4.3 and 5.5).
Run from the repository root:  python scripts/e8_target_labels.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR  # noqa: E402

import time, warnings  # noqa: E402
import numpy as np, pandas as pd, lightgbm as lgb  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
warnings.filterwarnings("ignore")

OUT, LIMIT, R, TEST_CAP, SRC_N = OUT_DIR, 1e15, 5, 60_000, 100_000
BUDGETS = [100, 1_000, 10_000]


def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))


def f1_fast(y, pred):
    tp = np.sum(pred & y); fp = np.sum(pred & ~y); fn = np.sum(~pred & y)
    return 2 * tp / max(2 * tp + fp + fn, 1)


# Same test samples as E2/E5 (same generator, same order of draws).
rng0 = np.random.default_rng(42)
D = {}
for k in ["unsw", "ton", "cic"]:
    tr = pd.read_parquet(f"{OUT}/{k}_train.parquet")
    te = pd.read_parquet(f"{OUT}/{k}_test.parquet")
    feats = [c for c in tr.columns if c not in ("Label", "Attack", "LONGEST_FLOW_PKT")]
    te = te.iloc[rng0.choice(len(te), min(TEST_CAP, len(te)), replace=False)]
    D[k] = dict(Xtr=sanitize(tr[feats]).reset_index(drop=True), ytr=tr["Label"].values,
                Xte=sanitize(te[feats]), yte=te["Label"].values.astype(bool))


def make_models(seed):
    return {
        "DecisionTree": make_pipeline(SimpleImputer(strategy="median"),
                                      DecisionTreeClassifier(max_depth=12, random_state=seed)),
        "LightGBM100": lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1,
                                          random_state=seed, verbose=-1),
    }


def fit(model, X, y, w=None):
    if w is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):  # pipeline: pass the weight to the last step
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def draw(k, n, rng):
    idx = rng.choice(len(D[k]["Xtr"]), min(n, len(D[k]["Xtr"])), replace=False)
    return D[k]["Xtr"].iloc[idx], D[k]["ytr"][idx]


def score(model, t):
    y = D[t]["yte"]
    p = model.predict_proba(D[t]["Xte"])[:, 1]
    return dict(roc=roc_auc_score(y, p), pr=average_precision_score(y, p),
                f1=f1_fast(y, p >= 0.5))


rows = []
for r in range(R):
    t0 = time.time()
    rng = np.random.default_rng(500 + r)
    for t in D:
        sources = [s for s in D if s != t]
        S = {s: draw(s, SRC_N, rng) for s in sources}
        for name in make_models(r):
            for s in sources:               # n = 0: source only
                m = fit(make_models(r)[name], *S[s])
                rows.append(dict(rep=r, model=name, target=t, source=s, budget=0,
                                 regime="source-only", **score(m, t)))
        for n in BUDGETS:
            Xt, yt = draw(t, n, rng)
            if yt.min() == yt.max():        # a draw with only one class cannot train
                continue
            for name in make_models(r):
                m = fit(make_models(r)[name], Xt, yt)
                rows.append(dict(rep=r, model=name, target=t, source="-", budget=n,
                                 regime="target-only", **score(m, t)))
                for s in sources:
                    Xs, ys = S[s]
                    X = pd.concat([Xs, Xt], ignore_index=True)
                    y = np.concatenate([ys, yt])
                    m = fit(make_models(r)[name], X, y)
                    rows.append(dict(rep=r, model=name, target=t, source=s, budget=n,
                                     regime="pooled", **score(m, t)))
                    w = np.concatenate([np.ones(len(ys)), np.full(len(yt), len(ys) / len(yt))])
                    m = fit(make_models(r)[name], X, y, w)
                    rows.append(dict(rep=r, model=name, target=t, source=s, budget=n,
                                     regime="pooled-weighted", **score(m, t)))
    print(f"repeat {r+1}/{R} done ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_e8_target_labels.csv", index=False)
chance_pr = np.mean([D[k]["yte"].mean() for k in D])
triv = {k: 2 * D[k]["yte"].mean() / (1 + D[k]["yte"].mean()) for k in D}
res["f1_above_trivial"] = [row.f1 > triv[row.target] for row in res.itertuples()]

# Mean over targets (and sources) per repeat, then mean and SD over repeats.
order = [("source-only", 0)] + [(g, n) for n in BUDGETS
                                for g in ("target-only", "pooled", "pooled-weighted")]
for name in ["LightGBM100", "DecisionTree"]:
    sub = res[res.model == name]
    per_rep = sub.groupby(["regime", "budget", "rep"])[["roc", "pr", "f1"]].mean()
    agg = per_rep.groupby(["regime", "budget"]).agg(["mean", "std"])
    print(f"\n=== {name}: mean over the three targets; mean±SD over {R} repeats "
          f"(chance PR-AUC {chance_pr:.3f})")
    print(f"{'regime':16s} {'n':>6s}  {'ROC-AUC':>13s}  {'PR-AUC':>13s}  {'F1':>13s}  F1>trivial")
    for g, n in order:
        if (g, n) not in agg.index:
            continue
        a = agg.loc[(g, n)]
        share = sub[(sub.regime == g) & (sub.budget == n)].f1_above_trivial.mean()
        print(f"{g:16s} {n:6d}  {a[('roc','mean')]:.3f}±{a[('roc','std')]:.3f}  "
              f"{a[('pr','mean')]:.3f}±{a[('pr','std')]:.3f}  "
              f"{a[('f1','mean')]:.3f}±{a[('f1','std')]:.3f}  {share:.2f}")

print("\n=== LightGBM100 ROC-AUC by target (mean over sources and repeats)")
sub = res[res.model == "LightGBM100"]
print(sub.pivot_table(index=["regime", "budget"], columns="target", values="roc",
                      aggfunc="mean").round(3).to_string())
