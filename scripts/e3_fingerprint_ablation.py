"""E3: leave-one-dataset-out training with the top-k fingerprint features removed
(ranked on source data only). Writes results_step7_protocol.csv (Table 11).

Paper: "Accurate but Not Reliably Transferable" (Section 4.3, 5.5).
Run from the repository root:  python scripts/e3_fingerprint_ablation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR, DATA as DATA_DIR, get_csv  # noqa: E402

import time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

OUT, LIMIT, CAP, R = OUT_DIR, 1e15, 100_000, 5
KS = [0, 2, 4, 6, 8, 10]

def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))

D = {}
for k in ["unsw", "ton", "cic"]:
    tr = pd.read_parquet(f"{OUT}/{k}_train.parquet")
    te = pd.read_parquet(f"{OUT}/{k}_test.parquet")
    feats = [c for c in tr.columns if c not in ("Label", "Attack", "LONGEST_FLOW_PKT")]
    D[k] = dict(Xtr=sanitize(tr[feats]), ytr=tr["Label"].values,
                Xte=sanitize(te[feats]), yte=te["Label"].values)
ALL = feats
print("Features used:", len(ALL))

def sample(k, rng, n=CAP):
    idx = rng.choice(len(D[k]["Xtr"]), min(n, len(D[k]["Xtr"])), replace=False)
    return D[k]["Xtr"].iloc[idx], D[k]["ytr"][idx]

def make_model(seed):
    return lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1,
                              random_state=seed, verbose=-1)

def fingerprint_ranking(sources, rng):
    """Features ranked by how well they reveal WHICH source dataset a flow
    came from. Uses training data of the source datasets only."""
    Xs, ys = [], []
    for i, s in enumerate(sources):
        X, _ = sample(s, rng, 30_000)
        Xs.append(X); ys.append(np.full(len(X), i))
    m = lgb.LGBMClassifier(n_estimators=60, num_leaves=31, n_jobs=-1,
                           random_state=0, verbose=-1, importance_type="gain")
    m.fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys))
    imp = pd.Series(m.feature_importances_, index=ALL)
    return imp.sort_values(ascending=False).index.tolist()

rows, tops = [], {}
for t in D:
    sources = [x for x in D if x != t]
    y = D[t]["yte"]
    for r in range(R):
        t0 = time.time()
        rng = np.random.default_rng(100 + r)
        samp = [sample(s, rng) for s in sources]
        Xs = pd.concat([x for x, _ in samp], ignore_index=True)
        ys = np.concatenate([yy for _, yy in samp])
        order = fingerprint_ranking(sources, rng)
        tops.setdefault(t, []).append(order[:8])
        Xin, yin = sample(t, rng)
        for k in KS:
            cols = [c for c in ALL if c not in order[:k]]
            for kind, (Xa, ya) in {"lodo": (Xs, ys), "in": (Xin, yin)}.items():
                m = make_model(r).fit(Xa[cols], ya)
                p = m.predict_proba(D[t]["Xte"][cols])[:, 1]
                rows.append(dict(target=t, rep=r, k=k, kind=kind,
                                 roc_auc=roc_auc_score(y, p),
                                 pr_auc=average_precision_score(y, p),
                                 f1=f1_score(y, p >= 0.5)))
        print(f"target {t} | repeat {r+1}/{R} done ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_step7_protocol.csv", index=False)

def table(kind, metric):
    sub = res[res.kind == kind]
    g = sub.groupby(["target", "k"])[metric].agg(["mean", "std"])
    out = g.apply(lambda s: f"{s['mean']:.3f}±{s['std']:.3f}", axis=1).unstack("k")
    avg = sub.groupby(["rep", "k"])[metric].mean().groupby("k").agg(["mean", "std"])
    out.loc["AVERAGE"] = [f"{avg.loc[k, 'mean']:.3f}±{avg.loc[k, 'std']:.3f}"
                          for k in out.columns]
    return out

print("\n=== LEAVE-ONE-DATASET-OUT ROC-AUC (columns = number of fingerprint features dropped)")
print(table("lodo", "roc_auc").to_string())
print("\n=== LEAVE-ONE-DATASET-OUT PR-AUC")
print(table("lodo", "pr_auc").to_string())
print("\n=== LEAVE-ONE-DATASET-OUT F1 (trivial all-attack F1: unsw 0.293, ton 0.859, cic 0.118)")
print(table("lodo", "f1").to_string())
print("\n=== IN-DATASET ROC-AUC (does dropping features cost accuracy?)")
print(table("in", "roc_auc").to_string())

print("\n=== Stability: how often each feature was in the top-8 fingerprint set (out of 5 repeats)")
for t, lst in tops.items():
    cnt = pd.Series([f for l in lst for f in l]).value_counts()
    print(t, cnt.head(10).to_dict())
