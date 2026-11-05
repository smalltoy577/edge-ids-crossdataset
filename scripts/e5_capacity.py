"""E5: model-capacity sweep (decision-tree depth, LightGBM leaves).
Writes results_step12_capacity.csv (Table 13, Fig. 2).

Paper: "Accurate but Not Reliably Transferable" (Section 4.3, 5.5).
Run from the repository root:  python scripts/e5_capacity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR, DATA as DATA_DIR, get_csv  # noqa: E402

import time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

OUT, LIMIT, R, TEST_CAP = OUT_DIR, 1e15, 5, 60_000

def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))

rng0 = np.random.default_rng(42)
D = {}
for k in ["unsw", "ton", "cic"]:
    tr = pd.read_parquet(f"{OUT}/{k}_train.parquet")
    te = pd.read_parquet(f"{OUT}/{k}_test.parquet")
    feats = [c for c in tr.columns if c not in ("Label", "Attack", "LONGEST_FLOW_PKT")]
    te = te.iloc[rng0.choice(len(te), min(TEST_CAP, len(te)), replace=False)]
    D[k] = dict(Xtr=sanitize(tr[feats]), ytr=tr["Label"].values,
                Xte=sanitize(te[feats]), yte=te["Label"].values.astype(bool))
chance_pr = np.mean([D[k]["yte"].mean() for k in D])

DEPTHS, LEAVES = [2, 3, 5, 8, 12, 20], [3, 7, 15, 31, 63]
ORDER = [f"DT_depth{d}" for d in DEPTHS] + [f"LGBM_leaves{l}" for l in LEAVES]

def configs(seed):
    cfg = {}
    for d in DEPTHS:
        cfg[f"DT_depth{d}"] = make_pipeline(SimpleImputer(strategy="median"),
                                            DecisionTreeClassifier(max_depth=d, random_state=seed))
    for l in LEAVES:
        cfg[f"LGBM_leaves{l}"] = lgb.LGBMClassifier(n_estimators=100, num_leaves=l, n_jobs=-1,
                                                    random_state=seed, verbose=-1)
    return cfg

rows = []
for r in range(R):
    t0 = time.time()
    rng = np.random.default_rng(400 + r)
    for trk in D:
        idx = rng.choice(len(D[trk]["Xtr"]), min(100_000, len(D[trk]["Xtr"])), replace=False)
        Xa, ya = D[trk]["Xtr"].iloc[idx], D[trk]["ytr"][idx]
        for name, model in configs(r).items():
            model.fit(Xa, ya)
            for tek in D:
                y = D[tek]["yte"]
                p = model.predict_proba(D[tek]["Xte"])[:, 1]
                rows.append(dict(rep=r, model=name, kind="in" if tek == trk else "cross",
                                 roc=roc_auc_score(y, p), pr=average_precision_score(y, p)))
    print(f"repeat {r+1}/{R} done ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_step12_capacity.csv", index=False)

per_rep = res.groupby(["model", "kind", "rep"])[["roc", "pr"]].mean()
agg = per_rep.groupby(["model", "kind"]).agg(["mean", "std"])
def f(m, kind, col):
    a = agg.loc[(m, kind), (col, "mean")]; s = agg.loc[(m, kind), (col, "std")]
    return f"{a:.3f}±{s:.3f}"
tab = pd.DataFrame({
    "in-dataset ROC": [f(m, "in", "roc") for m in ORDER],
    "cross ROC": [f(m, "cross", "roc") for m in ORDER],
    "cross PR": [f(m, "cross", "pr") for m in ORDER]}, index=ORDER)
print(f"\nChance PR-AUC (mean attack rate of targets): {chance_pr:.3f}")
print("\n=== Capacity sweep (mean±std over 5 repeats; cross = mean over 6 directions)")
print(tab.to_string())
