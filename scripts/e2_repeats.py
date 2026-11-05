"""E2: cross-dataset runs repeated five times with new training subsamples and seeds.
Writes results_step11_model_family.csv (Tables 7 and 9, off-diagonal of Fig. 1).

Paper: "Accurate but Not Reliably Transferable" (Section 4.3, 5.3).
Run from the repository root:  python scripts/e2_repeats.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR, DATA as DATA_DIR, get_csv  # noqa: E402

import time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

OUT, LIMIT, R, TEST_CAP = OUT_DIR, 1e15, 5, 60_000

def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def f1_fast(y, pred):
    tp = np.sum(pred & y); fp = np.sum(pred & ~y); fn = np.sum(~pred & y)
    return 2 * tp / max(2 * tp + fp + fn, 1)

rng0 = np.random.default_rng(42)
D = {}
for k in ["unsw", "ton", "cic"]:
    tr = pd.read_parquet(f"{OUT}/{k}_train.parquet")
    te = pd.read_parquet(f"{OUT}/{k}_test.parquet")
    feats = [c for c in tr.columns if c not in ("Label", "Attack", "LONGEST_FLOW_PKT")]
    te = te.iloc[rng0.choice(len(te), min(TEST_CAP, len(te)), replace=False)]
    D[k] = dict(Xtr=sanitize(tr[feats]), ytr=tr["Label"].values,
                Xte=sanitize(te[feats]), yte=te["Label"].values.astype(bool))
triv = {k: 2 * D[k]["yte"].mean() / (1 + D[k]["yte"].mean()) for k in D}

def make_models(seed):
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "DecisionTree": make_pipeline(imp(), DecisionTreeClassifier(max_depth=12, random_state=seed)),
        "RandomForest30": make_pipeline(imp(), RandomForestClassifier(
            n_estimators=30, max_depth=12, n_jobs=-1, random_state=seed)),
        "LightGBM100": lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1,
                                          random_state=seed, verbose=-1),
        "LightGBM30small": lgb.LGBMClassifier(n_estimators=30, num_leaves=15, n_jobs=-1,
                                              random_state=seed, verbose=-1),
        "LogReg": make_pipeline(imp(), FunctionTransformer(signed_log), StandardScaler(),
                                LogisticRegression(max_iter=300)),
    }

rows = []
for r in range(R):
    t0 = time.time()
    rng = np.random.default_rng(300 + r)
    for trk in D:
        idx = rng.choice(len(D[trk]["Xtr"]), min(100_000, len(D[trk]["Xtr"])), replace=False)
        Xa, ya = D[trk]["Xtr"].iloc[idx], D[trk]["ytr"][idx]
        for name, model in make_models(r).items():
            model.fit(Xa, ya)
            for tek in D:
                if tek == trk:
                    continue
                y = D[tek]["yte"]
                p = model.predict_proba(D[tek]["Xte"])[:, 1]
                rows.append(dict(rep=r, model=name, direction=f"{trk}->{tek}", target=tek,
                                 roc=roc_auc_score(y, p),
                                 pr=average_precision_score(y, p),
                                 f1=f1_fast(y, p >= 0.5)))
    print(f"repeat {r+1}/{R} done ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_step11_model_family.csv", index=False)

g = res.groupby(["model", "direction"])["roc"].agg(["mean", "std"])
tab = g.apply(lambda s: f"{s['mean']:.3f}±{s['std']:.3f}", axis=1).unstack("direction")
print("\n=== Cross-dataset ROC-AUC, mean±std over 5 repeats (0.5 = random)")
print(tab.to_string())

res["roc_above_chance"] = res["roc"] > 0.5
res["f1_above_trivial"] = [row.f1 > triv[row.target] for row in res.itertuples()]
summ = res.groupby("model").agg(
    mean_roc=("roc", "mean"), mean_pr=("pr", "mean"),
    share_roc_above_0_5=("roc_above_chance", "mean"),
    share_f1_above_trivial=("f1_above_trivial", "mean"))
print("\n=== Summary over all 6 directions x 5 repeats = 30 cells per model")
print(summ.round(3).to_string())
