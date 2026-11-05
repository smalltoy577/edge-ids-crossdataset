"""E1: cross-dataset matrix of all five models with 95% bootstrap intervals.
Writes results_bootstrap_ci.csv (Tables 5 and 8, diagonal of Fig. 1).

Paper: "Accurate but Not Reliably Transferable" (Section 4.3, 5.1, 5.3).
Run from the repository root:  python scripts/e1_bootstrap.py
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
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

OUT, LIMIT, SEED, TEST_CAP, B = OUT_DIR, 1e15, 42, 60_000, 100

def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def f1_fast(y, pred):
    tp = np.sum(pred & y); fp = np.sum(pred & ~y); fn = np.sum(~pred & y)
    return 2 * tp / max(2 * tp + fp + fn, 1)

rng = np.random.default_rng(SEED)
D = {}
for k in ["unsw", "ton", "cic"]:
    tr = pd.read_parquet(f"{OUT}/{k}_train.parquet")
    te = pd.read_parquet(f"{OUT}/{k}_test.parquet")
    feats = [c for c in tr.columns if c not in ("Label", "Attack", "LONGEST_FLOW_PKT")]
    tr = tr.iloc[rng.choice(len(tr), min(100_000, len(tr)), replace=False)]
    te = te.iloc[rng.choice(len(te), min(TEST_CAP, len(te)), replace=False)]
    D[k] = dict(Xtr=sanitize(tr[feats]), ytr=tr["Label"].values,
                Xte=sanitize(te[feats]), yte=te["Label"].values.astype(bool))
rb = np.random.default_rng(7)
IDX = {k: rb.integers(0, len(D[k]["yte"]), size=(B, len(D[k]["yte"]))) for k in D}

print("Trivial 'flag everything' F1 on the test samples:",
      {k: round(2*D[k]["yte"].mean()/(1+D[k]["yte"].mean()), 3) for k in D})

def make_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "DecisionTree": make_pipeline(imp(), DecisionTreeClassifier(max_depth=12, random_state=SEED)),
        "RandomForest30": make_pipeline(imp(), RandomForestClassifier(
            n_estimators=30, max_depth=12, n_jobs=-1, random_state=SEED)),
        "LightGBM100": lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1,
                                          random_state=SEED, verbose=-1),
        "LightGBM30small": lgb.LGBMClassifier(n_estimators=30, num_leaves=15, n_jobs=-1,
                                              random_state=SEED, verbose=-1),
        "LogReg": make_pipeline(imp(), FunctionTransformer(signed_log), StandardScaler(),
                                LogisticRegression(max_iter=300)),
    }

rows = []
for trk in D:
    t0 = time.time()
    for name, model in make_models().items():
        model.fit(D[trk]["Xtr"], D[trk]["ytr"])
        for tek in D:
            y = D[tek]["yte"]
            p = model.predict_proba(D[tek]["Xte"])[:, 1]
            pred = p >= 0.5
            roc_b = np.array([roc_auc_score(y[i], p[i]) for i in IDX[tek]])
            f1_b = np.array([f1_fast(y[i], pred[i]) for i in IDX[tek]])
            rows.append(dict(model=name, train=trk, test=tek,
                             kind="in" if trk == tek else "cross",
                             roc=roc_auc_score(y, p), roc_lo=np.percentile(roc_b, 2.5),
                             roc_hi=np.percentile(roc_b, 97.5),
                             f1=f1_fast(y, pred), f1_lo=np.percentile(f1_b, 2.5),
                             f1_hi=np.percentile(f1_b, 97.5)))
    print(f"train={trk} done ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_bootstrap_ci.csv", index=False)

def fmt(metric):
    s = res.apply(lambda r: f"{r[metric]:.3f} [{r[metric+'_lo']:.3f},{r[metric+'_hi']:.3f}]", axis=1)
    return res.assign(v=s).pivot(index=["model", "train"], columns="test", values="v")

print("\n=== ROC-AUC with 95% bootstrap CI (diagonal = in-dataset)")
print(fmt("roc").to_string())
print("\n=== F1 at threshold 0.5 with 95% bootstrap CI")
print(fmt("f1").to_string())
