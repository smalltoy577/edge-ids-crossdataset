"""E6: trivial baselines, single-feature ROC-AUC per dataset, and dataset
identification accuracy. Writes single_feature_auc.csv (Table 10).

Paper: "Accurate but Not Reliably Transferable" (Section 4.3, 5.4).
Run from the repository root:  python scripts/e6_identifiability.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR, DATA as DATA_DIR, get_csv  # noqa: E402

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

OUT, LIMIT = OUT_DIR, 1e15

def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))

D = {}
for k in ["unsw", "ton", "cic"]:
    te = pd.read_parquet(f"{OUT}/{k}_test.parquet")
    feats = [c for c in te.columns if c not in ("Label", "Attack")]
    D[k] = dict(X=sanitize(te[feats]), y=te["Label"].values)

# --- 1. Trivial baselines ---
print("=== Trivial 'flag everything as attack' F1 per test dataset")
for k in D:
    p = D[k]["y"].mean()
    print(f"  {k}: attack rate {p:.4f} -> trivial F1 {2*p/(1+p):.3f}")

# --- 2. Single-feature AUC per dataset ---
rows = []
for k in D:
    X, y = D[k]["X"], D[k]["y"]
    for c in X.columns:
        v = X[c].fillna(X[c].median())
        auc = 0.5 if v.nunique() < 2 else roc_auc_score(y, v)
        rows.append((c, k, auc))
A = (pd.DataFrame(rows, columns=["feature", "dataset", "auc"])
       .pivot(index="feature", columns="dataset", values="auc"))
A["spread"] = A.max(axis=1) - A.min(axis=1)
A["sign_flip"] = (A[["unsw", "ton", "cic"]].max(axis=1) > 0.6) & \
                 (A[["unsw", "ton", "cic"]].min(axis=1) < 0.4)
print("\n=== Single-feature attack-vs-benign AUC (0.5 = no signal; "
      ">0.5 = higher value means more attack)")
print(A.sort_values("spread", ascending=False).round(3).to_string())
print("\nFeatures that flip direction between datasets:",
      A.index[A["sign_flip"]].tolist())
A.to_csv(f"{OUT}/single_feature_auc.csv")

# --- 3. Can a model tell the datasets apart? ---
rng = np.random.default_rng(0)
Xs, ys = [], []
for i, k in enumerate(D):
    idx = rng.choice(len(D[k]["X"]), 40000, replace=False)
    Xs.append(D[k]["X"].iloc[idx]); ys.append(np.full(40000, i))
X_all = pd.concat(Xs, ignore_index=True); y_all = np.concatenate(ys)
perm = rng.permutation(len(X_all))
cut = int(0.7 * len(perm))
tr_i, te_i = perm[:cut], perm[cut:]
m = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1,
                       random_state=0, verbose=-1,
                       importance_type="gain")
m.fit(X_all.iloc[tr_i], y_all[tr_i])
acc = (m.predict(X_all.iloc[te_i]) == y_all[te_i]).mean()
print(f"\n=== Dataset identification accuracy (3 classes, chance = 0.333): {acc:.4f}")
imp = pd.Series(m.feature_importances_, index=X_all.columns)
print("Top 8 features that reveal the dataset:")
print((imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_string())
