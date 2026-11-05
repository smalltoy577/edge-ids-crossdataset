"""E4: attack-type matching on the denial-of-service family.
Writes results_step8_dos_matched.csv (Table 12).

Paper: "Accurate but Not Reliably Transferable" (Section 4.3, 5.5).
Run from the repository root:  python scripts/e4_dos_matching.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR, DATA as DATA_DIR, get_csv  # noqa: E402

import time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

OUT, LIMIT, R = OUT_DIR, 1e15, 5

def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))

D = {}
for k in ["unsw", "ton", "cic"]:
    tr = pd.read_parquet(f"{OUT}/{k}_train.parquet")
    te = pd.read_parquet(f"{OUT}/{k}_test.parquet")
    for df in (tr, te):
        a = df["Attack"].str.strip().str.lower()
        df["fam"] = np.where(a == "benign", 0, np.where(a.str.contains("dos"), 1, -1))
    D[k] = dict(tr=tr, te=te)

FEATS = [c for c in D["unsw"]["tr"].columns
         if c not in ("Label", "Attack", "LONGEST_FLOW_PKT", "fam")]

for k in D:
    te = D[k]["te"]; te = te[te.fam >= 0]
    D[k]["Xte"], D[k]["yte"] = sanitize(te[FEATS]), te["fam"].values
    tr = D[k]["tr"]
    print(f"{k}: train DoS rows {(tr.fam==1).sum():,} | test benign {(D[k]['yte']==0).sum():,}"
          f" | test DoS {(D[k]['yte']==1).sum():,} | base rate {D[k]['yte'].mean():.4f}")

def make_model(seed):
    return lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1,
                              random_state=seed, verbose=-1)

def sample_family(k, rng):
    tr = D[k]["tr"]; tr = tr[tr.fam >= 0]
    pos, neg = tr[tr.fam == 1], tr[tr.fam == 0]
    pos = pos.iloc[rng.choice(len(pos), min(len(pos), 50_000), replace=False)]
    neg = neg.iloc[rng.choice(len(neg), min(len(neg), 100_000), replace=False)]
    d = pd.concat([pos, neg])
    return sanitize(d[FEATS]), d["fam"].values

def sample_all(k, rng):
    tr = D[k]["tr"]
    d = tr.iloc[rng.choice(len(tr), min(len(tr), 100_000), replace=False)]
    return sanitize(d[FEATS]), d["Label"].values

rows = []
for t in D:
    src = [x for x in D if x != t]
    y = D[t]["yte"]
    for r in range(R):
        t0 = time.time()
        rng = np.random.default_rng(200 + r)
        PA = [sample_all(s, rng) for s in src]
        PB = [sample_family(s, rng) for s in src]
        regimes = {
            "A_all-attack-labels": (pd.concat([p[0] for p in PA], ignore_index=True),
                                    np.concatenate([p[1] for p in PA])),
            "B_DoS-matched": (pd.concat([p[0] for p in PB], ignore_index=True),
                              np.concatenate([p[1] for p in PB])),
            "C_in-dataset-DoS": sample_family(t, rng),
        }
        for name, (X, yy) in regimes.items():
            p = make_model(r).fit(X, yy).predict_proba(D[t]["Xte"])[:, 1]
            rows.append(dict(target=t, rep=r, regime=name,
                             roc_auc=roc_auc_score(y, p),
                             pr_auc=average_precision_score(y, p),
                             f1=f1_score(y, p >= 0.5), base=float(y.mean())))
        print(f"target {t} | repeat {r+1}/{R} done ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_step8_dos_matched.csv", index=False)

def table(metric):
    g = res.groupby(["target", "regime"])[metric].agg(["mean", "std"])
    return g.apply(lambda s: f"{s['mean']:.3f}±{s['std']:.3f}", axis=1).unstack("regime")

print("\n=== Chance PR-AUC (= DoS share in the test set):",
      res.groupby("target")["base"].first().round(4).to_dict())
for m in ["roc_auc", "pr_auc", "f1"]:
    print(f"\n=== {m.upper()} on the DoS-only test set")
    print(table(m).to_string())
