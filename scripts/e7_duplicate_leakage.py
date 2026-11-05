"""E7: duplicate leakage in a random 70/30 split of raw (non-deduplicated) samples.
Writes results_e7_leak_share.csv and results_e7_leakage.csv (Table 6).

Paper: "Accurate but Not Reliably Transferable" (Section 4.3, 5.2).
Run from the repository root:  python scripts/e7_duplicate_leakage.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR, DATA as DATA_DIR, get_csv  # noqa: E402

import os, glob, zipfile, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings("ignore")

DATA = DATA_DIR
OUT  = OUT_DIR
KEYS = {"unsw": "NF-UNSW-NB15-v2.csv", "ton": "NF-ToN-IoT-v2.csv", "cic": "NF-CSE-CIC-IDS2018-v2.csv"}
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "LONGEST_FLOW_PKT"]
TARGET, MIN_PER_CLASS, SEED, CH, LIMIT = 1_000_000, 500, 42, 1_000_000, 1e15


def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))

def ev(y, p):
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = np.nan
    return auc, f1_score(y, p >= 0.5), float(np.mean(y))

rows, leak = [], []
for key, csv_name in KEYS.items():
    t0 = time.time()
    path = get_csv(csv_name)
    counts = pd.Series(dtype="int64")
    for ch in pd.read_csv(path, usecols=["Attack"], chunksize=2_000_000):
        counts = counts.add(ch["Attack"].str.strip().value_counts(), fill_value=0)
    counts = counts.astype(int); total = int(counts.sum())
    want = {c: min(n, max(round(n * TARGET / total), MIN_PER_CLASS)) for c, n in counts.items()}
    prob = {c: want[c] / counts[c] for c in counts.index}

    rng = np.random.default_rng(SEED); parts = []
    for ch in pd.read_csv(path, chunksize=CH):
        ch["Attack"] = ch["Attack"].str.strip()
        keep = rng.random(len(ch)) < ch["Attack"].map(prob).fillna(0).values
        parts.append(ch[keep].drop(columns=DROP))
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in ("Label", "Attack")]
    df[feats] = sanitize(df[feats])

    tr, te = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["Attack"])
    f_tr = set(pd.util.hash_pandas_object(tr[feats], index=False).values)
    s_tr = set(pd.util.hash_pandas_object(tr[feats + ["Label"]], index=False).values)
    twin_any = pd.util.hash_pandas_object(te[feats], index=False).isin(f_tr).values
    twin_same = pd.util.hash_pandas_object(te[feats + ["Label"]], index=False).isin(s_tr).values
    conflict_only = twin_any & ~twin_same
    leak.append(dict(dataset=key, raw_sample=len(df), test=len(te),
                     twin_same_label_pct=round(100 * twin_same.mean(), 1),
                     twin_other_label_only_pct=round(100 * conflict_only.mean(), 1),
                     no_twin_pct=round(100 * (~twin_any).mean(), 1)))
    print(leak[-1])

    groups = {"all test flows": np.ones(len(te), bool),
              "flow has same-label twin in train": twin_same,
              "no twin in train": ~twin_any}
    models = {
        "DecisionTree": make_pipeline(SimpleImputer(strategy="median"),
                                      DecisionTreeClassifier(max_depth=12, random_state=SEED)),
        "LightGBM100": lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1,
                                          random_state=SEED, verbose=-1)}
    y = te["Label"].values
    for mname, m in models.items():
        m.fit(tr[feats], tr["Label"].values)
        p = m.predict_proba(te[feats])[:, 1]
        for g, mask in groups.items():
            if mask.sum() == 0:
                continue
            auc, f1, ar = ev(y[mask], p[mask])
            rows.append(dict(dataset=key, model=mname, group=g, n=int(mask.sum()),
                             attack_share=round(ar, 4), roc_auc=round(auc, 4), f1=round(f1, 4)))
    print(f"{key} done ({time.time()-t0:.0f}s)")

pd.DataFrame(leak).to_csv(f"{OUT}/results_e7_leak_share.csv", index=False)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/results_e7_leakage.csv", index=False)
print("\n=== Share of the test flows with an exact twin in the training split (raw, not deduplicated)")
print(pd.DataFrame(leak).to_string(index=False))
print("\n=== Scores by group")
print(res.to_string(index=False))
