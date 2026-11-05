"""Step 0: deduplicate (identifiers removed), sample about 1M unique flows per
dataset, and make the 70/30 stratified train/test splits.
Writes {OUT}/<key>_sample.parquet, <key>_train.parquet, <key>_test.parquet and
sampling_summary.json.

Paper: "Accurate but Not Reliably Transferable" (Section 3).
Run from the repository root:  python scripts/00_prepare_data.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR, DATA as DATA_DIR, get_csv  # noqa: E402

import os, glob, zipfile, json
import numpy as np
import pandas as pd

DATA = DATA_DIR
OUT  = OUT_DIR
os.makedirs(OUT, exist_ok=True)

KEYS = {
    "unsw": "NF-UNSW-NB15-v2.csv",
    "ton":  "NF-ToN-IoT-v2.csv",
    "cic":  "NF-CSE-CIC-IDS2018-v2.csv",
}
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
TARGET, MIN_PER_CLASS, SEED, CH = 1_000_000, 500, 42, 1_000_000


summary = {}
for key, csv_name in KEYS.items():
    path = get_csv(csv_name)
    print(f"\n=== {key}: {path}")

    # Pass 1: hash every row (identifiers excluded) and mark first occurrences
    hashes = []
    for ch in pd.read_csv(path, chunksize=CH):
        ch = ch.drop(columns=DROP)
        ch["Attack"] = ch["Attack"].str.strip()
        hashes.append(pd.util.hash_pandas_object(ch, index=False).values)
    h = np.concatenate(hashes); del hashes
    total_rows = len(h)
    _, first_idx = np.unique(h, return_index=True)
    is_first = np.zeros(total_rows, dtype=bool)
    is_first[first_idx] = True
    del h, first_idx
    n_unique = int(is_first.sum())
    print(f"full rows: {total_rows:,} | unique rows: {n_unique:,} "
          f"({100*(total_rows-n_unique)/total_rows:.1f}% duplicates)")

    # Pass 2: count unique rows per attack class
    counts = pd.Series(dtype="int64"); off = 0
    for ch in pd.read_csv(path, usecols=["Attack"], chunksize=CH):
        m = is_first[off:off+len(ch)]; off += len(ch)
        counts = counts.add(ch.loc[m, "Attack"].str.strip().value_counts(),
                            fill_value=0)
    counts = counts.astype(int)
    want = {c: min(n, max(round(n * TARGET / n_unique), MIN_PER_CLASS))
            for c, n in counts.items()}
    prob = {c: want[c] / counts[c] for c in counts.index}

    # Pass 3: sample only from the unique rows
    rng = np.random.default_rng(SEED)
    parts, off = [], 0
    for ch in pd.read_csv(path, chunksize=CH):
        m = is_first[off:off+len(ch)]; off += len(ch)
        ch["Attack"] = ch["Attack"].str.strip()
        keep = m & (rng.random(len(ch)) < ch["Attack"].map(prob).fillna(0).values)
        parts.append(ch[keep].drop(columns=DROP))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(f"{OUT}/{key}_sample.parquet", index=False)

    summary[key] = {
        "full_rows": total_rows,
        "unique_rows": n_unique,
        "sample_rows": len(df),
        "attack_pct": round(float(100 * df["Label"].mean()), 2),
        "columns": len(df.columns),
    }
    print(summary[key])
    print(df["Attack"].value_counts().to_string())

with open(f"{OUT}/sampling_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE")

# ---------- Train/test split (70/30, stratified by attack class) ----------
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, roc_auc_score)

OUT, SEED = OUT_DIR, 42

# ---------- Part 1: clean + split ----------
for key in ["unsw", "ton", "cic"]:
    df = pd.read_parquet(f"{OUT}/{key}_sample.parquet")
    feats = [c for c in df.columns if c not in ("Label", "Attack")]
    n0 = len(df)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.30, random_state=SEED,
                              stratify=df["Attack"])
    tr.to_parquet(f"{OUT}/{key}_train.parquet", index=False)
    te.to_parquet(f"{OUT}/{key}_test.parquet", index=False)
    print(f"{key}: {n0:,} -> {len(df):,} rows after dedup "
          f"({100*(n0-len(df))/n0:.2f}% removed) | "
          f"train {len(tr):,} | test {len(te):,} | attack {100*df['Label'].mean():.2f}%")
