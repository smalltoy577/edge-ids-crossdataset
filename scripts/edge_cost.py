"""Edge cost: model size, memory, single-flow latency and batch throughput.

Trains the five models on 100,000 ToN-IoT training flows, saves them, and
benchmarks each one three times in a fresh Python process pinned to one CPU
core (Linux `taskset`). Writes results_edge_cost_v2.csv (Table 14).

Paper: "Accurate but Not Reliably Transferable" (Sections 4.4 and 5.6).
Run from the repository root:  python scripts/edge_cost.py
"""
import json
import os
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as OUT_DIR, WORK  # noqa: E402

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

warnings.filterwarnings("ignore")

OUT, LIMIT, SEED = OUT_DIR, 1e15, 42
WORK = os.path.abspath(WORK)
MODELS = os.path.join(WORK, "models")
os.makedirs(MODELS, exist_ok=True)

# Helper module, so that the saved pipelines can be loaded in a separate process.
with open(os.path.join(WORK, "edgeutil.py"), "w") as f:
    f.write('''
import numpy as np
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))
def single_thread(model):
    try:
        for step in getattr(model, "steps", []):
            if hasattr(step[1], "n_jobs"):
                step[1].set_params(n_jobs=1)
        if hasattr(model, "n_jobs"):
            model.set_params(n_jobs=1)
    except Exception:
        pass
''')
sys.path.insert(0, WORK)
from edgeutil import signed_log  # noqa: E402

# Benchmark script, run in a fresh process pinned to one core.
# argv: model name, models folder, work folder
with open(os.path.join(WORK, "bench.py"), "w") as f:
    f.write('''
import sys, time, json, warnings
import numpy as np, joblib
warnings.filterwarnings("ignore")
name, models, work = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, work)
import sklearn.pipeline, sklearn.tree, sklearn.ensemble, sklearn.linear_model
import sklearn.impute, sklearn.preprocessing, lightgbm
from edgeutil import signed_log, single_thread

def status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1]) / 1024

X = np.load(models + "/bench_X.npy")
before = status("VmRSS")                      # libraries already imported
model = joblib.load(f"{models}/{name}.joblib")
single_thread(model)
after_load = status("VmRSS")
for i in range(200):                          # warm-up
    model.predict_proba(X[i:i+1])
lat = []
for i in range(2000):
    row = X[i:i+1]
    t0 = time.perf_counter()
    model.predict_proba(row)
    lat.append((time.perf_counter() - t0) * 1000)
tp = []
for _ in range(15):
    t0 = time.perf_counter()
    model.predict_proba(X[:1000])
    tp.append(1000 / (time.perf_counter() - t0))
print(json.dumps(dict(
    p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)),
    throughput=float(np.median(tp)),
    model_load_mb=after_load - before,
    run_mb=status("VmRSS") - before,
    process_peak_mb=status("VmHWM"))))
''')


def sanitize(X):
    X = X.astype("float64")
    return X.mask((X.abs() > LIMIT) | np.isinf(X))


tr = pd.read_parquet(f"{OUT}/ton_train.parquet")
te = pd.read_parquet(f"{OUT}/ton_test.parquet")
feats = [c for c in tr.columns if c not in ("Label", "Attack", "LONGEST_FLOW_PKT")]
tr = tr.sample(100_000, random_state=SEED)
Xtr, ytr = sanitize(tr[feats]).values, tr["Label"].values
Xte, yte = sanitize(te[feats]).values, te["Label"].values
np.save(os.path.join(MODELS, "bench_X.npy"), Xte[:3000])

imp = lambda: SimpleImputer(strategy="median")  # noqa: E731
models = {
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

# 1) train, save, and record file size and in-dataset accuracy on the ToN-IoT test split
info = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    path = os.path.join(MODELS, f"{name}.joblib")
    joblib.dump(m, path)
    info[name] = dict(size_mb=os.path.getsize(path) / 1e6,
                      roc_auc_ton=roc_auc_score(yte, p), f1_ton=f1_score(yte, p >= 0.5))
    print("trained:", name)

# 2) benchmark every model three times in a fresh, single-core process
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
rows = []
for name in models:
    for rep in range(3):
        out = subprocess.run(["taskset", "-c", "0", sys.executable,
                              os.path.join(WORK, "bench.py"), name, MODELS, WORK],
                             capture_output=True, text=True, env=env)
        if out.returncode != 0:
            print(name, "FAILED:", out.stderr[-400:])
            break
        rows.append(dict(model=name, rep=rep, **json.loads(out.stdout.strip().splitlines()[-1])))
    print("benchmarked:", name)

new = pd.DataFrame(rows)
summ = (new.groupby("model").median(numeric_only=True).drop(columns="rep")
        .join(pd.DataFrame(info).T))
summ.to_csv(f"{OUT}/results_edge_cost_v2.csv")
print("\n=== Edge cost (median of 3 runs, 1 pinned core)")
print(summ[["size_mb", "model_load_mb", "run_mb", "process_peak_mb", "p50_ms",
            "p95_ms", "throughput", "roc_auc_ton", "f1_ton"]].round(3).to_string())
