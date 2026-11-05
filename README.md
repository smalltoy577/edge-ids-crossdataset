# Accurate but Not Reliably Transferable

Code and results for the paper

> M. H. Shanto, *Accurate but Not Reliably Transferable: A Deduplicated Cross-Dataset
> Evaluation of Lightweight Machine Learning Intrusion Detection on NetFlow Data*
> (submitted to *Computers & Security*).

The scripts evaluate five lightweight classifiers (decision tree, random forest,
two LightGBM ensembles, logistic regression) on three NetFlow v2 datasets
(NF-UNSW-NB15-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2) within and across
datasets. They also measure duplicate leakage the cost of deploying each model, and how far a small number of labelled
target flows helps.

## 1. Get the data

The datasets are **not** included in this repository. Download the three zip files from the
University of Queensland NetFlow dataset page:
<https://staff.itee.uq.edu.au/marius/NIDS_datasets/>. Put them, unchanged, in a folder,
for example `data/`:

```
data/
  <...>NF-UNSW-NB15-v2<...>.zip
  <...>NF-ToN-IoT-v2<...>.zip
  <...>NF-CSE-CIC-IDS2018-v2<...>.zip
```

The scripts find each CSV inside the zips by its file name, so you can keep the
original zip names. If you have already extracted the CSV files, you can put them in `data/` too.

Please cite the dataset papers (see the paper's reference list) if you use the data.

## 2. Install

Python 3.13 with the versions used in the paper:

```
pip install -r requirements.txt
```

The edge-cost script needs Linux (it reads `/proc/self/status` and uses `taskset`).

## 3. Run

From the repository root, in this order:

| Script | Paper | Output in `processed/` | Time on free Colab |
|---|---|---|---|
| `scripts/00_prepare_data.py` | Sec. 3, Tables 1–2 | `*_sample/train/test.parquet`, `sampling_summary.json` | longest step |
| `scripts/e1_bootstrap.py` | E1: Tables 5, 8, Fig. 1 | `results_bootstrap_ci.csv` | ~3 min |
| `scripts/e2_repeats.py` | E2: Tables 7, 9, Fig. 1 | `results_step11_model_family.csv` | ~4 min |
| `scripts/e3_fingerprint_ablation.py` | E3: Table 11 | `results_step7_protocol.csv` | ~14 min |
| `scripts/e4_dos_matching.py` | E4: Table 12 | `results_step8_dos_matched.csv` | ~4 min |
| `scripts/e5_capacity.py` | E5: Table 13, Fig. 2 | `results_step12_capacity.csv` | ~6 min |
| `scripts/e6_identifiability.py` | E6: Table 10, Sec. 5.4 | `single_feature_auc.csv` | ~1 min |
| `scripts/e7_duplicate_leakage.py` | E7: Table 6 | `results_e7_leak_share.csv`, `results_e7_leakage.csv` | ~8 min |
| `scripts/e8_target_labels.py` | E8: Table 14 | `results_e8_target_labels.csv` | ~17 min |
| `scripts/edge_cost.py` | Table 15 | `results_edge_cost_v2.csv` | ~2 min |
| `scripts/make_figures.py` | Figs. 1–2 | `figures/*.pdf` | seconds |

`00_prepare_data.py` must run first. The other scripts read its output and can
run in any order. `make_figures.py` needs E1, E2 and E5.

Paths can be changed with environment variables:

```
export EDGE_IDS_DATA=data          # downloaded zip files
export EDGE_IDS_OUT=processed      # samples, splits and result CSVs
export EDGE_IDS_WORK=work          # extracted CSVs and edge-cost models
```

On Google Colab, open `run_all_colab.ipynb`. It mounts Google Drive, sets these
paths and runs the scripts in order.

## 4. Results

`results/` contains the result CSV files produced for the paper, so the tables and
figures can be checked without re-running anything. Python 3.13.18,
scikit-learn 1.6.1, LightGBM 4.6.0, NumPy 2.4.6 and pandas 2.2.3 on a free
Colab instance (Intel Xeon CPU @ 2.20 GHz, 2 vCPUs).

All random seeds are fixed in the scripts. Timing results (latency, throughput)
depend on the machine and vary between runs.

## 5. Notes

* The four identifier columns (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`,
  `L4_DST_PORT`) are removed before deduplication. `LONGEST_FLOW_PKT` is identical
  to `MAX_IP_PKT_LEN` and is dropped in E1–E5, E7, E8 and the edge-cost script. E6 was
  run with it, as stated in the paper.
* The scripts are the cells of the original notebook, unchanged except for the file
  paths, so they reproduce the paper's numbers.

## License

Code: MIT (see `LICENSE`). The datasets have their own licence; see the
dataset page.
