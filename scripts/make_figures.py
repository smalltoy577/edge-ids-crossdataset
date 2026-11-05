"""Figures 1 and 2 of the paper, drawn from the result CSV files.

Needs results_bootstrap_ci.csv (E1), results_step11_model_family.csv (E2) and
results_step12_capacity.csv (E5) in the results folder ($EDGE_IDS_OUT).
Writes figures/fig_cross_matrix.pdf and figures/fig_capacity.pdf.
Run from the repository root:  python scripts/make_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#8a8a85",
                     "axes.linewidth": 0.6, "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a",
                     "text.color": "#1f1f1e", "axes.labelcolor": "#1f1f1e", "pdf.fonttype": 42})

KEYS = ["unsw", "ton", "cic"]
NAMES = {"unsw": "UNSW", "ton": "ToN-IoT", "cic": "CIC"}
MODELS = ["DecisionTree", "RandomForest30", "LightGBM100", "LightGBM30small", "LogReg"]

# ---------- Figure 1: 3x3 ROC-AUC matrix per model ----------
e1 = pd.read_csv(f"{OUT}/results_bootstrap_ci.csv")      # diagonal: in-dataset, single run
e2 = pd.read_csv(f"{OUT}/results_step11_model_family.csv")  # off-diagonal: mean of 5 repeats
e2m = e2.groupby(["model", "direction"])["roc"].mean()

cmap = LinearSegmentedColormap.from_list(
    "div", ["#a8322e", "#e34948", "#f0a7a1", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
fig = plt.figure(figsize=(5.0, 3.5))
W, H, xs, y1, y2 = 0.25, 0.34, [0.11, 0.40, 0.69], 0.60, 0.10
axs = [fig.add_axes([x, y1, W, H]) for x in xs] + [fig.add_axes([x, y2, W, H]) for x in xs[:2]]
cax = fig.add_axes([0.74, y2 + 0.03, 0.025, H - 0.06])
for ax, m in zip(axs, MODELS):
    M = np.zeros((3, 3))
    for i, a in enumerate(KEYS):
        for j, b in enumerate(KEYS):
            if i == j:
                M[i, j] = e1[(e1.model == m) & (e1.train == a) & (e1.test == b)]["roc"].iloc[0]
            else:
                M[i, j] = e2m.loc[(m, f"{a}->{b}")]
    im = ax.imshow(M, cmap=cmap, norm=norm)
    for i in range(3):
        for j in range(3):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v - 0.5) > 0.33 else "#1f1f1e",
                    fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="#1f1f1e", lw=0.9))
    labels = [NAMES[k] for k in KEYS]
    ax.set_xticks(range(3), labels, fontsize=6.5)
    ax.set_yticks(range(3), labels if m in (MODELS[0], MODELS[3]) else ["", "", ""], fontsize=6.5)
    ax.tick_params(length=0)
    ax.set_title(m, fontsize=7.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 3), minor=True)
    ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
fig.text(0.015, 0.5, "Training dataset", rotation=90, va="center", fontsize=8)
fig.text(0.5, 0.01, "Test dataset", ha="center", fontsize=8)
cb = fig.colorbar(im, cax=cax, ticks=[0, .25, .5, .75, 1])
cb.ax.tick_params(labelsize=6.5, length=2)
cb.set_label("ROC-AUC", fontsize=7)
cb.outline.set_visible(False)
fig.savefig(f"{FIG}/fig_cross_matrix.pdf")
plt.close(fig)

# ---------- Figure 2: capacity sweep ----------
e5 = pd.read_csv(f"{OUT}/results_step12_capacity.csv")
per_rep = e5.groupby(["model", "kind", "rep"])["roc"].mean()
agg = per_rep.groupby(["model", "kind"]).agg(["mean", "std"])


def series(prefix, values):
    names = [f"{prefix}{v}" for v in values]
    return dict(x=values,
                ins=[agg.loc[(n, "in"), "mean"] for n in names],
                ins_sd=[agg.loc[(n, "in"), "std"] for n in names],
                cr=[agg.loc[(n, "cross"), "mean"] for n in names],
                cr_sd=[agg.loc[(n, "cross"), "std"] for n in names])


dt = series("DT_depth", [2, 3, 5, 8, 12, 20])
lg = series("LGBM_leaves", [3, 7, 15, 31, 63])
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.4), sharey=True, constrained_layout=True)
for ax, d, t, xl in [(axs[0], dt, "Decision tree", "Maximum depth"),
                     (axs[1], lg, "LightGBM", "Leaves per tree")]:
    x = np.arange(len(d["x"]))
    ax.axhline(.5, color="#8a8a85", lw=.8, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(x, d["ins"], yerr=d["ins_sd"], color="#2a78d6", lw=1.6, marker="o", ms=4.5,
                mec="white", mew=1, capsize=2, zorder=3)
    ax.errorbar(x, d["cr"], yerr=d["cr_sd"], color="#eb6834", lw=1.6, marker="s", ms=4.5,
                mec="white", mew=1, capsize=2, zorder=3)
    ax.set_xticks(x, [str(v) for v in d["x"]])
    ax.set_xlabel(xl)
    ax.set_title(t, fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.grid(axis="y", color="#e4e3dd", lw=.5)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(x[-1] + .15, d["ins"][-1] - .035, "in-dataset", color="#3d3d3a", va="center", fontsize=7)
    ax.text(x[-1] + .15, d["cr"][-1], "cross-dataset", color="#3d3d3a", va="center", fontsize=7)
    ax.set_xlim(-.4, len(x) - 1 + 1.6)
axs[0].text(.0, .515, "chance", color="#5f5e5a", fontsize=6.5)
axs[0].set_ylabel("ROC-AUC")
fig.legend(handles=[Line2D([], [], color="#2a78d6", marker="o", label="In-dataset (mean of 3 datasets)"),
                    Line2D([], [], color="#eb6834", marker="s", label="Cross-dataset (mean of 6 directions)")],
           loc="outside upper center", ncol=2, frameon=False, fontsize=7)
fig.savefig(f"{FIG}/fig_capacity.pdf")
plt.close(fig)
print("Saved figures to", FIG)
