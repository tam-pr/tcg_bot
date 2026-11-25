"""Figures for the paper (task T14). Run after the PCA notebook and the baseline script:

    uv run python reports/figures/make_figures.py

Writes pca_scree.png and baselines.png next to this file. ablation.png and training_curve.png come from
notebooks/evaluation.ipynb.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tcg_bot.features import BYPASS, COLUMNS_V2, card_features_v2, parallel_analysis, standardize  # noqa: E402
from tcg_bot.paths import PROC, REPORTS  # noqa: E402

FIG = REPORTS / "figures"
cards = pd.read_parquet(PROC / "cards.parquet")
moves = pd.read_parquet(PROC / "moves.parquet")
F = card_features_v2(cards, moves)
Z, _ = standardize(F[COLUMNS_V2])
bulk = [c for c in COLUMNS_V2 if c not in BYPASS]
k, ev, thr = parallel_analysis(Z[bulk].to_numpy(), n_iter=50)
fig, ax = plt.subplots(figsize=(5.5, 3.3))
ax.plot(range(1, len(ev) + 1), ev, "o-", ms=3, label="card features (v2)")
ax.plot(range(1, len(thr) + 1), thr, "--", label="random, 95th percentile")
ax.axvline(k, color="grey", lw=0.8)
ax.set_xlabel("component"); ax.set_ylabel("eigenvalue"); ax.set_title(f"Parallel analysis keeps {k} components")
ax.legend(); plt.tight_layout(); fig.savefig(FIG / "pca_scree.png", dpi=130)

base = json.loads((REPORTS / "results" / "baselines.json").read_text())
rows = [(f"{m['a']} vs {m['b'].replace('_b', '')}", m["a_win_rate"], *m["ci95"], m["games"]) for m in base["matches"]]
fig, ax = plt.subplots(figsize=(6, 3.2))
for i, (label, w, lo, hi, n) in enumerate(rows):
    ax.errorbar(w, i, xerr=[[w - lo], [hi - w]], fmt="o", capsize=3)
    ax.text(1.01, i, f"n={n}", va="center", fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.axvline(0.5, color="grey", lw=0.7)
ax.set_xlim(0, 1); ax.set_xlabel("win rate (95% Wilson interval)"); ax.set_title("Baseline league")
plt.tight_layout(); fig.savefig(FIG / "baselines.png", dpi=130)
print("wrote", FIG / "pca_scree.png", FIG / "baselines.png")
