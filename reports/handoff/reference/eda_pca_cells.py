# Reference for task T2. Each '#%% md' or '#%% code' block is one notebook cell, in order.
#%% md
# PCA on the card features

Goal: a shorter numeric vector for each card, to feed the model and to describe decks.

Rows are cards (the whole pool, not only the two decks). Columns are stats, a few flags, type / weakness / resistance indicators and the effect verbs from `tcg_bot.features`. Family and card kind are left out of the matrix; they reach the model separately.
#%% code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from tcg_bot.features import card_features, standardize, tag_effects, VERBS, STATS

sns.set_theme(style="whitegrid", context="notebook")
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
PROC = ROOT / "data" / "processed"
cards = pd.read_parquet(PROC / "cards.parquet")
moves = pd.read_parquet(PROC / "moves.parquet")
deck_cards = pd.read_parquet(PROC / "deck_cards.parquet")

X = card_features(cards, moves)
info = cards.set_index("card_id").loc[X.index]
is_pokemon = info["family"].eq("pokemon").to_numpy()
assert len(X) == len(cards) and X.index.is_unique
assert (X.nunique() > 1).all(), "a column is constant"
assert X.loc[~is_pokemon & ~info["is_fossil"].to_numpy(bool), "hp"].isna().all()      # fossils are Items with HP
print("cards x columns:", X.shape, "| missing cells (Trainer and Energy stats):", int(X.isna().sum().sum()))
#%% md
## 1. Missing stats

Trainers and Energy cards have no HP or attack cost. Filling those with 0 makes the first component mostly answer "is this a Pokémon?", which we already know. Instead, each column is scaled using only the cards that have the value, and a missing value becomes 0 after scaling, so it pulls in no direction.
#%% code
zero = X.fillna(0)
Z_zero = (zero - zero.mean()) / zero.std(ddof=0)
mean, std = X.mean(), X.std(ddof=0)
Z = standardize(X, mean, std)

for name, M in [("missing = 0", Z_zero), ("missing = neutral", Z)]:
    scores = PCA(3, svd_solver="full").fit_transform(M)
    corr = [abs(np.corrcoef(scores[:, i], is_pokemon)[0, 1]) for i in range(3)]
    print(f"{name:18s} |correlation with is_pokemon|  PC1 {corr[0]:.2f}  PC2 {corr[1]:.2f}  PC3 {corr[2]:.2f}")
#%% md
## 2. Is the data suited to PCA?

The Kaiser-Meyer-Olkin (KMO) measure asks how much of the correlation between columns is shared. A common reading is: below 0.5 poor, 0.6 to 0.7 mediocre, above 0.8 good. It is not designed for sets of indicator columns that exclude each other (a card has one type), so it is shown by block.
#%% code
def kmo(M):
    R = np.corrcoef(M.to_numpy(), rowvar=False)
    S = np.linalg.pinv(R)
    partial = -S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    np.fill_diagonal(R, 0); np.fill_diagonal(partial, 0)
    return (R ** 2).sum() / ((R ** 2).sum() + (partial ** 2).sum())

types = [c for c in X if c.startswith(("type_", "weak_", "resist_"))]
verbs = list(VERBS)
stats_flags = [c for c in X if c not in types + verbs]
for name, cols in [("all columns", list(X)), ("stats and flags", stats_flags), ("effect verbs", verbs), ("type / weakness / resistance", types)]:
    print(f"{name:30s} {len(cols):2d} columns  KMO {kmo(Z[cols]):.2f}")
#%% md
## 3. How many components, and which columns skip the PCA

**Parallel analysis** keeps a component only if it explains more than the same position does in shuffled data (95th percentile of 200 shuffles).

**Bypass rule:** a column that the kept components rebuild poorly (R² below 0.5) is taken out of the PCA and passed to the model unchanged. The PCA is refitted until no column falls under the line.
#%% code
def parallel_analysis(M, n=200, q=95, seed=0):
    rng = np.random.default_rng(seed)
    A = M.to_numpy()
    eig = lambda B: np.linalg.eigvalsh(np.cov(B, rowvar=False))[::-1]
    real = eig(A)
    null = np.array([eig(np.column_stack([rng.permutation(col) for col in A.T])) for _ in range(n)])
    threshold = np.percentile(null, q, axis=0)
    above = real > threshold
    k = len(real) if above.all() else int(np.argmin(above))
    return k, real, threshold

def column_r2(M, k):
    p = PCA(k, svd_solver="full").fit(M)
    A, rebuilt = M.to_numpy(), p.inverse_transform(p.transform(M))
    return pd.Series(1 - ((A - rebuilt) ** 2).sum(0) / ((A - A.mean(0)) ** 2).sum(0), index=M.columns)

bypass = []
while True:
    bulk = [c for c in X if c not in bypass]
    k, eigenvalues, threshold = parallel_analysis(Z[bulk])
    r2 = column_r2(Z[bulk], k)
    new = list(r2[r2 < 0.5].index)
    print(f"{len(bulk)} columns in the PCA -> {k} components | newly below 0.5: {new}")
    if not new:
        break
    bypass += new

pca = PCA(k, svd_solver="full").fit(Z[bulk])
cumulative = np.cumsum(PCA(svd_solver="full").fit(Z[bulk]).explained_variance_ratio_)
print(f"\nPCA: {len(bulk)} columns -> {k} components, {cumulative[k - 1]:.1%} of their variance")
print(f"bypass ({len(bypass)}): {bypass}")
print(f"model input per card: {k} + {len(bypass)} = {k + len(bypass)}, down from {X.shape[1]}")
#%% code
fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1, 1.1]})
n = np.arange(1, len(eigenvalues) + 1)
axes[0].plot(n, eigenvalues, marker="o", ms=3, label="card data")
axes[0].plot(n, threshold, ls="--", label="shuffled data (95th percentile)")
axes[0].axvline(k, color="k", lw=1); axes[0].set_title(f"Parallel analysis: keep {k}")
axes[0].set_xlabel("component"); axes[0].set_ylabel("eigenvalue"); axes[0].legend()
axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative)
axes[1].axvline(k, color="k", lw=1); axes[1].axhline(cumulative[k - 1], color="k", lw=1, ls=":")
axes[1].set_title("Cumulative variance"); axes[1].set_xlabel("components kept"); axes[1].set_ylim(0, 1.02)
worst = r2.sort_values().head(15)
axes[2].barh(worst.index, worst.values, color="#4c72b0"); axes[2].axvline(0.5, color="#c44e52", ls="--")
axes[2].set_title("15 worst-rebuilt columns that stay in"); axes[2].set_xlabel("R²"); axes[2].set_xlim(0, 1)
plt.tight_layout(); plt.show()
#%% md
## 4. What the components mean
#%% code
scores = pd.DataFrame(pca.transform(Z[bulk]), index=X.index, columns=[f"PC{i + 1}" for i in range(k)])
loadings = pd.DataFrame(pca.components_.T, index=bulk, columns=scores.columns)
for pc in scores.columns[:5]:
    top = loadings[pc].abs().sort_values(ascending=False).head(6).index
    print(pc, f"({pca.explained_variance_ratio_[int(pc[2:]) - 1]:.1%}):", ", ".join(f"{c} {loadings.at[c, pc]:+.2f}" for c in top))

shown = loadings.iloc[:, :5]
shown = shown[shown.abs().max(axis=1) >= 0.2]
fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1.4]})
sns.heatmap(shown, cmap="RdBu_r", center=0, ax=axes[0]); axes[0].set_title("Loadings, first 5 components (|loading| >= 0.2)")
for fam, colour in [("pokemon", "#4c72b0"), ("trainer", "#dd8452"), ("energy", "#55a868")]:
    pick = info["family"].eq(fam)
    axes[1].scatter(scores.loc[pick, "PC1"], scores.loc[pick, "PC2"], s=10, alpha=0.35, color=colour, label=fam)
in_deck = scores.loc[deck_cards["card_id"].unique()]
axes[1].scatter(in_deck["PC1"], in_deck["PC2"], s=40, facecolor="none", edgecolor="k", label="in one of the two decks")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].set_title("All cards on the first two components"); axes[1].legend()
plt.tight_layout(); plt.show()
#%% md
## 5. Robustness

- **Bootstrap:** refit on resampled cards. The *subspace* score asks whether the same space is found; the per-component score asks whether each axis keeps its meaning.
- **FAMD-style variant:** weight the type / weakness / resistance indicators by 1 / sqrt(frequency), as mixed-data methods do, and see whether card neighbours change.
- **Neighbours:** how many of a card's 5 nearest cards in the full scaled table are still its nearest in the short vector.
#%% code
rng = np.random.default_rng(0)
A = Z[bulk].to_numpy()
subspace, axis = [], []
for _ in range(100):
    b = PCA(k, svd_solver="full").fit(A[rng.integers(0, len(A), len(A))]).components_
    overlap = pca.components_ @ b.T
    subspace.append((np.linalg.svd(overlap, compute_uv=False) ** 2).mean())
    axis.append(np.abs(overlap).max(axis=1))
axis = np.median(axis, axis=0)
print(f"subspace stability {np.mean(subspace):.2f} (1 = identical)")
print("components that keep their meaning (median |cos| >= 0.8):", [f"PC{i + 1}" for i in range(k) if axis[i] >= 0.8])

def neighbours(M, n=5):
    V = np.asarray(M, dtype=float)
    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    S = V @ V.T
    np.fill_diagonal(S, -9)
    return np.argsort(-S, axis=1)[:, :n]

def shared(a, b):
    return np.mean([len(set(i) & set(j)) / len(i) for i, j in zip(a, b)])

famd = Z[bulk].copy()
for col in [c for c in bulk if c in types]:
    p = X[col].mean()
    famd[col] = (X[col] - p) / np.sqrt(p)
vectors = np.column_stack([scores, Z[bypass]])                # what the model would get
raw_nn = neighbours(Z)
print(f"PCA vs FAMD-style, shared neighbours: {shared(neighbours(pca.transform(Z[bulk])), neighbours(PCA(k, svd_solver='full').fit_transform(famd))):.2f}")
print(f"final vector vs full table, shared neighbours: {shared(neighbours(vectors), raw_nn):.2f}")

names = info["card_name"].to_numpy()
nn = neighbours(vectors, 4)
for name in ["Mega Lopunny ex", "Mega Lucario ex", "Riolu", "Solrock", "Ultra Ball", "Poké Pad", "Switch", "Boss's Orders"]:
    hit = np.flatnonzero(names == name)
    if len(hit):
        print(f"  {name:18s} -> {', '.join(names[j] for j in nn[hit[0]])}")
#%% md
## 6. Save the fitted PCA and the card vectors

The PCA is fitted once on the whole pool and then frozen, so training and play use the same vectors. The last check lists cards in the two decks that the vector cannot tell apart.
#%% code
card_vectors = pd.concat([scores, Z[bypass]], axis=1).reset_index()
card_vectors.to_parquet(PROC / "card_vectors.parquet", index=False)
joblib.dump({"columns": list(X), "mean": mean, "std": std, "bulk": bulk, "bypass": bypass, "pca": pca},
            PROC / "card_pca.joblib")

again = joblib.load(PROC / "card_pca.joblib")
Z_again = standardize(card_features(cards, moves)[again["columns"]], again["mean"], again["std"])
assert np.allclose(again["pca"].transform(Z_again[again["bulk"]]), scores)
print("saved", card_vectors.shape, "and card_pca.joblib")

deck_rows = card_vectors.set_index("card_id").loc[deck_cards["card_id"].unique()].round(6)
twins = deck_rows[deck_rows.duplicated(keep=False)]
print(f"deck cards: {len(deck_rows)} | sharing a vector with another deck card: {len(twins)}")
for _, group in twins.groupby(list(twins.columns)):
    print("   identical:", [info.at[i, "card_name"] for i in group.index])
#%% md
## What this establishes, and what it does not

- The short vector is a moderate reduction, not a dramatic one: the effect verbs are close to independent of each other, so they compress little.
- KMO is low over all columns. PCA is justified here by the reduction and the neighbour checks, not by strong factor structure.
- Only the first few components can be named. The model uses the space as a whole.
- Effects are described by verbs only: no amounts and no filters, so a search for Basic Pokémon and a search for Energy can look alike.
- Verbs come from the regex patterns; the hand-labelled holdout has not been scored yet.
- Whether PCA helps the model play better is not tested here. That needs the ablation (PCA, PCA plus bypass, raw columns, learned embedding).
