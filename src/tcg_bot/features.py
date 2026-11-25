"""Card feature matrix and PCA (tasks T2 and T7).

One row per card (1,431). Columns: Pokémon stats, damage modifiers, type, Weakness and Resistance
one-hots, rule flags and the effect verbs found in the card's text. Family and kind columns are left
out of the PCA on purpose (otherwise PC1 is just Pokémon vs. Trainer); they reach the model by a
separate input.

Missing values: stats that only exist for Pokémon (HP, attacks, Weakness, ...) are NaN for Energy and
non-Fossil Trainers before `standardize`. The default ("neutral") puts them at the column mean after
scaling, so they carry no signal; "zero" fills them with 0 before scaling (the comparison in the notebook).

The verb patterns live here and only here (task T3 imports them). `switch` is split into
`switch_self` and `switch_opp`; MERGE maps the two back to the single `switch` label that the tagging
notebook, the gold labels and the holdout use.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- verbs (one regex per building block)
VERBS = {
    "search_deck":     r"[Ss]earch your deck",
    "look_top":        r"[Ll]ook at the top",
    "recover_discard": r"[Pp]ut .{0,60}from your discard pile into your hand",
    "attach_energy":   r"[Aa]ttach .{0,80}Energy",
    "discard":         r"\b[Dd]iscards?\b(?! pile)",
    "shuffle_hand":    r"shuffles? (?:their|your) hand|[Ss]huffle your hand",
    "return_to_hand":  r"into your hand",
    "draw":            r"[Dd]raws? (?:a|an|\d+|up to \d+|that many) cards?|draws? \d+",
    "switch_self":     r"[Ss]witch (?:this Pokémon|your Active)|[Ss]witch it with",
    "switch_opp":      r"[Ss]witch (?:in )?.{0,20}your opponent's|opponent switches|[Ss]witch out your opponent",
    "heal":            r"[Hh]eal",
    "status":          r"is now (?:Asleep|Burned|Confused|Paralyzed|Poisoned)",
    "recoil":          r"damage to itself",
    "lockout":         r"can't use (?!more than)|can't attack|can't retreat",
    "prevent":         r"[Pp]revent",
    "modify_stat":     r"attacks used by .{0,60}do \d+ more damage|[-+]\d+ HP|Retreat Cost.{0,80}less|takes \d+ less damage",
    "coin_flip":       r"[Ff]lips? (?:a|\d+) coins?",
    "damage_modifier": r"this attack does .{0,40}more damage|damage for each|isn't affected|does nothing|This attack's damage",
}
MERGE = {"switch_self": "switch", "switch_opp": "switch"}     # features verb -> holdout / notebook label
SWITCH_ANY = r"[Ss]witch"                                     # the notebook's single switch pattern


def notebook_verbs() -> dict[str, str]:
    """The 17 notebook / holdout labels: the feature patterns with switch_self and switch_opp replaced by a
    single `switch` pattern (SWITCH_ANY), in the order the tagging notebook uses."""
    out: dict[str, str] = {}
    for k, v in VERBS.items():
        if k in MERGE:
            out.setdefault(MERGE[k], SWITCH_ANY)
        else:
            out[k] = v
    return out


def merged_labels(row_verbs: dict[str, bool]) -> dict[str, bool]:
    """Feature verbs -> the 17 notebook labels (switch_self / switch_opp -> switch)."""
    out: dict[str, bool] = {}
    for k, v in row_verbs.items():
        key = MERGE.get(k, k)
        out[key] = out.get(key, False) or bool(v)
    return out


TYPES = ["G", "R", "W", "L", "P", "F", "D", "M", "N", "C"]
WEAK = ["R", "F", "L", "G", "D", "W", "M", "P"]
RESIST = ["F", "G"]
FLAGS = ["is_mega", "is_tera", "has_rule_box", "is_ace_spec", "is_ancient", "is_future", "has_owner"]   # is_ex == has_rule_box in this pool
STATS = ["hp", "retreat_cost", "n_attacks", "n_abilities", "max_damage", "mean_damage", "min_attack_cost",
         "max_attack_cost"]
COUNTS = ["n_effects"]                                        # moves with effect text, for every card
DMG_MODS = ["dmg_times", "dmg_minus"]
POKEMON_ONLY = STATS + DMG_MODS + [f"weak_{t}" for t in WEAK] + [f"resist_{t}" for t in RESIST]
VERB_COLS = list(VERBS)
COLUMNS = STATS + COUNTS + DMG_MODS + [f"type_{t}" for t in TYPES] + [f"weak_{t}" for t in WEAK] + \
    [f"resist_{t}" for t in RESIST] + FLAGS + VERB_COLS
# columns appended raw instead of compressed: rare, nearly independent of the rest (R² < 0.5 at the chosen k)
# and used by the engine or the effect filters. The rule is checked in the notebook.
BYPASS = ["is_ace_spec", "is_ancient", "has_owner", "switch_opp", "heal", "recoil", "modify_stat", "coin_flip", "status"]
ID_COLUMNS = ["family", "card_kind"]                          # kept out of the PCA; separate model input


def _eq(v, s) -> bool:
    return bool(pd.notna(v) and v == s)


def verb_flags(text: str | None) -> dict[str, bool]:
    t = text or ""
    return {k: bool(re.search(p, t)) for k, p in VERBS.items()}


def card_features(cards: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    """Raw features, one row per card (index card_id). Pokémon-only columns are NaN for other cards."""
    rows = []
    mv = moves.groupby("card_id")
    for c in cards.itertuples():
        m = mv.get_group(c.card_id) if c.card_id in mv.groups else moves.iloc[0:0]
        atk = m[m["move_kind"] == "attack"]
        r = {"card_id": c.card_id, "family": c.family, "card_kind": c.card_kind}
        is_poke = c.family == "pokemon"
        if is_poke:
            dmg = atk["damage_base"].astype("float").fillna(0)
            cost = atk["cost_total"].astype("float").fillna(0)
            r.update(hp=float(c.hp) if pd.notna(c.hp) else np.nan,
                     retreat_cost=float(c.retreat_cost) if pd.notna(c.retreat_cost) else np.nan,
                     n_attacks=float(len(atk)), n_abilities=float((m["move_kind"] == "ability").sum()),
                     max_damage=float(dmg.max()) if len(atk) else 0.0, mean_damage=float(dmg.mean()) if len(atk) else 0.0,
                     min_attack_cost=float(cost.min()) if len(atk) else 0.0,
                     max_attack_cost=float(cost.max()) if len(atk) else 0.0,
                     dmg_times=float((atk["damage_mod"] == "times").any()),
                     dmg_minus=float((atk["damage_mod"] == "minus").any()))
            for t in WEAK:
                r[f"weak_{t}"] = float(_eq(c.weakness, f"{{{t}}}"))
            for t in RESIST:
                r[f"resist_{t}"] = float(_eq(c.resistance, f"{{{t}}}"))
        else:
            for k in POKEMON_ONLY:
                r[k] = np.nan
            if bool(c.is_fossil):                             # a Fossil is played as a Basic Pokémon and has HP
                r["hp"] = float(c.hp) if pd.notna(c.hp) else np.nan
        for t in TYPES:
            r[f"type_{t}"] = float(_eq(c.type, f"{{{t}}}"))
        for f in FLAGS:
            r[f] = float(pd.notna(c.trainer_owner)) if f == "has_owner" else float(bool(getattr(c, f)))
        r["n_effects"] = float(m["effect_text"].notna().sum())
        verbs = {k: False for k in VERBS}
        for t in m["effect_text"].dropna():
            for k, v in verb_flags(t).items():
                verbs[k] |= v
        r.update({k: float(v) for k, v in verbs.items()})
        rows.append(r)
    df = pd.DataFrame(rows).set_index("card_id")
    return df[ID_COLUMNS + COLUMNS]


def standardize(X: pd.DataFrame, missing: str = "neutral", stats: dict | None = None):
    """z-scores. missing='neutral': NaN -> 0 after scaling (the column mean). missing='zero': NaN -> 0 before.
    Returns (Z, stats) where stats holds the means and scales for reuse on new cards."""
    X = X.astype(float)
    if missing == "zero":
        X = X.fillna(0.0)
    elif missing != "neutral":
        raise ValueError(f"missing must be 'neutral' or 'zero', got {missing!r}")
    if stats is None:
        mu, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
        stats = {"mean": mu, "std": sd}
    Z = ((X - stats["mean"]) / stats["std"]).fillna(0.0)
    return Z, stats


# ---------------------------------------------------------------- diagnostics
def kmo(Z: pd.DataFrame) -> tuple[float, pd.Series]:
    """Kaiser-Meyer-Olkin sampling adequacy (overall, per column), from correlations and partial correlations."""
    R = np.corrcoef(Z.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(R)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    P = -inv / d
    np.fill_diagonal(R, 0)
    np.fill_diagonal(P, 0)
    r2, p2 = R ** 2, P ** 2
    per = r2.sum(0) / (r2.sum(0) + p2.sum(0))
    return float(r2.sum() / (r2.sum() + p2.sum())), pd.Series(per, index=Z.columns)


def parallel_analysis(Z: np.ndarray, n_iter: int = 50, q: float = 95, seed: int = 0) -> tuple[int, np.ndarray, np.ndarray]:
    """Horn's parallel analysis with column permutations. Returns (k, eigenvalues, the random threshold)."""
    rng = np.random.default_rng(seed)
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    sims = []
    for _ in range(n_iter):
        P = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        sims.append(np.sort(np.linalg.eigvalsh(np.corrcoef(P, rowvar=False)))[::-1])
    thr = np.percentile(np.array(sims), q, axis=0)
    k = int(np.argmax(ev <= thr)) if np.any(ev <= thr) else len(ev)
    return k, ev, thr


def reconstruction_r2(Z: np.ndarray, k: int) -> np.ndarray:
    """R² of each column when rebuilt from the first k principal components."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=k).fit(Z)
    rec = p.inverse_transform(p.transform(Z))
    ss = ((Z - Z.mean(0)) ** 2).sum(0)
    return 1 - ((Z - rec) ** 2).sum(0) / np.where(ss == 0, 1, ss)


def subspace_stability(Z: np.ndarray, k: int, n_boot: int = 50, seed: int = 0) -> tuple[float, np.ndarray]:
    """Bootstrap: mean squared cosine of principal angles between the full-data and resampled k-subspaces,
    plus per-component |Tucker congruence| after matching."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    base = PCA(n_components=k).fit(Z).components_
    sims, cong = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(Z), len(Z))
        comp = PCA(n_components=k).fit(Z[idx]).components_
        s = np.linalg.svd(base @ comp.T, compute_uv=False)
        sims.append(float(np.mean(s ** 2)))
        cong.append(np.abs(np.sum(base * comp, axis=1)))
    return float(np.mean(sims)), np.mean(np.array(cong), axis=0)


def knn_preserved(A: np.ndarray, B: np.ndarray, k: int = 5) -> float:
    """Share of each card's k nearest neighbours in A that are also among its k nearest in B."""
    from sklearn.neighbors import NearestNeighbors
    na = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A, return_distance=False)[:, 1:]
    nb = NearestNeighbors(n_neighbors=k + 1).fit(B).kneighbors(B, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(na, nb)]))


def famd_scale(X: pd.DataFrame, binary: list[str], missing: str = "neutral") -> pd.DataFrame:
    """FAMD-style scaling: numeric columns as z-scores, binary columns centred and divided by sqrt(p)."""
    Z, _ = standardize(X.drop(columns=binary), missing)
    B = X[binary].astype(float).fillna(0.0)
    p = B.mean().replace(0, 1.0)
    out = pd.concat([Z, (B - p) / np.sqrt(p)], axis=1)
    return out[X.columns]


# ---------------------------------------------------------------- the fitted transform
class CardPCA:
    """standardize -> PCA on the bulk columns; bypass columns are appended raw (0/1)."""

    def __init__(self, k: int, bypass: list[str], missing: str = "neutral"):
        self.k, self.bypass, self.missing = k, list(bypass), missing

    def fit(self, F: pd.DataFrame) -> "CardPCA":
        from sklearn.decomposition import PCA
        self.bulk = [c for c in COLUMNS if c not in self.bypass]
        Z, self.stats = standardize(F[self.bulk], self.missing)
        self.pca = PCA(n_components=self.k, random_state=0).fit(Z.to_numpy())
        return self

    def transform(self, F: pd.DataFrame) -> pd.DataFrame:
        Z, _ = standardize(F[self.bulk], self.missing, self.stats)
        pcs = self.pca.transform(Z.to_numpy())
        out = pd.DataFrame(pcs, index=F.index, columns=[f"pc{i + 1}" for i in range(self.k)])
        for c in self.bypass:
            out[c] = F[c].fillna(0.0).astype(float).to_numpy()
        return out

    @property
    def n_inputs(self) -> int:
        return self.k + len(self.bypass)


# ================================================================ v2 (task T7): amounts, search filters, identity
AMOUNTS = {
    "draw_n":   r"[Dd]raws? (\d+|a) cards?",
    "heal_n":   r"[Hh]eal (\d+|all) damage",
    "bonus_n":  r"does (\d+) more damage",
    "search_n": r"[Ss]earch your deck for (?:up to )?(\d+|a|an) ",
}
SEARCH_FILTERS = {
    "find_pokemon":   r"Pokémon",
    "find_basic":     r"Basic (?!\{|Energy)",
    "find_evolution": r"Evolution|Stage 1|Stage 2",
    "find_energy":    r"Energy",
    "find_supporter": r"Supporter",
    "find_trainer":   r"Trainer|Item|Tool|Stadium",
}
AMOUNT_COLS = list(AMOUNTS)
FILTER_COLS = list(SEARCH_FILTERS)
COLUMNS_V2 = COLUMNS + AMOUNT_COLS + FILTER_COLS
HEAL_ALL = 300                     # "Heal all damage" counted as a large heal (the largest HP in the pool is 340)


def amount_features(text: str | None) -> dict[str, float]:
    t = text or ""
    out = {}
    for k, p in AMOUNTS.items():
        vals = []
        for m in re.finditer(p, t):
            v = m.group(1)
            vals.append(HEAL_ALL if v == "all" else 1 if v in ("a", "an") else int(v))
        out[k] = float(max(vals)) if vals else 0.0
    clauses = re.findall(r"[Ss]earch your deck for ([^.]*)", t)
    for k, p in SEARCH_FILTERS.items():
        out[k] = float(any(re.search(p, c) for c in clauses))
    return out


def card_features_v2(cards: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    """v1 columns plus effect amounts and search filters (largest value over the card's moves)."""
    F = card_features(cards, moves)
    extra = {cid: {k: 0.0 for k in AMOUNT_COLS + FILTER_COLS} for cid in F.index}
    for r in moves.dropna(subset=["effect_text"]).itertuples():
        a = amount_features(r.effect_text)
        e = extra[r.card_id]
        for k, v in a.items():
            e[k] = max(e[k], v)
    E = pd.DataFrame.from_dict(extra, orient="index").loc[F.index]
    return pd.concat([F, E], axis=1)


def card_identity(cards: pd.DataFrame) -> pd.DataFrame:
    """Integer inputs next to the PCA vector: the card's own index (for a learned embedding) and the index of
    the card it evolves from (-1 if none). Every card gets a different card_index."""
    ids = cards.sort_values("card_id")["card_id"].tolist()
    index = {c: i for i, c in enumerate(ids)}
    first_by_name: dict[str, int] = {}
    for c in cards.sort_values("card_id").itertuples():
        if c.family == "pokemon":
            first_by_name.setdefault(c.card_name, c.card_id)
    rows = []
    for c in cards.itertuples():
        src = first_by_name.get(c.previous_stage) if pd.notna(c.previous_stage) else None
        rows.append({"card_id": c.card_id, "card_index": index[c.card_id],
                     "evolves_from_index": index[src] if src is not None else -1})
    return pd.DataFrame(rows).set_index("card_id")


class CardPCAv2(CardPCA):
    """Same transform on the v2 columns."""

    def fit(self, F: pd.DataFrame) -> "CardPCAv2":
        from sklearn.decomposition import PCA
        self.bulk = [c for c in COLUMNS_V2 if c not in self.bypass]
        Z, self.stats = standardize(F[self.bulk], self.missing)
        self.pca = PCA(n_components=self.k, random_state=0).fit(Z.to_numpy())
        return self
