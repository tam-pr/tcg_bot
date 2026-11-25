"""Full-pool effect tagger (task T8).

Every move with effect text gets verb tags (one boolean column v_<verb> per label, plus `labels` as text) from the 17 holdout labels (Appendix A of TASKS.md):
1. gold: the 54 hand-written effects of the two decks (effects_data.py) give exact labels;
2. regex: the patterns in tcg_bot.features (one source of truth);
3. classifier: a local TF-IDF + logistic regression model, trained on the regex-labelled rows, labels the rest;
   a row the classifier cannot label with probability >= threshold gets status `unknown`.

Holdout rows (configs/tag_holdout.csv) are never used for training or for tuning patterns: their move_ids are
removed from the training set, and score_holdout only reads them to report scores.
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd

from tcg_bot.effects import (ApplyStatus, CoinFlip, DamageSelf, Draw, Heal, Lockout, ModifyStat, MoveCards, Prevent,
                             SwitchActive, build)
from tcg_bot.effects_data import EFFECTS
from tcg_bot.features import notebook_verbs
from tcg_bot.paths import CONFIGS, PROC

LABELS = list(notebook_verbs())
MIN_EXAMPLES = 15          # a label needs this many regex examples before the classifier predicts it
THRESHOLD = 0.5


def _walk(x):
    if dataclasses.is_dataclass(x):
        yield x
        for f in dataclasses.fields(x):
            yield from _walk(getattr(x, f.name))
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from _walk(y)


def gold_labels(effect) -> set[str]:
    """The 17 labels implied by a hand-written effect (same rules as eda_tagging section 3)."""
    g = set()
    for n in _walk(effect):
        if isinstance(n, MoveCards):
            if n.look_n:
                g.add("look_top")
            if n.source == "deck":
                g.add("search_deck")
            if n.source == "discard" and n.destination == "hand":
                g.add("recover_discard")
            if n.destination == "attached_to_bench":
                g.add("attach_energy")
            if n.destination == "discard":
                g.add("discard")
            if n.source == "hand" and n.destination in ("deck_shuffle", "deck_bottom"):
                g.add("shuffle_hand")
            if n.destination == "hand":
                g.add("return_to_hand")
        elif isinstance(n, Draw):
            g.add("draw")
        elif isinstance(n, SwitchActive):
            g.add("switch")
        elif isinstance(n, Heal):
            g.add("heal")
        elif isinstance(n, ApplyStatus):
            g.add("status")
        elif isinstance(n, DamageSelf):
            g.add("recoil")
        elif isinstance(n, Lockout):
            g.add("lockout")
        elif isinstance(n, Prevent):
            g.add("prevent")
        elif isinstance(n, ModifyStat):
            g.add("modify_stat")
        elif isinstance(n, CoinFlip):
            g.add("coin_flip")
    if effect.damage is not None or effect.ignore_defender_effects or effect.ignore_weakness_resistance or \
            effect.works_only_if:
        g.add("damage_modifier")
    return g


def mask_factory(cards: pd.DataFrame):
    names = sorted({n for n in cards.loc[cards["family"] == "pokemon", "card_name"].dropna().unique()
                    if re.fullmatch(r"[A-Z][a-zé'’\-]{4,}", n)}, key=len, reverse=True)
    name_re = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b")

    def mask(s: str) -> str:
        s = name_re.sub("<NAME>", s)
        s = re.sub(r"\{[A-Za-z ]+\}", "<E>", s)
        s = re.sub(r"\b\d+\b", "<N>", s)
        s = re.sub(r"\b(Asleep|Burned|Confused|Paralyzed|Poisoned)\b", "<STATUS>", s)
        return re.sub(r"\b(heads|tails)\b", "<HT>", s)
    return mask


def regex_labels(text: str) -> set[str]:
    return {k for k, p in notebook_verbs().items() if re.search(p, text)}


def holdout_ids(path=None) -> set[str]:
    path = path or CONFIGS / "tag_holdout.csv"
    try:
        return set(pd.read_csv(path)["move_id"].astype(str))
    except FileNotFoundError:
        return set()


def tag_pool(moves: pd.DataFrame, cards: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """Tag every move with effect text. `exclude` (default: the holdout) is never used for training."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    exclude = holdout_ids() if exclude is None else exclude
    fx = moves[moves["effect_text"].notna()][["move_id", "card_id", "move_kind", "effect_text"]].copy()
    mask = mask_factory(cards)
    fx["masked"] = fx["effect_text"].map(mask)
    rx = [regex_labels(t) for t in fx["effect_text"]]
    gold = {o.move_id: gold_labels(o) for o in (build(e) for e in EFFECTS)}

    train = np.array([bool(r) and m not in exclude for r, m in zip(rx, fx["move_id"])])
    usable = [k for k in LABELS if sum(k in r for r, t in zip(rx, train) if t) >= MIN_EXAMPLES]
    Y = np.array([[k in r for k in usable] for r in rx], dtype=int)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(fx["masked"][train])
    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=10, class_weight="balanced"))
    clf.fit(vec.transform(fx["masked"][train]), Y[train])
    proba = clf.predict_proba(vec.transform(fx["masked"]))

    rows = []
    for i, r in enumerate(fx.itertuples()):
        if r.move_id in gold:
            labels, source = gold[r.move_id], "gold"
        elif rx[i]:
            labels, source = rx[i], "regex"
        else:
            labels = {k for k, p in zip(usable, proba[i]) if p >= THRESHOLD}
            source = "classifier"
        status = "ok" if labels else "unknown"
        rows.append({"move_id": r.move_id, "card_id": int(r.card_id), "move_kind": r.move_kind,
                     "labels": ";".join(sorted(labels)) if labels else "unknown", "source": source, "status": status,
                     "max_proba": float(proba[i].max()) if source == "classifier" else np.nan,
                     **{f"v_{k}": k in labels for k in LABELS}})
    out = pd.DataFrame(rows)
    out.attrs["usable_labels"] = usable
    return out


def score_holdout(tags: pd.DataFrame, path=None) -> pd.DataFrame:
    """Macro F1 on the person-labelled holdout, reported separately for rows tagged by regex and by the
    classifier (the two groups). Raises until the holdout is fully labelled (task H3)."""
    from sklearn.metrics import f1_score
    path = path or CONFIGS / "tag_holdout.csv"
    hold = pd.read_csv(path)
    if not {"move_id", "labels"} <= set(hold.columns):
        raise ValueError(f"{path} needs columns move_id and labels, has {hold.columns.tolist()}")
    lab = hold["labels"].astype("string")
    if lab.isna().any() or (lab.str.strip() == "").any():
        raise ValueError(f"{path}: {int(lab.isna().sum() + (lab.str.strip() == '').sum())} rows are not labelled yet (H3)")
    truth = {m: set() if x.strip() == "none" else set(x.replace(" ", "").split(";"))
             for m, x in zip(hold["move_id"].astype(str), lab)}
    bad = {l for s in truth.values() for l in s} - set(LABELS)
    if bad:
        raise ValueError(f"unknown labels in the holdout: {sorted(bad)}")
    t = tags.set_index("move_id").loc[list(truth)]
    rows = []
    for group, g in t.groupby(t["source"].where(t["source"] != "gold", "regex")):
        Yt = np.array([[k in truth[m] for k in LABELS] for m in g.index], dtype=int)
        Yp = g[[f"v_{k}" for k in LABELS]].to_numpy(dtype=int)
        rows.append({"group": group, "n": len(g), "macro_f1": f1_score(Yt, Yp, average="macro", zero_division=0),
                     "micro_f1": f1_score(Yt, Yp, average="micro", zero_division=0),
                     "exact_match": float((Yt == Yp).all(axis=1).mean()),
                     "unknown": int((g["status"] == "unknown").sum())})
    return pd.DataFrame(rows).set_index("group")


def build_move_tags() -> pd.DataFrame:
    cards = pd.read_parquet(PROC / "cards.parquet")
    moves = pd.read_parquet(PROC / "moves.parquet")
    tags = tag_pool(moves, cards)
    tags.to_parquet(PROC / "move_tags.parquet", index=False)
    return tags


if __name__ == "__main__":
    t = build_move_tags()
    print(t["source"].value_counts().to_dict(), t["status"].value_counts().to_dict())
