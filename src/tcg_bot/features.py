"""Card feature matrix for the PCA: stats, types and effect verbs, one row per card."""

import pandas as pd

VERBS = {
    "search_deck":     r"[Ss]earch your deck",
    "look_top":        r"[Ll]ook at the top",
    "recover_discard": r"[Pp]ut .{0,60}from your discard pile into your hand",
    "attach_energy":   r"[Aa]ttach .{0,80}Energy",
    "discard":         r"\b[Dd]iscards?\b(?! pile)",
    "shuffle_hand":    r"shuffles? (?:their|your) hand|[Ss]huffle your hand",
    "return_to_hand":  r"into your hand",
    "draw":            r"[Dd]raws? (?:a|an|\d+|up to \d+|that many) cards?|draws? \d+",
    "switch_self":     r"[Ss]witch (?:this|it\b|(?:your|1 of your)(?! opponent))|[Ss]witch in 1 of your (?!opponent)|[Ss]witch their Active",
    "switch_opp":      r"[Ss]witch(?: in| out)?[^.]{0,60}opponent|opponent switches",
    "heal":            r"[Hh]eal",
    "status":          r"is now (?:Asleep|Burned|Confused|Paralyzed|Poisoned)",
    "recoil":          r"damage to itself",
    "lockout":         r"can't use (?!more than)|can't attack|can't retreat",
    "prevent":         r"[Pp]revent",
    "modify_stat":     r"attacks used by .{0,60}do \d+ more damage|[-+]\d+ HP|Retreat Cost.{0,80}less|takes \d+ less damage",
    "coin_flip":       r"[Ff]lips? (?:a|\d+) coins?",
    "damage_modifier": r"this attack does .{0,40}more damage|damage for each|isn't affected|does nothing|This attack's damage",
}
STATS = ["hp", "retreat_cost", "n_attacks", "n_abilities", "max_damage",
         "min_cost", "max_cost", "max_typed_cost", "text_length"]
FLAGS = ["is_ex", "is_mega", "is_ace_spec", "is_tera", "is_ancient", "is_future",
         "is_fossil", "has_owner", "scaling_damage"]
TYPES = list("GRWLPFDMNC")


def tag_effects(text: pd.Series) -> pd.DataFrame:
    """One True/False column per verb, for a Series of effect texts."""
    text = text.fillna("")
    return pd.DataFrame({v: text.str.contains(p, regex=True).astype(bool) for v, p in VERBS.items()},
                        index=text.index)


def card_features(cards: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    """One row per card, indexed by card_id. A stat is NaN when the card cannot have it (Trainers, Energy)."""
    c = cards.set_index("card_id").sort_index()
    m = moves.join(tag_effects(moves["effect_text"]))
    m["is_attack"] = m["move_kind"].eq("attack")
    m["is_ability"] = m["move_kind"].eq("ability")
    m["typed_cost"] = m["cost_total"] - m["cost_C"]
    m["times"] = m["damage_mod"].eq("times").fillna(False).astype(bool)
    m["chars"] = m["effect_text"].str.len().fillna(0)
    per = m.groupby("card_id").agg(
        n_attacks=("is_attack", "sum"), n_abilities=("is_ability", "sum"),
        max_damage=("damage_base", "max"), min_cost=("cost_total", "min"), max_cost=("cost_total", "max"),
        max_typed_cost=("typed_cost", "max"), scaling_damage=("times", "max"), text_length=("chars", "sum"),
        **{v: (v, "max") for v in VERBS}).reindex(c.index)

    pokemon = c["family"].eq("pokemon")
    X = pd.DataFrame(index=c.index)
    X["hp"] = c["hp"].astype("Float64").astype(float)
    X["retreat_cost"] = c["retreat_cost"].astype("Float64").astype(float)
    for col in ["n_attacks", "n_abilities", "max_damage", "min_cost", "max_cost", "max_typed_cost"]:
        X[col] = per[col].astype("Float64").astype(float).fillna(0).where(pokemon)
    X["text_length"] = per["text_length"].astype("Float64").astype(float).fillna(0)

    for col in FLAGS[:7]:
        X[col] = c[col].astype(float)
    X["has_owner"] = c["trainer_owner"].notna().astype(float)
    X["scaling_damage"] = per["scaling_damage"].astype("boolean").fillna(False).astype(float)

    for prefix, col in [("type", "type"), ("weak", "weakness"), ("resist", "resistance")]:
        for t in TYPES:
            hit = c[col].eq("{" + t + "}").fillna(False).astype(float)
            if hit.sum():
                X[f"{prefix}_{t}"] = hit

    for v in VERBS:
        X[v] = per[v].astype("boolean").fillna(False).astype(float)
    return X


def standardize(X: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """Scale with statistics from the cards that have the value; a missing stat becomes 0 (neutral)."""
    return ((X - mean) / std).fillna(0.0)