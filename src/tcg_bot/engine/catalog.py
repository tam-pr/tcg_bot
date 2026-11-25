"""Build the static card Catalog for a game from the processed tables (task T4).

Tables are read once and cached. Every card that a deck uses must have its effects written as data
in effects_data.py; a card whose text has no effect data raises instead of being skipped.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from tcg_bot.decks import expand_deck
from tcg_bot.effects import build
from tcg_bot.effects_data import EFFECTS
from tcg_bot.paths import PROC

from .types import UID_STRIDE, Attack, Catalog, CardRef, CardStatic

COST_TYPES = ("C", "D", "F", "G", "R", "W", "L", "P", "M")


@lru_cache(maxsize=1)
def tables() -> dict[str, pd.DataFrame]:
    names = ("cards", "moves", "deck_cards", "game_constants", "plays", "energy_provides", "special_conditions")
    return {n: pd.read_parquet(PROC / f"{n}.parquet") for n in names}


def _opt_str(v):
    return None if pd.isna(v) else str(v)


def _opt_int(v):
    return None if pd.isna(v) else int(v)


@lru_cache(maxsize=1)
def built_effects() -> dict:
    return {o.move_id: o for o in (build(e) for e in EFFECTS)}


@lru_cache(maxsize=None)
def card_static(card_id: int) -> CardStatic:
    t = tables()
    c = t["cards"].set_index("card_id").loc[card_id]
    mv = t["moves"][t["moves"]["card_id"] == card_id].sort_values("move_idx")
    fx = built_effects()
    attacks, abilities, effects = [], [], []
    for r in mv.itertuples():
        if r.move_kind == "attack":
            cost = tuple((k, int(getattr(r, f"cost_{k}"))) for k in COST_TYPES
                         if pd.notna(getattr(r, f"cost_{k}")) and int(getattr(r, f"cost_{k}")) > 0)
            attacks.append(Attack(r.move_id, str(r.move_name), cost, _opt_int(r.damage_base) or 0, _opt_str(r.damage_mod)))
        elif r.move_kind == "ability":
            abilities.append(r.move_id)
        elif r.move_kind in ("trainer_effect", "energy_effect"):
            effects.append(r.move_id)
        else:
            raise NotImplementedError(f"card {card_id}: move kind {r.move_kind!r} is not supported by the engine")
        if r.move_id not in fx:
            raise NotImplementedError(f"card {card_id} ({c['card_name']}): move {r.move_id} has no effect data")
    provides: tuple[str, ...] = ()
    if c["family"] == "energy":
        ep = t["energy_provides"].set_index("card_id")
        if card_id not in ep.index:
            raise NotImplementedError(f"energy card {card_id} has no row in energy_provides")
        row = ep.loc[card_id]
        if pd.notna(row["condition"]) or pd.notna(row["attach_restriction"]) or bool(row["discard_end_of_turn"]):
            raise NotImplementedError(f"energy card {card_id}: conditional Energy is not supported yet")
        if "|" in str(row["provides"]) or str(row["provides"]) == "ALL":
            raise NotImplementedError(f"energy card {card_id}: multi-type Energy is not supported yet")
        provides = (str(row["provides"]),) * int(row["amount"])
    if c["family"] == "trainer" and c["card_kind"] in ("Item", "Supporter", "Stadium", "Pokémon Tool") and not effects:
        raise NotImplementedError(f"trainer {card_id} has no effect")
    if bool(c["is_fossil"]):
        raise NotImplementedError(f"card {card_id}: Fossils are not supported yet")
    return CardStatic(
        card_id=int(card_id), name=str(c["card_name"]), family=str(c["family"]), kind=str(c["card_kind"]),
        evolves_from=_opt_str(c["previous_stage"]), hp=_opt_int(c["hp"]), type=_opt_str(c["type"]),
        weakness=_opt_str(c["weakness"]), resistance=_opt_str(c["resistance"]), retreat_cost=_opt_int(c["retreat_cost"]),
        is_ex=bool(c["is_ex"]), is_mega=bool(c["is_mega"]), has_rule_box=bool(c["has_rule_box"]),
        is_ace_spec=bool(c["is_ace_spec"]), attacks=tuple(attacks), ability_ids=tuple(abilities),
        effect_ids=tuple(effects), provides=provides)


@lru_cache(maxsize=1)
def constants() -> dict[str, int]:
    gc = tables()["game_constants"]
    out = {}
    for r in gc.itertuples():
        if r.status == "verified" and pd.notna(r.value) and str(r.value) != "":
            out[r.key] = int(r.value)
    return out


@lru_cache(maxsize=16)
def make_catalog(deck0: str, deck1: str) -> Catalog:
    t = tables()
    refs: dict[int, CardRef] = {}
    lists = []
    for p, deck_id in enumerate((deck0, deck1)):
        inst = expand_deck(t["deck_cards"], t["cards"], deck_id)
        lists.append(tuple(i.card_id for i in inst))
        for i in inst:
            uid = p * UID_STRIDE + i.uid
            refs[uid] = CardRef(uid, p, f"p{p}_{i.tag}", card_static(i.card_id))
    fx = built_effects()
    return Catalog(decks=(deck0, deck1), refs=refs, effects=fx, constants=constants(),
                   deck_lists=(lists[0], lists[1]))
