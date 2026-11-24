import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from tcg_bot.effects import Effect, Filter, build
from tcg_bot.effects_data import EFFECTS

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
cards = pd.read_parquet(PROC / "cards.parquet")
moves = pd.read_parquet(PROC / "moves.parquet")
deck_cards = pd.read_parquet(PROC / "deck_cards.parquet")
in_scope = moves[moves["card_id"].isin(deck_cards["card_id"])].set_index("move_id")
built = [build(e) for e in EFFECTS]

# effects that are allowed to be anything other than "ok", and why
NOT_OK = {"1237_0": "complex", "858_0": "no_effect_in_scope"}


def walk(x):
    """Yield every dataclass object inside an effect."""
    if dataclasses.is_dataclass(x):
        yield x
        for f in dataclasses.fields(x):
            yield from walk(getattr(x, f.name))
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from walk(y)


def test_one_effect_per_in_scope_move():
    ids = [e["move_id"] for e in EFFECTS]
    assert len(ids) == len(set(ids)), "duplicate move_id"
    assert set(ids) == set(in_scope.index)


def test_every_effect_builds():
    assert all(isinstance(o, Effect) for o in built)


def test_only_known_effects_are_not_ok():
    assert {o.move_id: o.status for o in built if o.status != "ok"} == NOT_OK


def test_attack_timing_matches_the_move_kind():
    for o in built:
        is_attack = in_scope.loc[o.move_id, "move_kind"] == "attack"
        assert is_attack == (o.timing == "attack"), o.move_id


def test_plain_damage_attacks_are_exactly_the_moves_without_text():
    plain = {o.move_id for o in built if o.timing == "attack" and not o.steps and o.damage is None
             and not o.works_only_if and not o.ignore_defender_effects and not o.ignore_weakness_resistance}
    assert plain == set(in_scope.index[in_scope["effect_text"].isna()])


def test_ability_usage_tags_match_the_card_text():
    for o in built:
        if in_scope.loc[o.move_id, "move_kind"] != "ability":
            continue
        text = in_scope.loc[o.move_id, "effect_text"]
        name = in_scope.loc[o.move_id, "move_name"]
        if "can't use more than 1" in text:
            assert o.usage.scope in ("per_name", "per_name_contains"), o.move_id
            if o.usage.scope == "per_name":
                assert o.usage.key == name, o.move_id
        elif "Once during your" in text:
            assert o.usage.scope == "per_pokemon", o.move_id
        else:
            assert o.usage.scope == "none", o.move_id
        assert o.usage.first_turn_only == ("Once during your first turn" in text), o.move_id


def test_filters_only_use_values_that_exist_in_the_cards_table():
    kinds, types, families = set(cards["card_kind"].dropna()), set(cards["type"].dropna()), set(cards["family"])
    for o in built:
        for f in (n for n in walk(o) if isinstance(n, Filter)):
            assert set(f.card_kind) <= kinds, (o.move_id, f.card_kind)
            assert f.type is None or f.type in types, (o.move_id, f.type)
            assert f.family is None or f.family in families, (o.move_id, f.family)


@pytest.mark.parametrize("bad", [
    {"block": "teleport"},
    {"block": "draw", "count": {"block": "fixed", "n": 1}, "colour": "red"},
    {"block": "move_cards", "source": "graveyard", "destination": "hand"},
    {"block": "cond", "name": "is_raining"},
    {"block": "apply_status", "status": "Frozen", "target": {"block": "target"}},
    {"block": "effect", "move_id": "x", "timing": "whenever"},
])
def test_schema_rejects_mistakes(bad):
    with pytest.raises((ValueError, TypeError)):
        build(bad)