import dataclasses
from pathlib import Path

import pandas as pd

from tcg_bot.effects import SwitchActive, build
from tcg_bot.effects_data import EFFECTS
from tcg_bot.features import card_features, standardize, tag_effects

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
cards = pd.read_parquet(PROC / "cards.parquet")
moves = pd.read_parquet(PROC / "moves.parquet")
X = card_features(cards, moves)
info = cards.set_index("card_id").loc[X.index]


def walk(x):
    """Yield every dataclass object inside an effect."""
    if dataclasses.is_dataclass(x):
        yield x
        for f in dataclasses.fields(x):
            yield from walk(getattr(x, f.name))
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from walk(y)


def test_one_row_per_card():
    assert len(X) == len(cards)
    assert X.index.is_unique


def test_no_missing_values_after_standardize():
    Z = standardize(X, X.mean(), X.std(ddof=0))
    assert not Z.isna().any().any()


def test_trainers_and_energy_have_no_hp_before_standardize():
    no_hp = info["family"].ne("pokemon") & ~info["is_fossil"].astype(bool)      # fossils are Items with HP
    assert no_hp.any()
    assert X.loc[no_hp, "hp"].isna().all()


def test_switch_verbs_agree_with_the_hand_written_effects():
    text = moves.set_index("move_id")["effect_text"]
    ids = [e["move_id"] for e in EFFECTS if pd.notna(text[e["move_id"]])]
    tags = tag_effects(text.loc[ids])
    sides_seen = set()
    for e in EFFECTS:
        if e["move_id"] not in tags.index:
            continue
        sides = {s.side for s in walk(build(e)) if isinstance(s, SwitchActive)}
        sides_seen |= sides
        assert tags.at[e["move_id"], "switch_self"] == ("you" in sides), e["move_id"]
        assert tags.at[e["move_id"], "switch_opp"] == ("opponent" in sides), e["move_id"]
    assert {"you", "opponent"} <= sides_seen, "the check would be empty"