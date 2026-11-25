"""T2: the feature matrix and the fitted PCA transform."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from tcg_bot.effects import SwitchActive, build
from tcg_bot.effects_data import EFFECTS
from tcg_bot.features import (BYPASS, COLUMNS, MERGE, VERBS, CardPCA, card_features, merged_labels, standardize,
                              verb_flags)
from tcg_bot.paths import PROC

cards = pd.read_parquet(PROC / "cards.parquet")
moves = pd.read_parquet(PROC / "moves.parquet")
F = card_features(cards, moves)


def walk(x):
    if dataclasses.is_dataclass(x):
        yield x
        for f in dataclasses.fields(x):
            yield from walk(getattr(x, f.name))
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from walk(y)


def test_one_row_per_card():
    assert len(F) == len(cards) and F.index.is_unique and set(F.index) == set(cards["card_id"])


def test_no_missing_values_after_standardize():
    for missing in ("neutral", "zero"):
        Z, _ = standardize(F[COLUMNS], missing)
        assert not Z.isna().any().any()


def test_energy_and_non_fossil_trainers_have_no_hp_before_standardize():
    fossil = cards.set_index("card_id")["is_fossil"].reindex(F.index).astype(bool)
    other = (F["family"] != "pokemon") & ~fossil
    assert F.loc[other, "hp"].isna().all()
    assert F.loc[F["family"] == "pokemon", "hp"].notna().all()


def test_switch_verbs_agree_with_the_hand_written_effects():
    text = moves.set_index("move_id")["effect_text"]
    for o in (build(e) for e in EFFECTS):
        sides = {n.side for n in walk(o) if isinstance(n, SwitchActive)}
        if pd.isna(text.get(o.move_id)):
            continue
        v = verb_flags(text[o.move_id])
        assert v["switch_self"] == ("you" in sides), o.move_id
        assert v["switch_opp"] == ("opponent" in sides), o.move_id


def test_switch_mapping_to_the_single_holdout_label():
    assert set(MERGE.values()) == {"switch"}
    assert merged_labels({"switch_self": False, "switch_opp": True, "draw": False}) == {"switch": True, "draw": False}
    assert len({MERGE.get(k, k) for k in VERBS}) == 17


def test_pca_transform_shape_and_bypass_are_raw():
    m = CardPCA(5, BYPASS).fit(F)
    V = m.transform(F)
    assert V.shape == (len(F), 5 + len(BYPASS)) and m.n_inputs == 5 + len(BYPASS)
    assert set(np.unique(V[BYPASS].to_numpy())) <= {0.0, 1.0}
    assert not V.isna().any().any()


def test_saved_vectors_match_the_saved_model():
    import joblib
    from tcg_bot.features import card_features_v2, card_identity
    saved = pd.read_parquet(PROC / "card_vectors.parquet").set_index("card_id")
    bundle = joblib.load(PROC / "card_pca.joblib")
    assert bundle["version"] == 2
    again = bundle["model"].transform(card_features_v2(cards, moves))
    cols = list(again.columns)
    assert np.allclose(again[cols].to_numpy(), saved.loc[again.index, cols].to_numpy(), atol=1e-8)


def test_v2_no_two_deck_cards_share_a_model_input():
    """T7 done-when: with the identity input, every one of the 42 deck cards has its own model input."""
    deck_ids = sorted(pd.read_parquet(PROC / "deck_cards.parquet")["card_id"].unique())
    saved = pd.read_parquet(PROC / "card_vectors.parquet").set_index("card_id")
    assert len(deck_ids) == 42
    assert not saved.loc[deck_ids].round(6).duplicated().any()
    assert saved["card_index"].is_unique


def test_amounts_and_filters():
    from tcg_bot.features import amount_features
    a = amount_features("Search your deck for up to 2 Basic Pokémon with 70 HP or less and put them onto your Bench.")
    assert a["search_n"] == 2 and a["find_basic"] == 1 and a["find_pokemon"] == 1 and a["find_energy"] == 0
    assert amount_features("Heal all damage from 1 of your Pokémon.")["heal_n"] == 300
    assert amount_features("Draw a card.")["draw_n"] == 1 and amount_features("draw 4 cards")["draw_n"] == 4
    assert amount_features("this attack does 170 more damage")["bonus_n"] == 170
