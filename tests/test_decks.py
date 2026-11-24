from pathlib import Path

import pandas as pd
import pytest

from tcg_bot.decks import expand_deck

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
cards = pd.read_parquet(PROC / "cards.parquet")
deck_cards = pd.read_parquet(PROC / "deck_cards.parquet")
DECKS = sorted(deck_cards["deck_id"].unique())


@pytest.mark.parametrize("deck_id", DECKS)
def test_every_deck_expands_to_60_instances(deck_id):
    inst = expand_deck(deck_cards, cards, deck_id)
    assert len(inst) == 60
    assert [i.uid for i in inst] == list(range(60))


@pytest.mark.parametrize("deck_id", DECKS)
def test_tags_are_unique_and_follow_the_pattern(deck_id):
    inst = expand_deck(deck_cards, cards, deck_id)
    assert len({i.tag for i in inst}) == 60
    for fam in ("pokemon", "trainer", "energy"):
        numbers = [int(i.tag.rsplit("_", 1)[1]) for i in inst if i.tag.startswith(f"{deck_id}_{fam}_")]
        assert numbers == list(range(1, len(numbers) + 1))      # 1, 2, 3, ... with no gaps


@pytest.mark.parametrize("deck_id", DECKS)
def test_instance_counts_match_copies(deck_id):
    inst = expand_deck(deck_cards, cards, deck_id)
    counts = pd.Series([i.card_id for i in inst]).value_counts().sort_index().to_dict()
    expected = deck_cards[deck_cards["deck_id"] == deck_id].set_index("card_id")["copies"].sort_index().to_dict()
    assert counts == expected


@pytest.mark.parametrize("deck_id", DECKS)
def test_family_in_the_tag_matches_the_card(deck_id):
    family = cards.set_index("card_id")["family"]
    for i in expand_deck(deck_cards, cards, deck_id):
        assert i.tag.split("_")[-2] == family[i.card_id]


def test_two_dunsparce_versions_stay_separate():
    inst = expand_deck(deck_cards, cards, "lopunny")
    assert {i.card_id for i in inst if i.card_id in (65, 305)} == {65, 305}


def test_expansion_is_deterministic():
    assert expand_deck(deck_cards, cards, "lopunny") == expand_deck(deck_cards, cards, "lopunny")


def test_unknown_deck_raises():
    with pytest.raises(ValueError):
        expand_deck(deck_cards, cards, "no_such_deck")