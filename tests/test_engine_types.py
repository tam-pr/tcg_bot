from pathlib import Path

import pandas as pd
import pytest

from tcg_bot.decks import expand_deck
from tcg_bot.engine.types import PLAYS, Action, ActionKind, PokemonInPlay, State, owner_of

ROOT = Path(__file__).resolve().parents[1]
cards = pd.read_parquet(ROOT / "data" / "processed" / "cards.parquet")
deck_cards = pd.read_parquet(ROOT / "data" / "processed" / "deck_cards.parquet")


def deck(deck_id):
    return [c.card_id for c in expand_deck(deck_cards, cards, deck_id)]


def new_state(seed=0):
    return State.initial((deck("lopunny"), deck("mega_lucario")), seed)


def test_mirror_match_has_120_distinct_cards():
    s = State.initial((deck("lopunny"), deck("lopunny")), seed=0)
    assert sorted(s.all_uids()) == list(range(120))
    assert [owner_of(u) for u in s.players[1].deck] == [1] * 60


def test_wrong_deck_size_raises():
    with pytest.raises(ValueError):
        State.initial((deck("lopunny")[:59], deck("lopunny")), seed=0)


def test_clone_is_independent():
    s = new_state()
    p = s.players[0]
    p.active = PokemonInPlay(stack=[p.deck.pop()], in_play_since=0)
    c = s.clone()
    c.players[0].hand.append(c.players[0].deck.pop())
    c.players[0].active.energy.append(c.players[0].deck.pop())
    c.players[0].flags.names_used.add("Fan Call")
    assert p.hand == [] and p.active.energy == [] and p.flags.names_used == set()
    assert sorted(s.all_uids()) == sorted(c.all_uids()) == list(range(120))


def test_clone_keeps_the_same_random_sequence():
    s = new_state(seed=7)
    c = s.clone()
    assert [s.rng.random() for _ in range(5)] == [c.rng.random() for _ in range(5)]


def test_pokemon_id_stays_when_it_evolves():
    mon = PokemonInPlay(stack=[3], in_play_since=1)
    mon.stack.append(9)
    assert (mon.uid, mon.card) == (3, 9)


def test_first_turn_of_each_player():
    s = new_state()
    s.first_player, s.turn = 1, 2
    assert s.is_first_turn_of(0) and not s.is_first_turn_of(1)


def test_actions_can_be_hashed_and_compared():
    a = Action(ActionKind.END_TURN, player=0)
    assert a == Action(ActionKind.END_TURN, player=0) and len({a, a}) == 1


def test_every_row_of_plays_is_accounted_for():
    plays = pd.read_csv(ROOT / "configs" / "plays.csv", encoding="utf-8")
    assert set(plays["play_id"]) == set(PLAYS)