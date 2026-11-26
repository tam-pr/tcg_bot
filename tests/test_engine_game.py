import random

import pytest

from tcg_bot.engine.catalog import catalog
from tcg_bot.engine.game import card_of, legal_actions, new_game, start_turn, step
from tcg_bot.engine.types import Action, ActionKind, Phase

DECKS = ("lopunny", "mega_lucario")


def play_opening(seed):
    """Random legal actions until turn 1 starts. Returns the state and the bonus cards each player drew."""
    pick = random.Random(seed)
    state, bonus = new_game(DECKS, seed), [0, 0]
    while state.phase == Phase.SETUP:
        action = pick.choice(legal_actions(state))
        if state.pending and state.pending.kind == "mulligan_draw":
            bonus[action.player] = action.option
        state = step(state, action)
    return state, bonus


@pytest.mark.parametrize("seed", range(200))
def test_every_opening_is_valid(seed):
    state, bonus = play_opening(seed)
    assert state.phase == Phase.MAIN and state.turn == 1 and state.current == state.first_player
    assert sorted(state.all_uids()) == list(range(120))                          # nothing created or lost
    for i, p in enumerate(state.players):
        assert len(p.prizes) == catalog().constants["prize_count"]
        assert card_of(state, p.active.card).is_basic_pokemon
        assert len(p.bench) <= catalog().constants["bench_max"]
        assert all(card_of(state, m.card).is_basic_pokemon for m in p.bench)
        drawn = 1 if i == state.first_player else 0                              # the turn draw
        assert len(p.hand) + len(p.pokemon()) == 7 + bonus[i] + drawn
        assert len(p.deck) == 60 - 6 - 7 - bonus[i] - drawn


def test_same_seed_same_game():
    a, b = play_opening(3)[0], play_opening(3)[0]
    assert [p.deck for p in a.players] == [p.deck for p in b.players]
    assert [p.hand for p in a.players] == [p.hand for p in b.players]


def test_step_does_not_change_the_state_it_is_given():
    state = new_game(DECKS, 1)
    before = (state.pending, [p.deck.copy() for p in state.players], state.rng.getstate())
    step(state, legal_actions(state)[0])
    assert (state.pending, [p.deck for p in state.players], state.rng.getstate()) == before


@pytest.mark.parametrize("option", [0, 1])
def test_the_coin_flip_winner_decides_who_goes_first(option):
    state = new_game(DECKS, 5)
    winner = state.pending.player
    after = step(state, Action(ActionKind.CHOOSE, winner, option=option))
    assert after.first_player == (winner if option == 1 else 1 - winner)


def test_only_the_player_with_fewer_mulligans_draws_the_difference():
    seen = 0
    for seed in range(200):
        state = new_game(DECKS, seed)
        state = step(state, legal_actions(state)[0])
        a, b = (p.mulligans for p in state.players)
        if a == b:
            assert state.pending is None
            continue
        seen += 1
        assert state.pending.player == (0 if a < b else 1)
        assert state.pending.options == tuple(range(abs(a - b) + 1))
    assert seen, "no seed had a mulligan difference"


def test_an_illegal_action_raises():
    state = new_game(DECKS, 0)
    with pytest.raises(ValueError):
        step(state, Action(ActionKind.END_TURN, 0))


def test_a_player_who_cannot_draw_loses():
    state, _ = play_opening(0)
    loser = 1 - state.current
    state.players[loser].discard += state.players[loser].deck
    state.players[loser].deck.clear()
    start_turn(state, loser)
    assert state.phase == Phase.OVER and state.winner == 1 - loser and state.end_reason == "deck_out"
    assert legal_actions(state) == []


def test_main_phase_says_it_is_not_written():
    with pytest.raises(NotImplementedError):
        legal_actions(play_opening(0)[0])


def test_unknown_deck_raises():
    with pytest.raises(ValueError):
        new_game(("lopunny", "nope"), 0)