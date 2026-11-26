import random
from collections import Counter

import numpy as np
import pytest
from test_engine_attacks import LOP, LUC, F
from test_engine_cards import board, give, play
from test_engine_observe import scrambled

from tcg_bot.engine.game import legal_actions, new_game, step
from tcg_bot.engine.observe import observe
from tcg_bot.engine.types import Phase
from tcg_bot.model.encode import GAME_FEATURES, SLOT_FEATURES, card_idx, encode, vocabulary

ARRAYS = ("slots", "slot_cards", "counts", "game", "context_cards", "moves", "move_cards")


def walk(seed, decks=("lopunny", "mega_lucario")):
    """Every decision of a random game: the state, who decides, and the legal actions."""
    pick = random.Random(seed)
    state = new_game(decks, seed)
    while state.phase != Phase.OVER:
        actions = legal_actions(state)
        yield state, observe(state, 0).waiting_for, actions, pick
        busy = [a for a in actions if a.kind != "end_turn"]
        state = step(state, pick.choice(busy if busy and pick.random() < 0.85 else actions))


def same(a, b):
    return all(np.array_equal(getattr(a, n), getattr(b, n)) for n in ARRAYS)


@pytest.mark.parametrize("seed", range(20))
def test_shapes_are_fixed_and_numbers_are_small(seed):
    cards = len(vocabulary().card_ids)
    for state, player, actions, _ in walk(seed):
        e = encode(observe(state, player), actions)
        assert e.slots.shape == (12, len(SLOT_FEATURES) + 30) and e.counts.shape == (5, cards)
        assert e.game.shape == (len(GAME_FEATURES),) and e.context_cards.shape == (3,)
        assert len(e.moves) == len(e.move_cards) == len(e.actions) >= 1 and set(e.actions) <= set(actions)
        for name in ARRAYS:
            array = getattr(e, name)
            assert np.isfinite(array).all() and (array.dtype == np.int64 or array.dtype == np.float32)
            if array.dtype == np.float32:
                assert np.abs(array).max() < 20


@pytest.mark.parametrize("seed", range(15))
def test_hidden_cards_do_not_change_the_numbers(seed):
    for state, player, actions, pick in walk(seed):
        other = scrambled(state, player, pick)
        assert same(encode(observe(state, player), actions), encode(observe(other, player), legal_actions(other)))


def outcome(state):
    """The game after a move, ignoring which of two identical copies is where."""
    def cards(uids):
        return tuple(sorted(state.card_ids[u] for u in uids))

    def mon(m):
        return (cards(m.stack), cards(m.energy), None if m.tool is None else state.card_ids[m.tool], m.damage,
                m.rotation, m.in_play_since)

    players = tuple((cards(p.hand), cards(p.discard), cards(p.deck + p.prizes), cards(p.playing),
                     None if p.active is None else mon(p.active), tuple(mon(m) for m in p.bench))
                    for p in state.players)
    pending = state.pending and (state.pending.kind, state.pending.player, Counter(
        state.card_ids[o] if state.pending.kind in ("pick", "attach_pick", "retreat_cost", "energy_to_hand") else o
        for o in state.pending.options))
    return (players, state.phase, state.turn, state.winner, pending,
            None if state.stadium is None else state.card_ids[state.stadium])


@pytest.mark.parametrize("seed", range(10))
def test_merged_moves_really_do_the_same_thing(seed):
    merged = 0
    for state, player, actions, _ in walk(seed):
        view = observe(state, player)
        kept = encode(view, actions)
        for action in set(actions) - set(kept.actions):
            row = encode(view, [action])
            twin = next(a for n, a in enumerate(kept.actions)
                        if kept.move_cards[n] == row.move_cards[0] and np.array_equal(kept.moves[n], row.moves[0]))
            first, second = step(state, action), step(state, twin)
            if first.rng.getstate() == state.rng.getstate():   # a shuffle or coin flip may land differently
                assert outcome(first) == outcome(second), (action, twin)
                merged += 1
    assert merged, "no identical moves came up"


def test_two_dunsparce_like_cards_stay_different():
    assert card_idx(65) != card_idx(848) != card_idx(305)


def test_an_unknown_card_raises():
    with pytest.raises(ValueError):
        card_idx(1)


def test_keeping_and_discarding_look_different():
    state = board(LUC)
    give(state, LUC, "Ultra Ball", F, F)
    state = play(state, "Ultra Ball")
    cost = encode(observe(state, LUC), legal_actions(state))
    assert len(cost.actions) == 1                                    # two identical Energy: one move
    for _ in range(2):                                               # pay both
        state = step(state, legal_actions(state)[0])
    search = encode(observe(state, LUC), legal_actions(state))
    assert not np.array_equal(cost.game, search.game) and cost.context_cards[1] == search.context_cards[1]


def test_the_other_players_actions_are_refused():
    state = board(LOP)
    with pytest.raises(ValueError):
        encode(observe(state, LUC), legal_actions(state))
