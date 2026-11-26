"""A player's view must not depend on anything that player cannot see."""
import random
from dataclasses import replace

import pytest
from test_engine_attacks import LOP, LUC
from test_engine_cards import abilities, board, choose_first, give, play

from tcg_bot.engine.game import legal_actions, new_game, step
from tcg_bot.engine.observe import CARD_CHOICES, observe
from tcg_bot.engine.types import Phase


def hidden_from(state, viewer):
    """For each player, the uids whose identity `viewer` has no way of knowing."""
    out = []
    for player, p in enumerate(state.players):
        uids = p.deck + p.prizes
        if player != viewer:
            uids = uids + [u for u in p.hand if u not in p.revealed]
            if state.phase == Phase.SETUP:
                uids = uids + [u for m in p.pokemon() for u in m.stack]
        elif state.pending and state.pending.player == viewer and state.pending.kind in CARD_CHOICES:
            uids = [u for u in uids if u not in state.pending.options + state.pending.picked]
        out.append(uids)
    return out


def scrambled(state, viewer, pick):
    """The same game with the unknown cards swapped around among themselves."""
    ids = list(state.card_ids)
    for uids in hidden_from(state, viewer):
        cards = [ids[u] for u in uids]
        pick.shuffle(cards)
        for u, c in zip(uids, cards, strict=True):
            ids[u] = c
    return replace(state, card_ids=tuple(ids))


@pytest.mark.parametrize("seed", range(40))
def test_the_view_does_not_change_when_unseen_cards_are_swapped(seed):
    pick = random.Random(seed)
    state = new_game(("lopunny", "mega_lucario"), seed)
    while state.phase != Phase.OVER:
        for viewer in (0, 1):
            view = observe(state, viewer)
            assert view == observe(scrambled(state, viewer, pick), viewer)
            assert not set(view.card_ids) & {u for uids in hidden_from(state, viewer) for u in uids}
        mover = observe(state, 0).waiting_for         # the moves on offer must not give anything away either
        assert legal_actions(state) == legal_actions(scrambled(state, mover, pick))
        for p in state.players:                       # a card shown once is forgotten when shuffled away
            assert not p.revealed & set(p.deck + p.prizes)
        busy = [a for a in legal_actions(state) if a.kind != "end_turn"]
        state = step(state, pick.choice(busy if busy and pick.random() < 0.85 else legal_actions(state)))


def test_changing_the_view_does_not_change_the_game():
    state = board(LUC)
    view = observe(state, LUC)
    view.me.active.energy.append(99)
    view.me.flags.names_used.add("x")
    view.card_ids.clear()
    assert 99 not in state.players[LUC].active.energy and not state.players[LUC].flags.names_used


def test_the_opponent_sees_how_many_cards_you_hold_but_not_which():
    state = board(LUC)
    give(state, LUC, "Switch", "Judge")
    view = observe(state, LOP)
    assert view.opponent.hand == () and view.opponent.hand_count == 2
    assert len(observe(state, LUC).me.hand) == 2


def test_a_card_shown_by_a_search_is_known_until_it_is_shuffled_away():
    state = board(LOP)
    give(state, LOP, "Poké Pad")
    state = choose_first(play(state, "Poké Pad"))
    [found] = state.players[LOP].hand
    assert observe(state, LUC).opponent.hand == (found,)
    give(state, LUC, "Judge")
    state.current = LUC
    state = play(state, "Judge")
    assert found not in observe(state, LUC).card_ids or found in state.players[LOP].discard


def test_a_pokemon_shuffled_back_is_hidden_again():
    state = board(LOP, active="Dudunsparce", bench=["Psyduck"])
    state.players[LOP].revealed.update(state.players[LOP].active.stack)     # as if a search had shown it
    mine = state.players[LOP].active.stack.copy()
    state = step(state, abilities(state)[0])
    assert not state.players[LOP].revealed & set(state.players[LOP].deck) and mine


def test_only_the_chooser_sees_the_choice():
    state = board(LOP)
    give(state, LOP, "Poké Pad")
    state = play(state, "Poké Pad")
    mine, theirs = observe(state, LOP), observe(state, LUC)
    assert mine.pending and theirs.pending is None and theirs.waiting_for == LOP
    assert set(state.pending.options) <= set(mine.card_ids) and not set(state.pending.options) & set(theirs.card_ids)


def test_face_down_pokemon_are_hidden_during_setup():
    state = new_game(("lopunny", "mega_lucario"), 1)
    while state.phase == Phase.SETUP and not (state.players[0].active and state.players[1].active):
        state = step(state, legal_actions(state)[-1])
    if state.phase == Phase.SETUP:
        view = observe(state, 0)
        assert view.opponent.active is None and view.opponent.face_down >= 1
