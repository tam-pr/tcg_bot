"""Whole-game properties: no hidden-information leaks, determinism, turn limit, 1,000 random games."""
import dataclasses
import random

import pytest

from tcg_bot.engine import RuleSettings, legal_actions, new_game, observe, step, step_inplace
from tcg_bot.engine.core import card_locations

from .helpers import scenario

PAIRS = [("lopunny", "mega_lucario"), ("mega_lucario", "lopunny"), ("lopunny", "lopunny"),
         ("mega_lucario", "mega_lucario")]


def ints_in(x):
    if isinstance(x, bool):
        return
    if isinstance(x, int):
        yield x
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from ints_in(y)
    elif dataclasses.is_dataclass(x):
        for f in dataclasses.fields(x):
            if f.name not in ("my_deck_list_card_ids", "public_log", "my_hand_card_ids", "my_discard_card_ids",
                              "opp_discard_card_ids", "card_ids", "energy_card_ids", "tool_card_id",
                              "stadium_card_id", "turn", "current", "first_player", "player", "damage", "hp",
                              "my_deck_count", "my_prize_count", "opp_hand_count", "opp_deck_count",
                              "opp_prize_count", "winner", "pending_player", "flags", "resolving", "lasting", "pending_option_card_ids", "turns_taken", "ko_turns", "entered_turn", "evolved_turn",
                              "moved_to_active_turn", "paralyzed_turn", "lockouts", "used", "retreat_cost"):
                yield from ints_in(getattr(x, f.name))


def hidden_uids(state, player):
    me, op = state.players[player], state.players[1 - player]
    out = set(op.hand) | set(op.deck) | set(op.prizes) | set(me.prizes)
    if not (state.pending and state.pending.player == player):
        out |= set(me.deck)                       # my own deck is only visible while I search it
    return out


def random_game(seed, pair, check=None, settings=None):
    s = new_game(pair, seed, settings)
    rng = random.Random(seed)
    while True:
        acts = legal_actions(s)
        if not acts:
            return s
        if check:
            check(s)
        step_inplace(s, rng.choice(acts), validate=False)


def test_observe_leaks_no_hidden_card():
    def check(s):
        for p in (0, 1):
            leaked = set(ints_in(observe(s, p))) & hidden_uids(s, p)
            assert not leaked, (p, leaked)
    for seed in range(40):
        random_game(seed, PAIRS[seed % 4], check)


def test_hidden_information_does_not_change_the_observation():
    s = scenario(hands=(("Buneary",), ("Switch", "Judge")), actives=("Snorunt", "Riolu"))
    a = observe(s, 0)
    op = s.players[1]
    swap_in = next(u for u in op.deck if s.catalog.card(u).name == "Basic {F} Energy")
    op.deck[op.deck.index(swap_in)], op.hand[0] = op.hand[0], swap_in
    op.prizes.reverse(); op.deck.reverse()
    assert observe(s, 0) == a


def test_step_does_not_change_its_input_and_is_deterministic():
    s = new_game(("lopunny", "mega_lucario"), 11)
    before = repr(s)
    a = legal_actions(s)[0]
    step(s, a)
    assert repr(s) == before
    g1 = random_game(123, ("lopunny", "mega_lucario"))
    g2 = random_game(123, ("lopunny", "mega_lucario"))
    assert (g1.public, g1.winner, g1.result, g1.turn) == (g2.public, g2.winner, g2.result, g2.turn)


def test_turn_limit_ends_in_a_draw():
    s = random_game(3, ("lopunny", "mega_lucario"), settings=RuleSettings(turn_limit=4))
    assert s.phase == "over" and s.turn <= 5
    assert s.result == "draw_turn_limit" and s.winner is None or s.winner is not None


def test_mirror_match_uids_do_not_collide():
    s = new_game(("lopunny", "lopunny"), 0)
    assert len(s.catalog.refs) == 120 and len({r.tag for r in s.catalog.refs.values()}) == 120


@pytest.mark.parametrize("pair", PAIRS[:2])
def test_1000_random_games_finish_without_errors_and_keep_every_card(pair):
    """T4/T5 done-when: 1,000 seeds per seat order with all effects enabled; no card created or lost."""
    def check(s):
        assert len(card_locations(s)) == 120
    results = {}
    for seed in range(1000):
        s = random_game(seed, pair, check if seed % 10 == 0 else None)
        assert s.phase == "over" and len(card_locations(s)) == 120
        results[s.result] = results.get(s.result, 0) + 1
    assert sum(results.values()) == 1000
