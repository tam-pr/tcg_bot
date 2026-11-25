"""T10: random vs random is near 50%, no illegal actions, opening-without-a-Basic rate matches the math."""
import math
import random

import pytest

from tcg_bot.agents import Agent, HeuristicAgent, RandomAgent
from tcg_bot.engine import new_game
from tcg_bot.engine.types import Observation
from tcg_bot.evaluate import IllegalAgentAction, p_opening_without_basic, play_game, play_match, wilson


def test_wilson_interval_hand_case():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_random_vs_random_is_near_50_percent():
    r = play_match(RandomAgent(1), RandomAgent(2), 200, seed0=0)
    lo, hi = r.interval()
    assert lo <= 0.5 <= hi, r.summary()
    seats = {(g.seats[0] == r.a) for g in r.games}
    decks = {g.decks[g.seats.index(r.a)] for g in r.games}
    assert seats == {True, False} and decks == {"lopunny", "mega_lucario"}      # both seats, both decks


def test_heuristic_beats_random():
    r = play_match(HeuristicAgent(1), RandomAgent(2), 100, seed0=500)
    assert r.interval()[0] > 0.6, r.summary()


class Spy(Agent):
    name = "spy"

    def __init__(self):
        self.rng = random.Random(0)

    def act(self, obs, legal):
        assert isinstance(obs, Observation)
        assert not hasattr(obs, "players") and not hasattr(obs, "rng")
        return self.rng.choice(legal)


class Cheater(Agent):
    name = "cheater"

    def act(self, obs, legal):
        from tcg_bot.engine import Action
        return Action("end_turn", obs.player) if legal[0].kind != "end_turn" else Action("setup_done", obs.player)


def test_agents_see_observations_only_and_illegal_actions_raise():
    play_game((Spy(), RandomAgent(3)), ("lopunny", "mega_lucario"), 7)
    with pytest.raises(IllegalAgentAction):
        play_game((Cheater(), RandomAgent(3)), ("lopunny", "mega_lucario"), 7)


@pytest.mark.parametrize("deck", ["lopunny", "mega_lucario"])
def test_opening_without_a_basic_rate_matches_the_hypergeometric(deck):
    p = p_opening_without_basic(deck)
    n = 3000
    hits = sum(new_game((deck, deck), seed).players[0].mulligans > 0 for seed in range(n))
    se = math.sqrt(p * (1 - p) / n)
    assert abs(hits / n - p) < 3 * se, (hits / n, p)
