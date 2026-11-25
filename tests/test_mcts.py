"""T11: determinization uses only the Observation; the agent returns legal actions and finishes games."""
import random

from tcg_bot.agents import RandomAgent
from tcg_bot.agents.mcts import MCTSAgent, determinize
from tcg_bot.beliefs import OpponentDeckBelief
from tcg_bot.engine import legal_actions, observe
from tcg_bot.engine.core import card_locations
from tcg_bot.evaluate import play_match

from tests.engine.helpers import scenario


def _pos():
    return scenario(hands=(("Buneary", "Basic {W} Energy"), ("Switch", "Judge")), actives=("Snorunt", "Riolu"),
                    benches=(("Psyduck",), ("Solrock",)), energy={(0, 0): ["Basic {W} Energy"]})


def test_determinize_is_consistent_with_the_observation():
    s = _pos()
    obs = observe(s, 0)
    d = determinize(obs, OpponentDeckBelief(), random.Random(0))
    assert len(card_locations(d)) == 120
    assert set(legal_actions(d)) == set(legal_actions(s))
    assert len(d.players[1].hand) == obs.opp_hand_count and len(d.players[0].prizes) == obs.my_prize_count
    assert observe(d, 0).my_hand == obs.my_hand


def test_determinize_never_reads_the_true_hidden_state():
    a, b = _pos(), _pos()
    op = b.players[1]
    op.hand[0], op.deck[5] = op.deck[5], op.hand[0]          # change only what player 0 cannot see
    b.players[0].deck.reverse(); b.players[0].prizes.reverse()
    da = determinize(observe(a, 0), OpponentDeckBelief(), random.Random(1))
    db = determinize(observe(b, 0), OpponentDeckBelief(), random.Random(1))
    assert da.players[1].hand == db.players[1].hand and da.players[0].prizes == db.players[0].prizes


def test_mcts_returns_a_legal_action_and_plays_a_game():
    s = _pos()
    ag = MCTSAgent(determinizations=2, iterations=6, rollout_actions=10, seed=0)
    assert ag.act(observe(s, 0), legal_actions(s)) in legal_actions(s)
    r = play_match(MCTSAgent(determinizations=1, iterations=4, rollout_actions=8, seed=0), RandomAgent(1), 1, 77)
    assert r.n == 1
