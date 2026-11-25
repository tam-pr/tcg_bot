"""T6: exact probabilities on hand-computed cases, and probabilities that sum correctly."""
import random
from collections import Counter
from math import comb

import pytest

from tcg_bot.beliefs import OpponentDeckBelief, PrizeBelief, archetype_lists, hypergeom_pmf, opponent_seen
from tcg_bot.engine import legal_actions, new_game, observe, step_inplace


def test_hypergeometric_hand_computed():
    # 4 copies among 54 unseen cards, 6 of them are Prizes
    assert hypergeom_pmf(0, 54, 4, 6) == pytest.approx(comb(50, 6) / comb(54, 6))
    assert hypergeom_pmf(1, 54, 4, 6) == pytest.approx(4 * comb(50, 5) / comb(54, 6))
    assert hypergeom_pmf(0, 54, 4, 6) == pytest.approx(0.6153, abs=1e-4)      # hand value: 15,890,700 / 25,827,165
    assert sum(hypergeom_pmf(k, 54, 4, 6) for k in range(5)) == pytest.approx(1.0)
    assert hypergeom_pmf(5, 54, 4, 6) == 0.0


def _game_after(seed, n_actions):
    s = new_game(("lopunny", "mega_lucario"), seed)
    rng = random.Random(seed)
    for _ in range(n_actions):
        acts = legal_actions(s)
        if not acts:
            break
        step_inplace(s, rng.choice(acts), validate=False)
    return s


def test_prize_belief_matches_the_true_unseen_pool_and_sums():
    for seed in range(10):
        s = _game_after(seed, 40)
        for p in (0, 1):
            obs = observe(s, p)
            b = PrizeBelief(obs)
            truth = Counter(s.catalog.card(u).card_id for u in s.players[p].deck + s.players[p].prizes)
            assert b.unseen == dict(truth)                                     # counting is exact
            for c in b.unseen:
                assert sum(b.copies_pmf(c)) == pytest.approx(1.0)
            assert sum(b.expected_prized(c) for c in b.unseen) == pytest.approx(obs.my_prize_count)


def test_prize_belief_hand_case():
    s = _game_after(4, 0)
    while s.phase == "setup":
        step_inplace(s, legal_actions(s)[0], validate=False)
    obs = observe(s, 0)
    b = PrizeBelief(obs)
    assert b.k == 6 and b.n_unseen == obs.my_deck_count + 6
    c, K = next((c, k) for c, k in b.unseen.items() if k >= 2)
    N = b.n_unseen
    assert b.p_any_prized(c) == pytest.approx(1 - comb(N - K, 6) / comb(N, 6))
    assert b.expected_prized(c) == pytest.approx(6 * K / N)


def test_opponent_deck_posterior_sums_to_one_and_rules_out_impossible_lists():
    lists = archetype_lists()
    assert set(lists) == {"lopunny", "mega_lucario"}
    b = OpponentDeckBelief()
    assert b.update(observe(new_game(("lopunny", "mega_lucario"), 0), 0)) == pytest.approx(
        {"lopunny": 0.5, "mega_lucario": 0.5})
    for seed in range(10):
        s = _game_after(seed, 60)
        post = OpponentDeckBelief().update(observe(s, 0))
        assert sum(post.values()) == pytest.approx(1.0)
        if sum(opponent_seen(observe(s, 0)).values()) and set(opponent_seen(observe(s, 0))) - {6}:
            assert post["mega_lucario"] == pytest.approx(1.0)              # player 1 plays mega_lucario


def test_likelihood_hand_case():
    deck = {1: 4, 2: 56}
    seen = Counter({1: 2})
    assert OpponentDeckBelief.likelihood(seen, deck) == pytest.approx(comb(4, 2) / comb(60, 2))
    assert OpponentDeckBelief.likelihood(Counter({1: 5}), deck) == 0.0


def test_expected_remaining_is_never_negative():
    s = _game_after(2, 80)
    b = OpponentDeckBelief()
    b.update(observe(s, 1))
    rem = b.expected_remaining()
    assert all(v >= 0 for v in rem.values())
    assert sum(rem.values()) <= 60
