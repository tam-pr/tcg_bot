"""Belief trackers (task T6). Input is an Observation only, never the true State.

PrizeBelief: what may be in my own Prize cards. Unseen = my deck list minus every card I can see
(hand, discard, my Pokémon in play, my Stadium). The unseen cards are split between my deck and my
Prizes at random, so the number of copies of a card in the Prizes is hypergeometric.

OpponentDeckBelief: which known deck list the opponent is playing, by Bayes' rule over the known
archetypes, using the cards the opponent has shown. With only two archetypes in scope
(lopunny, mega_lucario) this model is small: it separates them after one distinctive card.
Broader priors need more deck lists from the owner.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import comb

import pandas as pd

from tcg_bot.engine.types import UID_STRIDE, Observation, SlotView
from tcg_bot.paths import PROC



# ---------------------------------------------------------------- helpers
def _slot_card_ids(v: SlotView | None) -> list[int]:
    if v is None:
        return []
    out = list(v.card_ids) + list(v.energy_card_ids)
    if v.tool_card_id is not None:
        out.append(v.tool_card_id)
    return out


def hypergeom_pmf(k: int, N: int, K: int, n: int) -> float:
    """P(exactly k successes) drawing n from N items of which K are successes."""
    if k < 0 or k > K or k > n or n - k > N - K or N < 0 or n > N:
        return 0.0
    return comb(K, k) * comb(N - K, n - k) / comb(N, n)


# ---------------------------------------------------------------- my Prize cards
class PrizeBelief:
    def __init__(self, obs: Observation):
        self.update(obs)

    def update(self, obs: Observation) -> None:
        seen = Counter(obs.my_hand_card_ids) + Counter(obs.my_discard_card_ids)
        for v in (obs.my_active, *obs.my_bench):
            seen.update(_slot_card_ids(v))
        if obs.stadium is not None and obs.stadium // UID_STRIDE == obs.player:
            seen[obs.stadium_card_id] += 1
        seen.update(c for u, c in obs.resolving if u // UID_STRIDE == obs.player)
        listed = Counter(obs.my_deck_list_card_ids)
        unseen = listed - seen
        if sum((seen - listed).values()):
            raise ValueError("more copies seen than the deck list holds")
        self.unseen = dict(unseen)
        self.n_unseen = sum(unseen.values())
        self.k = obs.my_prize_count
        if self.n_unseen != obs.my_deck_count + obs.my_prize_count:
            raise ValueError(f"unseen {self.n_unseen} != deck {obs.my_deck_count} + prizes {obs.my_prize_count}")

    def copies_pmf(self, card_id: int) -> list[float]:
        """P(exactly j copies of card_id are in my Prizes), j = 0..copies unseen."""
        K = self.unseen.get(card_id, 0)
        return [hypergeom_pmf(j, self.n_unseen, K, self.k) for j in range(K + 1)]

    def p_any_prized(self, card_id: int) -> float:
        return 1.0 - self.copies_pmf(card_id)[0]

    def expected_prized(self, card_id: int) -> float:
        return self.k * self.unseen.get(card_id, 0) / self.n_unseen if self.n_unseen else 0.0

    def p_in_deck_at_least_one(self, card_id: int) -> float:
        """P(at least one copy is still in my deck), i.e. not all unseen copies are prized."""
        K = self.unseen.get(card_id, 0)
        return 1.0 - (self.copies_pmf(card_id)[K] if K else 1.0)


# ---------------------------------------------------------------- the opponent's deck
@lru_cache(maxsize=1)
def archetype_lists() -> dict[str, dict[int, int]]:
    """Known deck lists: deck_id -> {card_id: copies}, from deck_cards (exact lists)."""
    dc = pd.read_parquet(PROC / "deck_cards.parquet")
    return {d: dict(zip(g["card_id"].astype(int), g["copies"].astype(int))) for d, g in dc.groupby("deck_id")}


def opponent_seen(obs: Observation) -> Counter:
    """Cards the opponent certainly owns right now: their discard pile, their Pokémon in play (with attached
    cards) and their Stadium. Cards revealed into the hand are left out on purpose: counting events could
    count one card twice (played, recovered, played again), which would wrongly rule out a deck list."""
    seen = Counter(obs.opp_discard_card_ids)
    for v in (obs.opp_active, *obs.opp_bench):
        seen.update(_slot_card_ids(v))
    if obs.stadium is not None and obs.stadium // UID_STRIDE != obs.player:
        seen[obs.stadium_card_id] += 1
    seen.update(c for u, c in obs.resolving if u // UID_STRIDE != obs.player)
    return seen


class OpponentDeckBelief:
    def __init__(self, prior: dict[str, float] | None = None, lists: dict[str, dict[int, int]] | None = None):
        self.lists = lists or archetype_lists()
        p = prior or {d: 1.0 for d in self.lists}
        z = sum(p.values())
        self.prior = {d: v / z for d, v in p.items()}
        self.posterior = dict(self.prior)
        self.unexplained = False

    @staticmethod
    def likelihood(seen: Counter, deck: dict[int, int]) -> float:
        """Multivariate hypergeometric probability of the seen multiset from the 60-card list."""
        n = sum(seen.values())
        total = sum(deck.values())
        num = 1
        for c, s in seen.items():
            K = deck.get(c, 0)
            if s > K:
                return 0.0
            num *= comb(K, s)
        return num / comb(total, n)

    def update(self, obs: Observation) -> dict[str, float]:
        seen = opponent_seen(obs)
        w = {d: self.prior[d] * self.likelihood(seen, lst) for d, lst in self.lists.items()}
        z = sum(w.values())
        self.unexplained = z == 0
        self.posterior = dict(self.prior) if z == 0 else {d: v / z for d, v in w.items()}
        self.seen = seen
        return self.posterior

    def expected_remaining(self) -> dict[int, float]:
        """Expected copies of each card the opponent still holds unseen (hand, deck, Prizes)."""
        out: dict[int, float] = {}
        for d, p in self.posterior.items():
            for c, k in self.lists[d].items():
                out[c] = out.get(c, 0.0) + p * max(0, k - self.seen.get(c, 0))
        return out

    def most_likely(self) -> str:
        return max(self.posterior, key=self.posterior.get)
