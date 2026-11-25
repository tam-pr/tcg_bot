"""Evaluation harness (task T10): fixed seeds, both seats, both decks, confidence intervals.

Agents only ever receive observe(state, player) and the legal actions. An action that is not legal
raises immediately, so a finished match proves that no illegal action occurred.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import comb, sqrt

from tcg_bot.engine import RuleSettings, acting_player, legal_actions, new_game, observe, step_inplace
from tcg_bot.engine.catalog import card_static, make_catalog

DECKS = ("lopunny", "mega_lucario")


class IllegalAgentAction(RuntimeError):
    pass


@dataclass
class GameRecord:
    seed: int
    decks: tuple[str, str]
    seats: tuple[str, str]          # agent name in seat 0 and seat 1
    winner: int | None
    result: str | None
    turns: int
    actions: int
    mulligans: tuple[int, int]
    first_player: int


@dataclass
class MatchResult:
    a: str
    b: str
    games: list[GameRecord] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.games)

    def wins(self, who: str) -> int:
        return sum(1 for g in self.games if g.winner is not None and g.seats[g.winner] == who)

    @property
    def draws(self) -> int:
        return sum(1 for g in self.games if g.winner is None)

    def win_rate(self, who: str | None = None) -> float:
        """Wins / games, counting a draw as half a win."""
        who = who or self.a
        return (self.wins(who) + 0.5 * self.draws) / self.n if self.n else float("nan")

    def interval(self, who: str | None = None, z: float = 1.96) -> tuple[float, float]:
        who = who or self.a
        return wilson(self.wins(who) + 0.5 * self.draws, self.n, z)

    def summary(self) -> dict:
        lo, hi = self.interval()
        by_deck = {}
        for d in DECKS:
            gs = [g for g in self.games if g.decks[g.seats.index(self.a)] == d]
            if gs:
                w = sum(1 for g in gs if g.winner is not None and g.seats[g.winner] == self.a) + \
                    0.5 * sum(1 for g in gs if g.winner is None)
                by_deck[d] = (round(w / len(gs), 3), len(gs))
        return {"a": self.a, "b": self.b, "games": self.n, "a_win_rate": round(self.win_rate(), 3),
                "ci95": (round(lo, 3), round(hi, 3)), "draws": self.draws, "a_win_rate_by_a_deck": by_deck,
                "mean_turns": round(sum(g.turns for g in self.games) / max(1, self.n), 1),
                "games_per_second": round(self.n / self.seconds, 1) if self.seconds else None}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def play_game(agents, decks: tuple[str, str], seed: int, settings: RuleSettings | None = None,
              max_actions: int = 20000) -> GameRecord:
    s = new_game(decks, seed, settings)
    for i, ag in enumerate(agents):
        ag.reset(seed * 2 + i)
    mull = (s.players[0].mulligans, s.players[1].mulligans)
    n = 0
    while True:
        legal = legal_actions(s)
        if not legal:
            break
        p = acting_player(s)
        a = agents[p].act(observe(s, p), legal)
        if a not in legal:
            raise IllegalAgentAction(f"{agents[p].name} chose an illegal action {a}")
        step_inplace(s, a, validate=False)
        n += 1
        if n > max_actions:
            raise RuntimeError(f"game {seed} exceeded {max_actions} actions")
    return GameRecord(seed, decks, (agents[0].name, agents[1].name), s.winner, s.result, s.turn, n, mull,
                      s.first_player)


def play_match(agent_a, agent_b, n_games: int, seed0: int = 0, decks=DECKS,
               settings: RuleSettings | None = None) -> MatchResult:
    """Each block of 4 games rotates: A in seat 0 or 1, and A on deck 0 or deck 1 (both seats, both decks)."""
    if agent_a.name == agent_b.name:
        agent_b.name = agent_b.name + "_b"
    res = MatchResult(agent_a.name, agent_b.name)
    t = time.time()
    for i in range(n_games):
        a_seat = i % 2
        a_deck = decks[(i // 2) % 2]
        b_deck = decks[1 - (i // 2) % 2] if len(set(decks)) > 1 else decks[0]
        seat_decks = (a_deck, b_deck) if a_seat == 0 else (b_deck, a_deck)
        agents = (agent_a, agent_b) if a_seat == 0 else (agent_b, agent_a)
        res.games.append(play_game(agents, seat_decks, seed0 + i, settings))
    res.seconds = time.time() - t
    return res


def p_opening_without_basic(deck_id: str) -> float:
    """Hypergeometric chance that 7 cards from the 60-card list hold no Basic Pokémon."""
    cat = make_catalog(deck_id, deck_id)
    ids = cat.deck_lists[0]
    basics = sum(card_static(c).is_basic_pokemon for c in ids)
    hand = cat.constants["opening_hand_size"]
    return comb(len(ids) - basics, hand) / comb(len(ids), hand)
