"""Baseline agents (task T10). Every agent sees an Observation and the legal actions, never the State."""
from __future__ import annotations

import random

from tcg_bot.engine.types import Action, Observation


class Agent:
    name = "agent"

    def reset(self, seed: int | None = None) -> None:
        """Called before each game."""

    def act(self, obs: Observation, legal: list[Action]) -> Action:
        raise NotImplementedError


class RandomAgent(Agent):
    """Picks a legal action uniformly at random."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def act(self, obs: Observation, legal: list[Action]) -> Action:
        return self.rng.choice(legal)


from .heuristic import HeuristicAgent  # noqa: E402

__all__ = ["Agent", "RandomAgent", "HeuristicAgent"]
