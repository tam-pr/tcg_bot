"""How often does the model win? Against the stand-in and a random player, from both decks and both seats.

    uv run python -m tcg_bot.model.evaluate models/first --games 400
"""
import argparse
import math
from collections import Counter
from pathlib import Path

from tcg_bot.model.net import Net, load
from tcg_bot.model.player import MATCHUPS, Game, play

EVAL_SEED = 1_000_000_000          # training never uses seeds this high


def win_rate(net: Net, opponent: str, games: int, seed: int = EVAL_SEED) -> dict:
    """The model picks its best-scored move. Half the games per deck, half per seat."""
    batch = []
    for n in range(games):
        seat = (n // 2) % 2
        seats = ("learner", opponent) if seat == 0 else (opponent, "learner")
        batch.append(Game(seed=seed + n, decks=MATCHUPS[n % 2], seats=seats))
    results = Counter()
    by_deck: dict[str, Counter] = {}
    reasons = Counter()
    for start in range(0, games, 100):
        for game in play(batch[start:start + 100], net, greedy=True):
            seat = game.seats.index("learner")
            outcome = "draw" if game.state.winner is None else ("win" if game.state.winner == seat else "loss")
            results[outcome] += 1
            by_deck.setdefault(game.decks[seat], Counter())[outcome] += 1
            reasons[game.state.end_reason] += 1
    rate = results["win"] / games
    return {"opponent": opponent, "games": games, "win_rate": rate,
            "plus_minus": 1.96 * math.sqrt(rate * (1 - rate) / games),        # 95% interval
            "by_deck": {d: c["win"] / sum(c.values()) for d, c in sorted(by_deck.items())},
            "draws": results["draw"], "end_reasons": dict(reasons)}


def describe(r: dict) -> str:
    decks = ", ".join(f"as {d} {v:.0%}" for d, v in r["by_deck"].items())
    return f"vs {r['opponent']:9s} wins {r['win_rate']:.1%} +/- {r['plus_minus']:.1%} of {r['games']} ({decks})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Win rate of a saved model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--games", type=int, default=400)
    args = parser.parse_args()
    net = load(args.model)
    for opponent in ("random", "stand_in"):
        print(describe(win_rate(net, opponent, args.games)))


if __name__ == "__main__":
    main()
