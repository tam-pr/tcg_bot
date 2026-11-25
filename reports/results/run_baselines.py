"""Baseline league for the report: random, heuristic and determinized MCTS (fixed seeds, both seats, both decks).

Usage: uv run python reports/results/run_baselines.py [games_per_pair] [mcts_games]
Writes reports/results/baselines.json.
"""
import json
import sys
import time

from tcg_bot.agents import HeuristicAgent, RandomAgent
from tcg_bot.agents.mcts import MCTSAgent
from tcg_bot.engine import RuleSettings
from tcg_bot.evaluate import p_opening_without_basic, play_match
from tcg_bot.paths import REPORTS

n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
m = int(sys.argv[2]) if len(sys.argv) > 2 else 20
out = {"settings_assumptions": RuleSettings().assumptions(), "matches": []}
for a, b, games, seed in [(RandomAgent(1), RandomAgent(2), n, 0), (HeuristicAgent(1), RandomAgent(2), n, 10_000)]:
    r = play_match(a, b, games, seed)
    out["matches"].append(r.summary())
    print(r.summary(), flush=True)
for opp, seed in ((RandomAgent(2), 20_000), (HeuristicAgent(2), 30_000)):
    ag = MCTSAgent(seed=1)
    t = time.time()
    r = play_match(ag, opp, m, seed)
    s = r.summary()
    s["mcts_seconds_per_searched_move"] = round(ag.seconds / max(1, ag.decisions), 3)
    s["mcts_searched_moves"] = ag.decisions
    s["mcts_budget"] = {"determinizations": ag.D, "iterations": ag.iters, "rollout_actions": ag.depth}
    out["matches"].append(s)
    print(s, flush=True)
out["p_opening_without_basic"] = {d: round(p_opening_without_basic(d), 4) for d in ("lopunny", "mega_lucario")}
(REPORTS / "results").mkdir(parents=True, exist_ok=True)
(REPORTS / "results" / "baselines.json").write_text(json.dumps(out, indent=2, default=str))
print("wrote", REPORTS / "results" / "baselines.json")
