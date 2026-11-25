"""PCA ablation (task T13): equal training budgets, several seeds, every baseline.

    uv run python reports/results/run_ablation.py [iterations] [games_per_iter] [eval_games] [seeds...]

Writes reports/results/ablation.csv (one row per variant, seed and opponent).
"""
import csv
import sys
import time

from tcg_bot.agents import HeuristicAgent, RandomAgent
from tcg_bot.evaluate import play_match
from tcg_bot.model import NeuralAgent
from tcg_bot.model.ppo import TrainConfig, train
from tcg_bot.paths import REPORTS

VARIANTS = {                                  # name: (card_repr, use_identity)
    "pca_only": ("pca", False),
    "pca_bypass": ("pca_bypass", False),
    "pca_bypass_identity": ("pca_bypass", True),
    "raw_columns": ("raw", False),
    "learned_embedding": ("learned", True),
}
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 40
gpi = int(sys.argv[2]) if len(sys.argv) > 2 else 16
n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 100
seeds = [int(x) for x in sys.argv[4:]] or [1, 2, 3]
out = REPORTS / "results" / "ablation.csv"
rows = []
for seed in seeds:
    for name, (rep, ident) in VARIANTS.items():
        t = time.time()
        cfg = TrainConfig(name=f"abl_{name}_s{seed}", seed=seed, iterations=iters, games_per_iter=gpi,
                          card_repr=rep, use_identity=ident, stage1_iters=iters // 3, snapshot_every=max(1, iters // 3))
        res = train(cfg, None, log=lambda r: None)
        agent = NeuralAgent(res["net"], res["encoder"], greedy=True, name=name)
        for opp, s0 in ((RandomAgent(7), 700_000), (HeuristicAgent(7), 800_000)):
            r = play_match(agent, opp, n_eval, s0 + seed * 10_000).summary()
            rows.append({"variant": name, "card_repr": rep, "identity": ident, "seed": seed, "opponent": opp.name,
                         "train_games": iters * gpi, "eval_games": r["games"], "win_rate": r["a_win_rate"],
                         "ci_low": r["ci95"][0], "ci_high": r["ci95"][1], "train_seconds": res["seconds"],
                         "final_train_win_rate": res["history"][-1]["train_win_rate"]})
            print(rows[-1], flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
print("wrote", out)
