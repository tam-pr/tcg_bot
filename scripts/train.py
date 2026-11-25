"""Train the policy by PPO self-play, then evaluate it (task T12).

    uv run python scripts/train.py --name main --iterations 100 --games-per-iter 32 --seed 0

Writes models/<name>/: config.json, metrics.csv, train_summary.json, curves.png, eval.json
(and policy.pt, which git ignores). Everything is seeded; the same command gives the same run.
"""
import argparse
import json
from dataclasses import fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tcg_bot.agents import HeuristicAgent, RandomAgent  # noqa: E402
from tcg_bot.evaluate import play_match  # noqa: E402
from tcg_bot.model import NeuralAgent  # noqa: E402
from tcg_bot.model.ppo import TrainConfig, train  # noqa: E402
from tcg_bot.paths import ROOT  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, type=lambda x: x.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    args = ap.parse_args(argv)
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    run = ROOT / "models" / cfg.name
    res = train(cfg, run, log=lambda r: print(r, flush=True))
    h = pd.DataFrame(res["history"])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].plot(h["games"], h["train_win_rate"].rolling(5, min_periods=1).mean()); ax[0].set_title("training win rate (5-iter mean)")
    ax[1].plot(h["games"], h["value_loss"]); ax[1].set_title("value loss")
    ax[2].plot(h["games"], h["entropy"]); ax[2].set_title("policy entropy")
    for a in ax:
        a.set_xlabel("training games")
        a.axvline(cfg.stage1_iters * cfg.games_per_iter, color="grey", lw=0.7)
    plt.tight_layout(); fig.savefig(run / "curves.png", dpi=110)
    out = {}
    if cfg.eval_games:
        agent = NeuralAgent(res["net"], res["encoder"], greedy=True, name=cfg.name)
        for opp, seed in ((RandomAgent(7), 900_000), (HeuristicAgent(7), 950_000)):
            r = play_match(agent, opp, cfg.eval_games, seed)
            out[opp.name] = r.summary()
            print(out[opp.name], flush=True)
    (run / "eval.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
