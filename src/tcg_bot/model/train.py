"""Training by playing: PPO against itself, the stand-in and earlier copies of itself.

    uv run python -m tcg_bot.model.train --name first --iterations 200

Writes models/<name>/: model.pt (the network), metrics.csv (one row per iteration) and config.json.
Stop with Ctrl+C at any time; the model saved after the last finished iteration stays. Add --resume to go on.
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import torch

from tcg_bot.model.evaluate import win_rate
from tcg_bot.model.net import Net, collate, load, save
from tcg_bot.model.player import MATCHUPS, Game, Step, play

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {"games": 64, "lr": 3e-4, "gamma": 0.999, "lam": 0.95, "clip": 0.2, "epochs": 4, "minibatch": 512,
            "entropy": 0.01, "value_weight": 0.5, "stand_in": 0.35, "past": 0.15, "snapshot_every": 10,
            "eval_every": 10, "eval_games": 200, "seed": 0}


def advantages(steps: list[Step], gamma: float, lam: float) -> list[tuple[float, float]]:
    """Generalised advantage estimation over one player's decisions in one game: (advantage, target value)."""
    out, running, next_value = [], 0.0, 0.0
    for s in reversed(steps):
        delta = s.reward + gamma * next_value - s.value
        running = delta + gamma * lam * running
        out.append((running, running + s.value))
        next_value = s.value
    return out[::-1]


def new_games(config: dict, iteration: int, pick: random.Random, has_past: bool) -> list[Game]:
    games = []
    for n in range(config["games"]):
        roll, seat = pick.random(), pick.randrange(2)
        if roll < config["stand_in"]:
            other = "stand_in"
        elif has_past and roll < config["stand_in"] + config["past"]:
            other = "past"
        else:
            other = "learner"                                         # self-play: both seats learn
        seats = ("learner", other) if seat == 0 else (other, "learner")
        games.append(Game(seed=config["seed"] * 10_000_000 + iteration * 10_000 + n,
                          decks=MATCHUPS[n % 2], seats=seats))
    return games


def update(net: Net, optimiser: torch.optim.Optimizer, games: list[Game], config: dict) -> dict:
    steps, targets = [], []
    for game in games:
        for side in game.steps:
            steps += side
            targets += advantages(side, config["gamma"], config["lam"])
    advantage = torch.tensor([t[0] for t in targets])
    returns = torch.tensor([t[1] for t in targets])
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
    old_logp = torch.tensor([s.logp for s in steps])
    rows = torch.tensor([s.row for s in steps])
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    batches = 0
    net.train()
    for _ in range(config["epochs"]):
        for index in torch.randperm(len(steps)).split(config["minibatch"]):
            logits, value = net(collate([steps[i].encoded for i in index.tolist()]))
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, rows[index][:, None]).squeeze(1)
            ratio = torch.exp(logp - old_logp[index])
            clipped = torch.clamp(ratio, 1 - config["clip"], 1 + config["clip"])
            policy_loss = -torch.min(ratio * advantage[index], clipped * advantage[index]).mean()
            value_loss = (value - returns[index]).pow(2).mean()
            entropy = -(logp_all.exp() * logp_all).sum(dim=-1).mean()
            loss = policy_loss + config["value_weight"] * value_loss - config["entropy"] * entropy
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimiser.step()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            batches += 1
    net.eval()
    return {"decisions": len(steps), **{k: v / batches for k, v in stats.items()}}


def train(name: str, iterations: int, resume: bool = False, folder: Path | None = None, say=print, **changes) -> Path:
    config = DEFAULTS | changes
    folder = folder or ROOT / "models" / name
    if (folder / "model.pt").exists() and not resume:
        raise SystemExit(f"{folder} already holds a model. Pick another --name or add --resume.")
    torch.manual_seed(config["seed"])
    start, net = 0, Net()
    if resume:
        net = load(folder)
        start = json.loads((folder / "config.json").read_text())["iterations_done"]
    optimiser = torch.optim.Adam(net.parameters(), lr=config["lr"])
    past = copy.deepcopy(net).eval() if resume else None
    folder.mkdir(parents=True, exist_ok=True)
    columns = ["iteration", "games", "decisions", "seconds", "vs_stand_in_training", "policy_loss", "value_loss",
               "entropy", "eval_random", "eval_stand_in"]
    if not resume:
        with (folder / "metrics.csv").open("w", newline="") as f:
            csv.writer(f).writerow(columns)
    began = time.time()
    for iteration in range(start + 1, start + iterations + 1):
        pick = random.Random(config["seed"] * 100_003 + iteration)
        games = play(new_games(config, iteration, pick, past is not None), net, past)
        row = {"iteration": iteration, "games": iteration * config["games"]} | update(net, optimiser, games, config)
        against = [g.state.winner == g.seats.index("learner") for g in games if "stand_in" in g.seats]
        row["vs_stand_in_training"] = sum(against) / len(against) if against else ""
        if iteration % config["snapshot_every"] == 0:
            past = copy.deepcopy(net).eval()
        if config["eval_every"] and iteration % config["eval_every"] == 0:
            for opponent in ("random", "stand_in"):
                row[f"eval_{opponent}"] = win_rate(net, opponent, config["eval_games"])["win_rate"]
        row["seconds"] = round(time.time() - began, 1)
        with (folder / "metrics.csv").open("a", newline="") as f:
            csv.DictWriter(f, columns, restval="").writerow(row)
        save(net, folder / "model.pt", iteration=iteration)
        (folder / "config.json").write_text(json.dumps(config | {"iterations_done": iteration}, indent=1))
        tested = (f" | TEST vs random {row['eval_random']:.0%}, vs stand-in {row['eval_stand_in']:.0%}"
                  if "eval_random" in row else "")
        say(f"iteration {iteration:4d} | games {row['games']:6d} | {row['seconds']:7.0f}s | "
            f"entropy {row['entropy']:.2f} | value loss {row['value_loss']:.3f}{tested}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model by playing.")
    parser.add_argument("--name", required=True, help="folder under models/")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = vars(parser.parse_args())
    torch.set_num_threads(1)                        # a small network runs fastest on one thread
    try:
        train(args.pop("name"), args.pop("iterations"), args.pop("resume"), **args)
    except KeyboardInterrupt:
        print("\nStopped. The last saved model is kept.")


if __name__ == "__main__":
    main()
