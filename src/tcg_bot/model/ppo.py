"""PPO self-play training (task T12). Trained from random weights, locally, on the CPU.

Opponents: one fixed opponent (random) for the first `stage1_iters` iterations, then a rotation of
random, heuristic and frozen snapshots of the policy itself. Reward: +1 win, -1 loss, 0 draw at the
end, plus a small shaping term of `shaping` per Prize card taken (and minus that per Prize card lost).
Seeds and the full config are saved with every run; checkpoints stay out of git (models/.gitignore).
"""
from __future__ import annotations

import copy
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tcg_bot.agents import HeuristicAgent, RandomAgent
from tcg_bot.engine import RuleSettings, acting_player, legal_actions, new_game, observe, step_inplace

from . import NeuralAgent, batch, save
from .encode import Encoder, EncoderConfig
from .net import PolicyValueNet

DECKS = ("lopunny", "mega_lucario")


@dataclass
class TrainConfig:
    name: str = "ppo"
    seed: int = 0
    iterations: int = 30
    games_per_iter: int = 16
    card_repr: str = "pca_bypass"
    use_identity: bool = True
    emb: int = 8
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    shaping: float = 0.05
    stage1_iters: int = 10
    snapshot_every: int = 10
    eval_games: int = 100


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def play_and_record(net, enc, opponent, decks, seat, seed, shaping):
    """One game; returns the learner's decisions with rewards, and the winner."""
    s = new_game(decks, seed)
    me = NeuralAgent(net, enc, greedy=False, seed=seed)
    opponent.reset(seed + 1)
    traj = []
    while True:
        legal = legal_actions(s)
        if not legal:
            break
        p = acting_player(s)
        if p == seat and len(legal) > 1:
            item = me.encode(observe(s, p), legal)
            logp, v = me.policy(item)
            i = int(torch.multinomial(logp.exp(), 1, generator=me.gen))
            item.update(a=i, logp=float(logp[i]), v=v, r=0.0)
            traj.append(item)
            a = legal[i]
        elif p == seat:
            a = legal[0]
        else:
            a = opponent.act(observe(s, p), legal)
        mine_before, theirs_before = len(s.players[seat].prizes), len(s.players[1 - seat].prizes)
        step_inplace(s, a, validate=False)
        if traj and not s.sudden_death:
            traj[-1]["r"] += shaping * ((mine_before - len(s.players[seat].prizes))
                                        - (theirs_before - len(s.players[1 - seat].prizes)))
    if traj:
        traj[-1]["r"] += 0.0 if s.winner is None else (1.0 if s.winner == seat else -1.0)
    return traj, s.winner


def gae(traj, gamma, lam):
    adv, last = np.zeros(len(traj), np.float32), 0.0
    for t in reversed(range(len(traj))):
        nv = traj[t + 1]["v"] if t + 1 < len(traj) else 0.0
        delta = traj[t]["r"] + gamma * nv - traj[t]["v"]
        last = delta + gamma * lam * last
        adv[t] = last
    for t, it in enumerate(traj):
        it["adv"], it["ret"] = float(adv[t]), float(adv[t] + it["v"])


def ppo_update(net, opt, items, cfg: TrainConfig, gen: torch.Generator):
    adv = torch.tensor([it["adv"] for it in items])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "n": 0}
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(items), generator=gen).tolist()
        for k in range(0, len(items), cfg.minibatch):
            idx = perm[k:k + cfg.minibatch]
            mb = [items[i] for i in idx]
            tensors = batch(mb)
            amask = tensors[5]
            logits, values = net(*tensors)
            logp_all = torch.log_softmax(logits, dim=1)
            a = torch.tensor([it["a"] for it in mb])
            new_logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
            old_logp = torch.tensor([it["logp"] for it in mb])
            ratio = torch.exp(new_logp - old_logp)
            ad = adv[idx]
            pl = -torch.min(ratio * ad, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ad).mean()
            ret = torch.tensor([it["ret"] for it in mb])
            vl = ((values - ret) ** 2).mean()
            safe = logp_all.masked_fill(~amask, 0.0)          # no -inf in the product, so no NaN gradients
            ent = -(safe.exp() * safe).masked_fill(~amask, 0.0).sum(1).mean()
            loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats["policy_loss"] += pl.item(); stats["value_loss"] += vl.item(); stats["entropy"] += ent.item()
            stats["approx_kl"] += (old_logp - new_logp.detach()).mean().item(); stats["n"] += 1
    n = max(1, stats.pop("n"))
    return {k: v / n for k, v in stats.items()}


def train(cfg: TrainConfig, run_dir: Path | None = None, log=print) -> dict:
    set_seeds(cfg.seed)
    enc = Encoder(EncoderConfig(cfg.card_repr, cfg.use_identity))
    net = PolicyValueNet(enc.state_dim, enc.action_dim, enc.n_cards, cfg.emb, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    snapshots: list = []
    history = []
    t0 = time.time()
    for it in range(cfg.iterations):
        if it < cfg.stage1_iters:
            pool = [RandomAgent(cfg.seed + it)]
        else:
            pool = [RandomAgent(cfg.seed + it), HeuristicAgent(cfg.seed + it)] + \
                [NeuralAgent(s, enc, greedy=False, seed=cfg.seed + it, name=f"snap{j}") for j, s in enumerate(snapshots)]
        items, wins, games = [], 0, 0
        for g in range(cfg.games_per_iter):
            game_seed = cfg.seed * 1_000_003 + it * 1000 + g
            seat = g % 2
            pair = DECKS if (g // 2) % 2 == 0 else DECKS[::-1]
            opp = pool[rng.randrange(len(pool))]
            traj, w = play_and_record(net, enc, opp, pair, seat, game_seed, cfg.shaping)
            gae(traj, cfg.gamma, cfg.lam)
            items.extend(traj)
            wins += (w == seat) + 0.5 * (w is None)
            games += 1
        net.train()
        st = ppo_update(net, opt, items, cfg, gen)
        net.eval()
        if (it + 1) % cfg.snapshot_every == 0:
            snap = copy.deepcopy(net)
            snap.eval()
            snapshots.append(snap)
        row = {"iteration": it + 1, "games": (it + 1) * cfg.games_per_iter, "decisions": len(items),
               "train_win_rate": wins / games, "opponents": len(pool), "seconds": round(time.time() - t0, 1), **st}
        history.append(row)
        log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    result = {"config": asdict(cfg), "settings_assumptions": RuleSettings().assumptions(), "history": history,
              "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}
    if run_dir is not None:
        run_dir = Path(run_dir)
        save(run_dir, net, {**asdict(cfg), "name": cfg.name})
        with open(run_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        (run_dir / "train_summary.json").write_text(json.dumps(result, indent=2, default=str))
    result["net"], result["encoder"] = net, enc
    return result
