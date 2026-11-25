"""The learned player (task T12): encoder, policy-value network and an Agent that uses it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tcg_bot.agents import Agent
from tcg_bot.beliefs import OpponentDeckBelief
from tcg_bot.engine.types import Action, Observation

from .encode import Encoder, EncoderConfig
from .net import PolicyValueNet


def batch(items):
    """Stack per-decision encodings into padded tensors."""
    B = len(items)
    A = max(it["ax"].shape[0] for it in items)
    F = items[0]["ax"].shape[1]
    ax = np.zeros((B, A, F), np.float32)
    aids = np.zeros((B, A, 2), np.int64)
    amask = np.zeros((B, A), bool)
    pad = items[0]["pad"]
    aids[:] = pad
    for i, it in enumerate(items):
        n = it["ax"].shape[0]
        ax[i, :n] = it["ax"]
        aids[i, :n] = it["aids"]
        amask[i, :n] = True
    return (torch.from_numpy(np.stack([it["x"] for it in items])), torch.from_numpy(np.stack([it["slots"] for it in items])),
            torch.from_numpy(np.stack([it["hand"] for it in items])), torch.from_numpy(ax), torch.from_numpy(aids),
            torch.from_numpy(amask))


class NeuralAgent(Agent):
    """Plays by the network's policy. Sees only the Observation (plus its own belief tracker)."""

    def __init__(self, net: PolicyValueNet, encoder: Encoder, greedy: bool = True, seed: int = 0, name: str = "model"):
        self.net, self.enc, self.greedy, self.name = net, encoder, greedy, name
        self.gen = torch.Generator().manual_seed(seed)
        self.belief = OpponentDeckBelief()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.gen.manual_seed(seed)
        self.belief = OpponentDeckBelief()

    def encode(self, obs: Observation, legal: list[Action]) -> dict:
        post = self.belief.update(obs)
        x, slots, hand = self.enc.state(obs, (post.get("lopunny", 0.5), post.get("mega_lucario", 0.5)))
        ax, aids = self.enc.actions(obs, legal)
        return {"x": x, "slots": slots, "hand": hand, "ax": ax, "aids": aids, "pad": self.enc.pad}

    @torch.no_grad()
    def policy(self, item: dict):
        logits, value = self.net(*batch([item]))
        return torch.log_softmax(logits[0], dim=0), float(value[0])

    def act(self, obs: Observation, legal: list[Action]) -> Action:
        if len(legal) == 1:
            return legal[0]
        logp, _ = self.policy(self.encode(obs, legal))
        i = int(torch.argmax(logp)) if self.greedy else int(torch.multinomial(logp.exp(), 1, generator=self.gen))
        return legal[i]


def save(run_dir: Path, net: PolicyValueNet, cfg: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), run_dir / "policy.pt")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def load(run_dir: Path) -> NeuralAgent:
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    enc = Encoder(EncoderConfig(cfg["card_repr"], cfg["use_identity"]))
    net = PolicyValueNet(enc.state_dim, enc.action_dim, enc.n_cards, cfg["emb"], cfg["hidden"])
    net.load_state_dict(torch.load(Path(run_dir) / "policy.pt"))
    net.eval()
    return NeuralAgent(net, enc, name=cfg.get("name", "model"))


__all__ = ["Encoder", "EncoderConfig", "PolicyValueNet", "NeuralAgent", "batch", "save", "load"]
