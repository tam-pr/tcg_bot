"""Policy-value network (task T12): a flat-slot MLP that scores each legal action, plus a value head.

state  -> [features, mean identity embedding of the 12 slots (per slot, concatenated), mean of the hand] -> h
action -> [action features, identity embeddings of its source and option cards, h] -> one logit
value  -> h -> win value in [-1, 1]
Illegal actions are never scored: only the legal ones are encoded, and padding is masked with -inf.
"""
from __future__ import annotations

import torch
from torch import nn

from .encode import N_SLOTS


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_cards: int, emb: int = 8, hidden: int = 128):
        super().__init__()
        self.pad = n_cards
        self.emb = nn.Embedding(n_cards + 1, emb, padding_idx=n_cards)
        self.state_net = nn.Sequential(nn.Linear(state_dim + N_SLOTS * emb + emb, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden), nn.ReLU())
        self.action_net = nn.Sequential(nn.Linear(action_dim + 2 * emb + hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, 1))
        self.value_net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1), nn.Tanh())

    def encode_state(self, x, slot_ids, hand_ids):
        se = self.emb(slot_ids).flatten(1)
        he = self.emb(hand_ids)
        mask = (hand_ids != self.pad).unsqueeze(-1).float()
        hmean = (he * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.state_net(torch.cat([x, se, hmean], dim=1))

    def forward(self, x, slot_ids, hand_ids, ax, aids, amask):
        """x: (B, S); ax: (B, A, F); aids: (B, A, 2); amask: (B, A) bool. Returns logits (B, A) and value (B,)."""
        h = self.encode_state(x, slot_ids, hand_ids)
        B, A, _ = ax.shape
        ae = self.emb(aids).flatten(2)
        hh = h.unsqueeze(1).expand(B, A, h.shape[-1])
        logits = self.action_net(torch.cat([ax, ae, hh], dim=2)).squeeze(-1)
        logits = logits.masked_fill(~amask, float("-inf"))
        return logits, self.value_net(h).squeeze(-1)
