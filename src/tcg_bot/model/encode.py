"""Encode an Observation and its legal actions as arrays (task T12). Observation only, never the State.

Cards enter through a CardTable. The representation is configurable for the ablation (T13):
- "pca_bypass" (default): PCA components + raw bypass columns (card_vectors.parquet, v2)
- "pca":        PCA components only
- "raw":        the standardized raw v2 feature columns
- "learned":    no card features; only the learned identity embedding
`use_identity` adds the card's index (for a learned embedding) next to the features.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from tcg_bot.engine.board import can_pay
from tcg_bot.engine.catalog import card_static
from tcg_bot.engine.types import ACTION_KINDS, Action, Observation, SlotView
from tcg_bot.paths import PROC

N_SLOTS = 12                                   # my Active + 5 Bench, then the opponent's
PENDING_GROUPS = ("promote", "switch_target", "heal_target", "move", "retreat_discard", "use_ability", "other")
CONDITIONS = ("Asleep", "Burned", "Confused", "Paralyzed", "Poisoned")
FLAG_KEYS = ("energy_attached", "supporter_played", "retreated", "stadium_played", "attacked")


@dataclass(frozen=True)
class EncoderConfig:
    card_repr: str = "pca_bypass"
    use_identity: bool = True


@lru_cache(maxsize=8)
def card_table(card_repr: str) -> tuple[dict[int, np.ndarray], dict[int, int], int, int]:
    """card_id -> feature vector, card_id -> card_index, feature dim, number of cards."""
    vec = pd.read_parquet(PROC / "card_vectors.parquet").set_index("card_id")
    pcs = [c for c in vec.columns if c.startswith("pc")]
    bypass = [c for c in vec.columns if c not in pcs and c not in ("card_index", "evolves_from_index")]
    if card_repr == "pca_bypass":
        cols, M = pcs + bypass, vec[pcs + bypass]
    elif card_repr == "pca":
        cols, M = pcs, vec[pcs]
    elif card_repr == "raw":
        from tcg_bot.features import COLUMNS_V2, card_features_v2, standardize
        cards = pd.read_parquet(PROC / "cards.parquet")
        moves = pd.read_parquet(PROC / "moves.parquet")
        M, _ = standardize(card_features_v2(cards, moves)[COLUMNS_V2])
        cols = COLUMNS_V2
    elif card_repr == "learned":
        cols, M = [], vec[[]]
    else:
        raise ValueError(f"unknown card_repr {card_repr!r}")
    feats = {int(c): M.loc[c].to_numpy(dtype=np.float32) if cols else np.zeros(0, np.float32) for c in vec.index}
    index = {int(c): int(i) for c, i in vec["card_index"].items()}
    return feats, index, len(cols), len(vec)


class Encoder:
    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        self.cfg = cfg
        self.feats, self.index, self.D, self.n_cards = card_table(cfg.card_repr)
        self.pad = self.n_cards                      # embedding padding index
        self.slot_dim = 1 + self.D + 17
        self.state_dim = 21 + N_SLOTS * self.slot_dim + (self.D + 1) + 2 * self.D
        self.action_dim = len(ACTION_KINDS) + (self.D + 1) + (N_SLOTS + 1) + 4 + 1 + len(PENDING_GROUPS) + \
            (self.D + 1) + 1 + 2

    # ------------------------------------------------------------ helpers
    def _f(self, cid) -> np.ndarray:
        return self.feats[int(cid)] if cid is not None else np.zeros(self.D, np.float32)

    def _i(self, cid) -> int:
        return self.index[int(cid)] if (cid is not None and self.cfg.use_identity) else self.pad

    @staticmethod
    def _slots(obs: Observation) -> list[SlotView | None]:
        mine = [obs.my_active] + list(obs.my_bench) + [None] * (5 - len(obs.my_bench))
        theirs = [obs.opp_active] + list(obs.opp_bench) + [None] * (5 - len(obs.opp_bench))
        return mine + theirs

    def _slot_vec(self, v: SlotView | None) -> np.ndarray:
        out = np.zeros(self.slot_dim, np.float32)
        if v is None:
            return out
        top = v.card_ids[-1]
        c = card_static(int(top))
        prov = []
        for e in v.energy_card_ids:
            prov.extend(card_static(int(e)).provides)
        ready = float(any(can_pay(a.cost, prov) for a in c.attacks))
        extra = [max(0, v.hp - v.damage) / 340, v.damage / 340, v.hp / 340, len(prov) / 5,
                 prov.count("W") / 4, prov.count("F") / 4, prov.count("C") / 4,
                 (len(prov) - prov.count("W") - prov.count("F") - prov.count("C")) / 4,
                 float(v.tool is not None)] + [float(k in v.conditions) for k in CONDITIONS] + \
            [ready, v.retreat_cost / 4, float(c.is_ex)]
        out[0] = 1.0
        out[1:1 + self.D] = self._f(top)
        out[1 + self.D:] = extra
        return out

    # ------------------------------------------------------------ the observation
    def state(self, obs: Observation, posterior: tuple[float, float] = (0.5, 0.5)):
        me = obs.player
        flags = dict(obs.flags)
        g = [obs.turn / 100, float(obs.current == me), float(obs.first_player == me), obs.my_prize_count / 6,
             obs.opp_prize_count / 6, obs.my_deck_count / 60, obs.opp_deck_count / 60, len(obs.my_hand) / 20,
             obs.opp_hand_count / 20] + [float(bool(flags.get(k))) for k in FLAG_KEYS] + \
            [float(obs.stadium is not None and obs.stadium // 100 == me),
             float(obs.stadium is not None and obs.stadium // 100 != me), posterior[0], posterior[1],
             obs.turns_taken[me] / 50, len(obs.my_discard) / 30, len(obs.opp_discard) / 30]
        slots = self._slots(obs)
        sv = np.concatenate([self._slot_vec(v) for v in slots])
        hand = np.sum([self._f(c) for c in obs.my_hand_card_ids], axis=0) if obs.my_hand_card_ids else np.zeros(self.D, np.float32)
        hand = np.concatenate([hand / 5, [len(obs.my_hand) / 10]])
        dm = np.sum([self._f(c) for c in obs.my_discard_card_ids], axis=0) / 10 if obs.my_discard_card_ids else np.zeros(self.D)
        do = np.sum([self._f(c) for c in obs.opp_discard_card_ids], axis=0) / 10 if obs.opp_discard_card_ids else np.zeros(self.D)
        x = np.concatenate([np.asarray(g, np.float32), sv, hand, dm, do]).astype(np.float32)
        slot_ids = np.array([self._i(v.card_ids[-1]) if v is not None else self.pad for v in slots], np.int64)
        hand_ids = np.full(20, self.pad, np.int64)
        for i, c in enumerate(obs.my_hand_card_ids[:20]):
            hand_ids[i] = self._i(c)
        return x, slot_ids, hand_ids

    # ------------------------------------------------------------ the legal actions
    def actions(self, obs: Observation, legal: list[Action]):
        slots = self._slots(obs)
        pid_slot = {v.pid: i for i, v in enumerate(slots) if v is not None}
        uid_cid = dict(zip(obs.my_hand, obs.my_hand_card_ids))
        for v in slots:
            if v is not None:
                uid_cid.update(zip(v.cards, v.card_ids))
                uid_cid.update(zip(v.energy, v.energy_card_ids))
                if v.tool is not None:
                    uid_cid[v.tool] = v.tool_card_id
        uid_cid.update(zip(obs.my_discard, obs.my_discard_card_ids))
        uid_cid.update(zip(obs.opp_discard, obs.opp_discard_card_ids))
        opt_cid = dict(zip(obs.pending_options, obs.pending_option_card_ids))
        pk = obs.pending_kind or ""
        group = next((i for i, gname in enumerate(PENDING_GROUPS) if pk.startswith(gname)), len(PENDING_GROUPS) - 1)
        X = np.zeros((len(legal), self.action_dim), np.float32)
        ids = np.full((len(legal), 2), self.pad, np.int64)
        K = len(ACTION_KINDS)
        for r, a in enumerate(legal):
            row = X[r]
            row[ACTION_KINDS.index(a.kind)] = 1.0
            o = K
            src_cid = uid_cid.get(a.source) if a.source is not None else None
            if a.kind in ("use_ability", "attack") and a.source in pid_slot:
                src_cid = slots[pid_slot[a.source]].card_ids[-1]
            if src_cid is not None:
                row[o:o + self.D] = self._f(src_cid)
                row[o + self.D] = 1.0
                ids[r, 0] = self._i(src_cid)
            o += self.D + 1
            tgt = a.target
            opt_c = None
            if a.kind == "choose":
                ch = a.choice
                if isinstance(ch, tuple):
                    tgt = ch[1]
                    opt_c = opt_cid.get(ch, (None, None))[0]
                elif isinstance(ch, int) and not isinstance(ch, bool):
                    if ch in pid_slot:
                        tgt = ch
                    opt_c = opt_cid.get(ch)
            row[o + (pid_slot[tgt] if tgt in pid_slot else N_SLOTS)] = 1.0
            o += N_SLOTS + 1
            if a.kind == "attack":
                atk = next((x for x in card_static(int(src_cid)).attacks if x.move_id == a.choice), None)
                if atk is not None:
                    row[o:o + 4] = [atk.damage / 300, sum(n for _, n in atk.cost) / 4,
                                    float(atk.damage_mod == "times"), float(a.choice.endswith("_1"))]
            o += 4
            row[o] = float(a.kind == "use_ability")
            o += 1
            if a.kind == "choose":
                row[o + group] = 1.0
            o += len(PENDING_GROUPS)
            if opt_c is not None:
                row[o:o + self.D] = self._f(opt_c)
                row[o + self.D] = 1.0
                ids[r, 1] = self._i(opt_c)
            o += self.D + 1
            row[o] = float(a.kind == "choose" and a.choice == "stop")
            row[o + 1] = float(a.choice == "yes")
            row[o + 2] = float(a.choice == "no")
        return X, ids
