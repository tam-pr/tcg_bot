"""A hand-written heuristic player (task T10). It reads only the Observation and public card data.

Plan for a turn: evolve, fill the Bench, play search Items, use Abilities, attach Energy where it
gets an attack online soonest, play a Supporter, gust a Knock Out target, retreat into a better
attacker, then attack with the best expected damage (a Knock Out first). Choices inside effects
follow simple preferences (search what helps evolve or attack, discard what helps least).
"""
from __future__ import annotations

import random

from tcg_bot.engine.board import can_pay
from tcg_bot.engine.catalog import card_static
from tcg_bot.engine.types import Action, Observation, SlotView

from . import Agent

DRAW_SUPPORTERS = {"Lillie's Determination", "Judge", "Lucian"}


def _c(cid):
    return card_static(int(cid))


def _provided(v: SlotView) -> list[str]:
    out: list[str] = []
    for cid in v.energy_card_ids:
        out.extend(_c(cid).provides)
    return out


def _missing(v: SlotView, cost) -> int:
    """How many more Energy cards the cheapest-to-power attack needs (0 = ready)."""
    have = _provided(v)
    if can_pay(cost, have):
        return 0
    need = sum(n for _, n in cost)
    return max(1, need - len(have))


def _best_attack_damage(v: SlotView, defender: SlotView | None) -> int:
    top = _c(v.card_ids[-1])
    best = 0
    for a in top.attacks:
        if can_pay(a.cost, _provided(v)):
            best = max(best, _expected(top, a, defender))
    return best


def _expected(attacker, atk, defender: SlotView | None) -> int:
    d = atk.damage
    if atk.damage_mod == "times":
        d = atk.damage * 3
    if defender is not None and d > 0:
        dc = _c(defender.card_ids[-1])
        if dc.weakness and dc.weakness == attacker.type:
            d *= 2
        if dc.resistance and dc.resistance == attacker.type:
            d = max(0, d - 30)
    return d


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    # ------------------------------------------------------------ helpers
    def _my_views(self, obs):
        return [v for v in (obs.my_active, *obs.my_bench) if v is not None]

    def _view(self, obs, pid):
        for v in (obs.my_active, *obs.my_bench, obs.opp_active, *obs.opp_bench):
            if v is not None and v.pid == pid:
                return v
        return None

    def _hand_card(self, obs, uid):
        return _c(dict(zip(obs.my_hand, obs.my_hand_card_ids))[uid])

    def _power(self, v: SlotView) -> int:
        top = _c(v.card_ids[-1])
        return max((a.damage for a in top.attacks), default=0) + (100 if top.is_ex else 0)

    # ------------------------------------------------------------ choices inside effects
    def _choose(self, obs: Observation, legal: list[Action]) -> Action:
        kind = obs.pending_kind or ""
        opts = list(obs.pending_options)
        cids = list(obs.pending_option_card_ids) + [None] * (len(opts) - len(obs.pending_option_card_ids))
        by = {a.choice: a for a in legal}

        def pick(score):
            scored = [(score(o, c), i) for i, (o, c) in enumerate(zip(opts, cids)) if o in by]
            return by[opts[max(scored)[1]]]

        if kind == "use_ability":
            return by.get("yes", legal[0])
        if kind == "promote" or (kind == "switch_target" and all(self._view(obs, o) in obs.my_bench for o in opts if self._view(obs, o))):
            def s(o, c):
                v = self._view(obs, o)
                if v is None:
                    return -1
                return (_best_attack_damage(v, obs.opp_active) if v in obs.my_bench else 0) * 10 + v.hp - v.damage
            return pick(s)
        if kind == "switch_target":                             # gusting an opponent's Pokémon
            def s(o, c):
                v = self._view(obs, o)
                if v is None:
                    return -1
                me = obs.my_active
                dmg = _best_attack_damage(me, v) if me else 0
                ko = dmg >= v.hp - v.damage
                return (1000 if ko else 0) + (500 if _c(v.card_ids[-1]).is_ex else 0) - (v.hp - v.damage)
            return pick(s)
        if kind == "heal_target":
            return pick(lambda o, c: self._view(obs, o).damage if self._view(obs, o) else 0)
        if kind.startswith("move:") and kind.endswith(">discard") and "hand" in kind:
            # paying a cost: discard what helps least (extra Energy, then Trainers we have more of)
            counts = {}
            for c in obs.my_hand_card_ids:
                counts[c] = counts.get(c, 0) + 1
            return pick(lambda o, c: (counts.get(c, 0), _c(c).family == "energy") if c is not None else (-1, False))
        if kind.startswith("move:") and kind.endswith(">attached_to_bench"):
            def s(o, c):
                v = self._view(obs, o[1])
                return self._power(v) - 10 * len(v.energy_card_ids) if v else -1
            return pick(s)
        if kind.startswith("move:"):                            # searching or recovering cards: take the most useful
            in_play = {_c(v.card_ids[-1]).name for v in self._my_views(obs)}
            hand = {_c(c).name for c in obs.my_hand_card_ids}

            def s(o, c):
                if c is None or isinstance(c, tuple):
                    return 0
                card = _c(c)
                if card.family == "pokemon" and card.evolves_from in in_play:
                    return 50 + (20 if card.is_mega else 0)
                if card.family == "pokemon" and card.is_basic_pokemon:
                    return 30 + (10 if any(_c(x).evolves_from == card.name for x in obs.my_deck_list_card_ids) else 0)
                if card.family == "energy":
                    return 20
                if card.kind == "Supporter":
                    return 25 if card.name not in hand else 5
                return 10
            stop = by.get("stop")
            best = pick(s)
            return best if best.choice != "stop" or stop is None else best
        if kind == "retreat_discard":
            return legal[0]
        return self.rng.choice(legal)

    # ------------------------------------------------------------ main phase
    def act(self, obs: Observation, legal: list[Action]) -> Action:
        if legal[0].kind == "choose":
            return self._choose(obs, legal)
        kinds = {a.kind for a in legal}
        if "choose_first" in kinds:
            return next(a for a in legal if a.choice == "first")
        if "set_active" in kinds:
            return max((a for a in legal if a.kind == "set_active"),
                       key=lambda a: (self._hand_card(obs, a.source).hp or 0))
        if "set_bench" in kinds or "setup_done" in kinds:
            bench = [a for a in legal if a.kind == "set_bench"]
            if bench and len(obs.my_bench) < 4:
                return bench[0]
            return next(a for a in legal if a.kind == "setup_done")

        me = obs.my_active
        opp = obs.opp_active
        by_kind: dict[str, list[Action]] = {}
        for a in legal:
            by_kind.setdefault(a.kind, []).append(a)

        if "evolve" in by_kind:
            return max(by_kind["evolve"], key=lambda a: self._hand_card(obs, a.source).is_mega)
        if "bench_basic" in by_kind and len(obs.my_bench) < 4:
            return by_kind["bench_basic"][0]
        for a in by_kind.get("play_item", []):
            name = self._hand_card(obs, a.source).name
            if name == "Premium Power Pro":
                if me and _c(me.card_ids[-1]).type == "{F}" and _best_attack_damage(me, opp) > 0 and "attack" in by_kind:
                    return a
                continue
            if name == "Switch":
                if me and _best_attack_damage(me, opp) == 0 and any(_best_attack_damage(v, opp) for v in obs.my_bench):
                    return a
                continue
            if name == "Ultra Ball" and len(obs.my_hand) < 5:
                continue
            if name in ("Unfair Stamp", "Special Red Card") and len(obs.my_hand) > 5:
                continue
            return a
        if "use_ability" in by_kind:
            return by_kind["use_ability"][0]
        if "attach_energy" in by_kind:
            def s(a):
                v = self._view(obs, a.target)
                top = _c(v.card_ids[-1])
                miss = min((_missing(v, at.cost) for at in top.attacks), default=9)
                return (v is me and miss > 0, miss > 0, self._power(v), -miss)
            return max(by_kind["attach_energy"], key=s)
        if "attach_tool" in by_kind and me is not None:
            return next((a for a in by_kind["attach_tool"] if a.target == me.pid), by_kind["attach_tool"][0])
        for a in by_kind.get("play_supporter", []):
            name = self._hand_card(obs, a.source).name
            if name == "Boss's Orders":
                if me and any(_best_attack_damage(me, v) >= v.hp - v.damage for v in obs.opp_bench):
                    return a
                continue
            if name == "Wally's Compassion":
                if any(_c(v.card_ids[-1]).is_mega and v.damage >= v.hp / 2 for v in self._my_views(obs)):
                    return a
                continue
            if name in DRAW_SUPPORTERS and len(obs.my_hand) > 5:
                continue
            return a
        if "play_stadium" in by_kind:
            return by_kind["play_stadium"][0]
        if "retreat" in by_kind and me is not None and _best_attack_damage(me, opp) == 0:
            best = max(by_kind["retreat"], key=lambda a: _best_attack_damage(self._view(obs, a.target), opp))
            if _best_attack_damage(self._view(obs, best.target), opp) > 0:
                return best
        if "attack" in by_kind:
            def s(a):
                top = _c(me.card_ids[-1])
                atk = next(x for x in top.attacks if x.move_id == a.choice)
                d = _expected(top, atk, opp)
                ko = opp is not None and d >= opp.hp - opp.damage
                return (ko, d)
            return max(by_kind["attack"], key=s)
        return by_kind.get("end_turn", legal)[0]
