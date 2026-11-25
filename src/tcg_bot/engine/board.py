"""Low-level board operations and value lookups shared by the core rules and the effect interpreter.

Values (HP, Retreat Cost, damage) are always read through the lookups here, which apply card
modifiers generically from effect data (Air Balloon, Gravity Mountain, Mist Energy, Battle Cage, ...).
There is no code per card: a passive effect is found by its block type and parameters.
"""
from __future__ import annotations

from tcg_bot.effects import (Cond, Filter, Fixed, IfThen, ModifyStat, PerCount, Prevent, Sum)

from .types import CardStatic, Slot, State

CONDITION_ROTATION = ("Asleep", "Confused", "Paralyzed")    # special_conditions.exclusive_group == rotation


class EngineError(RuntimeError):
    """Raised for input the engine does not support. Never silently skipped."""


def opp(p: int) -> int:
    return 1 - p


def card(state: State, uid: int) -> CardStatic:
    return state.catalog.refs[uid].card


def const(state: State, key: str) -> int:
    try:
        return state.catalog.constants[key]
    except KeyError:
        raise EngineError(f"game constant {key!r} is missing or not verified in game_constants") from None


def log(state: State, player: int, what: str, uid: int) -> None:
    state.public.append((state.turn, player, what, card(state, uid).card_id))


# ---------------------------------------------------------------- finding Pokémon
def slots(state: State, p: int) -> list[Slot]:
    return state.players[p].in_play()


def find_slot(state: State, pid: int) -> tuple[int, Slot]:
    for p in (0, 1):
        for s in state.players[p].in_play():
            if s.pid == pid:
                return p, s
    raise EngineError(f"no Pokémon in play with pid {pid}")


def is_active(state: State, p: int, slot: Slot) -> bool:
    return state.players[p].active is slot


def has_name_in_play(state: State, p: int, name: str, bench_only: bool = False) -> bool:
    pool = state.players[p].bench if bench_only else slots(state, p)
    return any(card(state, s.top).name == name for s in pool)


# ---------------------------------------------------------------- passive effects (generic)
def _passive_sources(state: State):
    """(owner, uid, attached_to_slot or None) for every card in play whose effect is passive."""
    if state.stadium is not None:
        yield state.catalog.owner(state.stadium), state.stadium, None
    for p in (0, 1):
        for s in slots(state, p):
            if s.tool is not None:
                yield p, s.tool, s
            for e in s.energy:
                yield p, e, s
            if card(state, s.top).ability_ids:
                yield p, s.top, s


def _passive_steps(state: State, uid: int):
    c = card(state, uid)
    for mid in c.effect_ids + c.ability_ids:
        eff = state.catalog.effects[mid]
        if eff.timing != "passive" or eff.status == "no_effect_in_scope":
            continue
        if eff.status != "ok":
            raise EngineError(f"passive effect {mid} has status {eff.status!r}")
        yield from eff.steps


def matches(state: State, uid: int, f: Filter | None) -> bool:
    if f is None:
        return True
    c = card(state, uid)
    if f.any_of:
        return any(matches(state, uid, g) for g in f.any_of)
    if f.card_kind and c.kind not in f.card_kind:
        return False
    if f.family is not None and c.family != f.family:
        return False
    if f.type is not None and c.type != f.type:
        return False
    if f.hp_max is not None and (c.hp is None or c.hp > f.hp_max):
        return False
    if f.rule_box is not None and c.has_rule_box != f.rule_box:
        return False
    if f.is_mega is not None and c.is_mega != f.is_mega:
        return False
    if f.name is not None and c.name != f.name:
        return False
    return True


def _side_applies(side: str, source_owner: int, target_owner: int) -> bool:
    return {"you": source_owner == target_owner, "opponent": source_owner != target_owner,
            "both": True}.get(side, False)


def max_hp(state: State, slot: Slot) -> int:
    p, _ = find_slot(state, slot.pid)
    hp = card(state, slot.top).hp or 0
    for owner, uid, attached in _passive_sources(state):
        for st in _passive_steps(state, uid):
            if isinstance(st, ModifyStat) and st.stat == "hp":
                if st.side == "attached":
                    ok = attached is slot
                else:
                    ok = _side_applies(st.side, owner, p)
                if ok and matches(state, slot.top, st.applies_to):
                    hp += st.amount
    return hp


def retreat_cost(state: State, slot: Slot) -> int:
    p, _ = find_slot(state, slot.pid)
    cost = card(state, slot.top).retreat_cost or 0
    for owner, uid, attached in _passive_sources(state):
        for st in _passive_steps(state, uid):
            if isinstance(st, ModifyStat) and st.stat == "retreat_cost":
                ok = attached is slot if st.side == "attached" else _side_applies(st.side, owner, p)
                if ok and matches(state, slot.top, st.applies_to):
                    cost += st.amount
    return max(0, cost)


def attack_effects_prevented(state: State, slot: Slot) -> bool:
    """Effects of the opponent's attacks done to this Pokémon are prevented (Mist Energy, Dig heads)."""
    for owner, uid, attached in _passive_sources(state):
        for st in _passive_steps(state, uid):
            if isinstance(st, Prevent) and st.what in ("attack_effects", "damage_and_effects") \
                    and st.to == "attached_pokemon" and attached is slot:
                return True
    return any(le["kind"] == "prevent_attack" and le["pid"] == slot.pid for le in state.lasting)


def attack_damage_prevented(state: State, slot: Slot) -> bool:
    for owner, uid, attached in _passive_sources(state):
        for st in _passive_steps(state, uid):
            if isinstance(st, Prevent) and st.what == "damage_and_effects" and st.to == "attached_pokemon" \
                    and attached is slot:
                return True
    return any(le["kind"] == "prevent_attack" and le["pid"] == slot.pid and le["what"] == "damage_and_effects"
               for le in state.lasting)


def bench_counters_prevented(state: State, target_owner: int, source_owner: int) -> bool:
    """Damage counters placed on Benched Pokémon by effects of the opponent's attacks or Abilities."""
    if target_owner == source_owner:
        return False
    for owner, uid, attached in _passive_sources(state):
        for st in _passive_steps(state, uid):
            if isinstance(st, Prevent) and st.what == "damage_counters" and st.to == "all_benched":
                return True
    return False


def attack_damage_bonus(state: State, p: int, attacker: Slot) -> int:
    bonus = 0
    for le in state.lasting:
        if le["kind"] == "attack_bonus" and le["player"] == p and matches(state, attacker.top, le["filter"]):
            bonus += le["amount"]
    return bonus


# ---------------------------------------------------------------- Energy
def provided(state: State, slot: Slot) -> list[str]:
    out: list[str] = []
    for e in slot.energy:
        out.extend(card(state, e).provides)
    return out


def can_pay(cost: tuple[tuple[str, int], ...], provided_types: list[str]) -> bool:
    pool = list(provided_types)
    colorless = 0
    for t, n in cost:
        if t == "C":
            colorless += n
            continue
        for _ in range(n):
            if t in pool:
                pool.remove(t)
            else:
                return False
    return len(pool) >= colorless


# ---------------------------------------------------------------- moving cards
def draw(state: State, p: int, n: int) -> int:
    pl = state.players[p]
    k = 0
    for _ in range(n):
        if not pl.deck:
            break
        pl.hand.append(pl.deck.pop())
        k += 1
    return k


def shuffle_deck(state: State, p: int) -> None:
    state.rng.shuffle(state.players[p].deck)


def flip(state: State) -> bool:
    """A coin flip; True means heads."""
    return state.rng.random() < 0.5


def dedupe(state: State, uids) -> list[int]:
    """One uid per card_id (the lowest), sorted by card_id: identical cards are interchangeable."""
    best: dict[int, int] = {}
    for u in uids:
        cid = card(state, u).card_id
        if cid not in best or u < best[cid]:
            best[cid] = u
    return [best[c] for c in sorted(best)]


def all_cards_of(slot: Slot) -> list[int]:
    return list(slot.cards) + list(slot.energy) + ([slot.tool] if slot.tool is not None else [])


def remove_slot(state: State, p: int, slot: Slot) -> None:
    pl = state.players[p]
    if pl.active is slot:
        pl.active = None
    else:
        pl.bench.remove(slot)
    clear_attack_effects(state, slot)


def clear_attack_effects(state: State, slot: Slot) -> None:
    """Moving to the Bench or evolving ends Special Conditions and effects of attacks on the Pokémon."""
    slot.conditions.clear()
    slot.lockouts.clear()
    slot.paralyzed_turn = -1
    state.lasting = [le for le in state.lasting if le.get("pid") != slot.pid]


def switch_active(state: State, p: int, bench_pid: int) -> None:
    pl = state.players[p]
    new = next(s for s in pl.bench if s.pid == bench_pid)
    pl.bench.remove(new)
    old = pl.active
    pl.active = new
    new.moved_to_active_turn = state.turn
    if old is not None:
        clear_attack_effects(state, old)
        pl.bench.append(old)


def promote(state: State, p: int, bench_pid: int) -> None:
    pl = state.players[p]
    if pl.active is not None:
        raise EngineError("promote called while an Active Pokémon is in place")
    new = next(s for s in pl.bench if s.pid == bench_pid)
    pl.bench.remove(new)
    pl.active = new
    new.moved_to_active_turn = state.turn


def apply_condition(state: State, slot: Slot, cond: str) -> None:
    if cond in CONDITION_ROTATION:
        slot.conditions -= set(CONDITION_ROTATION)
    slot.conditions.add(cond)
    if cond == "Paralyzed":
        slot.paralyzed_turn = state.turn


def prize_value(state: State, slot: Slot) -> int:
    c = card(state, slot.top)
    if c.is_mega and c.is_ex:
        return state.settings.mega_ex_prize_value
    if c.is_ex:
        return const(state, "ex_prize_value")
    return 1


# ---------------------------------------------------------------- amounts and conditions
def eval_cond(state: State, ctx: dict, c: Cond) -> bool:
    me = ctx["player"]
    n = c.name
    if n == "has_in_play":
        v = has_name_in_play(state, me, c.arg)
    elif n == "has_on_bench":
        v = has_name_in_play(state, me, c.arg, bench_only=True)
    elif n == "stadium_in_play":
        v = state.stadium is not None
    elif n == "moved_to_active_this_turn":
        _, s = find_slot(state, ctx["pid"])
        v = s.moved_to_active_turn == state.turn
    elif n == "prizes_remaining_equals":
        v = len(state.players[me].prizes) == c.arg
    elif n == "opponent_prizes_at_most":
        v = len(state.players[opp(me)].prizes) <= c.arg
    elif n == "own_pokemon_knocked_out_last_opponent_turn":
        v = (state.turn - 1) in state.players[me].ko_turns and state.turn > 1
    elif n == "drew_any":
        v = ctx.get("drew", 0) > 0
    elif n == "healed_any":
        v = bool(ctx.get("healed_any", False))
    elif n == "either_player_put_cards_on_bottom":
        v = any(x > 0 for x in ctx.get("bottom", {}).values())
    elif n == "they_put_cards_on_bottom":
        v = ctx.get("bottom", {}).get(opp(me), 0) > 0
    else:
        raise EngineError(f"condition {n!r} is not implemented")
    return (not v) if c.negate else v


def eval_amount(state: State, ctx: dict, a) -> int:
    if isinstance(a, int):
        return a
    if isinstance(a, Fixed):
        return a.n
    if isinstance(a, PerCount):
        if a.counter == "opp_hand_size":
            return a.each * len(state.players[opp(ctx["player"])].hand)
        raise EngineError(f"counter {a.counter!r} is not implemented")
    if isinstance(a, Sum):
        return sum(eval_amount(state, ctx, x) for x in a.parts)
    if isinstance(a, IfThen):
        return eval_amount(state, ctx, a.then if eval_cond(state, ctx, a.condition) else a.otherwise)
    raise EngineError(f"amount {a!r} is not supported")
