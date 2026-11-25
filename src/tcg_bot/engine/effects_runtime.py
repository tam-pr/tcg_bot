"""Effect interpreter (task T5).

Runs the step verbs of tcg_bot.effects as data. There are no branches per card: a step is executed
from its block type and parameters only. The interpreter is resumable: its position lives in a
Frame of plain values, so a State can be copied in the middle of an effect (for example while the
player is choosing which card to search for) and both copies continue independently.

Unknown blocks, sources, destinations or durations raise EngineError.
"""
from __future__ import annotations

from tcg_bot.effects import (ApplyStatus, CoinFlip, DamageSelf, Draw, Effect, ForEachPlayer, Heal, If, Lockout,
                             ModifyStat, MoveCards, Prevent, SwitchActive)

from . import board as B
from .types import Frame, Pending, Slot, State

STOP = "stop"
DONE, WAIT, PUSHED = "done", "wait", "pushed"


# ---------------------------------------------------------------- building frames
def effect_frame(state: State, move_id: str, owner: int, pid: int | None = None, card_uid: int | None = None,
                 parts: tuple[str, ...] = ("cost", "steps"), extra_ctx: dict | None = None) -> Frame:
    eff: Effect = state.catalog.effects[move_id]
    if eff.status == "no_effect_in_scope":
        raise B.EngineError(f"effect {move_id} has no effect in scope and must not be run")
    steps: tuple = ()
    for part in parts:
        steps = steps + tuple(getattr(eff, part))
    ctx = {"player": owner, "pid": pid, "card": card_uid, "move_id": move_id, "bottom": {}}
    if extra_ctx:
        ctx.update(extra_ctx)
    return Frame("effect", {"move_id": move_id, "stack": [[steps, 0, owner]], "ctx": ctx, "sub": None})


# ---------------------------------------------------------------- can it be used?
def _sides(side: str, me: int, that_player: int | None = None) -> list[int]:
    if side == "you":
        return [me]
    if side == "opponent":
        return [B.opp(me)]
    if side == "both":
        return [me, B.opp(me)]
    if side == "that_player":
        if that_player is None:
            raise B.EngineError("side 'that_player' outside a for_each_player block")
        return [that_player]
    raise B.EngineError(f"side {side!r} is not supported")


def _hand_pool(state: State, p: int, ctx: dict) -> list[int]:
    return [u for u in state.players[p].hand if u != ctx.get("card")]


def _cost_payable(state: State, st, ctx: dict) -> bool:
    me = ctx["player"]
    if isinstance(st, MoveCards) and st.source == "hand" and st.destination == "discard":
        n = st.count if isinstance(st.count, int) else 1
        return sum(B.matches(state, u, st.filter) for u in _hand_pool(state, me, ctx)) >= n
    raise B.EngineError(f"cost step {st!r} is not supported")


def _feasible(state: State, st, ctx: dict) -> bool:
    me = ctx["player"]
    pl = state.players[me]
    if isinstance(st, MoveCards):
        s = st.source
        if s in ("deck", "top_of_deck"):
            if st.destination == "bench" and len(pl.bench) >= B.const(state, "bench_max"):
                return False
            return any(state.players[p].deck for p in _sides(st.side, me))
        if s == "discard":
            if st.destination == "attached_to_bench" and not pl.bench:
                return False
            return any(B.matches(state, u, st.filter) for u in pl.discard)
        if s == "attached_tool_any":
            return any(sl.tool is not None and B.matches(state, sl.tool, st.filter)
                       for p in (0, 1) for sl in B.slots(state, p))
        if s == "stadium_in_play":
            return state.stadium is not None
        if s == "attached_to_this":
            _, sl = B.find_slot(state, ctx["pid"])
            return any(B.matches(state, u, st.filter) for u in sl.energy)
        return True
    if isinstance(st, Draw):
        return any(state.players[p].deck for p in _sides(st.side, me, me))
    if isinstance(st, SwitchActive):
        return bool(state.players[me if st.side == "you" else B.opp(me)].bench)
    if isinstance(st, Heal):
        if st.target.which == "chosen":
            return any(B.matches(state, sl.top, st.target.filter) for sl in B.slots(state, me))
        return True
    return True


def can_use(state: State, move_id: str, owner: int, pid: int | None = None, card_uid: int | None = None) -> bool:
    """Requirements hold, the cost can be paid, and at least one step can do something."""
    eff: Effect = state.catalog.effects[move_id]
    if eff.status == "no_effect_in_scope":
        return False
    ctx = {"player": owner, "pid": pid, "card": card_uid, "move_id": move_id}
    if not all(B.eval_cond(state, ctx, c) for c in eff.requires):
        return False
    if not all(_cost_payable(state, st, ctx) for st in eff.cost):
        return False
    if not eff.steps:
        return True
    primary = [st for st in eff.steps if not isinstance(st, (If, CoinFlip, ForEachPlayer))] or list(eff.steps)
    return any(_feasible(state, st, ctx) for st in primary)


# ---------------------------------------------------------------- the interpreter
def run_effect(state: State, frame: Frame, answer) -> bool:
    d = frame.data
    while True:
        stack = d["stack"]
        if not stack:
            return True
        top = stack[-1]
        steps, idx, tp = top
        if idx >= len(steps):
            stack.pop()
            continue
        res = _exec(state, frame, steps[idx], tp, answer)
        answer = None
        if res == WAIT:
            return False
        if res == DONE:
            top[1] += 1
            d["sub"] = None
        # PUSHED: the step advanced its parent and pushed a branch; keep running


def _push(frame: Frame, branch: tuple, tp: int) -> str:
    frame.data["stack"][-1][1] += 1
    frame.data["sub"] = None
    if branch:
        frame.data["stack"].append([tuple(branch), 0, tp])
    return PUSHED


def _ask(state: State, chooser: int, kind: str, options, stop=None, label: str = "", public: bool = False) -> str:
    state.pending = Pending(chooser, kind, tuple(options), label, stop, public)
    return WAIT


def _exec(state: State, frame: Frame, st, tp: int, answer) -> str:
    d = frame.data
    ctx = d["ctx"]
    me = ctx["player"]
    if isinstance(st, MoveCards):
        return _move_cards(state, frame, st, tp, answer)
    if isinstance(st, Draw):
        n = B.eval_amount(state, ctx, st.count)
        for p in _sides(st.side, me, tp):
            k = B.draw(state, p, n)
            if p == me:
                ctx["drew"] = ctx.get("drew", 0) + k
        return DONE
    if isinstance(st, SwitchActive):
        p = me if st.side == "you" else B.opp(me)
        if st.side not in ("you", "opponent"):
            raise B.EngineError(f"switch side {st.side!r} is not supported")
        if st.chosen_by != "you":
            raise B.EngineError(f"switch chosen_by {st.chosen_by!r} is not supported")
        bench = [s.pid for s in state.players[p].bench]
        if state.players[p].active is None or not bench:
            return DONE
        if answer is None and len(bench) > 1:
            return _ask(state, me, "switch_target", bench, label="choose the Benched Pokémon to switch in", public=True)
        B.switch_active(state, p, answer if answer is not None else bench[0])
        return DONE
    if isinstance(st, Heal):
        t = st.target
        if t.which == "chosen":
            opts = [s.pid for s in B.slots(state, me) if B.matches(state, s.top, t.filter)]
        elif t.which == "this_pokemon":
            opts = [ctx["pid"]]
        elif t.which == "your_active":
            opts = [state.players[me].active.pid] if state.players[me].active else []
        else:
            raise B.EngineError(f"heal target {t.which!r} is not supported")
        if not opts:
            return DONE
        if answer is None and len(opts) > 1:
            return _ask(state, me, "heal_target", opts, label="choose the Pokémon to heal", public=True)
        pid = answer if answer is not None else opts[0]
        _, sl = B.find_slot(state, pid)
        before = sl.damage
        sl.damage = 0 if st.amount == "all" else max(0, sl.damage - B.eval_amount(state, ctx, st.amount))
        ctx["healed_any"] = sl.damage < before
        ctx["target_pid"] = pid
        return DONE
    if isinstance(st, ApplyStatus):
        if st.target.which != "opponent_active":
            raise B.EngineError(f"status target {st.target.which!r} is not supported")
        sl = state.players[B.opp(me)].active
        if sl is None:
            return DONE
        if B.attack_effects_prevented(state, sl):
            return DONE
        B.apply_condition(state, sl, st.status)
        return DONE
    if isinstance(st, DamageSelf):
        _, sl = B.find_slot(state, ctx["pid"])
        amount = st.amount
        if state.settings.recoil_applies_weakness_resistance:
            c = B.card(state, sl.top)
            if c.weakness and c.weakness == c.type:
                amount *= B.const(state, "weakness_multiplier")
            if c.resistance and c.resistance == c.type:
                amount = max(0, amount - state.settings.resistance_amount)
        sl.damage += amount
        return DONE
    if isinstance(st, Lockout):
        mid = ctx["move_id"] if st.attack == "this_attack" else st.attack
        offset = {"this_turn": 0, "opponent_next_turn": 1, "your_next_turn": 2}.get(st.duration)
        if offset is None:
            raise B.EngineError(f"lockout duration {st.duration!r} is not supported")
        _, sl = B.find_slot(state, ctx["pid"])
        sl.lockouts[mid] = state.turn + offset
        return DONE
    if isinstance(st, Prevent):
        if st.duration is None:
            raise B.EngineError("a standing Prevent must be a passive effect, not a step that runs")
        if st.to != "this_pokemon":
            raise B.EngineError(f"prevent target {st.to!r} at run time is not supported")
        offset = {"this_turn": 0, "opponent_next_turn": 1, "your_next_turn": 2}[st.duration]
        state.lasting.append({"kind": "prevent_attack", "what": st.what, "pid": ctx["pid"],
                              "until_turn": state.turn + offset})
        return DONE
    if isinstance(st, ModifyStat):
        if st.duration != "this_turn" or st.stat != "attack_damage" or st.side != "you":
            raise B.EngineError(f"modify_stat {st.stat}/{st.side}/{st.duration} at run time is not supported")
        state.lasting.append({"kind": "attack_bonus", "player": me, "amount": st.amount, "filter": st.applies_to,
                              "until_turn": state.turn})
        return DONE
    if isinstance(st, CoinFlip):
        heads = B.flip(state)
        state.public.append((state.turn, tp if st.side == "that_player" else me, "coin_heads" if heads else "coin_tails", -1))
        return _push(frame, st.heads if heads else st.tails, tp)
    if isinstance(st, If):
        return _push(frame, st.then if B.eval_cond(state, ctx, st.condition) else st.otherwise, tp)
    if isinstance(st, ForEachPlayer):
        d["stack"][-1][1] += 1
        d["sub"] = None
        for p in (B.opp(me), me):                # stack: the owner runs first
            d["stack"].append([tuple(st.steps), 0, p])
        return PUSHED
    raise B.EngineError(f"step {type(st).__name__} is not supported")


# ---------------------------------------------------------------- MoveCards
def _owner_zone(state: State, uid: int):
    """(player, zone name) of a card that is in a list zone."""
    for p in (0, 1):
        pl = state.players[p]
        for zname in ("deck", "hand", "discard", "prizes"):
            if uid in getattr(pl, zname):
                return p, zname
    return None, None


def _put(state: State, frame: Frame, st: MoveCards, uid, ctx: dict) -> None:
    """Move one chosen card (or (energy, bench pid) pair) to the destination."""
    dst = st.destination
    if st.source in ("deck", "top_of_deck", "discard", "hand"):
        e_uid = uid[0] if isinstance(uid, tuple) else uid
        p, zone = _owner_zone(state, e_uid)
        if p is None:
            raise B.EngineError(f"card {e_uid} is not in a zone")
        getattr(state.players[p], zone).remove(e_uid)
    elif st.source in ("attached_to_this", "attached_to_target"):
        e_uid = uid
        pid = ctx["pid"] if st.source == "attached_to_this" else ctx["target_pid"]
        p, sl = B.find_slot(state, pid)
        sl.energy.remove(e_uid)
    elif st.source == "attached_tool_any":
        e_uid = uid
        p = None
        for q in (0, 1):
            for sl in B.slots(state, q):
                if sl.tool == e_uid:
                    sl.tool, p = None, q
        if p is None:
            raise B.EngineError(f"tool {e_uid} is not attached")
    else:
        raise B.EngineError(f"move_cards source {st.source!r} is not supported for chosen cards")
    owner = state.catalog.owner(e_uid)
    if st.reveal:
        B.log(state, owner, "reveal", e_uid)
    if dst == "hand":
        state.players[owner].hand.append(e_uid)
    elif dst == "discard":
        state.players[owner].discard.append(e_uid)
    elif dst == "bench":
        if len(state.players[owner].bench) >= B.const(state, "bench_max"):
            raise B.EngineError("the Bench is full")
        state.players[owner].bench.append(Slot(pid=e_uid, cards=[e_uid], entered_turn=state.turn))
        B.log(state, owner, "bench", e_uid)
    elif dst == "attached_to_bench":
        _, sl = B.find_slot(state, uid[1])
        sl.energy.append(e_uid)
        B.log(state, owner, "attach", e_uid)
    else:
        raise B.EngineError(f"move_cards destination {dst!r} is not supported for chosen cards")


def _move_cards(state: State, frame: Frame, st: MoveCards, tp: int, answer) -> str:
    d = frame.data
    ctx = d["ctx"]
    me = ctx["player"]
    src, dst = st.source, st.destination

    # ---- whole-zone moves, no choice
    if src == "hand" and st.count == "all":
        for p in _sides(st.side, me, tp):
            pl = state.players[p]
            moved = [u for u in pl.hand if u != ctx.get("card")]
            pl.hand = [u for u in pl.hand if u == ctx.get("card")]
            if dst == "deck_shuffle":
                pl.deck.extend(moved)
                B.shuffle_deck(state, p)
            elif dst == "deck_bottom":
                state.rng.shuffle(moved)
                if state.settings.bottom_of_deck_order != "random":
                    raise B.EngineError("only random bottom-of-deck order is implemented")
                pl.deck[0:0] = moved
                ctx.setdefault("bottom", {})[p] = len(moved)
            else:
                raise B.EngineError(f"hand -> {dst!r} for the whole hand is not supported")
        return DONE
    if src == "this_pokemon_and_attached":
        p, sl = B.find_slot(state, ctx["pid"])
        cards = B.all_cards_of(sl)
        B.remove_slot(state, p, sl)
        if dst == "hand":
            state.players[p].hand.extend(cards)
            for u in cards:
                B.log(state, p, "to_hand", u)
        elif dst == "deck_shuffle":
            state.players[p].deck.extend(cards)
            B.shuffle_deck(state, p)
        else:
            raise B.EngineError(f"this_pokemon_and_attached -> {dst!r} is not supported")
        return DONE
    if src == "stadium_in_play":
        if dst != "discard":
            raise B.EngineError(f"stadium -> {dst!r} is not supported")
        if state.stadium is not None:
            state.players[state.catalog.owner(state.stadium)].discard.append(state.stadium)
            state.stadium = None
        return DONE
    if src == "attached_to_target" and st.count == "all":
        if "target_pid" not in ctx:
            return DONE
        p, sl = B.find_slot(state, ctx["target_pid"])
        moved = [u for u in sl.energy if B.matches(state, u, st.filter)]
        sl.energy = [u for u in sl.energy if u not in moved]
        if dst != "hand":
            raise B.EngineError(f"attached_to_target -> {dst!r} is not supported")
        state.players[p].hand.extend(moved)
        return DONE

    # ---- chosen one at a time
    sides = _sides(st.side, me, tp)
    sub = d["sub"]
    if sub is None:
        if st.count == "all":
            raise B.EngineError(f"count 'all' from {src!r} is not supported")
        remaining = B.eval_amount(state, ctx, st.count)
        if dst == "bench":
            remaining = min(remaining, B.const(state, "bench_max") - len(state.players[me].bench))
        sub = d["sub"] = {"remaining": remaining}
        if src == "top_of_deck":
            sub["window"] = list(state.players[me].deck[-st.look_n:]) if st.look_n else []

    if answer is not None:
        if answer == STOP:
            sub["remaining"] = 0
        else:
            _put(state, frame, st, answer, ctx)
            sub["remaining"] -= 1

    while True:
        if sub["remaining"] <= 0:
            break
        pool = _pool(state, st, sub, ctx, sides)
        cands = [u for u in pool if B.matches(state, u, st.filter)]
        if not cands:
            break
        if src == "attached_tool_any":
            opts = sorted(cands)
        else:
            opts = B.dedupe(state, cands)
        if dst == "attached_to_bench":
            bench = [s.pid for s in state.players[me].bench]
            if not bench:
                break
            opts = [(e, b) for e in opts for b in bench]
        can_stop = bool(st.up_to or st.optional or
                        (src in ("deck", "top_of_deck") and st.filter is not None and state.settings.search_may_find_fewer))
        if len(opts) == 1 and not can_stop:
            _put(state, frame, st, opts[0], ctx)
            sub["remaining"] -= 1
            continue
        public = src in ("discard", "attached_to_this", "attached_tool_any", "attached_to_target")
        return _ask(state, me, f"move:{src}>{dst}", opts, stop=STOP if can_stop else None,
                    label=f"choose a card ({src} -> {dst})", public=public)

    if st.shuffle_after or st.rest_to == "deck_shuffle":
        for p in sides:
            B.shuffle_deck(state, p)
    elif st.rest_to is not None:
        raise B.EngineError(f"rest_to {st.rest_to!r} is not supported")
    return DONE


def _pool(state: State, st: MoveCards, sub: dict, ctx: dict, sides: list[int]) -> list[int]:
    me = ctx["player"]
    s = st.source
    if s == "deck":
        return [u for p in sides for u in state.players[p].deck]
    if s == "top_of_deck":
        return [u for u in sub["window"] if u in state.players[me].deck]
    if s == "discard":
        return [u for p in sides for u in state.players[p].discard]
    if s == "hand":
        return [u for p in sides for u in _hand_pool(state, p, ctx)]
    if s == "attached_to_this":
        _, sl = B.find_slot(state, ctx["pid"])
        return list(sl.energy)
    if s == "attached_to_target":
        _, sl = B.find_slot(state, ctx["target_pid"])
        return list(sl.energy)
    if s == "attached_tool_any":
        return [sl.tool for p in sides for sl in B.slots(state, p) if sl.tool is not None]
    raise B.EngineError(f"move_cards source {s!r} is not supported")
