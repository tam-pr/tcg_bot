"""Core rules (task T4): setup, turns, actions, attacks, Knock Outs, Checkup, win checks.

Every rule cites its row in configs/plays.csv by play_id. Numbers come from game_constants (through
board.const) or from RuleSettings when a value is not yet confirmed. Card text is run by
effects_runtime. The public API is new_game / legal_actions / step / observe / winner.
"""
from __future__ import annotations

import random

from . import board as B
from . import effects_runtime as R
from .catalog import make_catalog
from .types import (UID_STRIDE, Action, Frame, Observation, Pending, PlayerState, RuleSettings, Slot, SlotView,
                    State)


class IllegalActionError(ValueError):
    """The action is not in legal_actions(state)."""


# =============================================================== setup
def new_game(deck_ids: tuple[str, str], seed: int, settings: RuleSettings | None = None) -> State:
    """Shuffle, draw 7, resolve mulligans, flip the coin (play_ids setup_coin_flip, setup_draw_hand, setup_mulligan).
    The flip winner then chooses to go first or second; then each player places Active and Bench."""
    catalog = make_catalog(deck_ids[0], deck_ids[1])
    state = State(catalog=catalog, settings=settings or RuleSettings(), players=[PlayerState(), PlayerState()],
                  rng=random.Random(seed))
    _deal(state)
    return state


def _deal(state: State) -> None:
    cat = state.catalog
    for p in (0, 1):
        pl = state.players[p]
        pl.deck = [u for u in cat.refs if cat.refs[u].owner == p]
        pl.deck.sort()
    hand_size = B.const(state, "opening_hand_size")
    for p in (0, 1):                                            # setup_draw_hand, setup_mulligan
        pl = state.players[p]
        while True:
            B.shuffle_deck(state, p)
            B.draw(state, p, hand_size)
            if any(B.card(state, u).is_basic_pokemon for u in pl.hand):
                break
            for u in pl.hand:
                B.log(state, p, "mulligan_reveal", u)
            pl.deck.extend(pl.hand)
            pl.hand = []
            pl.mulligans += 1
    if state.settings.take_mulligan_bonus_cards:                # setup_mulligan: 1 extra card per extra mulligan
        m0, m1 = state.players[0].mulligans, state.players[1].mulligans
        if m0 > m1:
            B.draw(state, 1, m0 - m1)
        elif m1 > m0:
            B.draw(state, 0, m1 - m0)
    winner = 0 if B.flip(state) else 1                           # setup_coin_flip
    state.flags = {"flip_winner": winner}
    state.phase = "setup"
    state.current = winner
    state.first_player = -1


def _restart_sudden_death(state: State) -> None:
    """win_check: if both players win at once, play Sudden Death with sudden_death_prize_count Prize cards."""
    for p in (0, 1):
        state.players[p] = PlayerState()
    state.stadium = None
    state.lasting = []
    state.frames = []
    state.pending = None
    state.sudden_death = True
    _deal(state)


def _finish_setup(state: State) -> None:
    n = B.const(state, "sudden_death_prize_count") if state.sudden_death else B.const(state, "prize_count")
    for p in (0, 1):                                            # setup_set_prizes
        pl = state.players[p]
        pl.prizes = [pl.deck.pop() for _ in range(min(n, len(pl.deck)))]
        for s in pl.in_play():
            B.log(state, p, "reveal_setup", s.top)
    state.phase = "main"
    state.flags = {}
    state.frames.append(Frame("core:start_turn", {"player": state.first_player}))
    _run(state, None)


# =============================================================== legal actions
def legal_actions(state: State) -> list[Action]:
    """Every action the acting player may take. Empty only when the game is over."""
    if state.phase == "over":
        return []
    if state.pending is not None:
        pd = state.pending
        acts = [Action("choose", pd.player, choice=o) for o in pd.options]
        if pd.stop is not None:
            acts.append(Action("choose", pd.player, choice=pd.stop))
        return acts
    if state.phase == "setup":
        return _setup_actions(state)
    return _main_actions(state)


def acting_player(state: State) -> int:
    return state.pending.player if state.pending is not None else state.current


def _setup_actions(state: State) -> list[Action]:
    p = state.current
    if state.first_player == -1:
        return [Action("choose_first", p, choice="first"), Action("choose_first", p, choice="second")]
    pl = state.players[p]
    basics = B.dedupe(state, [u for u in pl.hand if B.card(state, u).is_basic_pokemon])
    if pl.active is None:                                       # setup_place_active
        return [Action("set_active", p, source=u) for u in basics]
    acts = []
    if len(pl.bench) < B.const(state, "bench_max"):            # setup_place_bench
        acts += [Action("set_bench", p, source=u) for u in basics]
    acts.append(Action("setup_done", p))
    return acts


def _first_turn_of_first_player(state: State, p: int) -> bool:
    return p == state.first_player and state.players[p].turns_taken == 1


def _main_actions(state: State) -> list[Action]:
    p = state.current
    pl = state.players[p]
    f = state.flags
    acts: list[Action] = []
    hand = B.dedupe(state, pl.hand)
    my_slots = pl.in_play()
    bench_max = B.const(state, "bench_max")
    for u in hand:
        c = B.card(state, u)
        if c.is_basic_pokemon and len(pl.bench) < bench_max:                         # main_bench_basic
            acts.append(Action("bench_basic", p, source=u))
        elif c.family == "pokemon" and c.stage >= 1 and pl.turns_taken > 1:          # main_evolve
            for s in my_slots:
                if (B.card(state, s.top).name == c.evolves_from and s.entered_turn < state.turn
                        and s.evolved_turn < state.turn):
                    acts.append(Action("evolve", p, source=u, target=s.pid))
        elif c.family == "energy" and not f.get("energy_attached"):                 # main_attach_energy
            acts += [Action("attach_energy", p, source=u, target=s.pid) for s in my_slots]
        elif c.kind == "Item" and _trainer_ok(state, p, u):                          # main_play_item
            acts.append(Action("play_item", p, source=u))
        elif c.kind == "Supporter" and not f.get("supporter_played") \
                and not _first_turn_of_first_player(state, p) and _trainer_ok(state, p, u):   # main_play_supporter
            acts.append(Action("play_supporter", p, source=u))
        elif c.kind == "Stadium" and not f.get("stadium_played") and \
                (state.stadium is None or B.card(state, state.stadium).name != c.name):     # main_play_stadium
            acts.append(Action("play_stadium", p, source=u))
        elif c.kind == "Pokémon Tool":                                                # main_attach_tool
            acts += [Action("attach_tool", p, source=u, target=s.pid) for s in my_slots if s.tool is None]
        elif c.family == "pokemon" and c.stage < 0:
            raise B.EngineError(f"card {c.name} has an unsupported Pokémon kind {c.kind!r}")
    act = pl.active
    if act is not None and not f.get("retreated") and pl.bench \
            and not ({"Asleep", "Paralyzed"} & act.conditions) \
            and len(B.provided(state, act)) >= B.retreat_cost(state, act):            # main_retreat
        acts += [Action("retreat", p, target=s.pid) for s in pl.bench]
    for s in my_slots:                                                                # main_use_ability
        for mid in B.card(state, s.top).ability_ids:
            eff = state.catalog.effects[mid]
            if eff.timing == "activated" and _usage_ok(state, p, s, eff) and R.can_use(state, mid, p, pid=s.pid):
                acts.append(Action("use_ability", p, source=s.pid, choice=mid))
    if act is not None and not _first_turn_of_first_player(state, p) \
            and not ({"Asleep", "Paralyzed"} & act.conditions):                        # attack_use
        have = B.provided(state, act)
        for a in B.card(state, act.top).attacks:
            if act.lockouts.get(a.move_id) == state.turn:
                continue
            if B.can_pay(a.cost, have):
                acts.append(Action("attack", p, source=act.pid, choice=a.move_id))
    if state.settings.may_end_turn_without_attack or not acts:                       # turn_end
        acts.append(Action("end_turn", p))
    return acts


def _trainer_ok(state: State, p: int, uid: int) -> bool:
    c = B.card(state, uid)
    return all(R.can_use(state, mid, p, card_uid=uid) for mid in c.effect_ids)


def _usage_ok(state: State, p: int, s: Slot, eff) -> bool:
    u = eff.usage
    if u.first_turn_only and state.players[p].turns_taken != 1:
        return False
    if u.scope == "per_pokemon":
        return s.used.get(eff.move_id) != state.turn
    if u.scope in ("per_name", "per_name_contains"):
        return u.key not in state.flags.get("names_used", [])
    if u.scope == "none":
        return True
    raise B.EngineError(f"usage scope {u.scope!r} is not supported")


def _mark_usage(state: State, s: Slot, eff) -> None:
    u = eff.usage
    s.used[eff.move_id] = state.turn
    if u.scope in ("per_name", "per_name_contains"):
        state.flags.setdefault("names_used", []).append(u.key)


# =============================================================== step
def step(state: State, action: Action, validate: bool = True) -> State:
    """Apply an action to a copy of the state and return it. The input state is unchanged."""
    new = state.copy()
    step_inplace(new, action, validate)
    return new


def step_inplace(state: State, action: Action, validate: bool = True) -> None:
    if validate and action not in legal_actions(state):
        raise IllegalActionError(f"illegal action {action}")
    state.action_count += 1
    k = action.kind
    if k == "choose":
        state.pending = None
        _run(state, action.choice)
        return
    if state.phase == "setup":
        _setup_step(state, action)
        return
    p = action.player
    pl = state.players[p]
    f = state.flags
    if k == "bench_basic":
        pl.hand.remove(action.source)
        pl.bench.append(Slot(pid=action.source, cards=[action.source], entered_turn=state.turn))
        B.log(state, p, "bench", action.source)
        s = pl.bench[-1]
        state.frames.append(Frame("core:after_main", {}))
        for mid in B.card(state, s.top).ability_ids:
            eff = state.catalog.effects[mid]
            if eff.timing == "on_play_bench" and _usage_ok(state, p, s, eff) and R.can_use(state, mid, p, pid=s.pid):
                state.frames.append(Frame("core:offer_ability", {"pid": s.pid, "move_id": mid, "player": p}))
    elif k == "evolve":
        pl.hand.remove(action.source)
        _, s = B.find_slot(state, action.target)
        s.cards.append(action.source)
        s.evolved_turn = state.turn
        B.clear_attack_effects(state, s)                       # main_evolve: Special Conditions and attack effects end
        B.log(state, p, "evolve", action.source)
        c = B.card(state, action.source)
        if c.is_mega and state.settings.mega_evolution_ends_turn:      # main_mega_turn_end (setting)
            state.frames.append(Frame("core:end_turn", {}))
        state.frames.append(Frame("core:after_main", {}))
        for mid in c.ability_ids:
            eff = state.catalog.effects[mid]
            if eff.timing == "on_evolve" and _usage_ok(state, p, s, eff) and R.can_use(state, mid, p, pid=s.pid):
                state.frames.append(Frame("core:offer_ability", {"pid": s.pid, "move_id": mid, "player": p}))
    elif k == "attach_energy":
        pl.hand.remove(action.source)
        _, s = B.find_slot(state, action.target)
        s.energy.append(action.source)
        f["energy_attached"] = True
        B.log(state, p, "attach", action.source)
        state.frames.append(Frame("core:after_main", {}))
        for mid in B.card(state, action.source).effect_ids:
            if state.catalog.effects[mid].timing == "on_attach_from_hand":
                state.frames.append(R.effect_frame(state, mid, p, pid=s.pid, card_uid=action.source))
    elif k in ("play_item", "play_supporter"):
        pl.hand.remove(action.source)
        B.log(state, p, "play", action.source)
        if k == "play_supporter":
            f["supporter_played"] = True
        state.frames.append(Frame("core:after_main", {}))
        state.frames.append(Frame("core:discard_played", {"uid": action.source, "player": p}))
        for mid in reversed(B.card(state, action.source).effect_ids):
            state.frames.append(R.effect_frame(state, mid, p, card_uid=action.source))
    elif k == "play_stadium":
        pl.hand.remove(action.source)
        B.log(state, p, "play", action.source)
        if state.stadium is not None:
            state.players[state.catalog.owner(state.stadium)].discard.append(state.stadium)
        state.stadium = action.source
        f["stadium_played"] = True
        state.frames.append(Frame("core:after_main", {}))
    elif k == "attach_tool":
        pl.hand.remove(action.source)
        _, s = B.find_slot(state, action.target)
        s.tool = action.source
        B.log(state, p, "attach", action.source)
        state.frames.append(Frame("core:after_main", {}))
    elif k == "retreat":
        f["retreated"] = True
        state.frames.append(Frame("core:after_main", {}))
        state.frames.append(Frame("core:retreat", {"player": p, "to": action.target,
                                                   "to_pay": B.retreat_cost(state, pl.active)}))
    elif k == "use_ability":
        _, s = B.find_slot(state, action.source)
        eff = state.catalog.effects[action.choice]
        _mark_usage(state, s, eff)
        state.frames.append(Frame("core:after_main", {}))
        state.frames.append(R.effect_frame(state, action.choice, p, pid=s.pid))
    elif k == "attack":
        f["attacked"] = True
        state.frames.append(Frame("core:attack", {"player": p, "pid": action.source, "move_id": action.choice,
                                                  "stage": 0}))
    elif k == "end_turn":
        state.frames.append(Frame("core:end_turn", {}))
    else:
        raise IllegalActionError(f"action kind {k!r} is not valid in the main phase")
    _run(state, None)


def _setup_step(state: State, a: Action) -> None:
    p = a.player
    pl = state.players[p]
    if a.kind == "choose_first":
        state.first_player = p if a.choice == "first" else B.opp(p)
        state.current = 0
    elif a.kind == "set_active":
        pl.hand.remove(a.source)
        pl.active = Slot(pid=a.source, cards=[a.source], entered_turn=0)
    elif a.kind == "set_bench":
        pl.hand.remove(a.source)
        pl.bench.append(Slot(pid=a.source, cards=[a.source], entered_turn=0))
    elif a.kind == "setup_done":
        pl.setup_done = True
        if not state.players[B.opp(p)].setup_done:
            state.current = B.opp(p)
        else:
            _finish_setup(state)
    else:
        raise IllegalActionError(f"action kind {a.kind!r} is not valid during setup")


# =============================================================== the frame runner
def _run(state: State, answer) -> None:
    while state.frames and state.pending is None and state.phase != "over":
        fr = state.frames[-1]
        handler = HANDLERS.get(fr.kind)
        if handler is None:
            raise B.EngineError(f"no handler for frame {fr.kind!r}")
        res = handler(state, fr, answer)
        answer = None
        if res is True:
            for i in range(len(state.frames) - 1, -1, -1):
                if state.frames[i] is fr:
                    del state.frames[i]
                    break
    if state.phase == "over":
        state.frames = []
        state.pending = None


def _h_effect(state, fr, answer):
    return R.run_effect(state, fr, answer)


def _h_after_main(state, fr, answer):
    state.frames.pop()
    state.frames.append(Frame("core:ko_resolve", {"stage": 0}))
    return None


def _h_discard_played(state, fr, answer):
    state.players[fr.data["player"]].discard.append(fr.data["uid"])
    return True


def _h_offer_ability(state, fr, answer):
    d = fr.data
    eff = state.catalog.effects[d["move_id"]]
    try:
        p, s = B.find_slot(state, d["pid"])
    except B.EngineError:
        return True
    if answer is None:
        if not eff.optional:
            answer = "yes"
        else:
            state.pending = Pending(d["player"], "use_ability", ("yes", "no"), f"use {d['move_id']}?", None, True)
            return False
    state.frames.pop()
    if answer == "yes":
        _mark_usage(state, s, eff)
        state.frames.append(R.effect_frame(state, d["move_id"], d["player"], pid=s.pid))
    return None


def _h_retreat(state, fr, answer):
    """main_retreat: discard Energy equal to the Retreat Cost, then swap with the chosen Benched Pokémon."""
    d = fr.data
    p = d["player"]
    act = state.players[p].active
    if answer is not None:
        act.energy.remove(answer)
        state.players[p].discard.append(answer)
        d["to_pay"] -= 1
    while d["to_pay"] > 0:
        opts = B.dedupe(state, act.energy)
        if len(opts) > 1:
            state.pending = Pending(p, "retreat_discard", tuple(opts), "choose an Energy to discard", None, True)
            return False
        act.energy.remove(opts[0])
        state.players[p].discard.append(opts[0])
        d["to_pay"] -= 1
    B.switch_active(state, p, d["to"])
    return True


def _h_attack(state, fr, answer):
    """attack_use: Confusion check, works-only-if, damage with Weakness and Resistance, then the effect steps."""
    d = fr.data
    p = d["player"]
    if d["stage"] == 1:
        state.frames.pop()
        state.frames.append(Frame("core:end_turn", {}))
        state.frames.append(Frame("core:ko_resolve", {"stage": 0}))
        return None
    d["stage"] = 1
    try:
        _, me = B.find_slot(state, d["pid"])
    except B.EngineError:
        return False
    if "Confused" in me.conditions:                                     # special_conditions: Confused
        heads = B.flip(state)
        state.public.append((state.turn, p, "coin_heads" if heads else "coin_tails", -1))
        if not heads:
            me.damage += B.const(state, "confusion_tails_counters") * B.const(state, "damage_per_counter")
            return False
    eff = state.catalog.effects[d["move_id"]]
    ctx = {"player": p, "pid": me.pid, "card": None, "move_id": d["move_id"]}
    if not all(B.eval_cond(state, ctx, c) for c in eff.works_only_if):
        return False                                                    # "this attack does nothing"
    atk = next(a for a in B.card(state, me.top).attacks if a.move_id == d["move_id"])
    dmg = B.eval_amount(state, ctx, eff.damage) if eff.damage is not None else atk.damage
    defender = state.players[B.opp(p)].active
    if defender is not None and dmg > 0:
        dmg += B.attack_damage_bonus(state, p, me)
        a_type = B.card(state, me.top).type
        dc = B.card(state, defender.top)
        if not eff.ignore_weakness_resistance:
            if dc.weakness and dc.weakness == a_type:
                dmg *= B.const(state, "weakness_multiplier")
            if dc.resistance and dc.resistance == a_type:
                dmg -= state.settings.resistance_amount
        dmg = max(0, dmg)
        if eff.ignore_defender_effects or not B.attack_damage_prevented(state, defender):
            defender.damage += dmg
    if eff.steps:
        state.frames.append(R.effect_frame(state, d["move_id"], p, pid=me.pid, parts=("steps",)))
    return False


def _h_end_turn(state, fr, answer):
    """turn_end then checkup, Knock Outs, and the next player's turn."""
    state.frames.pop()
    state.frames.append(Frame("core:start_turn", {"player": B.opp(state.current)}))
    state.frames.append(Frame("core:ko_resolve", {"stage": 0}))
    state.frames.append(Frame("core:checkup", {}))
    return None


def _h_checkup(state, fr, answer):
    """checkup: Poisoned, Burned, Asleep, Paralyzed in that order, for both players."""
    per = B.const(state, "damage_per_counter")
    order = (state.current, B.opp(state.current))
    for cond in ("Poisoned", "Burned", "Asleep", "Paralyzed"):
        for p in order:
            s = state.players[p].active
            if s is None or cond not in s.conditions:
                continue
            if cond == "Poisoned":
                s.damage += B.const(state, "poison_counters_per_checkup") * per
            elif cond == "Burned":
                s.damage += B.const(state, "burn_counters_per_checkup") * per
                if B.flip(state):
                    s.conditions.discard("Burned")
            elif cond == "Asleep":
                if B.flip(state):
                    s.conditions.discard("Asleep")
            elif cond == "Paralyzed":
                if p == state.current and s.paralyzed_turn < state.turn:
                    s.conditions.discard("Paralyzed")
    return True


def _h_ko_resolve(state, fr, answer):
    """knock_out and win_check: discard Knocked Out Pokémon, take Prize cards, check wins, promote."""
    state.frames.pop()
    took_last = [False, False]
    for p in (state.current, B.opp(state.current)):
        for s in list(state.players[p].in_play()):
            if s.damage >= B.max_hp(state, s):
                value = B.prize_value(state, s)
                cards = B.all_cards_of(s)
                B.remove_slot(state, p, s)
                state.players[p].discard.extend(cards)
                state.players[p].ko_turns.append(state.turn)
                B.log(state, p, "knocked_out", s.top)
                q = B.opp(p)
                prizes = state.players[q].prizes
                for _ in range(min(value, len(prizes))):
                    state.players[q].hand.append(prizes.pop(state.rng.randrange(len(prizes))))
                if not prizes:
                    took_last[q] = True
    no_pokemon = [not state.players[p].in_play() for p in (0, 1)]
    wins = [took_last[p] or no_pokemon[B.opp(p)] for p in (0, 1)]
    if wins[0] or wins[1]:
        reason = ["prizes" if took_last[p] else "no_pokemon" for p in (0, 1)]
        if wins[0] and wins[1]:
            _restart_sudden_death(state)
            return None
        w = 0 if wins[0] else 1
        _game_over(state, w, ("sudden_death_" if state.sudden_death else "") + reason[w])
        return None
    for p in (B.opp(state.current), state.current):            # the current player promotes first
        pl = state.players[p]
        if pl.active is None and pl.bench:
            state.frames.append(Frame("core:promote", {"player": p}))
    return None


def _h_promote(state, fr, answer):
    p = fr.data["player"]
    pl = state.players[p]
    if pl.active is not None or not pl.bench:
        return True
    if answer is None:
        if len(pl.bench) == 1:
            answer = pl.bench[0].pid
        else:
            state.pending = Pending(p, "promote", tuple(s.pid for s in pl.bench), "choose a new Active Pokémon", None, True)
            return False
    B.promote(state, p, answer)
    return True


def _h_start_turn(state, fr, answer):
    """turn_draw: the next player draws 1 card; a player who cannot draw loses."""
    p = fr.data["player"]
    state.turn += 1
    if state.turn > state.settings.turn_limit:
        _game_over(state, None, "draw_turn_limit")
        return True
    state.current = p
    state.flags = {}
    state.players[p].turns_taken += 1
    state.lasting = [le for le in state.lasting if le["until_turn"] >= state.turn]
    if B.draw(state, p, 1) == 0:
        _game_over(state, B.opp(p), ("sudden_death_" if state.sudden_death else "") + "deck_out")
    return True


def _game_over(state: State, w: int | None, result: str) -> None:
    state.phase = "over"
    state.winner = w
    state.result = result
    state.pending = None
    state.frames = []


HANDLERS = {
    "effect": _h_effect,
    "core:after_main": _h_after_main,
    "core:discard_played": _h_discard_played,
    "core:offer_ability": _h_offer_ability,
    "core:retreat": _h_retreat,
    "core:attack": _h_attack,
    "core:end_turn": _h_end_turn,
    "core:checkup": _h_checkup,
    "core:ko_resolve": _h_ko_resolve,
    "core:promote": _h_promote,
    "core:start_turn": _h_start_turn,
}


# =============================================================== winner and observe
def winner(state: State) -> int | None:
    """The winning player, or None while the game runs or when it ended in a draw (see state.result)."""
    return state.winner if state.phase == "over" else None


def _view(state: State, s: Slot | None) -> SlotView | None:
    if s is None:
        return None
    cid = lambda u: B.card(state, u).card_id
    return SlotView(s.pid, tuple(s.cards), tuple(cid(u) for u in s.cards), tuple(s.energy),
                    tuple(cid(u) for u in s.energy), s.tool, cid(s.tool) if s.tool is not None else None,
                    s.damage, B.max_hp(state, s), tuple(sorted(s.conditions)), s.entered_turn, s.evolved_turn,
                    s.moved_to_active_turn, s.paralyzed_turn, tuple(sorted(s.lockouts.items())),
                    tuple(sorted(s.used.items())), B.retreat_cost(state, s))


def observe(state: State, player: int) -> Observation:
    """What `player` may see: own hand, both discard piles, Pokémon in play, counts of hidden zones."""
    me, op = state.players[player], state.players[B.opp(player)]
    cid = lambda u: B.card(state, u).card_id
    hide_opp_board = state.phase == "setup" and not (me.setup_done and op.setup_done)
    pend = state.pending
    mine = pend is not None and pend.player == player
    return Observation(
        player=player, phase=state.phase, turn=state.turn, current=state.current, first_player=state.first_player,
        my_hand=tuple(sorted(me.hand)), my_hand_card_ids=tuple(cid(u) for u in sorted(me.hand)),
        my_deck_count=len(me.deck), my_prize_count=len(me.prizes),
        my_discard=tuple(me.discard), my_discard_card_ids=tuple(cid(u) for u in me.discard),
        my_active=_view(state, me.active), my_bench=tuple(_view(state, s) for s in me.bench),
        opp_hand_count=len(op.hand), opp_deck_count=len(op.deck), opp_prize_count=len(op.prizes),
        opp_discard=tuple(op.discard), opp_discard_card_ids=tuple(cid(u) for u in op.discard),
        opp_active=None if hide_opp_board else _view(state, op.active),
        opp_bench=() if hide_opp_board else tuple(_view(state, s) for s in op.bench),
        stadium=state.stadium, stadium_card_id=cid(state.stadium) if state.stadium is not None else None,
        flags=_public_flags(state),
        pending_kind=pend.kind if pend is not None and (mine or pend.visible_to_opponent) else
        ("hidden" if pend is not None else None),
        pending_player=pend.player if pend is not None else None,
        pending_options=tuple(pend.options) + ((pend.stop,) if pend.stop is not None else ())
        if mine else (),
        pending_option_card_ids=tuple(_option_card_id(state, o) for o in pend.options) if mine else (),
        my_deck_list_card_ids=state.catalog.deck_lists[player],
        public_log=tuple(state.public),
        lasting=tuple(tuple(sorted(le.items())) for le in state.lasting),
        turns_taken=(state.players[0].turns_taken, state.players[1].turns_taken),
        ko_turns=(tuple(state.players[0].ko_turns), tuple(state.players[1].ko_turns)),
        resolving=tuple((fr.data["uid"], cid(fr.data["uid"])) for fr in state.frames if fr.kind == "core:discard_played"),
        winner=state.winner, result=state.result,
    )


def _option_card_id(state: State, o):
    if isinstance(o, tuple):
        return tuple(_option_card_id(state, x) for x in o)
    if isinstance(o, int) and not isinstance(o, bool):
        for p in (0, 1):
            for s in state.players[p].in_play():
                if s.pid == o:
                    return B.card(state, s.top).card_id
        return B.card(state, o).card_id
    return None


def _public_flags(state: State) -> tuple:
    out = []
    for k in sorted(state.flags):
        v = state.flags[k]
        out.append((k, tuple(v) if isinstance(v, list) else v))
    return tuple(out)


def uid_owner(uid: int) -> int:
    return uid // UID_STRIDE


def card_locations(state: State) -> dict[int, str]:
    """Where every card instance is. A Trainer being played is 'resolving' until it is discarded.
    Used by tests to prove that no card is ever created or lost."""
    loc: dict[int, str] = {}

    def put(u, z):
        if u in loc:
            raise B.EngineError(f"card {u} is in two places: {loc[u]} and {z}")
        loc[u] = z
    for p in (0, 1):
        pl = state.players[p]
        for z in ("deck", "hand", "discard", "prizes"):
            for u in getattr(pl, z):
                put(u, f"p{p}_{z}")
        for s in pl.in_play():
            for u in B.all_cards_of(s):
                put(u, f"p{p}_in_play")
    if state.stadium is not None:
        put(state.stadium, "stadium")
    for fr in state.frames:
        if fr.kind == "core:discard_played":
            put(fr.data["uid"], "resolving")
    return loc
