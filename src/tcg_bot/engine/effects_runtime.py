"""Running card effects, which are data (tcg_bot.effects_data). No code here is about one particular card.

Covered: the blocks used by the attacks of the two decks, and by the two Special Energy.
Anything else raises NotImplementedError instead of being skipped.

A step that needs the player to choose stays at the front of its Frame until the choice is complete.
"""
from dataclasses import replace

from tcg_bot import effects as fx
from tcg_bot.effects_data import EFFECTS
from tcg_bot.engine.board import card_of, coin_heads, draw, drop_lasting, find_pokemon, move_to_active, swap_active
from tcg_bot.engine.catalog import Card
from tcg_bot.engine.types import Frame, LastingEffect, PendingChoice, Phase, PokemonInPlay, State

EFFECT: dict[str, fx.Effect] = {e["move_id"]: fx.build(e) for e in EFFECTS}
CHOICES = {"switch_to", "energy_to_hand", "attach_pick", "attach_target"}


def effect_for(move_id: str) -> fx.Effect:
    if move_id not in EFFECT:
        raise NotImplementedError(f"no effect data for {move_id}")
    return EFFECT[move_id]


# ---------------------------------------------------------------- questions and numbers
def holds(state: State, cond: fx.Cond, player: int, source: int) -> bool:
    p = state.players[player]
    if cond.name == "stadium_in_play":
        answer = state.stadium is not None
    elif cond.name == "has_on_bench":
        answer = any(card_of(state, m.card).name == cond.arg for m in p.bench)
    elif cond.name == "has_in_play":
        answer = any(card_of(state, m.card).name == cond.arg for m in p.pokemon())
    elif cond.name == "moved_to_active_this_turn":
        answer = find_pokemon(state, player, source).moved_to_active_on_turn == state.turn
    else:
        raise NotImplementedError(f"condition {cond.name}")
    return answer != cond.negate


def amount(state: State, value, player: int, source: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, fx.Fixed):
        return value.n
    if isinstance(value, fx.PerCount) and value.counter == "opp_hand_size":
        return value.each * len(state.players[1 - player].hand)
    if isinstance(value, fx.Sum):
        return sum(amount(state, part, player, source) for part in value.parts)
    if isinstance(value, fx.IfThen):
        chosen = value.then if holds(state, value.condition, player, source) else value.otherwise
        return amount(state, chosen, player, source)
    raise NotImplementedError(f"amount {value}")


def matches(card: Card, wanted: fx.Filter | None) -> bool:
    if wanted is None:
        return True
    if wanted.hp_max is not None or wanted.rule_box is not None or wanted.is_mega is not None \
            or wanted.name is not None or wanted.any_of:
        raise NotImplementedError(f"filter {wanted}")
    return ((not wanted.card_kind or card.kind in wanted.card_kind)
            and (wanted.family is None or card.family == wanted.family)
            and (wanted.type is None or card.type == wanted.type))


# ---------------------------------------------------------------- protection
def damage_prevented(state: State, mon: PokemonInPlay) -> bool:
    return any(isinstance(e.block, fx.Prevent) and e.block.what == "damage_and_effects" and e.target == mon.uid
               and e.first_turn <= state.turn <= e.last_turn for e in state.lasting)


def effects_prevented(state: State, mon: PokemonInPlay) -> bool:
    """Is this Pokémon protected from the effects of the opponent's attacks?"""
    if damage_prevented(state, mon):
        return True
    return any(isinstance(step, fx.Prevent) and step.what == "attack_effects"
               for uid in mon.energy for move in card_of(state, uid).moves
               for step in effect_for(move.move_id).steps)


# ---------------------------------------------------------------- running steps
def push(state: State, controller: int, source: int, move_id: str) -> None:
    state.stack.append(Frame(controller=controller, source=source, move_id=move_id,
                             steps=effect_for(move_id).steps))


def _advance(state: State) -> None:
    state.stack[-1] = replace(state.stack[-1], steps=state.stack[-1].steps[1:])


def run(state: State) -> None:
    """Carry out steps until there are none left or a player has to choose."""
    while state.stack and not state.pending and state.phase != Phase.OVER:
        frame = state.stack[-1]
        if not frame.steps:
            state.stack.pop()
        else:
            _do(state, frame, frame.steps[0])


def _do(state: State, frame: Frame, step) -> None:
    me, source = frame.controller, frame.source
    p = state.players[me]

    def ask(kind: str, options: list[int], **more) -> None:
        if options:
            state.pending = PendingChoice(player=me, kind=kind, options=tuple(options), **more)
        else:
            _advance(state)                              # nothing to choose: as much as possible is done

    if isinstance(step, fx.CoinFlip):
        outcome = step.heads if coin_heads(state) else step.tails
        state.stack[-1] = replace(frame, steps=outcome + frame.steps[1:])
        return
    if isinstance(step, fx.SwitchActive) and step.side == "you":
        ask("switch_to", [m.uid for m in p.bench])
        return
    if isinstance(step, fx.MoveCards):
        route = (step.source, step.destination)
        if route == ("attached_to_this", "hand") and step.count == 1:
            mon = find_pokemon(state, me, source)
            ask("energy_to_hand", [u for u in mon.energy if matches(card_of(state, u), step.filter)], subject=source)
            return
        if route == ("discard", "attached_to_bench"):
            found = [u for u in p.discard if matches(card_of(state, u), step.filter)] if p.bench else []
            ask("attach_pick", found, min_count=0 if step.up_to else 1, max_count=step.count)
            return
        if route == ("this_pokemon_and_attached", "hand"):
            _return_to_hand(state, me, source)
            _advance(state)
            return
    if isinstance(step, fx.DamageSelf):
        find_pokemon(state, me, source).damage += step.amount
    elif isinstance(step, fx.ApplyStatus) and step.status == "Asleep" and step.target.side == "opponent":
        target = state.players[1 - me].active
        if target and not effects_prevented(state, target):
            target.rotation = "Asleep"
    elif isinstance(step, fx.Prevent) and step.duration == "opponent_next_turn":
        state.lasting.append(LastingEffect(block=step, controller=me, source=source, target=source,
                                           first_turn=state.turn + 1, last_turn=state.turn + 1))
    elif isinstance(step, fx.Lockout) and step.duration == "your_next_turn":
        locked = replace(step, attack=frame.move_id) if step.attack == "this_attack" else step
        state.lasting.append(LastingEffect(block=locked, controller=me, source=source, target=source,
                                           first_turn=state.turn + 2, last_turn=state.turn + 2))
    elif isinstance(step, fx.Draw) and step.side == "you":
        draw(state, me, amount(state, step.count, me, source))
    else:
        raise NotImplementedError(f"step {step}")
    _advance(state)


def _return_to_hand(state: State, player: int, uid: int) -> None:
    p = state.players[player]
    mon = find_pokemon(state, player, uid)
    cards = mon.stack + mon.energy + ([] if mon.tool is None else [mon.tool])
    p.hand += cards
    p.revealed.update(cards)                             # the opponent saw them in play
    if p.active is mon:
        p.active = None
    else:
        p.bench.remove(mon)
    drop_lasting(state, uid)


def answer(state: State, pending: PendingChoice, option: int | None) -> None:
    """Apply a choice made during an effect. `option` None means "done choosing"."""
    me = pending.player
    p = state.players[me]
    if pending.kind == "switch_to":
        swap_active(state, me, option)
        _advance(state)
    elif pending.kind == "energy_to_hand":
        find_pokemon(state, me, pending.subject).energy.remove(option)
        p.hand.append(option)
        p.revealed.add(option)
        _advance(state)
    elif pending.kind == "attach_pick":
        if option is None:
            _advance(state)
        else:                                            # now say which Benched Pokémon gets it
            state.pending = PendingChoice(player=me, kind="attach_target", options=tuple(m.uid for m in p.bench),
                                          picked=pending.picked, min_count=len(pending.picked) + 1,
                                          max_count=pending.max_count, subject=option)
    elif pending.kind == "attach_target":
        p.discard.remove(pending.subject)
        next(m for m in p.bench if m.uid == option).energy.append(pending.subject)
        done = pending.picked + (pending.subject,)
        step = state.stack[-1].steps[0]
        left = [u for u in p.discard if matches(card_of(state, u), step.filter)]
        if left and len(done) < pending.max_count:
            state.pending = PendingChoice(player=me, kind="attach_pick", options=tuple(left), picked=done,
                                          min_count=0, max_count=pending.max_count)
        else:
            _advance(state)
    else:
        raise NotImplementedError(f"choice {pending.kind!r} is not written yet")


__all__ = ["CHOICES", "EFFECT", "amount", "answer", "damage_prevented", "effect_for", "effects_prevented",
           "holds", "matches", "move_to_active", "push", "run"]
