"""Running card effects, which are data (tcg_bot.effects_data). No code here is about one particular card.

Covered: every block the 54 effects of the two decks use. Anything else raises NotImplementedError.

A step that needs the player to choose stays at the front of its Frame until the choice is complete.
Two passive cards need no code here: Damp (nothing in these decks for it to stop) and Battle Cage
(nothing in these decks puts damage counters on the Bench).
"""
from dataclasses import replace

from tcg_bot import effects as fx
from tcg_bot.engine.board import (
    EFFECT,
    card_of,
    coin_heads,
    draw,
    drop_lasting,
    effect_for,
    find_pokemon,
    matches,
    passive_steps,
    swap_active,
)
from tcg_bot.engine.catalog import catalog
from tcg_bot.engine.types import Frame, LastingEffect, PendingChoice, Phase, PokemonInPlay, State, owner_of

CHOICES = {"switch_to", "gust", "heal", "pick", "energy_to_hand", "attach_pick", "attach_target"}


# ---------------------------------------------------------------- questions and numbers
def holds(state: State, cond: fx.Cond, player: int, source: int | None, facts: frozenset[str] = frozenset()) -> bool:
    p, foe = state.players[player], state.players[1 - player]
    if cond.name == "stadium_in_play":
        answer = state.stadium is not None
    elif cond.name == "has_on_bench":
        answer = any(card_of(state, m.card).name == cond.arg for m in p.bench)
    elif cond.name == "has_in_play":
        answer = any(card_of(state, m.card).name == cond.arg for m in p.pokemon())
    elif cond.name == "moved_to_active_this_turn":
        answer = find_pokemon(state, player, source).moved_to_active_on_turn == state.turn
    elif cond.name == "prizes_remaining_equals":
        answer = len(p.prizes) == cond.arg
    elif cond.name == "opponent_prizes_at_most":
        answer = len(foe.prizes) <= cond.arg
    elif cond.name == "own_pokemon_knocked_out_last_opponent_turn":
        answer = p.knocked_out_on_turn == state.turn - 1
    elif cond.name == "drew_any":
        answer = "drew" in facts
    elif cond.name == "healed_any":
        answer = "healed" in facts
    elif cond.name in ("either_player_put_cards_on_bottom", "they_put_cards_on_bottom"):
        answer = "bottom" in facts
    else:
        raise NotImplementedError(f"condition {cond.name}")
    return answer != cond.negate


def amount(state: State, value, player: int, source: int | None) -> int:
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


# ---------------------------------------------------------------- protection and bonuses
def damage_prevented(state: State, mon: PokemonInPlay) -> bool:
    return any(isinstance(e.block, fx.Prevent) and e.block.what == "damage_and_effects" and e.target == mon.uid
               and e.first_turn <= state.turn <= e.last_turn for e in state.lasting)


def effects_prevented(state: State, mon: PokemonInPlay) -> bool:
    """Is this Pokémon protected from the effects of the opponent's attacks?"""
    return damage_prevented(state, mon) or any(
        isinstance(s, fx.Prevent) and s.what == "attack_effects" for uid in mon.energy for s in passive_steps(state, uid))


def damage_bonus(state: State, player: int, attacker: PokemonInPlay) -> int:
    card = card_of(state, attacker.card)
    return sum(e.block.amount for e in state.lasting
               if isinstance(e.block, fx.ModifyStat) and e.block.stat == "attack_damage" and e.controller == player
               and e.first_turn <= state.turn <= e.last_turn and matches(card, e.block.applies_to))


# ---------------------------------------------------------------- may it be used?
def _options(state: State, player: int, step: fx.MoveCards) -> list[int]:
    """The cards a choosing step can pick from right now."""
    p = state.players[player]
    if step.source == "attached_tool_any":
        return [m.tool for q in state.players for m in q.pokemon() if m.tool is not None]
    zones = {"deck": p.deck, "top_of_deck": p.deck[:step.look_n], "discard": p.discard, "hand": p.hand}
    if step.source not in zones:
        raise NotImplementedError(f"cards from {step.source}")
    return [u for u in zones[step.source] if matches(card_of(state, u), step.filter)]


def _room(state: State, player: int) -> int:
    return catalog().constants["bench_max"] - len(state.players[player].bench)


def _can_do_something(state: State, player: int, step, using: int | None) -> bool:
    p, foe = state.players[player], state.players[1 - player]
    if isinstance(step, fx.SwitchActive):
        return bool(p.bench if step.side == "you" else foe.bench)
    if isinstance(step, fx.Heal):
        return any(m.damage and matches(card_of(state, m.card), step.target.filter) for m in p.pokemon())
    if isinstance(step, fx.Draw):
        return bool(p.deck)
    if isinstance(step, fx.MoveCards):
        if step.source in ("deck", "top_of_deck"):
            return bool(p.deck) and (step.destination != "bench" or _room(state, player) > 0)
        if step.source in ("discard", "attached_tool_any"):
            return bool(_options(state, player, step))
        if step.source == "stadium_in_play":
            return state.stadium is not None
        if step.destination == "deck_bottom":
            mine = [u for u in p.hand if u != using] if step.side == "both" else []
            return bool(mine or foe.hand)
    return True


def can_use(state: State, player: int, move_id: str, using: int | None = None) -> bool:
    """Are the requirements met, can the cost be paid, and would it do anything? `using` is the card being played."""
    effect = effect_for(move_id)
    if not all(holds(state, c, player, using) for c in effect.requires):
        return False
    for cost in effect.cost:
        if len([u for u in _options(state, player, cost) if u != using]) < cost.count:
            return False
    return not effect.steps or _can_do_something(state, player, effect.steps[0], using)


# ---------------------------------------------------------------- running steps
def push(state: State, controller: int, source: int, move_id: str) -> None:
    effect = effect_for(move_id)
    state.stack.append(Frame(controller=controller, source=source, move_id=move_id, steps=effect.cost + effect.steps))


def _advance(state: State) -> None:
    state.stack[-1] = replace(state.stack[-1], steps=state.stack[-1].steps[1:])


def _note(state: State, fact: str) -> None:
    state.stack[-1] = replace(state.stack[-1], facts=state.stack[-1].facts | {fact})


def run(state: State) -> None:
    """Carry out steps until there are none left or a player has to choose."""
    while state.stack and not state.pending and state.phase != Phase.OVER:
        frame = state.stack[-1]
        if not frame.steps:
            state.stack.pop()
        else:
            _do(state, frame, frame.steps[0])


def _ask(state: State, player: int, kind: str, options: list[int], **more) -> None:
    if options:
        state.pending = PendingChoice(player=player, kind=kind, options=tuple(options), **more)
    else:
        _advance(state)                                  # nothing to choose: as much as possible is done


def _sides(player: int, side: str) -> list[int]:
    return {"you": [player], "that_player": [player], "opponent": [1 - player], "both": [player, 1 - player]}[side]


def _do(state: State, frame: Frame, step) -> None:
    me, source = frame.controller, frame.source
    p = state.players[me]
    if isinstance(step, fx.CoinFlip):
        outcome = step.heads if coin_heads(state) else step.tails
        state.stack[-1] = replace(frame, steps=outcome + frame.steps[1:])
        return
    if isinstance(step, fx.If):
        branch = step.then if holds(state, step.condition, me, source, frame.facts) else step.otherwise
        state.stack[-1] = replace(frame, steps=branch + frame.steps[1:])
        return
    if isinstance(step, fx.ForEachPlayer):
        _advance(state)
        for player in (1 - me, me):                      # pushed last, so the card's owner goes first
            state.stack.append(Frame(controller=player, source=source, move_id=frame.move_id, steps=step.steps))
        return
    if isinstance(step, fx.SwitchActive):
        if step.side == "you":
            _ask(state, me, "switch_to", [m.uid for m in p.bench])
        else:
            _ask(state, me, "gust", [m.uid for m in state.players[1 - me].bench])
        return
    if isinstance(step, fx.Heal) and step.target.which == "chosen" and step.amount == "all":
        _ask(state, me, "heal", [m.uid for m in p.pokemon()
                                 if m.damage and matches(card_of(state, m.card), step.target.filter)])
        return
    if isinstance(step, fx.MoveCards):
        _move(state, frame, step)
        return
    if isinstance(step, fx.Draw):
        count = amount(state, step.count, me, source)
        for player in _sides(me, step.side):
            if draw(state, player, count) and player == me:
                _note(state, "drew")
    elif isinstance(step, fx.DamageSelf):
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
    elif isinstance(step, fx.ModifyStat) and step.duration == "this_turn":
        state.lasting.append(LastingEffect(block=step, controller=me, source=source,
                                           first_turn=state.turn, last_turn=state.turn))
    else:
        raise NotImplementedError(f"step {step}")
    _advance(state)


# ---------------------------------------------------------------- moving cards
def _move(state: State, frame: Frame, step: fx.MoveCards) -> None:
    me, source = frame.controller, frame.source
    p = state.players[me]
    route = (step.source, step.destination)

    if route == ("hand", "deck_shuffle"):
        for player in _sides(me, step.side):
            q = state.players[player]
            q.deck += q.hand
            q.hand.clear()
            q.revealed.clear()
            state.rng.shuffle(q.deck)
    elif route == ("hand", "deck_bottom"):
        for player in _sides(me, step.side):
            q = state.players[player]
            if q.hand:
                cards = q.hand.copy()
                state.rng.shuffle(cards)                 # "shuffles their hand and puts it on the bottom"
                q.deck += cards
                q.hand.clear()
                q.revealed.difference_update(cards)
                _note(state, "bottom")
    elif step.source == "this_pokemon_and_attached":
        _leave_play(state, me, source, step.destination)
    elif route == ("stadium_in_play", "discard"):
        discard_stadium(state)
    elif route == ("attached_to_target", "hand"):
        mon = find_pokemon(state, me, frame.chosen)
        back = [u for u in mon.energy if matches(card_of(state, u), step.filter)]
        for uid in back:
            mon.energy.remove(uid)
            p.hand.append(uid)
            p.revealed.add(uid)
    elif route == ("attached_to_this", "hand") and step.count == 1:
        mon = find_pokemon(state, me, source)
        _ask(state, me, "energy_to_hand", [u for u in mon.energy if matches(card_of(state, u), step.filter)],
             subject=source)
        return
    elif route == ("discard", "attached_to_bench"):
        _ask(state, me, "attach_pick", _options(state, me, step) if p.bench else [],
             min_count=0 if step.up_to else 1, max_count=step.count)
        return
    else:                                                # the player picks cards one at a time
        options = _options(state, me, step)
        limit = len(options) if step.count == "all" else step.count
        if step.destination == "bench":
            limit = min(limit, _room(state, me))
        if options and limit > 0:
            may_stop = step.up_to or step.optional
            state.pending = PendingChoice(player=me, kind="pick", options=tuple(options), max_count=limit,
                                          min_count=0 if may_stop else min(limit, len(options)))
            return
        _after_picking(state, me, step)
    _advance(state)


def _take(state: State, player: int, step: fx.MoveCards, uid: int) -> None:
    p = state.players[player]
    route = (step.source, step.destination)
    if route in (("deck", "hand"), ("top_of_deck", "hand"), ("discard", "hand")):
        (p.discard if step.source == "discard" else p.deck).remove(uid)
        p.hand.append(uid)
        if step.reveal or step.source == "discard":
            p.revealed.add(uid)
    elif route == ("deck", "bench"):
        p.deck.remove(uid)
        p.bench.append(PokemonInPlay(stack=[uid], in_play_since=state.turn))
    elif route == ("hand", "discard"):
        p.hand.remove(uid)
        p.revealed.discard(uid)
        p.discard.append(uid)
    elif route == ("attached_tool_any", "discard"):
        mon = next(m for q in state.players for m in q.pokemon() if m.tool == uid)
        mon.tool = None
        state.players[owner_of(uid)].discard.append(uid)
    else:
        raise NotImplementedError(f"moving cards {route}")


def _after_picking(state: State, player: int, step: fx.MoveCards) -> None:
    if step.shuffle_after or step.rest_to == "deck_shuffle":
        state.rng.shuffle(state.players[player].deck)


def _leave_play(state: State, player: int, uid: int, destination: str) -> None:
    p = state.players[player]
    mon = find_pokemon(state, player, uid)
    cards = mon.stack + mon.energy + ([] if mon.tool is None else [mon.tool])
    if p.active is mon:
        p.active = None
    else:
        p.bench.remove(mon)
    drop_lasting(state, uid)
    if destination == "hand":
        p.hand += cards
        p.revealed.update(cards)                         # the opponent saw them in play
    elif destination == "deck_shuffle":
        p.deck += cards
        state.rng.shuffle(p.deck)
    else:
        raise NotImplementedError(f"a Pokémon leaving to {destination}")


def discard_stadium(state: State) -> None:
    if state.stadium is not None:
        state.players[owner_of(state.stadium)].discard.append(state.stadium)
        state.stadium = None


# ---------------------------------------------------------------- answers
def answer(state: State, pending: PendingChoice, option: int | None) -> None:
    """Apply a choice made during an effect. `option` None means "done choosing"."""
    me = pending.player
    p = state.players[me]
    if pending.kind == "switch_to":
        swap_active(state, me, option)
        _advance(state)
    elif pending.kind == "gust":
        swap_active(state, 1 - me, option)
        _advance(state)
    elif pending.kind == "heal":
        find_pokemon(state, me, option).damage = 0
        state.stack[-1] = replace(state.stack[-1], chosen=option)
        _note(state, "healed")
        _advance(state)
    elif pending.kind == "pick":
        step = state.stack[-1].steps[0]
        if option is not None:
            _take(state, me, step, option)
            done = pending.picked + (option,)
            left = [u for u in _options(state, me, step) if u not in done]
            if left and len(done) < pending.max_count:
                state.pending = replace(pending, options=tuple(left), picked=done)
                return
        _after_picking(state, me, step)
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
        left = _options(state, me, state.stack[-1].steps[0])
        if left and len(done) < pending.max_count:
            state.pending = PendingChoice(player=me, kind="attach_pick", options=tuple(left), picked=done,
                                          min_count=0, max_count=pending.max_count)
        else:
            _advance(state)
    else:
        raise NotImplementedError(f"choice {pending.kind!r} is not written yet")


__all__ = ["CHOICES", "EFFECT", "amount", "answer", "can_use", "damage_bonus", "damage_prevented", "discard_stadium",
           "effect_for", "effects_prevented", "holds", "matches", "push", "run"]
