"""One turn: main-phase actions, attacks, Knock Outs, Prizes, the Checkup and passing the turn.

NOT WRITTEN YET, so legal_actions does not offer them: Abilities and Trainer cards.
The only Special Condition the two decks' attacks cause is Asleep.
"""
from dataclasses import replace

from tcg_bot.effects import Lockout
from tcg_bot.engine import effects_runtime as runtime
from tcg_bot.engine.board import (
    attached_energy,
    card_of,
    clear_conditions,
    coin_heads,
    draw,
    drop_lasting,
    find_pokemon,
    max_hp,
    move_to_active,
    retreat_cost,
    swap_active,
)
from tcg_bot.engine.catalog import Card, Move, catalog
from tcg_bot.engine.types import PLAYERS, Action, ActionKind, PendingChoice, Phase, PokemonInPlay, State, TurnFlags

MAX_TURNS = 200                      # after this the game is a draw, so that none runs forever


def can_pay(state: State, mon: PokemonInPlay, move: Move) -> bool:
    have = attached_energy(state, mon)
    return all(have[t] >= n for t, n in move.typed_cost) and have.total() >= move.total_cost


def locked(state: State, mon: PokemonInPlay, move_id: str) -> bool:
    return any(isinstance(e.block, Lockout) and e.target == mon.uid and e.block.attack == move_id
               and e.first_turn <= state.turn <= e.last_turn for e in state.lasting)


def usable_attacks(state: State, mon: PokemonInPlay) -> list[Move]:
    return [m for m in card_of(state, mon.card).moves
            if m.kind == "attack" and can_pay(state, mon, m) and not locked(state, mon, m.move_id)]


# ---------------------------------------------------------------- the turn
def start_turn(state: State, player: int) -> None:
    state.turn += 1
    state.current = player
    state.lasting = [e for e in state.lasting if e.last_turn >= state.turn]
    state.players[player].flags = TurnFlags()
    for p in state.players:
        for mon in p.pokemon():
            mon.abilities_used.clear()
    if draw(state, player) == 0:                      # cannot draw at the start of the turn
        _finish(state, 1 - player, "deck_out")


def end_turn(state: State) -> None:
    _checkup(state)
    if state.turn >= MAX_TURNS:
        _finish(state, None, "turn_limit")
    else:
        start_turn(state, 1 - state.current)


def _checkup(state: State) -> None:
    """Between turns. Asleep: flip, heads wakes the Pokémon."""
    for p in state.players:
        if p.active and p.active.rotation == "Asleep" and coin_heads(state):
            p.active.rotation = None


def _finish(state: State, winner: int | None, reason: str) -> None:
    state.phase, state.winner, state.end_reason, state.pending = Phase.OVER, winner, reason, None
    state.stack.clear()


# ---------------------------------------------------------------- what can be done
def main_actions(state: State) -> list[Action]:
    me = state.current
    p = state.players[me]
    if any(m.poisoned or m.burned or m.rotation not in (None, "Asleep")
           for q in state.players for m in q.pokemon()):
        raise NotImplementedError("only the Asleep condition is written")
    asleep = p.active.rotation == "Asleep"
    out = [Action(ActionKind.END_TURN, me)]
    for uid in p.hand:
        card = card_of(state, uid)
        if card.is_basic_pokemon and len(p.bench) < catalog().constants["bench_max"]:
            out.append(Action(ActionKind.BENCH, me, card=uid))
        if card.family == "pokemon" and card.previous_stage and not state.is_first_turn_of(me):
            out += [Action(ActionKind.EVOLVE, me, card=uid, target=m.uid) for m in p.pokemon()
                    if card_of(state, m.card).name == card.previous_stage and m.in_play_since < state.turn]
        if card.family == "energy" and not p.flags.energy_attached:
            out += [Action(ActionKind.ATTACH_ENERGY, me, card=uid, target=m.uid) for m in p.pokemon()]
    if not asleep and not p.flags.retreated and attached_energy(state, p.active).total() >= retreat_cost(state, p.active):
        out += [Action(ActionKind.RETREAT, me, target=m.uid) for m in p.bench]
    if not asleep and state.turn > 1:                    # whoever goes first cannot attack on turn 1
        out += [Action(ActionKind.ATTACK, me, move_id=m.move_id) for m in usable_attacks(state, p.active)]
    return out


# ---------------------------------------------------------------- doing it
def apply_action(state: State, action: Action) -> None:
    me = action.player
    p = state.players[me]
    if action.kind == ActionKind.BENCH:
        p.hand.remove(action.card)
        p.bench.append(PokemonInPlay(stack=[action.card], in_play_since=state.turn))
    elif action.kind == ActionKind.EVOLVE:
        mon = find_pokemon(state, me, action.target)
        p.hand.remove(action.card)
        mon.stack.append(action.card)
        mon.in_play_since = state.turn
        mon.abilities_used.clear()
        clear_conditions(mon)
        drop_lasting(state, mon.uid)
    elif action.kind == ActionKind.ATTACH_ENERGY:
        p.hand.remove(action.card)
        find_pokemon(state, me, action.target).energy.append(action.card)
        p.flags.energy_attached = True
        for move in card_of(state, action.card).moves:   # a Special Energy that acts when attached
            if runtime.effect_for(move.move_id).timing == "on_attach_from_hand":
                runtime.push(state, me, action.target, move.move_id)
        runtime.run(state)
    elif action.kind == ActionKind.RETREAT:
        cost = retreat_cost(state, p.active)             # every Energy so far provides exactly 1
        if cost:
            state.pending = PendingChoice(player=me, kind="retreat_cost", options=tuple(p.active.energy),
                                          min_count=cost, max_count=cost, subject=action.target)
        else:
            _retreat(state, me, action.target)
    elif action.kind == ActionKind.ATTACK:
        _attack(state, action.move_id)
    elif action.kind == ActionKind.END_TURN:
        end_turn(state)
    else:
        raise NotImplementedError(f"{action.kind.value} is not written yet")


def answer_choice(state: State, pending: PendingChoice, option: int | None) -> None:
    p = state.players[pending.player]
    if pending.kind == "retreat_cost":
        picked = pending.picked + (option,)
        if len(picked) < pending.min_count:
            state.pending = replace(pending, picked=picked)
            return
        for uid in picked:
            p.active.energy.remove(uid)
            p.discard.append(uid)
        _retreat(state, pending.player, pending.subject)
    elif pending.kind == "new_active":                   # only attacks empty an Active Spot so far
        move_to_active(state, pending.player, option)
        _finish_attack(state)
    elif pending.kind in runtime.CHOICES:
        attacking = runtime.effect_for(state.stack[-1].move_id).timing == "attack"
        runtime.answer(state, pending, option)
        runtime.run(state)
        if attacking and not state.pending:
            _finish_attack(state)
    else:
        raise NotImplementedError(f"choice {pending.kind!r} is not written yet")


def _retreat(state: State, player: int, uid: int) -> None:
    swap_active(state, player, uid)
    state.players[player].flags.retreated = True


# ---------------------------------------------------------------- attacking
def _attack(state: State, move_id: str) -> None:
    """Damage first, then the text, then Knock Outs, then the turn ends."""
    me = state.current
    foe = 1 - me
    attacker = state.players[me].active
    defender = state.players[foe].active
    move = next(m for m in card_of(state, attacker.card).moves if m.move_id == move_id)
    effect = runtime.effect_for(move_id)
    if all(runtime.holds(state, c, me, attacker.uid) for c in effect.works_only_if):
        defender.damage += attack_damage(state, attacker, defender, move)
        runtime.push(state, me, attacker.uid, move_id)
        runtime.run(state)
    if not state.pending:
        _finish_attack(state)


def attack_damage(state: State, attacker: PokemonInPlay, defender: PokemonInPlay, move: Move) -> int:
    effect = runtime.effect_for(move.move_id)
    if effect.damage is None:
        damage = move.damage or 0
    else:
        damage = runtime.amount(state, effect.damage, state.current, attacker.uid)
    if not damage:
        return 0
    if not effect.ignore_weakness_resistance:
        kind, target = card_of(state, attacker.card).type, card_of(state, defender.card)
        if target.weakness == kind:
            damage *= catalog().constants["weakness_multiplier"]
        if target.resistance == kind:
            damage -= catalog().constants["resistance_amount"]
    if not effect.ignore_defender_effects and runtime.damage_prevented(state, defender):
        return 0
    return max(damage, 0)


def _finish_attack(state: State) -> None:
    _resolve_knock_outs(state)
    if state.phase == Phase.OVER:
        return
    for player in (1 - state.current, state.current):    # each empty Active Spot gets a new Pokémon
        p = state.players[player]
        if p.active is None:
            state.pending = PendingChoice(player=player, kind="new_active", options=tuple(m.uid for m in p.bench))
            return
    end_turn(state)


def prize_value(card: Card) -> int:
    if card.is_mega:
        return catalog().constants["mega_ex_prize_value"]
    return catalog().constants["ex_prize_value"] if card.is_ex else 1


def _resolve_knock_outs(state: State) -> None:
    """Knock Out everything with enough damage, hand out Prizes, then see who has won."""
    for player in PLAYERS:
        for mon in state.players[player].pokemon():
            if mon.damage >= max_hp(state, mon):
                _knock_out(state, player, mon)
    wins = [not state.players[i].prizes or not state.players[1 - i].pokemon() for i in PLAYERS]
    if all(wins):                                        # Sudden Death is a new game: not played out here
        _finish(state, None, "sudden_death")
    elif any(wins):
        winner = wins.index(True)
        _finish(state, winner, "prizes" if not state.players[winner].prizes else "no_pokemon")


def _knock_out(state: State, player: int, mon: PokemonInPlay) -> None:
    p = state.players[player]
    taker = state.players[1 - player]
    value = prize_value(card_of(state, mon.card))
    p.discard += mon.stack + mon.energy + ([] if mon.tool is None else [mon.tool])
    if p.active is mon:
        p.active = None
    else:
        p.bench.remove(mon)
    drop_lasting(state, mon.uid)
    p.knocked_out_on_turn = state.turn
    taker.hand += taker.prizes[:value]                   # face down, so which ones makes no difference
    del taker.prizes[:value]
