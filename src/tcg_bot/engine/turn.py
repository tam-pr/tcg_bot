"""One turn: main-phase actions, attacks, Knock Outs, Prizes and passing the turn.

NOT WRITTEN YET, so legal_actions does not offer them: attacks that have card text, Abilities,
Trainer cards and Special Energy. Special Conditions cannot occur until those exist.
"""
from collections import Counter
from dataclasses import replace

from tcg_bot.engine.catalog import Card, Move, catalog
from tcg_bot.engine.types import Action, ActionKind, PendingChoice, Phase, PokemonInPlay, State, TurnFlags

MAX_TURNS = 200                      # after this the game is a draw, so that none runs forever


# ---------------------------------------------------------------- looking things up
def card_of(state: State, uid: int) -> Card:
    return catalog().cards[state.card_ids[uid]]


def find_pokemon(state: State, player: int, uid: int) -> PokemonInPlay:
    return next(m for m in state.players[player].pokemon() if m.uid == uid)


def max_hp(state: State, mon: PokemonInPlay) -> int:
    return card_of(state, mon.card).hp                   # Stadium changes go here later


def retreat_cost(state: State, mon: PokemonInPlay) -> int:
    return card_of(state, mon.card).retreat_cost         # Tool changes go here later


def attached_energy(state: State, mon: PokemonInPlay) -> Counter:
    """Type letter -> how many Energy of that type the attached cards provide."""
    have: Counter = Counter()
    for uid in mon.energy:
        letter, amount = catalog().energy[state.card_ids[uid]]
        have[letter] += amount
    return have


def can_pay(state: State, mon: PokemonInPlay, move: Move) -> bool:
    have = attached_energy(state, mon)
    return all(have[t] >= n for t, n in move.typed_cost) and have.total() >= move.total_cost


def usable_attacks(state: State, mon: PokemonInPlay) -> list[Move]:
    return [m for m in card_of(state, mon.card).moves
            if m.kind == "attack" and m.text is None and can_pay(state, mon, m)]


# ---------------------------------------------------------------- drawing and the turn
def draw(state: State, player: int, count: int = 1) -> int:
    """Move up to `count` cards from the top of the deck to the hand. Returns how many moved."""
    p = state.players[player]
    n = min(count, len(p.deck))
    p.hand += p.deck[:n]
    del p.deck[:n]
    return n


def start_turn(state: State, player: int) -> None:
    state.turn += 1
    state.current = player
    state.players[player].flags = TurnFlags()
    for p in state.players:
        for mon in p.pokemon():
            mon.abilities_used.clear()
    if draw(state, player) == 0:                      # cannot draw at the start of the turn
        _finish(state, 1 - player, "deck_out")


def end_turn(state: State) -> None:
    if state.turn >= MAX_TURNS:
        _finish(state, None, "turn_limit")
    else:
        start_turn(state, 1 - state.current)


def _finish(state: State, winner: int | None, reason: str) -> None:
    state.phase, state.winner, state.end_reason, state.pending = Phase.OVER, winner, reason, None


# ---------------------------------------------------------------- what can be done
def main_actions(state: State) -> list[Action]:
    me = state.current
    p = state.players[me]
    if any(m.poisoned or m.burned or m.rotation for q in state.players for m in q.pokemon()):
        raise NotImplementedError("Special Conditions are not written yet")
    out = [Action(ActionKind.END_TURN, me)]
    for uid in p.hand:
        card = card_of(state, uid)
        if card.is_basic_pokemon and len(p.bench) < catalog().constants["bench_max"]:
            out.append(Action(ActionKind.BENCH, me, card=uid))
        if card.family == "pokemon" and card.previous_stage and not state.is_first_turn_of(me):
            out += [Action(ActionKind.EVOLVE, me, card=uid, target=m.uid) for m in p.pokemon()
                    if card_of(state, m.card).name == card.previous_stage and m.in_play_since < state.turn]
        if card.kind == "Basic Energy" and not p.flags.energy_attached:
            out += [Action(ActionKind.ATTACH_ENERGY, me, card=uid, target=m.uid) for m in p.pokemon()]
    if not p.flags.retreated and attached_energy(state, p.active).total() >= retreat_cost(state, p.active):
        out += [Action(ActionKind.RETREAT, me, target=m.uid) for m in p.bench]
    if state.turn > 1:                                   # whoever goes first cannot attack on turn 1
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
        _drop_lasting(state, mon.uid)
    elif action.kind == ActionKind.ATTACH_ENERGY:
        p.hand.remove(action.card)
        find_pokemon(state, me, action.target).energy.append(action.card)
        p.flags.energy_attached = True
    elif action.kind == ActionKind.RETREAT:
        cost = retreat_cost(state, p.active)             # every Energy used so far provides exactly 1
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


def answer_choice(state: State, pending: PendingChoice, option: int) -> None:
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
    elif pending.kind == "new_active":                   # only attacks Knock Out so far, so the turn ends
        _move_to_active(state, pending.player, option)
        end_turn(state)
    else:
        raise NotImplementedError(f"choice {pending.kind!r} is not written yet")


def _drop_lasting(state: State, uid: int) -> None:
    state.lasting = [e for e in state.lasting if e.target != uid]


def _move_to_active(state: State, player: int, uid: int) -> None:
    p = state.players[player]
    mon = next(m for m in p.bench if m.uid == uid)
    p.bench.remove(mon)
    mon.moved_to_active_on_turn = state.turn
    p.active = mon


def _retreat(state: State, player: int, uid: int) -> None:
    p = state.players[player]
    old = p.active
    _drop_lasting(state, old.uid)                        # attack effects end on leaving the Active Spot
    _move_to_active(state, player, uid)
    p.bench.append(old)
    p.flags.retreated = True


def _attack(state: State, move_id: str) -> None:
    me = state.current
    foe = 1 - me
    attacker = state.players[me].active
    defender = state.players[foe].active
    move = next(m for m in card_of(state, attacker.card).moves if m.move_id == move_id)
    kind = card_of(state, attacker.card).type
    target = card_of(state, defender.card)
    damage = move.damage or 0
    if damage and target.weakness == kind:
        damage *= catalog().constants["weakness_multiplier"]
    if damage and target.resistance == kind:
        damage -= catalog().constants["resistance_amount"]
    defender.damage += max(damage, 0)
    if defender.damage >= max_hp(state, defender):
        _knock_out(state, foe, defender)
    if state.phase == Phase.OVER:
        return
    if state.players[foe].active is None:
        state.pending = PendingChoice(player=foe, kind="new_active",
                                      options=tuple(m.uid for m in state.players[foe].bench))
    else:
        end_turn(state)


def prize_value(card: Card) -> int:
    if card.is_mega:
        return catalog().constants["mega_ex_prize_value"]
    return catalog().constants["ex_prize_value"] if card.is_ex else 1


def _knock_out(state: State, player: int, mon: PokemonInPlay) -> None:
    p = state.players[player]
    taker = state.players[1 - player]
    value = prize_value(card_of(state, mon.card))
    p.discard += mon.stack + mon.energy + ([] if mon.tool is None else [mon.tool])
    if p.active is mon:
        p.active = None
    else:
        p.bench.remove(mon)
    _drop_lasting(state, mon.uid)
    p.knocked_out_on_turn = state.turn
    taker.hand += taker.prizes[:value]                   # face down, so which ones makes no difference
    del taker.prizes[:value]
    if not taker.prizes:
        _finish(state, 1 - player, "prizes")
    elif not p.pokemon():
        _finish(state, 1 - player, "no_pokemon")
