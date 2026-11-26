"""Looking things up on the board, and the small moves that both turns and card effects use."""
from collections import Counter

from tcg_bot import effects as fx
from tcg_bot.effects_data import EFFECTS
from tcg_bot.engine.catalog import Card, catalog
from tcg_bot.engine.types import PokemonInPlay, State

EFFECT: dict[str, fx.Effect] = {e["move_id"]: fx.build(e) for e in EFFECTS}


def effect_for(move_id: str) -> fx.Effect:
    if move_id not in EFFECT:
        raise NotImplementedError(f"no effect data for {move_id}")
    return EFFECT[move_id]


def matches(card: Card, wanted: fx.Filter | None) -> bool:
    """Does the card fit a filter such as "a Basic Pokémon with 70 HP or less"?"""
    if wanted is None:
        return True
    if wanted.any_of:
        return any(matches(card, one) for one in wanted.any_of)
    return ((not wanted.card_kind or card.kind in wanted.card_kind)
            and (wanted.family is None or card.family == wanted.family)
            and (wanted.type is None or card.type == wanted.type)
            and (wanted.hp_max is None or (card.hp is not None and card.hp <= wanted.hp_max))
            and (wanted.rule_box is None or card.has_rule_box == wanted.rule_box)
            and (wanted.is_mega is None or card.is_mega == wanted.is_mega)
            and (wanted.name is None or card.name == wanted.name))


def card_of(state: State, uid: int) -> Card:
    return catalog().cards[state.card_ids[uid]]


def find_pokemon(state: State, player: int, uid: int) -> PokemonInPlay:
    return next(m for m in state.players[player].pokemon() if m.uid == uid)


def passive_steps(state: State, uid: int | None) -> list:
    """What a card that stays in play (Tool, Stadium, Energy) does just by being there."""
    if uid is None:
        return []
    return [step for move in card_of(state, uid).moves
            if effect_for(move.move_id).timing == "passive" for step in effect_for(move.move_id).steps]


def max_hp(state: State, mon: PokemonInPlay) -> int:
    card = card_of(state, mon.card)
    return card.hp + sum(s.amount for s in passive_steps(state, state.stadium)
                         if isinstance(s, fx.ModifyStat) and s.stat == "hp" and matches(card, s.applies_to))


def retreat_cost(state: State, mon: PokemonInPlay) -> int:
    cost = card_of(state, mon.card).retreat_cost
    cost += sum(s.amount for s in passive_steps(state, mon.tool)
                if isinstance(s, fx.ModifyStat) and s.stat == "retreat_cost" and s.side == "attached")
    return max(cost, 0)


def attached_energy(state: State, mon: PokemonInPlay) -> Counter:
    """Type letter -> how many Energy of that type the attached cards provide."""
    have: Counter = Counter()
    for uid in mon.energy:
        letter, amount = catalog().energy[state.card_ids[uid]]
        have[letter] += amount
    return have


def coin_heads(state: State) -> bool:
    return state.rng.random() < 0.5


def draw(state: State, player: int, count: int = 1) -> int:
    """Move up to `count` cards from the top of the deck to the hand. Returns how many moved."""
    p = state.players[player]
    n = min(count, len(p.deck))
    p.hand += p.deck[:n]
    del p.deck[:n]
    return n


def drop_lasting(state: State, uid: int) -> None:
    state.lasting = [e for e in state.lasting if e.target != uid]


def clear_conditions(mon: PokemonInPlay) -> None:
    mon.poisoned = mon.burned = False
    mon.rotation = mon.paralyzed_on_turn = None


def move_to_active(state: State, player: int, uid: int) -> None:
    """Promote a Benched Pokémon into an empty Active Spot."""
    p = state.players[player]
    mon = next(m for m in p.bench if m.uid == uid)
    p.bench.remove(mon)
    mon.moved_to_active_on_turn = state.turn
    p.active = mon


def swap_active(state: State, player: int, uid: int) -> None:
    """The Active goes to the Bench and a Benched Pokémon comes up."""
    p = state.players[player]
    old = p.active
    clear_conditions(old)                                # leaving the Active Spot ends conditions
    drop_lasting(state, old.uid)                         # and effects of attacks
    move_to_active(state, player, uid)
    p.bench.append(old)
