"""Looking things up on the board, and the small moves that both turns and card effects use."""
from collections import Counter

from tcg_bot.engine.catalog import Card, catalog
from tcg_bot.engine.types import PokemonInPlay, State


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
