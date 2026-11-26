import random

import pytest

from tcg_bot.engine.catalog import catalog
from tcg_bot.engine.game import card_of, legal_actions, new_game, step
from tcg_bot.engine.turn import prize_value
from tcg_bot.engine.types import Action, ActionKind, Phase, PokemonInPlay

DECKS = ("lopunny", "mega_lucario")


def opening(seed=0):
    state = new_game(DECKS, seed)
    while state.phase == Phase.SETUP:
        state = step(state, legal_actions(state)[-1])
    return state


def take(state, player, name):
    """Remove one copy of a card from the player's hidden zones and return its uid."""
    p = state.players[player]
    for zone in (p.deck, p.hand, p.prizes):
        for uid in zone:
            if card_of(state, uid).name == name:
                zone.remove(uid)
                return uid
    raise AssertionError(f"no {name} left")


def set_active(state, player, name, energy=()):
    p = state.players[player]
    p.discard += p.active.stack + p.active.energy
    p.active = PokemonInPlay(stack=[take(state, player, name)], in_play_since=0,
                             energy=[take(state, player, e) for e in energy])


def bench(state, player, name):
    state.players[player].bench.append(PokemonInPlay(stack=[take(state, player, name)], in_play_since=0))


def duel(attacker="Makuhita", defender="Dunsparce", energy=("Basic {F} Energy",)):
    """Turn 3: mega_lucario (player 1) to move with `attacker`, lopunny (player 0) defending."""
    state = opening()
    state.turn, state.current, state.first_player = 3, 1, 1
    state.players[1].flags.energy_attached = False
    set_active(state, 1, attacker, energy)
    set_active(state, 0, defender)
    bench(state, 0, "Buneary")
    return state


def attack(state):
    [action] = [a for a in legal_actions(state) if a.kind == ActionKind.ATTACK]
    return step(state, action)


@pytest.mark.parametrize("seed", range(100))
def test_random_games_finish_and_keep_every_card(seed):
    pick = random.Random(seed)
    state = new_game(DECKS, seed)
    while state.phase != Phase.OVER:
        state = step(state, pick.choice(legal_actions(state)))
        assert sorted(state.all_uids()) == list(range(120))
        for p in state.players:
            assert len(p.bench) <= catalog().constants["bench_max"]
    assert state.end_reason in {"prizes", "no_pokemon", "deck_out", "turn_limit"}


def test_no_attack_on_the_very_first_turn():
    state = duel()
    state.turn = 1
    assert not [a for a in legal_actions(state) if a.kind == ActionKind.ATTACK]


def test_weakness_doubles_the_damage():
    after = attack(duel("Makuhita", "Dunsparce"))                     # Corkscrew Punch 10, weak to {F}
    assert after.players[0].active.damage == 20


def test_resistance_subtracts_30():
    state = duel("Lunatone", "Fan Rotom", energy=("Basic {F} Energy", "Basic {F} Energy"))   # Power Gem 50
    assert attack(state).players[0].active.damage == 20


def test_attacking_ends_the_turn_and_the_other_player_draws():
    state = duel()
    after = attack(state)
    assert after.current == 0 and after.turn == 4
    assert len(after.players[0].hand) == len(state.players[0].hand) + 1


def test_knocking_out_a_mega_ex_gives_three_prizes_then_the_owner_picks():
    state = duel("Makuhita", "Mega Lopunny ex")
    state.players[0].active.damage = 320                                # 330 HP, weak to {F}
    after = attack(state)
    assert len(after.players[1].prizes) == 3
    assert after.pending.kind == "new_active" and after.pending.player == 0
    done = step(after, legal_actions(after)[0])
    assert done.players[0].active is not None and done.current == 0 and done.turn == 4


def test_knocking_out_the_last_pokemon_wins():
    state = duel()
    state.players[0].bench.clear()
    state.players[0].active.damage = 50
    after = attack(state)
    assert after.phase == Phase.OVER and after.winner == 1 and after.end_reason == "no_pokemon"


def test_prize_values():
    by_name = {c.name: c for c in catalog().cards.values()}
    assert [prize_value(by_name[n]) for n in ("Makuhita", "Meowth ex", "Mega Lucario ex")] == [1, 2, 3]


def test_only_one_energy_attachment_per_turn():
    state = duel()
    state.players[1].hand += [take(state, 1, "Basic {F} Energy"), take(state, 1, "Basic {F} Energy")]
    first = next(a for a in legal_actions(state) if a.kind == ActionKind.ATTACH_ENERGY)
    after = step(state, first)
    assert not [a for a in legal_actions(after) if a.kind == ActionKind.ATTACH_ENERGY]


def test_a_pokemon_cannot_evolve_the_turn_it_is_benched():
    state = duel()
    state.players[1].hand += [take(state, 1, "Riolu"), take(state, 1, "Mega Lucario ex")]
    riolu = next(u for u in state.players[1].hand if card_of(state, u).name == "Riolu")
    after = step(state, Action(ActionKind.BENCH, 1, card=riolu))
    assert not [a for a in legal_actions(after) if a.kind == ActionKind.EVOLVE and a.target == riolu]
    after.turn += 2
    assert [a for a in legal_actions(after) if a.kind == ActionKind.EVOLVE and a.target == riolu]


def test_retreating_discards_the_cost_and_works_once():
    state = duel(energy=("Basic {F} Energy", "Basic {F} Energy"))     # Makuhita: retreat cost 2
    bench(state, 1, "Solrock")
    state = step(state, next(a for a in legal_actions(state) if a.kind == ActionKind.RETREAT))
    while state.pending:
        state = step(state, legal_actions(state)[0])
    me = state.players[1]
    assert card_of(state, me.active.card).name == "Solrock" and me.active.moved_to_active_on_turn == 3
    assert me.bench[-1].energy == []
    assert not [a for a in legal_actions(state) if a.kind == ActionKind.RETREAT]
