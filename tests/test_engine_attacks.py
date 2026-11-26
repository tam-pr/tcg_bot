"""One scenario for each attack with card text in the two decks, plus the two Special Energy."""
import pytest

from tcg_bot.engine import effects_runtime, turn
from tcg_bot.engine.game import card_of, legal_actions, new_game, step
from tcg_bot.engine.types import Action, ActionKind, Phase, PokemonInPlay

LOP, LUC = 0, 1                                   # player numbers: lopunny, mega_lucario
W, F, MIST, RICH = "Basic {W} Energy", "Basic {F} Energy", "Mist Energy", "Enriching Energy"


def take(state, player, card):
    """Remove a card (by name, or by card_id) from the player's hidden zones and return its uid."""
    p = state.players[player]
    for zone in (p.deck, p.hand, p.prizes, p.discard):
        for uid in zone:
            if card in (card_of(state, uid).name, card_of(state, uid).card_id):
                zone.remove(uid)
                if zone is p.prizes:
                    p.prizes.append(p.deck.pop())
                return uid
    raise AssertionError(f"no {card} left")


def place(state, player, card, energy=()):
    mon = PokemonInPlay(stack=[take(state, player, card)], in_play_since=0)
    mon.energy = [take(state, player, e) for e in energy]
    return mon


def arrange(me, attacker, energy, defender, bench=(), foe_bench=("Psyduck",)):
    """Turn 3, `me` to move. A default Benched Pokémon keeps the defender from losing at once."""
    state = new_game(("lopunny", "mega_lucario"), 0)
    while state.phase == Phase.SETUP:
        state = step(state, legal_actions(state)[-1])
    state.turn, state.current, state.first_player = 3, me, me
    for p in state.players:
        p.deck += [u for m in p.pokemon() for u in m.stack]
        p.active, p.bench = None, []
    foe = 1 - me
    if foe == LUC and foe_bench == ("Psyduck",):
        foe_bench = ("Lunatone",)
    state.players[me].active = place(state, me, attacker, energy)
    state.players[me].bench = [place(state, me, c) for c in bench]
    state.players[foe].active = place(state, foe, defender)
    state.players[foe].bench = [place(state, foe, c) for c in foe_bench]
    return state


def attack(state, name):
    mon = state.players[state.current].active
    ids = {m.move_id: m.name for m in card_of(state, mon.card).moves}
    return step(state, next(a for a in legal_actions(state) if a.kind == ActionKind.ATTACK and ids[a.move_id] == name))


def choose(state, option):
    return step(state, Action(ActionKind.CHOOSE, state.pending.player, option=option))


def offered(state):
    return [a for a in legal_actions(state) if a.kind == ActionKind.ATTACK]


def end_turn(state):
    return step(state, Action(ActionKind.END_TURN, state.current))


def coins(monkeypatch, heads):
    monkeypatch.setattr(effects_runtime, "coin_heads", lambda state: heads)
    monkeypatch.setattr(turn, "coin_heads", lambda state: heads)


@pytest.mark.parametrize("card, name", [(848, "Run Around"), (305, "Trading Places")])
def test_switching_attacks_swap_with_the_bench(card, name):
    state = arrange(LOP, card, [MIST], "Solrock", bench=["Psyduck"])
    state = attack(state, name)
    assert state.pending.kind == "switch_to"
    state = choose(state, state.pending.options[0])
    assert card_of(state, state.players[LOP].active.card).name == "Psyduck" and state.current == LUC


def test_a_switching_attack_with_an_empty_bench_just_ends_the_turn():
    state = attack(arrange(LOP, 848, [MIST], "Solrock"), "Run Around")
    assert state.pending is None and state.current == LUC


@pytest.mark.parametrize("heads", [True, False])
def test_dig_protects_on_heads_only(monkeypatch, heads):
    coins(monkeypatch, heads)
    state = attack(arrange(LOP, 65, [MIST, MIST], "Lunatone"), "Dig")
    state.players[LUC].active.energy = [take(state, LUC, F), take(state, LUC, F)]
    state = attack(state, "Power Gem")                   # 50, doubled by weakness: enough for 60 HP Dunsparce
    survivor = state.players[LOP].active
    if heads:
        assert survivor.damage == 0
    else:
        assert survivor is None


def test_spiky_hopper_ignores_the_protection():
    state = arrange(LOP, 849, [MIST, MIST], "Solrock")
    foe = state.players[LUC].active
    state.lasting.append(effects_runtime.LastingEffect(
        block=effects_runtime.fx.Prevent("damage_and_effects", "this_pokemon", "opponent_attacks"),
        controller=LUC, source=foe.uid, target=foe.uid, first_turn=3, last_turn=3))
    state = attack(state, "Spiky Hopper")                           # 160 knocks out Solrock anyway
    assert card_of(state, state.players[LUC].discard[0]).name == "Solrock"


def test_icicle_loop_returns_one_energy():
    state = attack(arrange(LOP, 209, [W, W, MIST], "Mega Lucario ex"), "Icicle Loop")
    chosen = state.pending.options[0]
    state = choose(state, chosen)
    assert chosen in state.players[LOP].hand and len(state.players[LOP].active.energy) == 2
    assert state.players[LUC].active.damage == 120


@pytest.mark.parametrize("stadium, damage", [(False, 0), (True, 70)])
def test_assault_landing_needs_a_stadium(stadium, damage):
    state = arrange(LOP, 174, [MIST], "Mega Lucario ex")
    if stadium:
        state.stadium = take(state, LOP, "Battle Cage")
    assert attack(state, "Assault Landing").players[LUC].active.damage == damage


def test_wild_press_also_hurts_its_user():
    state = attack(arrange(LUC, 674, [F, F, F], "Mega Froslass ex"), "Wild Press")
    assert state.players[LUC].active.damage == 70 and state.players[LOP].active.damage == 210


def test_wild_press_can_knock_out_its_user_and_give_a_prize():
    state = arrange(LUC, 674, [F, F, F], "Mega Froslass ex", bench=["Solrock"])
    state.players[LUC].active.damage = 80                           # 150 HP
    state = attack(state, "Wild Press")
    assert len(state.players[LOP].prizes) == 5 and state.pending.player == LUC


def test_both_players_winning_at_once_is_sudden_death():
    state = arrange(LUC, 674, [F, F, F], "Dunsparce", bench=["Solrock"], foe_bench=())
    state.players[LOP].prizes = state.players[LOP].prizes[:1]
    state.players[LUC].active.damage = 80
    state = attack(state, "Wild Press")
    assert state.phase == Phase.OVER and state.winner is None and state.end_reason == "sudden_death"


def test_resentful_refrain_counts_the_opponents_hand():
    state = arrange(LOP, 861, [W], "Mega Lucario ex")
    state.players[LUC].hand = state.players[LUC].hand[:3]
    assert attack(state, "Resentful Refrain").players[LUC].active.damage == 150


@pytest.mark.parametrize("heads", [True, False])
def test_absolute_snow_puts_to_sleep_until_a_heads(monkeypatch, heads):
    coins(monkeypatch, heads)
    state = attack(arrange(LOP, 861, [W, MIST, MIST], "Mega Lucario ex"), "Absolute Snow")
    assert state.players[LUC].active.rotation == (None if heads else "Asleep")
    if not heads:
        state.players[LUC].active.energy = [take(state, LUC, F), take(state, LUC, F)]
        assert not offered(state)
        assert not [a for a in legal_actions(state) if a.kind == ActionKind.RETREAT]


def test_mist_energy_stops_the_sleep_but_not_the_damage(monkeypatch):
    coins(monkeypatch, False)
    state = arrange(LOP, 861, [W, MIST, MIST], "Mega Lucario ex")
    state.players[LUC].active.energy = [take(state, LOP, MIST)]     # put on the defender just for this check
    state = attack(state, "Absolute Snow")
    assert state.players[LUC].active.damage == 150 and state.players[LUC].active.rotation is None


@pytest.mark.parametrize("moved, damage", [(None, 60), (3, 230)])
def test_gale_thrust_hits_harder_after_moving_up(moved, damage):
    state = arrange(LOP, 849, [MIST], "Mega Lucario ex")
    state.players[LOP].active.moved_to_active_on_turn = moved
    assert attack(state, "Gale Thrust").players[LUC].active.damage == damage


def test_aura_jab_attaches_up_to_three_energy_from_the_discard():
    state = arrange(LUC, 678, [F], "Mega Lopunny ex", bench=["Riolu", "Solrock"])
    me = state.players[LUC]
    me.discard += [take(state, LUC, F) for _ in range(4)]
    state = attack(state, "Aura Jab")
    assert Action(ActionKind.CHOOSE_DONE, LUC) in legal_actions(state)
    for target in (0, 0, 1):
        state = choose(state, state.pending.options[0])
        state = choose(state, state.pending.options[target])
    assert state.pending is None and state.current == LOP
    assert [len(m.energy) for m in state.players[LUC].bench] == [2, 1]


def test_aura_jab_may_stop_early():
    state = arrange(LUC, 678, [F], "Mega Lopunny ex", bench=["Riolu"])
    state.players[LUC].discard.append(take(state, LUC, F))
    state = attack(state, "Aura Jab")
    state = step(state, Action(ActionKind.CHOOSE_DONE, LUC))
    assert state.players[LUC].bench[0].energy == [] and state.current == LOP


@pytest.mark.parametrize("card, name", [(678, "Mega Brave"), (677, "Accelerating Stab")])
def test_locked_attacks_come_back_a_turn_later(card, name):
    state = arrange(LUC, card, [F, F], "Mega Froslass ex")          # survives both attacks
    move_id = next(m.move_id for m in card_of(state, state.players[LUC].active.card).moves if m.name == name)
    state = end_turn(attack(state, name))                            # the opponent's turn goes by
    assert move_id not in [a.move_id for a in offered(state)]        # my next turn
    state = end_turn(end_turn(state))
    assert move_id in [a.move_id for a in offered(state)]


def test_tuck_tail_goes_back_to_the_hand():
    state = arrange(LUC, 1071, [F, F, F], "Mega Lopunny ex", bench=["Solrock"])
    mine = state.players[LUC].active.stack + state.players[LUC].active.energy
    state = attack(state, "Tuck Tail")
    assert set(mine) <= set(state.players[LUC].hand)
    assert state.pending.kind == "new_active" and state.pending.player == LUC
    assert choose(state, state.pending.options[0]).current == LOP


def test_tuck_tail_needs_another_pokemon_in_play():
    assert not offered(arrange(LUC, 1071, [F, F, F], "Mega Lopunny ex"))


@pytest.mark.parametrize("bench, damage", [([], 0), (["Lunatone"], 70)])
def test_cosmic_beam_needs_lunatone_and_ignores_weakness(bench, damage):
    state = arrange(LUC, 676, [F], "Dudunsparce", bench=bench)       # weak to {F}
    assert attack(state, "Cosmic Beam").players[LOP].active.damage == damage


def test_enriching_energy_draws_four_when_attached():
    state = arrange(LOP, 848, [], "Solrock")
    state.players[LOP].hand.append(take(state, LOP, RICH))
    before = len(state.players[LOP].hand)
    rich = state.players[LOP].hand[-1]
    state = step(state, Action(ActionKind.ATTACH_ENERGY, LOP, card=rich, target=state.players[LOP].active.uid))
    assert len(state.players[LOP].hand) == before - 1 + 4 and state.current == LOP
