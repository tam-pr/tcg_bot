"""One scenario for each Trainer card and Ability in the two decks."""
import pytest
from test_engine_attacks import LOP, LUC, MIST, F, arrange, attack, choose, coins, end_turn, take

from tcg_bot.engine.board import max_hp, retreat_cost
from tcg_bot.engine.catalog import catalog
from tcg_bot.engine.game import card_of, legal_actions, step
from tcg_bot.engine.types import Action, ActionKind

PLAY = (ActionKind.PLAY_ITEM, ActionKind.PLAY_SUPPORTER, ActionKind.PLAY_STADIUM, ActionKind.ATTACH_TOOL)


def board(me, active="", bench=(), foe_active="", foe_bench=(), energy=()):
    """A turn-3 board with empty hands, so each test says exactly what is held."""
    state = arrange(me, active or ("Buneary" if me == LOP else "Makuhita"), list(energy),
                    foe_active or ("Solrock" if me == LOP else "Dudunsparce"), bench=list(bench),
                    foe_bench=tuple(foe_bench) or ("Psyduck",))
    for p in state.players:
        p.deck += p.hand
        p.hand = []
    return state


def give(state, player, *names):
    uids = [take(state, player, n) for n in names]
    state.players[player].hand += uids
    return uids


def plays(state, name):
    return [a for a in legal_actions(state) if a.kind in PLAY and card_of(state, a.card).name == name]


def play(state, name):
    return step(state, plays(state, name)[0])


def done(state):
    return step(state, Action(ActionKind.CHOOSE_DONE, state.pending.player))


def names(state, uids):
    return sorted(card_of(state, u).name for u in uids)


def option(state, name):
    pool = state.pending.options
    return next(u for u in pool if card_of(state, u).name == name)


def abilities(state):
    return [a for a in legal_actions(state) if a.kind == ActionKind.USE_ABILITY]


# ---------------------------------------------------------------- Trainers
def test_switch_swaps_and_the_turn_goes_on():
    state = board(LUC, bench=["Solrock"])
    give(state, LUC, "Switch")
    state = choose_first(play(state, "Switch"))
    me = state.players[LUC]
    assert card_of(state, me.active.card).name == "Solrock" and state.current == LUC
    assert names(state, me.discard)[-1:] == ["Switch"] and me.playing == []


def choose_first(state):
    return choose(state, state.pending.options[0])


def test_a_card_that_would_do_nothing_is_not_offered():
    state = board(LUC)                                   # no Benched Pokémon
    give(state, LUC, "Switch")
    assert not plays(state, "Switch")


def test_bosses_orders_drags_up_a_benched_pokemon_once_per_turn():
    state = board(LUC, foe_bench=["Psyduck"])
    give(state, LUC, "Boss's Orders", "Boss's Orders")
    state = choose_first(play(state, "Boss's Orders"))
    assert card_of(state, state.players[LOP].active.card).name == "Psyduck"
    assert not plays(state, "Boss's Orders")


def test_no_supporter_on_the_very_first_turn():
    state = board(LUC)
    state.turn = 1
    give(state, LUC, "Lillie's Determination")
    assert not plays(state, "Lillie's Determination")


def test_buddy_buddy_poffin_benches_small_basics_and_may_stop():
    state = board(LOP)
    give(state, LOP, "Buddy-Buddy Poffin")
    state = play(state, "Buddy-Buddy Poffin")
    assert all(catalog().cards[state.card_ids[u]].hp <= 70 for u in state.pending.options)
    state = done(choose_first(state))
    assert len(state.players[LOP].bench) == 1 and state.pending is None


def test_fighting_gong_finds_only_fighting_basics():
    state = board(LUC)
    give(state, LUC, "Fighting Gong")
    state = play(state, "Fighting Gong")
    found = {card_of(state, u) for u in state.pending.options}
    assert all(c.type == "{F}" and c.kind in ("Basic Energy", "Basic Pokémon") for c in found)


def test_hilda_takes_both_cards():
    state = board(LOP)
    give(state, LOP, "Hilda")
    state = play(state, "Hilda")
    assert Action(ActionKind.CHOOSE_DONE, LOP) not in legal_actions(state)
    state = choose_first(choose_first(state))
    kinds = sorted(card_of(state, u).family for u in state.players[LOP].hand)
    assert kinds == ["energy", "pokemon"] and state.pending is None


def test_judge_gives_both_players_four_cards():
    state = board(LUC)
    give(state, LUC, "Judge", "Switch")
    state = play(state, "Judge")
    assert [len(p.hand) for p in state.players] == [4, 4]


@pytest.mark.parametrize("prizes, drawn", [(6, 8), (5, 6)])
def test_lillies_determination(prizes, drawn):
    state = board(LUC)
    state.players[LUC].prizes = state.players[LUC].prizes[:prizes]
    give(state, LUC, "Lillie's Determination", "Switch")
    assert len(play(state, "Lillie's Determination").players[LUC].hand) == drawn


@pytest.mark.parametrize("heads, drawn", [(True, 6), (False, 3)])
def test_lucian_flips_for_each_player(monkeypatch, heads, drawn):
    coins(monkeypatch, heads)
    state = board(LOP)
    give(state, LOP, "Lucian", "Hilda")
    give(state, LUC, "Switch")
    switch = state.players[LUC].hand[0]
    state = play(state, "Lucian")
    assert [len(p.hand) for p in state.players] == [drawn, drawn]
    assert switch in state.players[LUC].deck + state.players[LUC].hand


def test_night_stretcher_recovers_from_the_discard():
    state = board(LOP)
    give(state, LOP, "Night Stretcher")
    state.players[LOP].discard.append(take(state, LOP, "Buneary"))
    state = choose_first(play(state, "Night Stretcher"))
    assert names(state, state.players[LOP].hand) == ["Buneary"]


def test_poke_pad_cannot_find_a_rule_box_pokemon():
    state = board(LOP)
    give(state, LOP, "Poké Pad")
    state = play(state, "Poké Pad")
    assert not any(card_of(state, u).has_rule_box for u in state.pending.options)


def test_pokegear_looks_at_seven_and_may_take_nothing():
    state = board(LOP)
    give(state, LOP, "Pokégear 3.0")
    hilda = take(state, LOP, "Hilda")
    state.players[LOP].deck.insert(0, hilda)
    state = play(state, "Pokégear 3.0")
    top = state.players[LOP].deck[:7]
    assert hilda in state.pending.options and set(state.pending.options) <= set(top)
    assert done(state).players[LOP].hand == []


def test_premium_power_pro_adds_before_weakness():
    state = board(LUC, energy=[F])
    give(state, LUC, "Premium Power Pro")
    state = attack(play(state, "Premium Power Pro"), "Corkscrew Punch")       # (10 + 30) x 2
    assert state.players[LOP].active.damage == 80


def test_special_red_card_needs_three_prizes_or_fewer():
    state = board(LOP)
    give(state, LOP, "Special Red Card")
    assert not plays(state, "Special Red Card")
    state.players[LUC].prizes = state.players[LUC].prizes[:3]
    give(state, LUC, "Switch")
    assert len(play(state, "Special Red Card").players[LUC].hand) == 3


def test_team_rockets_petrel_finds_a_trainer():
    state = board(LUC)
    give(state, LUC, "Team Rocket's Petrel")
    state = play(state, "Team Rocket's Petrel")
    assert all(card_of(state, u).family == "trainer" for u in state.pending.options)


def test_tool_scrapper_discards_tools_from_either_side():
    state = board(LUC)
    state.players[LUC].active.tool = take(state, LUC, "Air Balloon")
    state.players[LOP].active.tool = take(state, LOP, "Air Balloon")
    give(state, LUC, "Tool Scrapper")
    state = choose_first(choose_first(play(state, "Tool Scrapper")))
    assert state.players[LUC].active.tool is None and state.players[LOP].active.tool is None
    assert "Air Balloon" in names(state, state.players[LOP].discard)


def test_ultra_ball_costs_two_other_cards():
    state = board(LUC)
    give(state, LUC, "Ultra Ball", "Switch")
    assert not plays(state, "Ultra Ball")
    give(state, LUC, "Judge")
    state = choose_first(choose_first(play(state, "Ultra Ball")))
    assert state.players[LUC].hand == [] and card_of(state, state.pending.options[0]).family == "pokemon"
    assert len(choose_first(state).players[LUC].hand) == 1


def test_unfair_stamp_needs_a_knock_out_last_turn():
    state = board(LUC)
    give(state, LUC, "Unfair Stamp")
    assert not plays(state, "Unfair Stamp")
    state.players[LUC].knocked_out_on_turn = 2
    state = play(state, "Unfair Stamp")
    assert [len(p.hand) for p in state.players] == [2, 5]


def test_wallys_compassion_heals_a_mega_and_returns_its_energy():
    state = board(LUC, active="Mega Lucario ex", energy=[F, F])
    give(state, LUC, "Wally's Compassion")
    assert not plays(state, "Wally's Compassion")                  # nothing to heal
    state.players[LUC].active.damage = 100
    state = choose_first(play(state, "Wally's Compassion"))
    me = state.players[LUC]
    assert me.active.damage == 0 and me.active.energy == [] and len(me.hand) == 2


def test_air_balloon_lowers_the_retreat_cost():
    state = board(LUC)
    give(state, LUC, "Air Balloon")
    state = play(state, "Air Balloon")
    assert retreat_cost(state, state.players[LUC].active) == 0      # Makuhita: 2 - 2
    assert not [a for a in plays(state, "Air Balloon")]


def test_gravity_mountain_takes_30_hp_from_stage_2_only():
    state = board(LUC)
    give(state, LUC, "Gravity Mountain")
    state = play(state, "Gravity Mountain")
    mon = state.players[LUC].active
    assert max_hp(state, mon) == 80
    stage_2 = next(c for c in catalog().cards.values() if c.kind == "Stage 2 Pokémon")
    ids = list(state.card_ids)
    ids[mon.card] = stage_2.card_id                                 # neither deck has a Stage 2, so swap one in
    state.card_ids = tuple(ids)
    assert max_hp(state, mon) == stage_2.hp - 30


def test_a_new_stadium_replaces_the_old_one():
    state = board(LUC)
    state.stadium = take(state, LOP, "Battle Cage")
    old = state.stadium
    give(state, LUC, "Gravity Mountain", "Gravity Mountain")
    state = play(state, "Gravity Mountain")
    assert old in state.players[LOP].discard and not plays(state, "Gravity Mountain")


# ---------------------------------------------------------------- Abilities
def bench_from_hand(state, player, name):
    [uid] = give(state, player, name)
    return step(state, Action(ActionKind.BENCH, player, card=uid))


@pytest.mark.parametrize("yes", [1, 0])
def test_snow_sink_may_discard_the_stadium(yes):
    state = board(LOP)
    state.stadium = take(state, LUC, "Gravity Mountain")
    state = bench_from_hand(state, LOP, "Chien-Pao")
    assert state.pending.kind == "use_ability"
    assert (choose(state, yes).stadium is None) == bool(yes)


def test_run_away_draw_shuffles_its_user_back_and_the_turn_goes_on():
    state = board(LOP, active="Dudunsparce", bench=["Psyduck"])
    mine = state.players[LOP].active.stack.copy()
    state = step(state, abilities(state)[0])
    assert state.pending.kind == "promote"
    state = choose_first(state)
    me = state.players[LOP]
    assert len(me.hand) + sum(u in me.deck for u in mine) >= 3 and state.current == LOP
    assert card_of(state, me.active.card).name == "Psyduck" and not any(m.uid == mine[0] for m in me.pokemon())


def test_fan_call_only_on_your_first_turn():
    state = board(LOP, bench=["Fan Rotom"])
    assert not abilities(state)
    state.turn, state.first_player = 1, LOP
    state = step(state, abilities(state)[0])
    assert all(card_of(state, u).type == "{C}" and card_of(state, u).hp <= 100 for u in state.pending.options)
    assert not abilities(done(state))


def test_heave_ho_catcher_drags_up_a_benched_pokemon():
    state = board(LUC, foe_bench=["Psyduck"])
    [uid] = give(state, LUC, "Hariyama")
    state = step(state, Action(ActionKind.EVOLVE, LUC, card=uid, target=state.players[LUC].active.uid))
    state = choose_first(choose(state, 1))
    assert card_of(state, state.players[LOP].active.card).name == "Psyduck"


def test_lunar_cycle_needs_solrock_and_works_once():
    state = board(LUC, active="Lunatone", bench=["Lunatone"])
    give(state, LUC, F, F)
    assert not abilities(state)
    state.players[LUC].bench.append(type(state.players[LUC].active)(stack=[take(state, LUC, "Solrock")], in_play_since=0))
    state = choose_first(step(state, abilities(state)[0]))
    assert len(state.players[LUC].hand) == 4 and not abilities(state)      # 2 - 1 discarded + 3 drawn


def test_last_ditch_catch_finds_a_supporter():
    state = board(LUC)
    state = choose(bench_from_hand(state, LUC, "Meowth ex"), 1)
    assert all(card_of(state, u).kind == "Supporter" for u in state.pending.options)
    assert len(choose_first(state).players[LUC].hand) == 1


def test_mist_energy_does_not_stop_bosses_orders():
    state = board(LUC, foe_bench=["Psyduck"])
    state.players[LOP].bench[0].energy = [take(state, LOP, MIST)]
    give(state, LUC, "Boss's Orders")
    state = choose_first(play(state, "Boss's Orders"))
    assert card_of(state, state.players[LOP].active.card).name == "Psyduck"
    assert end_turn(state).current == LOP
