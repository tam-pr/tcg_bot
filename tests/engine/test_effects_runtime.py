"""At least one scenario per in-scope effect (task T5). Each test names the move_ids it covers."""
import re
from dataclasses import replace
from pathlib import Path

import pytest

from tcg_bot.effects_data import EFFECTS
from tcg_bot.engine import Action, RuleSettings, legal_actions, step
from tcg_bot.engine import board as B
from tcg_bot.engine import effects_runtime as R
from tcg_bot.engine.core import card_locations
from tcg_bot.engine.types import CardRef

from .helpers import choose, do, has, names, scenario, slot_of

P1 = dict(current=1, turn=4, turns_taken=(2, 2))
LU = ("lopunny", "mega_lucario")


def covers(*ids):
    return ids


def resolve(s, pick=lambda opts: opts[0]):
    while s.pending is not None:
        opts = list(s.pending.options) + ([s.pending.stop] if s.pending.stop is not None else [])
        s = choose(s, pick(opts))
    return s


# ================================================================ Energy
def test_enriching_energy_draws_4_when_attached_from_hand():
    covers("13_0")
    s = scenario(hands=(("Enriching Energy",), ()), actives=("Snorunt", "Riolu"))
    s = do(s, kind="attach_energy")
    assert len(s.players[0].hand) == 4


def test_mist_energy_prevents_attack_effects_but_not_damage():
    covers("11_0", "861_1")
    asleep = {True: set(), False: set()}
    for seed in range(10):
        for mist in (True, False):
            s = scenario(seed=seed, decks=("lopunny", "lopunny"), **P1, actives=("Mega Lopunny ex", "Mega Froslass ex"),
                         energy={(1, 0): ["Basic {W} Energy"] * 3, **({(0, 0): ["Mist Energy"]} if mist else {})})
            s.rng.seed(seed)
            s = do(s, kind="attack", choice="861_1")
            assert slot_of(s, 0, 0).damage == 150                     # damage is not an effect
            asleep[mist].add("Asleep" in slot_of(s, 0, 0).conditions)  # Checkup may wake it up
    assert asleep[True] == {False} and True in asleep[False]


# ================================================================ plain damage attacks
@pytest.mark.parametrize("decks,attacker,energy,mid,defender,expected", [
    (LU, "Buneary", ["Basic {W} Energy"] * 2, "848_1", "Mega Lucario ex", 20),
    (LU, "Dudunsparce", ["Basic {W} Energy"] * 3, "66_1", "Mega Lucario ex", 90),
    (LU, "Dunsparce", ["Basic {W} Energy"], "65_0", "Mega Lucario ex", 10),
    (LU, "Dunsparce", ["Basic {W} Energy"] * 2, "305_1", "Mega Lucario ex", 20),
    (LU, "Psyduck", ["Basic {W} Energy"] * 2, "858_1", "Mega Lucario ex", 20),
    (LU, "Snorunt", ["Basic {W} Energy"], "860_0", "Mega Lucario ex", 10),
    (("mega_lucario", "lopunny"), "Lunatone", ["Basic {F} Energy"] * 2, "675_1", "Snorunt", 50),
    (("mega_lucario", "lopunny"), "Makuhita", ["Basic {F} Energy"], "673_0", "Snorunt", 10),
    (("mega_lucario", "lopunny"), "Makuhita", ["Basic {F} Energy"] * 2, "673_1", "Snorunt", 30),
])
def test_plain_damage(decks, attacker, energy, mid, defender, expected):
    covers("848_1", "66_1", "65_0", "305_1", "858_1", "860_0", "675_1", "673_0", "673_1")
    s = scenario(decks=decks, actives=(attacker, defender), energy={(0, 0): energy})
    if attacker == "Dunsparce" and mid == "305_1":
        s = scenario(decks=decks, actives=(None, defender), benches=(("Dunsparce", "Dunsparce"), ()))
        s.players[0].active = s.players[0].bench.pop([s.catalog.card(b.top).card_id for b in s.players[0].bench].index(305))
        s.players[0].active.energy = [u for u in s.players[0].deck if s.catalog.card(u).name == "Basic {W} Energy"][:2]
        s.players[0].deck = [u for u in s.players[0].deck if u not in s.players[0].active.energy]
    s = do(s, kind="attack", choice=mid)
    assert slot_of(s, 1, 0).damage == expected


# ================================================================ attacks with text
def test_run_around_and_trading_places_switch_after_damage():
    covers("848_0", "305_0")
    s = scenario(actives=("Buneary", "Riolu"), benches=(("Psyduck", "Snorunt"), ()),
                 energy={(0, 0): ["Basic {W} Energy"]})
    s = do(s, kind="attack", choice="848_0")
    assert s.pending.kind == "switch_target"
    s = choose(s, s.pending.options[1])
    assert s.catalog.card(slot_of(s, 0, 0).top).name == "Snorunt"
    t = scenario(actives=(None, "Riolu"), benches=(("Dunsparce", "Dunsparce", "Psyduck"), ()))
    pl = t.players[0]
    pl.active = pl.bench.pop([t.catalog.card(b.top).card_id for b in pl.bench].index(305))
    e = next(u for u in pl.deck if t.catalog.card(u).name == "Basic {W} Energy")
    pl.deck.remove(e); pl.active.energy = [e]
    t = resolve(do(t, kind="attack", choice="305_0"))
    assert t.catalog.card(slot_of(t, 0, 0).top).card_id != 305


def test_icicle_loop_returns_an_energy_to_hand():
    covers("209_1")
    s = scenario(actives=("Chien-Pao", "Mega Lucario ex"), energy={(0, 0): ["Basic {W} Energy"] * 3})
    s = do(s, kind="attack", choice="209_1")
    assert slot_of(s, 1, 0).damage == 120 and len(slot_of(s, 0, 0).energy) == 2
    assert "Basic {W} Energy" in names(s, s.players[0].hand)


def test_dig_heads_prevents_damage_and_spiky_hopper_ignores_it():
    covers("65_1", "849_1")
    found = {}
    for seed in range(40):
        s = scenario(seed=seed, actives=("Dunsparce", "Riolu"), energy={(0, 0): ["Basic {W} Energy"] * 2})
        s.rng.seed(seed)
        s = do(s, kind="attack", choice="65_1")
        heads = any(le["kind"] == "prevent_attack" for le in s.lasting)
        found[heads] = s
    assert set(found) == {True, False}
    s = found[True]
    assert slot_of(s, 1, 0).damage == 30
    f = next(u for u in s.players[1].deck if s.catalog.card(u).name == "Basic {F} Energy")
    s.players[1].deck.remove(f); slot_of(s, 1, 0).energy.append(f)
    s = do(s, kind="attack", choice="677_0")
    assert slot_of(s, 0, 0).damage == 0                               # prevented during the opponent's next turn
    t = scenario(actives=("Mega Lopunny ex", "Mega Lucario ex"), energy={(0, 0): ["Basic {W} Energy"] * 2})
    t.lasting.append({"kind": "prevent_attack", "what": "damage_and_effects", "pid": slot_of(t, 1, 0).pid,
                      "until_turn": t.turn})
    assert slot_of(do(t, kind="attack", choice="849_0"), 1, 0).damage == 0
    assert slot_of(do(t, kind="attack", choice="849_1"), 1, 0).damage == 160


def test_assault_landing_does_nothing_without_a_stadium():
    covers("174_1")
    s = scenario(actives=("Fan Rotom", "Riolu"), energy={(0, 0): ["Basic {W} Energy"]})
    assert slot_of(do(s, kind="attack", choice="174_1"), 1, 0).damage == 0
    s = scenario(actives=("Fan Rotom", "Riolu"), energy={(0, 0): ["Basic {W} Energy"]}, stadium=(0, "Battle Cage"))
    assert slot_of(do(s, kind="attack", choice="174_1"), 1, 0).damage == 70


def test_wild_press_recoil():
    covers("674_1")
    s = scenario(**P1, actives=("Psyduck", "Hariyama"), benches=(("Snorunt",), ("Riolu",)),
                 energy={(1, 0): ["Basic {F} Energy"] * 3})
    s = resolve(do(s, kind="attack", choice="674_1"))
    assert names(s, s.players[0].discard) == ["Psyduck"]
    assert slot_of(s, 1, 0).damage == 70


def test_resentful_refrain_counts_the_opponents_hand():
    covers("861_0")
    s = scenario(actives=("Mega Froslass ex", "Mega Lucario ex"), hands=((), ("Judge", "Switch", "Poké Pad")),
                 energy={(0, 0): ["Basic {W} Energy"]})
    assert slot_of(do(s, kind="attack", choice="861_0"), 1, 0).damage == 150


def test_gale_thrust_bonus_when_moved_to_active_this_turn():
    covers("849_0")
    s = scenario(actives=("Mega Lopunny ex", "Mega Lucario ex"), energy={(0, 0): ["Basic {W} Energy"]})
    assert slot_of(do(s, kind="attack", choice="849_0"), 1, 0).damage == 60
    slot_of(s, 0, 0).moved_to_active_turn = s.turn
    assert slot_of(do(s, kind="attack", choice="849_0"), 1, 0).damage == 230


def test_aura_jab_attaches_from_discard_without_using_the_hand_attach():
    covers("678_0")
    s = scenario(**P1, actives=("Mega Lopunny ex", "Mega Lucario ex"), benches=((), ("Riolu", "Solrock")),
                 energy={(1, 0): ["Basic {F} Energy"]}, discards=((), ["Basic {F} Energy"] * 4))
    s = do(s, kind="attack", choice="678_0")
    assert s.pending.kind == "move:discard>attached_to_bench" and s.pending.stop == R.STOP
    assert not s.flags.get("energy_attached")
    s = resolve(s)
    assert sum(len(b.energy) for b in s.players[1].bench) == 3
    assert slot_of(s, 0, 0).damage == 260                             # 130 x2, Mega Lopunny ex is weak to {F}


def test_mega_brave_and_accelerating_stab_lock_the_next_turn():
    covers("678_1", "677_0")
    for attacker, mid, en in (("Mega Lucario ex", "678_1", 2), ("Riolu", "677_0", 1)):
        s = scenario(**P1, actives=("Mega Lopunny ex", attacker), energy={(1, 0): ["Basic {F} Energy"] * en},
                     benches=(("Psyduck",), ()), prizes=(6, 6))
        s = do(s, kind="attack", choice=mid)
        s = resolve(s)
        s = do(s, kind="end_turn")
        assert s.current == 1 and not has(s, kind="attack", choice=mid)
        s = do(do(s, kind="end_turn"), kind="end_turn")
        assert has(s, kind="attack", choice=mid)


def test_tuck_tail_returns_meowth_and_its_cards_to_hand():
    covers("1071_1")
    s = scenario(**P1, actives=("Snorunt", "Meowth ex"), benches=((), ("Riolu",)),
                 energy={(1, 0): ["Basic {F} Energy"] * 3})
    s = resolve(do(s, kind="attack", choice="1071_1"))
    assert names(s, s.players[1].hand).count("Basic {F} Energy") == 3 and "Meowth ex" in names(s, s.players[1].hand)
    assert s.catalog.card(slot_of(s, 1, 0).top).name == "Riolu"
    assert slot_of(s, 0, 0).damage == 60


def test_cosmic_beam_needs_lunatone_and_ignores_weakness():
    covers("676_0")
    s = scenario(**P1, actives=("Buneary", "Solrock"), energy={(1, 0): ["Basic {F} Energy"]})
    assert slot_of(do(s, kind="attack", choice="676_0"), 0, 0).damage == 0
    s = scenario(**P1, actives=("Mega Lopunny ex", "Solrock"), benches=((), ("Lunatone",)),
                 energy={(1, 0): ["Basic {F} Energy"]})
    assert slot_of(do(s, kind="attack", choice="676_0"), 0, 0).damage == 70   # not doubled by Weakness


# ================================================================ abilities
def test_snow_sink_discards_the_stadium():
    covers("209_0")
    s = scenario(hands=(("Chien-Pao",), ()), actives=("Snorunt", "Riolu"), stadium=(1, "Gravity Mountain"))
    s = do(s, kind="bench_basic")
    assert s.pending.kind == "use_ability"
    s = choose(s, "yes")
    assert s.stadium is None and names(s, s.players[1].discard) == ["Gravity Mountain"]
    t = scenario(hands=(("Chien-Pao",), ()), actives=("Snorunt", "Riolu"))
    assert do(t, kind="bench_basic").pending is None              # nothing to discard: not offered


def test_run_away_draw():
    covers("66_0")
    s = scenario(actives=("Snorunt", "Riolu"), benches=(("Dudunsparce",), ()),
                 energy={(0, 1): ["Basic {W} Energy"]})
    deck = len(s.players[0].deck)
    s = do(s, kind="use_ability")
    assert len(s.players[0].hand) == 3 and not s.players[0].bench
    assert len(s.players[0].deck) == deck - 3 + 2                    # Dudunsparce and its Energy shuffled in
    t = scenario(actives=("Dudunsparce", "Riolu"), benches=(("Buneary", "Psyduck"), ()))
    t = do(t, kind="use_ability")
    assert t.pending.kind == "promote"                               # the Active left play: promote now
    t = resolve(t)
    assert t.players[0].active is not None and t.current == 0


def test_fan_call_first_turn_only():
    covers("174_0")
    s = scenario(actives=("Fan Rotom", "Riolu"), turn=1, turns_taken=(1, 0))
    s = do(s, kind="use_ability")
    picked = 0
    while s.pending is not None:
        s = choose(s, s.pending.options[0]); picked += 1
    assert picked == 3 and len(s.players[0].hand) == 3
    assert all(s.catalog.card(u).type == "{C}" and s.catalog.card(u).hp <= 100 for u in s.players[0].hand)
    assert not has(s, kind="use_ability")
    assert not has(scenario(actives=("Fan Rotom", "Riolu")), kind="use_ability")


def test_heave_ho_catcher_on_evolve():
    covers("674_0")
    s = scenario(**P1, hands=((), ("Hariyama",)), actives=("Snorunt", "Makuhita"), benches=(("Psyduck",), ()))
    s = do(s, kind="evolve")
    assert s.pending.kind == "use_ability"
    s = choose(s, "yes")
    assert s.catalog.card(slot_of(s, 0, 0).top).name == "Psyduck"


def test_lunar_cycle_needs_solrock_and_once_per_turn():
    covers("675_0")
    s = scenario(**P1, hands=((), ("Basic {F} Energy",)), actives=("Snorunt", "Lunatone"),
                 benches=((), ("Solrock", "Lunatone")))
    s = do(s, kind="use_ability")
    assert len(s.players[1].hand) == 3 and names(s, s.players[1].discard) == ["Basic {F} Energy"]
    s.players[1].hand.append(next(u for u in s.players[1].deck if s.catalog.card(u).name == "Basic {F} Energy"))
    s.players[1].deck.remove(s.players[1].hand[-1])
    assert not has(s, kind="use_ability")                            # 1 Lunar Cycle per turn
    t = scenario(**P1, hands=((), ("Basic {F} Energy",)), actives=("Snorunt", "Lunatone"))
    assert not has(t, kind="use_ability")                            # needs Solrock in play


def test_last_ditch_catch_on_bench():
    covers("1071_0")
    s = scenario(**P1, hands=((), ("Meowth ex",)), actives=("Snorunt", "Riolu"))
    s = choose(do(s, kind="bench_basic"), "yes")
    assert s.pending.kind == "move:deck>hand"
    s = choose(s, s.pending.options[0])
    assert s.catalog.card(s.players[1].hand[-1]).kind == "Supporter"


def test_damp_is_a_guarded_no_op():
    covers("858_0")
    s = scenario(actives=("Psyduck", "Riolu"))
    assert not R.can_use(s, "858_0", 0, pid=slot_of(s, 0, 0).pid)
    assert not any(a.kind == "use_ability" for a in legal_actions(s))
    with pytest.raises(B.EngineError):
        R.effect_frame(s, "858_0", 0)


# ================================================================ Trainers
def test_air_balloon_and_gravity_mountain_modify_values_through_lookups():
    covers("1174_0", "1252_0")
    s = scenario(actives=("Dudunsparce", "Riolu"), tools={(0, 0): "Air Balloon"})
    assert B.retreat_cost(s, slot_of(s, 0, 0)) == 1
    s = scenario(**P1, actives=("Snorunt", "Hariyama"), stadium=(1, "Gravity Mountain"))
    h = slot_of(s, 1, 0)
    assert B.max_hp(s, h) == 150                                      # Stage 1: not affected
    ref = s.catalog.refs[h.top]
    s.catalog.refs[h.top] = CardRef(ref.uid, ref.owner, ref.tag, replace(ref.card, kind="Stage 2 Pokémon"))
    try:
        assert B.max_hp(s, h) == 120                                  # a Stage 2 gets -30 HP
    finally:
        s.catalog.refs[h.top] = ref


def test_battle_cage_protects_benched_pokemon_from_opponent_effects():
    covers("1264_0")
    s = scenario(actives=("Snorunt", "Riolu"), stadium=(0, "Battle Cage"))
    assert B.bench_counters_prevented(s, target_owner=0, source_owner=1)
    assert not B.bench_counters_prevented(s, target_owner=0, source_owner=0)


def test_boss_orders_and_switch():
    covers("1182_0", "1123_0")
    s = scenario(hands=(("Boss's Orders",), ()), actives=("Snorunt", "Riolu"), benches=((), ("Solrock", "Lunatone")))
    s = do(s, kind="play_supporter")
    s = choose(s, s.pending.options[1])
    assert s.catalog.card(slot_of(s, 1, 0).top).name == "Lunatone"
    assert not has(scenario(hands=(("Boss's Orders",), ()), actives=("Snorunt", "Riolu")), kind="play_supporter")
    t = scenario(**P1, hands=((), ("Switch",)), actives=("Snorunt", "Riolu"), benches=((), ("Solrock",)),
                 conditions={(1, 0): {"Asleep"}})
    t = do(t, kind="play_item")
    assert t.catalog.card(slot_of(t, 1, 0).top).name == "Solrock"
    assert slot_of(t, 1, 1).conditions == set()                       # moving to the Bench ends conditions


def test_buddy_buddy_poffin_respects_the_bench_limit():
    covers("1086_0")
    s = scenario(hands=(("Buddy-Buddy Poffin",), ()), actives=("Snorunt", "Riolu"),
                 benches=(("Psyduck", "Buneary", "Buneary", "Snorunt"), ()))
    s = do(s, kind="play_item")
    s = resolve(s)
    assert len(s.players[0].bench) == 5
    assert all(s.catalog.card(b.top).hp <= 70 for b in s.players[0].bench[4:])


def test_fighting_gong_hilda_petrel_poke_pad_ultra_ball_searches():
    covers("1142_0", "1225_0", "1219_0", "1152_0", "1121_0")
    s = scenario(**P1, hands=((), ("Fighting Gong",)), actives=("Snorunt", "Riolu"))
    s = resolve(do(s, kind="play_item"))
    assert all(B.card(s, u).type == "{F}" for u in s.players[1].hand)
    s = scenario(hands=(("Hilda",), ()), actives=("Snorunt", "Riolu"))
    s = resolve(do(s, kind="play_supporter"))
    kinds = sorted(B.card(s, u).kind for u in s.players[0].hand)
    assert len(kinds) == 2 and "Stage 1 Pokémon" in kinds
    t = scenario(hands=(("Hilda",), ()), actives=("Snorunt", "Riolu"))
    t = do(t, kind="play_supporter")
    t = choose(t, t.pending.stop)                                      # Hilda may come up short (setting)
    t = resolve(t)
    assert len(t.players[0].hand) == 1
    s = scenario(**P1, hands=((), ("Team Rocket's Petrel",)), actives=("Snorunt", "Riolu"))
    s = resolve(do(s, kind="play_supporter"))
    assert B.card(s, s.players[1].hand[0]).family == "trainer"
    s = scenario(hands=(("Poké Pad",), ()), actives=("Snorunt", "Riolu"))
    s = do(s, kind="play_item")
    assert all(not B.card(s, u).has_rule_box for u in s.pending.options)
    s = scenario(hands=(("Ultra Ball", "Lucian", "Hilda", "Boss's Orders"), ()), actives=("Snorunt", "Riolu"))
    s = do(s, kind="play_item")
    assert s.pending.kind == "move:hand>discard" and s.pending.stop is None
    s = resolve(s)
    assert len(s.players[0].discard) == 3 and len(s.players[0].hand) == 2
    u = scenario(hands=(("Ultra Ball", "Hilda"), ()), actives=("Snorunt", "Riolu"))
    assert not has(u, kind="play_item")                               # needs 2 other cards


def test_judge_lillies_determination_unfair_stamp_hand_resets():
    covers("1213_0", "1227_0", "1080_0")
    s = scenario(**P1, hands=(("Buneary",), ("Judge", "Switch")), actives=("Snorunt", "Riolu"))
    s = do(s, kind="play_supporter")
    assert len(s.players[0].hand) == 4 and len(s.players[1].hand) == 4
    for prizes, n in ((6, 8), (5, 6)):
        t = scenario(hands=(("Lillie's Determination", "Buneary"), ()), actives=("Snorunt", "Riolu"),
                     prizes=(prizes, 3))
        assert len(do(t, kind="play_supporter").players[0].hand) == n
    u = scenario(**P1, hands=(("Buneary",), ("Unfair Stamp",)), actives=("Snorunt", "Riolu"))
    assert not has(u, kind="play_item")
    u.players[1].ko_turns.append(u.turn - 1)
    u = do(u, kind="play_item")
    assert len(u.players[1].hand) == 5 and len(u.players[0].hand) == 2


def test_lucian_both_hands_to_bottom_and_coin_per_player():
    covers("1237_0")
    seen = set()
    for seed in range(12):
        s = scenario(seed=seed, hands=(("Lucian", "Buneary"), ("Switch",)), **dict(P1, current=0, turns_taken=(2, 1)),
                     actives=("Snorunt", "Riolu"))
        s.rng.seed(seed)
        bottom1 = s.players[1].hand[0]
        s = do(s, kind="play_supporter")
        h = (len(s.players[0].hand), len(s.players[1].hand))
        assert all(x in (3, 6) for x in h)
        seen.add(h)
    assert len(seen) >= 3                                             # the two coins are independent


def test_night_stretcher_pokegear_special_red_card():
    covers("1097_0", "1122_0", "1413_0")
    s = scenario(hands=(("Night Stretcher",), ()), actives=("Snorunt", "Riolu"))
    assert not has(s, kind="play_item")                               # nothing to put back
    s = scenario(hands=(("Night Stretcher",), ()), actives=("Snorunt", "Riolu"), discards=(("Buneary", "Hilda"), ()))
    s = resolve(do(s, kind="play_item"))
    assert names(s, s.players[0].hand) == ["Buneary"]
    s = scenario(hands=(("Pokégear 3.0",), ()), actives=("Snorunt", "Riolu"),
                 decks_top=(("Buneary", "Hilda", "Psyduck"), ()))
    s = do(s, kind="play_item")
    assert names(s, s.pending.options) == ["Hilda"] and s.pending.stop == R.STOP
    s = choose(s, s.pending.options[0])
    assert names(s, s.players[0].hand) == ["Hilda"]
    t = scenario(hands=(("Special Red Card",), ("Switch", "Judge")), actives=("Snorunt", "Riolu"), prizes=(6, 4))
    assert not has(t, kind="play_item")
    t = scenario(hands=(("Special Red Card",), ("Switch", "Judge")), actives=("Snorunt", "Riolu"), prizes=(6, 3))
    bottom = sorted(t.players[1].hand)
    t = do(t, kind="play_item")
    assert len(t.players[1].hand) == 3 and sorted(t.players[1].deck[:2]) == bottom


def test_premium_power_pro_adds_30_this_turn():
    covers("1141_0")
    s = scenario(**P1, hands=((), ("Premium Power Pro",)), actives=("Snorunt", "Riolu"),
                 energy={(1, 0): ["Basic {F} Energy"]})
    s = do(do(s, kind="play_item"), kind="attack", choice="677_0")
    assert slot_of(s, 0, 0).damage == 60
    assert not any(le["kind"] == "attack_bonus" for le in do(s, kind="end_turn").lasting)


def test_tool_scrapper_discards_up_to_two_tools():
    covers("1137_0")
    s = scenario(decks=("mega_lucario", "lopunny"), hands=(("Tool Scrapper",), ()), actives=("Riolu", "Snorunt"),
                 benches=(("Solrock",), ()), tools={(0, 1): "Air Balloon", (1, 0): "Air Balloon"})
    s = resolve(do(s, kind="play_item"))
    assert slot_of(s, 1, 0).tool is None and slot_of(s, 0, 1).tool is None
    assert not has(scenario(decks=("mega_lucario", "lopunny"), hands=(("Tool Scrapper",), ()),
                            actives=("Riolu", "Snorunt")), kind="play_item")


def test_wallys_compassion_heals_and_returns_energy():
    covers("1229_0")
    s = scenario(hands=(("Wally's Compassion",), ()), actives=("Mega Lopunny ex", "Riolu"),
                 energy={(0, 0): ["Basic {W} Energy"] * 2}, damage={(0, 0): 100})
    s = resolve(do(s, kind="play_supporter"))
    assert slot_of(s, 0, 0).damage == 0 and not slot_of(s, 0, 0).energy
    assert names(s, s.players[0].hand) == ["Basic {W} Energy"] * 2
    t = scenario(hands=(("Wally's Compassion",), ()), actives=("Snorunt", "Riolu"))
    assert not has(t, kind="play_supporter")                          # no Mega Evolution Pokémon ex in play


# ================================================================ coverage and safety
def test_unknown_blocks_raise():
    s = scenario(actives=("Snorunt", "Riolu"))
    fr = R.effect_frame(s, "1123_0", 0)
    fr.data["stack"] = [[("not a step",), 0, 0]]
    with pytest.raises(B.EngineError):
        R.run_effect(s, fr, None)


def test_every_effect_has_a_scenario():
    cited = set(re.findall(r'"(\d+_\d+)"', " ".join(re.findall(r"covers\(([^)]*)\)", Path(__file__).read_text(encoding="utf-8")))))
    assert {e["move_id"] for e in EFFECTS} == cited, {e["move_id"] for e in EFFECTS} ^ cited


def test_no_card_is_lost_during_effects():
    s = scenario(hands=(("Ultra Ball", "Lucian", "Hilda"), ()), actives=("Snorunt", "Riolu"))
    s = do(s, kind="play_item")
    assert len(card_locations(s)) == 120 and "resolving" in card_locations(s).values()
