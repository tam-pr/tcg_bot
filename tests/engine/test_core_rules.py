"""One test (at least) per row of configs/plays.csv. Each test names its play_id."""
import pandas as pd
import pytest

from tcg_bot.engine import Action, IllegalActionError, RuleSettings, legal_actions, new_game, step
from tcg_bot.engine.catalog import card_static, tables
from tcg_bot.engine.core import card_locations

from .helpers import choose, do, find, has, names, scenario, slot_of

PLAYS = pd.read_csv(__import__("tcg_bot.paths", fromlist=["CONFIGS"]).CONFIGS / "plays.csv")
TESTED = set()


def covers(play_id):
    TESTED.add(play_id)
    return play_id


def finish_setup(s):
    """Place the first Basic as Active for both players and end setup."""
    while s.phase == "setup":
        acts = legal_actions(s)
        a = next((x for x in acts if x.kind == "choose_first"), None) or \
            next((x for x in acts if x.kind == "set_active"), None) or \
            next(x for x in acts if x.kind == "setup_done")
        s = step(s, a)
    return s


# ---------------------------------------------------------------- setup
def test_setup_coin_flip():
    covers("setup_coin_flip")
    for seed in range(6):
        s = new_game(("lopunny", "mega_lucario"), seed)
        w = s.flags["flip_winner"]
        assert {a.kind for a in legal_actions(s)} == {"choose_first"} and all(a.player == w for a in legal_actions(s))
        s2 = step(s, Action("choose_first", w, choice="second"))
        assert s2.first_player == 1 - w


def test_setup_draw_hand_and_mulligan():
    covers("setup_draw_hand"); covers("setup_mulligan")
    seen_mulligan = False
    for seed in range(60):
        s = new_game(("lopunny", "mega_lucario"), seed)
        m = [s.players[p].mulligans for p in (0, 1)]
        seen_mulligan |= max(m) > 0
        for p in (0, 1):
            pl = s.players[p]
            assert any(s.catalog.card(u).is_basic_pokemon for u in pl.hand)
            assert len(pl.hand) == 7 + max(0, m[1 - p] - m[p])
    assert seen_mulligan, "no seed produced a mulligan; widen the range"


def test_setup_place_active_and_bench():
    covers("setup_place_active"); covers("setup_place_bench")
    s = new_game(("lopunny", "mega_lucario"), 3)
    s = step(s, Action("choose_first", s.current, choice="first"))
    acts = legal_actions(s)
    assert acts and all(a.kind == "set_active" and s.catalog.card(a.source).is_basic_pokemon for a in acts)
    s = step(s, acts[0])
    assert all(a.kind in ("set_bench", "setup_done") for a in legal_actions(s))
    while has(s, kind="set_bench"):
        s = do(s, kind="set_bench")
    assert len(s.players[s.current].bench) <= 5


def test_setup_set_prizes_and_first_draw():
    covers("setup_set_prizes")
    s = finish_setup(new_game(("lopunny", "mega_lucario"), 5))
    assert s.phase == "main" and s.turn == 1 and s.current == s.first_player
    for p in (0, 1):
        assert len(s.players[p].prizes) == 6
    assert len(card_locations(s)) == 120


def test_opponent_setup_is_hidden_until_both_are_done():
    s = new_game(("lopunny", "mega_lucario"), 3)
    s = step(s, Action("choose_first", s.current, choice="first"))
    s = do(s, kind="set_active")
    from tcg_bot.engine import observe
    assert observe(s, 1 - s.current).opp_active is None


# ---------------------------------------------------------------- turn
def test_turn_draw_and_deck_out():
    covers("turn_draw")
    s = scenario(actives=("Buneary", "Riolu"), benches=(("Buneary",), ("Riolu",)))
    before = len(s.players[1].hand)
    s2 = do(s, kind="end_turn")
    assert s2.current == 1 and len(s2.players[1].hand) == before + 1
    s.players[1].discard.extend(s.players[1].deck); s.players[1].deck = []
    s3 = do(s, kind="end_turn")
    assert s3.phase == "over" and s3.winner == 0 and s3.result == "deck_out"


def test_end_turn_without_attacking():
    covers("turn_end")
    s = scenario(actives=("Buneary", "Riolu"))
    assert has(s, kind="end_turn")
    s2 = scenario(actives=("Buneary", "Riolu"))
    s2.settings = RuleSettings(may_end_turn_without_attack=False)
    assert has(s2, kind="end_turn") is False or not any(a.kind != "end_turn" for a in legal_actions(s2))


# ---------------------------------------------------------------- main phase
def test_bench_basic_respects_bench_max():
    covers("main_bench_basic")
    s = scenario(hands=(("Buneary",), ()), actives=("Snorunt", "Riolu"),
                 benches=(("Psyduck", "Fan Rotom", "Chien-Pao", "Dunsparce", "Snorunt"), ()))
    assert not has(s, kind="bench_basic")
    s = scenario(hands=(("Buneary",), ()), actives=("Snorunt", "Riolu"))
    s = do(s, kind="bench_basic")
    assert names(s, [b.pid for b in s.players[0].bench]) == ["Buneary"]


def test_evolve_rules():
    covers("main_evolve")
    s = scenario(hands=(("Mega Lopunny ex", "Buneary"), ()), actives=("Buneary", "Riolu"),
                 energy={(0, 0): ["Basic {W} Energy"]}, damage={(0, 0): 30}, conditions={(0, 0): {"Asleep"}})
    s = do(s, kind="evolve")
    a = slot_of(s, 0, 0)
    assert s.catalog.card(a.top).name == "Mega Lopunny ex" and a.damage == 30 and len(a.energy) == 1
    assert a.conditions == set()
    s = scenario(hands=(("Mega Lopunny ex",), ()), actives=("Buneary", "Riolu"), turn=1, turns_taken=(1, 0))
    assert not has(s, kind="evolve")                         # not on the player's first turn
    s = scenario(hands=(("Buneary", "Mega Lopunny ex"), ()), actives=("Snorunt", "Riolu"))
    s = do(s, kind="bench_basic")
    assert not has(s, kind="evolve")                         # not a Pokémon that came into play this turn


def test_mega_evolution_turn_end_is_a_setting():
    covers("main_mega_turn_end")
    s = scenario(hands=(("Mega Lopunny ex",), ()), actives=("Buneary", "Riolu"))
    assert do(s, kind="evolve").current == 0                 # default: does not end the turn (assumption)
    s.settings = RuleSettings(mega_evolution_ends_turn=True)
    assert do(s, kind="evolve").current == 1


def test_attach_energy_once_per_turn():
    covers("main_attach_energy")
    s = scenario(hands=(("Basic {W} Energy", "Basic {W} Energy"), ()), actives=("Snorunt", "Riolu"))
    s = do(s, kind="attach_energy")
    assert len(slot_of(s, 0, 0).energy) == 1 and not has(s, kind="attach_energy")


def test_play_item_goes_to_discard_and_items_are_unlimited():
    covers("main_play_item")
    s = scenario(hands=(("Poké Pad", "Poké Pad"), ()), actives=("Snorunt", "Riolu"), decks_top=(("Buneary",), ()))
    s = do(s, kind="play_item")
    while s.pending:
        s = choose(s, s.pending.options[0])
    assert names(s, s.players[0].discard) == ["Poké Pad"] and has(s, kind="play_item")


def test_supporter_once_per_turn_and_not_first_turn_of_first_player():
    covers("main_play_supporter")
    s = scenario(hands=((), ("Judge", "Judge")), current=1, actives=("Snorunt", "Riolu"), turns_taken=(1, 1), turn=2)
    s = do(s, kind="play_supporter")
    assert not has(s, kind="play_supporter")
    s = scenario(hands=((), ("Judge",)), current=1, first_player=1, actives=("Snorunt", "Riolu"),
                 turns_taken=(0, 1), turn=1)
    assert not has(s, kind="play_supporter")


def test_stadium_rules():
    covers("main_play_stadium")
    s = scenario(hands=((), ("Gravity Mountain", "Gravity Mountain")), current=1, actives=("Snorunt", "Riolu"),
                 stadium=(0, "Battle Cage"))
    s = do(s, kind="play_stadium")
    assert s.catalog.card(s.stadium).name == "Gravity Mountain"
    assert names(s, s.players[0].discard) == ["Battle Cage"]      # the old Stadium goes to its owner's discard
    assert not has(s, kind="play_stadium")                         # once per turn, and not the same name


def test_tool_one_per_pokemon():
    covers("main_attach_tool")
    s = scenario(hands=(("Air Balloon", "Air Balloon"), ()), actives=("Snorunt", "Riolu"))
    s = do(s, kind="attach_tool")
    assert not has(s, kind="attach_tool")


def test_retreat_cost_modifier_and_once_per_turn():
    covers("main_retreat")
    s = scenario(actives=("Dudunsparce", "Riolu"), benches=(("Buneary", "Psyduck"), ()),
                 energy={(0, 0): ["Basic {W} Energy"]}, tools={(0, 0): "Air Balloon"})
    assert has(s, kind="retreat")                                  # cost 3 - 2 = 1, one Energy attached
    s = do(s, kind="retreat")
    assert s.catalog.card(slot_of(s, 0, 0).top).name in ("Buneary", "Psyduck")
    assert names(s, s.players[0].discard) == ["Basic {W} Energy"]
    assert not has(s, kind="retreat")
    s2 = scenario(actives=("Dudunsparce", "Riolu"), benches=(("Buneary",), ()),
                  energy={(0, 0): ["Basic {W} Energy"]})
    assert not has(s2, kind="retreat")                             # 1 Energy < cost 3
    s3 = scenario(actives=("Buneary", "Riolu"), benches=(("Psyduck",), ()), energy={(0, 0): ["Basic {W} Energy"]},
                  conditions={(0, 0): {"Asleep"}})
    assert not has(s3, kind="retreat")


def test_ability_is_not_an_attack():
    covers("main_use_ability")
    s = scenario(actives=("Dudunsparce", "Riolu"), benches=(("Buneary",), ()),
                 energy={(0, 0): ["Basic {W} Energy"] * 3})
    s.players[0].active, s.players[0].bench[0] = s.players[0].bench[0], s.players[0].active
    s = do(s, kind="use_ability")
    assert s.current == 0 and not s.flags.get("attacked")


def test_fossils_are_rejected_not_skipped():
    covers("main_play_fossil")
    c = tables()["cards"]
    fossil = int(c.loc[c["is_fossil"], "card_id"].iloc[0])
    with pytest.raises(NotImplementedError):
        card_static(fossil)


# ---------------------------------------------------------------- attacks
def test_attack_rules_first_turn_cost_and_turn_end():
    covers("attack_use")
    s = scenario(actives=("Buneary", "Riolu"), energy={(0, 0): ["Basic {W} Energy"]}, turn=1, turns_taken=(1, 0))
    assert not has(s, kind="attack")                               # the player going first cannot attack turn 1
    s = scenario(actives=("Buneary", "Riolu"), energy={(0, 0): ["Basic {W} Energy"]})
    assert {a.choice for a in legal_actions(s) if a.kind == "attack"} == {"848_0"}   # Kick needs 2 Energy
    s = do(s, kind="attack", choice="848_0")
    assert s.current == 1                                          # the turn ends after an attack


def test_weakness_and_resistance():
    s = scenario(current=1, actives=("Buneary", "Riolu"), energy={(1, 0): ["Basic {F} Energy"]})
    s = do(s, kind="attack", choice="677_0")
    assert slot_of(s, 0, 0).damage == 60                           # 30 x2, Buneary is weak to {F}
    s = scenario(current=1, actives=("Fan Rotom", "Riolu"), energy={(1, 0): ["Basic {F} Energy"]})
    s = do(s, kind="attack", choice="677_0")
    assert slot_of(s, 0, 0).damage == 0                            # 30 - 30, Fan Rotom resists {F}


def test_confusion_coin():
    outcomes = set()
    for seed in range(30):
        s = scenario(seed=seed, current=1, actives=("Snorunt", "Riolu"), energy={(1, 0): ["Basic {F} Energy"]},
                     conditions={(1, 0): {"Confused"}})
        s.rng.seed(seed)
        s = do(s, kind="attack", choice="677_0")
        outcomes.add((slot_of(s, 0, 0).damage, slot_of(s, 1, 0).damage))
    assert outcomes == {(30, 0), (0, 30)}


# ---------------------------------------------------------------- checkup
def test_checkup_order_and_effects():
    covers("checkup")
    s = scenario(actives=("Buneary", "Riolu"), conditions={(0, 0): {"Poisoned", "Burned"}})
    s = do(s, kind="end_turn")
    assert slot_of(s, 0, 0).damage == 30                           # 1 + 2 damage counters
    woke = set()
    for seed in range(20):
        t = scenario(seed=seed, actives=("Buneary", "Riolu"), conditions={(0, 0): {"Asleep"}})
        t.rng.seed(seed)
        woke.add("Asleep" in slot_of(do(t, kind="end_turn"), 0, 0).conditions)
    assert woke == {True, False}
    p = scenario(current=1, actives=("Buneary", "Riolu"), conditions={(0, 0): {"Paralyzed"}})
    slot_of(p, 0, 0).paralyzed_turn = p.turn
    p = do(p, kind="end_turn")                                     # opponent's turn ends: still Paralyzed
    assert "Paralyzed" in slot_of(p, 0, 0).conditions
    p = do(p, kind="end_turn")                                     # its owner's turn ends: removed
    assert "Paralyzed" not in slot_of(p, 0, 0).conditions


# ---------------------------------------------------------------- knock out and winning
def test_knock_out_prizes_and_promotion():
    covers("knock_out")
    s = scenario(current=1, actives=("Buneary", "Riolu"), benches=(("Psyduck", "Snorunt"), ()),
                 energy={(1, 0): ["Basic {F} Energy"]}, damage={(0, 0): 10})
    hand = len(s.players[1].hand)
    s = do(s, kind="attack", choice="677_0")
    assert names(s, s.players[0].discard) == ["Buneary"]
    assert len(s.players[1].hand) == hand + 1 and len(s.players[1].prizes) == 2
    assert s.pending.kind == "promote" and s.pending.player == 0
    s = choose(s, s.pending.options[0])
    assert s.players[0].active is not None


@pytest.mark.parametrize("decks,active,attacker,energy,move,value", [
    (("mega_lucario", "lopunny"), "Meowth ex", "Snorunt", "Basic {W} Energy", "860_0", 2),
    (("lopunny", "mega_lucario"), "Mega Lopunny ex", "Riolu", "Basic {F} Energy", "677_0", 3)])
def test_prize_values(decks, active, attacker, energy, move, value):
    s = scenario(decks=decks, current=1, actives=(active, attacker), energy={(1, 0): [energy]},
                 damage={(0, 0): 300}, prizes=(6, 6), benches=((), ()))
    s.players[0].bench = []
    s = do(s, kind="attack", choice=move)
    assert len(s.players[1].prizes) == 6 - value


def test_win_by_prizes_and_by_no_pokemon():
    covers("win_check")
    s = scenario(current=1, actives=("Buneary", "Riolu"), benches=(("Psyduck",), ()),
                 energy={(1, 0): ["Basic {F} Energy"]}, damage={(0, 0): 60}, prizes=(3, 1))
    s = do(s, kind="attack", choice="677_0")
    assert (s.winner, s.result) == (1, "prizes")
    s = scenario(current=1, actives=("Buneary", "Riolu"), energy={(1, 0): ["Basic {F} Energy"]}, damage={(0, 0): 60})
    s = do(s, kind="attack", choice="677_0")
    assert (s.winner, s.result) == (1, "no_pokemon")


def test_sudden_death_when_both_win_at_once():
    s = scenario(current=1, actives=("Buneary", "Hariyama"), energy={(1, 0): ["Basic {F} Energy"] * 3},
                 damage={(1, 0): 100})
    s = do(s, kind="attack", choice="674_1")                      # Wild Press KOs both last Pokémon
    assert s.sudden_death and s.phase == "setup"
    s = finish_setup(s)
    assert all(len(s.players[p].prizes) == 1 for p in (0, 1))


def test_illegal_action_raises():
    s = scenario(actives=("Buneary", "Riolu"))
    with pytest.raises(IllegalActionError):
        step(s, Action("attack", 0, source=s.players[0].active.pid, choice="848_1"))


def test_every_plays_row_has_a_test():
    """Every play_id in configs/plays.csv is named by a covers(...) call in this file."""
    import re
    from pathlib import Path
    cited = set(re.findall(r'covers\("([a-z_]+)"\)', Path(__file__).read_text(encoding="utf-8")))
    assert set(PLAYS["play_id"]) == cited, set(PLAYS["play_id"]) ^ cited
