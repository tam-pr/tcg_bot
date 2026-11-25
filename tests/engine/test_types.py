"""T1: the interface types construct, compare and copy as the contract says."""
import random

import pytest

from tcg_bot.engine.types import (UID_STRIDE, Action, Frame, Pending, PlayerState, RuleSettings, Slot, State)


def test_action_is_a_hashable_plain_value():
    a = Action("attack", 0, source=3, choice="848_1")
    assert a == Action("attack", 0, source=3, choice="848_1")
    assert len({a, Action("attack", 0, source=3, choice="848_1"), Action("end_turn", 1)}) == 2


@pytest.mark.parametrize("bad", [dict(kind="teleport", player=0), dict(kind="end_turn", player=2)])
def test_action_rejects_unknown_kinds_and_players(bad):
    with pytest.raises(ValueError):
        Action(**bad)


def test_uids_are_unique_across_players():
    assert UID_STRIDE >= 60
    p0 = {0 * UID_STRIDE + i for i in range(60)}
    p1 = {1 * UID_STRIDE + i for i in range(60)}
    assert not p0 & p1


def test_settings_list_every_assumption():
    s = RuleSettings()
    assert set(s.assumptions()) == {
        "mega_evolution_ends_turn", "mega_ex_prize_value", "resistance_amount", "may_end_turn_without_attack",
        "effect_attach_uses_hand_attach", "bottom_of_deck_order", "search_may_find_fewer",
        "recoil_applies_weakness_resistance", "take_mulligan_bonus_cards"}


def _state():
    pl = [PlayerState(deck=[1, 2, 3], hand=[4], active=Slot(pid=5, cards=[5], energy=[6], conditions={"Asleep"}),
                      bench=[Slot(pid=7, cards=[7])]), PlayerState(deck=[101])]
    return State(catalog=None, settings=RuleSettings(), players=pl, rng=random.Random(1),
                 flags={"names_used": ["Fan Call"]}, lasting=[{"kind": "x", "until_turn": 3}],
                 pending=Pending(0, "promote", (7,)), frames=[Frame("effect", {"stack": [[(), 0, 0]], "ctx": {}})])


def test_copy_is_independent_and_shares_the_catalog():
    s = _state()
    c = s.copy()
    c.players[0].deck.append(9)
    c.players[0].active.energy.clear()
    c.players[0].active.conditions.clear()
    c.players[0].bench[0].damage = 50
    c.flags["names_used"].append("Lunar Cycle")
    c.lasting[0]["until_turn"] = 9
    c.frames[0].data["stack"][0][1] = 5
    assert s.players[0].deck == [1, 2, 3] and s.players[0].active.energy == [6]
    assert s.players[0].active.conditions == {"Asleep"} and s.players[0].bench[0].damage == 0
    assert s.flags["names_used"] == ["Fan Call"] and s.lasting[0]["until_turn"] == 3
    assert s.frames[0].data["stack"][0][1] == 0
    assert c.catalog is s.catalog and c.settings is s.settings


def test_copy_replays_the_same_random_numbers():
    s = _state()
    c = s.copy()
    assert [s.rng.random() for _ in range(5)] == [c.rng.random() for _ in range(5)]
