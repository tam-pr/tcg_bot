import random

import pytest

from tcg_bot.engine.catalog import catalog
from tcg_bot.engine.game import legal_actions, new_game, step
from tcg_bot.engine.types import Phase
from tcg_bot.narrate import board_lines, demo_choice, describe, opening_lines, write

DECKS = ("lopunny", "mega_lucario")


@pytest.mark.parametrize("seed", range(30))
def test_the_log_names_no_card_before_pokemon_are_turned_face_up(seed):
    pick = random.Random(seed)
    state = new_game(DECKS, seed)
    lines = opening_lines(state, DECKS)
    while state.phase == Phase.SETUP:
        action = pick.choice(legal_actions(state))
        after = step(state, action)
        lines += describe(state, action, after, DECKS)
        state = after
    reveal = next(i for i, line in enumerate(lines) if "turned face up" in line)
    names = {catalog().cards[c].name for deck in DECKS for c in catalog().decks[deck]}
    assert not [line for line in lines[:reveal] for name in names if name in line]
    assert any(line.startswith("--- Turn 1:") for line in lines)


def test_write_adds_to_the_file(tmp_path):
    path = tmp_path / "game.log"
    write(["a"], path)
    write(["b"], path)
    assert path.read_text(encoding="utf-8") == "a\nb\n"


@pytest.mark.parametrize("seed", range(20))
def test_a_whole_game_can_be_narrated(seed):
    pick = random.Random(seed)
    state = new_game(DECKS, seed)
    lines = []
    while state.phase != Phase.OVER:
        action = demo_choice(state, pick)
        after = step(state, action)
        lines += describe(state, action, after, DECKS)
        board_lines(after, DECKS)
        state = after
    assert lines[-1].startswith("GAME OVER")


def test_card_effects_that_draw_or_flip_are_reported_without_naming_cards(monkeypatch):
    from test_engine_attacks import LOP, LUC
    from test_engine_cards import board, give, play

    from tcg_bot.engine import effects_runtime

    monkeypatch.setattr(effects_runtime, "coin_heads", lambda state: True)
    state = board(LOP)
    give(state, LOP, "Lucian", "Hilda")
    give(state, LUC, "Switch", "Judge")
    after = play(state, "Lucian")
    action = next(a for a in legal_actions(state) if a.kind == "play_supporter")
    lines = describe(state, action, after, ("me", "them"))
    assert "me puts their hand (1 card(s)) on the bottom of their deck." in lines
    assert "them puts their hand (2 card(s)) on the bottom of their deck." in lines
    assert "Coin flip for me: HEADS." in lines and "them draws 6 card(s)." in lines
    assert not [line for line in lines for name in ("Hilda", "Switch", "Judge") if name in line]


def test_a_knock_out_is_reported_before_its_prizes_and_hp_stops_at_zero():
    from test_engine_attacks import LOP, LUC, F, arrange, attack

    from tcg_bot.engine.types import Action, ActionKind
    from tcg_bot.narrate import board_lines

    state = arrange(LUC, 678, [F, F], "Mega Lopunny ex")
    after = attack(state, "Mega Brave")                       # 270, doubled by Weakness: more than 330 HP
    move_id = next(m.move_id for m in catalog().cards[678].moves if m.name == "Mega Brave")
    lines = describe(state, Action(ActionKind.ATTACK, LUC, move_id=move_id), after, ("lop", "luc"))
    knocked = next(i for i, line in enumerate(lines) if "Knocked Out" in line)
    assert knocked < next(i for i, line in enumerate(lines) if "Prize card" in line)
    state.players[LOP].active.damage = 999
    assert " 0/330" in "\n".join(board_lines(state, ("lop", "luc")))
