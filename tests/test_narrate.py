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
