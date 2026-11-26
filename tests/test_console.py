import random

import pytest
from test_engine_observe import scrambled

from tcg_bot.console import card_text, find_moves, menu, play, table
from tcg_bot.engine.catalog import catalog
from tcg_bot.engine.game import new_game, step
from tcg_bot.engine.observe import observe
from tcg_bot.engine.types import Phase
from tcg_bot.narrate import demo_choice

NAMES = ("you", "computer")


def someone_typing(seed):
    """Types a small move number, with the odd command or typo in between."""
    pick, said = random.Random(seed), []

    def ask(_prompt):
        if pick.random() < 0.1:
            return pick.choice(["help", "board", "discard", "info hilda", "info zzz", "nonsense", "999", ""])
        return str(pick.randint(1, 4))

    return ask, said


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("deck", ["lopunny", "mega_lucario"])
def test_whole_games_can_be_played_by_typing(deck, seed):
    ask, said = someone_typing(seed)
    state = play(deck, seed, ask=ask, say=said.append)
    assert state.phase == Phase.OVER
    assert any(result in said[-1] for result in ("You win!", "You lose.", "It is a draw."))


@pytest.mark.parametrize("seed", range(10))
def test_nothing_hidden_is_shown(seed):
    """The table and your list of moves must be the same whatever the unseen cards are."""
    pick = random.Random(seed)
    state = new_game(("lopunny", "mega_lucario"), seed)
    while state.phase != Phase.OVER:
        if observe(state, 0).waiting_for == 0:
            other = scrambled(state, 0, pick)
            assert table(state, observe(state, 0), NAMES) == table(other, observe(other, 0), NAMES)
            assert [text for text, _ in menu(state)] == [text for text, _ in menu(other)]
        state = step(state, demo_choice(state, pick))


def test_a_move_can_be_typed_as_a_number_or_as_words():
    moves = [("Attach Basic {W} Energy to Buneary (Active, 70/70)", "a"), ("Bench Dunsparce", "b"), ("End turn", "c")]
    assert find_moves(moves, "2") == [moves[1]] and find_moves(moves, "end") == [moves[2]]
    assert find_moves(moves, "attach {w} buneary") == [moves[0]]
    assert find_moves(moves, "9") == [] and find_moves(moves, "pikachu") == []


def test_card_text_shows_what_a_card_does():
    hilda = next(c for c in catalog().cards.values() if c.name == "Hilda")
    assert "Search your deck" in "\n".join(card_text(hilda))


def test_end_of_input_stops_the_game():
    def closed(_prompt):
        raise EOFError

    said = []
    state = play("lopunny", 0, ask=closed, say=said.append)
    assert state.phase != Phase.OVER and said[-1] == "Game stopped."
