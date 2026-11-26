"""A running commentary of a game, written to a file that a second terminal can follow.

It reports only what both players may see: how many cards were drawn, never which ones.

Watch:  touch game.log && tail -f game.log
Play:   uv run python -m tcg_bot.narrate --delay 1
"""
import argparse
import time
from pathlib import Path

from tcg_bot.engine.game import card_of, legal_actions, new_game, step
from tcg_bot.engine.types import PLAYERS, Action, ActionKind, Phase, State

LOG = Path("game.log")


def opening_lines(state: State, names: tuple[str, str]) -> list[str]:
    return ["=== new game ===", f"{names[0]} vs {names[1]}",
            f"Coin flip: {names[state.pending.player]} wins and decides who goes first."]


def describe(before: State, action: Action, after: State, names: tuple[str, str]) -> list[str]:
    """What happened in one step, as lines of text."""
    who = names[action.player]
    out = []
    if action.kind == ActionKind.CHOOSE and before.pending.kind == "go_first":
        out.append(f"{names[after.first_player]} goes first.")
        for i in PLAYERS:
            times = after.players[i].mulligans
            if times:
                out.append(f"{names[i]} had no Basic Pokémon: {times} mulligan(s).")
            out.append(f"{names[i]} draws an opening hand of 7.")
    elif action.kind == ActionKind.CHOOSE and before.pending.kind == "mulligan_draw":
        out.append(f"{who} draws {action.option} bonus card(s) for the other player's mulligans.")
    elif action.kind == ActionKind.PLACE_ACTIVE:
        out.append(f"{who} places an Active Pokémon face down.")
    elif action.kind == ActionKind.PLACE_BENCH:
        out.append(f"{who} places a Benched Pokémon face down.")
    elif action.kind == ActionKind.SETUP_DONE:
        out.append(f"{who} is ready.")
    else:
        out.append(f"{who}: {action.kind.value}")          # kinds without their own sentence yet

    if before.phase == Phase.SETUP and after.phase != Phase.SETUP:
        out.append(f"Each player sets {after.prize_count} Prize cards. Pokémon are turned face up:")
        for i in PLAYERS:
            p = after.players[i]
            bench = ", ".join(card_of(after, m.card).name for m in p.bench) or "empty"
            out.append(f"  {names[i]}: Active {card_of(after, p.active.card).name} | Bench {bench}")
    if after.turn > before.turn:
        out.append(f"--- Turn {after.turn}: {names[after.current]} ---")
        if after.end_reason != "deck_out":
            out.append(f"{names[after.current]} draws 1 card.")
        out.append("Cards: " + " | ".join(
            f"{names[i]} hand {len(p.hand)}, deck {len(p.deck)}, prizes {len(p.prizes)}"
            for i, p in enumerate(after.players)))
    if after.phase == Phase.OVER and before.phase != Phase.OVER:
        result = "Draw" if after.winner is None else f"{names[after.winner]} wins"
        out.append(f"GAME OVER: {result} ({after.end_reason}).")
    return out


def write(lines: list[str], path: Path = LOG) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)


# ---------------------------------------------------------------- the board view
BOARD = Path("board.log")
CLEAR = "\033[2J\033[H"                  # tells the terminal to wipe the screen before the next board


def board_lines(state: State, names: tuple[str, str]) -> list[str]:
    """Everything on the table, BOTH hands included. For testing: a player must not watch this."""
    def name(uid: int) -> str:
        return card_of(state, uid).name

    title = "setup" if state.phase == Phase.SETUP else f"turn {state.turn}: {names[state.current]}"
    out = [f"=== {title} ==="]
    for i, p in enumerate(state.players):
        out += ["",
                f"{names[i]} (mulligans: {p.mulligans})",
                f"  active: {name(p.active.card) if p.active else '-'}",
                f"  bench:  {[name(m.card) for m in p.bench]}",
                f"  hand:   {[name(u) for u in p.hand]}",
                f"  deck {len(p.deck)} cards, prizes {len(p.prizes)}, discard {len(p.discard)}"]
    return out


def show_board(state: State, names: tuple[str, str], path: Path = BOARD) -> None:
    lines = board_lines(state, names)
    write([CLEAR + lines[0], *lines[1:]], path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an opening and log it.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between steps")
    args = parser.parse_args()

    decks = ("lopunny", "mega_lucario")
    state = new_game(decks, args.seed)
    write(opening_lines(state, decks))
    show_board(state, decks)
    while state.phase == Phase.SETUP:                      # main-phase actions are not written yet
        time.sleep(args.delay)
        action = legal_actions(state)[-1]
        after = step(state, action)
        write(describe(state, action, after, decks))
        show_board(after, decks)
        state = after
    print(f"Done. The game was written to {LOG} and {BOARD}.")


if __name__ == "__main__":
    main()

