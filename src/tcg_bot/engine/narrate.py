"""A running commentary of a game, written to a file that a second terminal can follow.

It reports only what both players may see: how many cards were drawn, never which ones.

Watch:  touch game.log && tail -f game.log        (public)
        touch board.log && tail -f board.log      (both hands, for testing)
Play:   uv run python -m tcg_bot.narrate --delay 1
"""
import argparse
import random
import time
from pathlib import Path

from tcg_bot.engine.catalog import catalog
from tcg_bot.engine.game import card_of, legal_actions, new_game, step
from tcg_bot.engine.turn import max_hp
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
        out.append(_turn_sentence(before, action, after, who))

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


def _turn_sentence(before: State, action: Action, after: State, who: str) -> str:
    """One sentence for a main-phase action. Every card named here is face up by now."""
    me = before.players[action.player]
    mons = {m.uid: card_of(before, m.card).name for p in before.players for m in p.pokemon()}

    def name(uid: int) -> str:
        return card_of(before, uid).name

    if action.kind == ActionKind.BENCH:
        return f"{who} puts {name(action.card)} on the Bench."
    if action.kind == ActionKind.EVOLVE:
        return f"{who} evolves {mons[action.target]} into {name(action.card)}."
    if action.kind == ActionKind.ATTACH_ENERGY:
        return f"{who} attaches {name(action.card)} to {mons[action.target]}."
    if action.kind == ActionKind.RETREAT:
        return f"{who} retreats {mons[me.active.uid]} for {mons[action.target]}."
    if action.kind == ActionKind.END_TURN:
        return f"{who} ends the turn without attacking."
    if action.kind == ActionKind.CHOOSE and before.pending.kind == "retreat_cost":
        return f"{who} discards {name(action.option)} to retreat."
    if action.kind == ActionKind.CHOOSE and before.pending.kind == "new_active":
        return f"{who} moves {mons[action.option]} to the Active Spot."
    if action.kind == ActionKind.ATTACK:
        foe = 1 - action.player
        attacker, defender = me.active, before.players[foe].active
        move = next(m for m in card_of(before, attacker.card).moves if m.move_id == action.move_id)
        text = f"{who}'s {mons[attacker.uid]} uses {move.name} on {mons[defender.uid]}"
        hit = next((m for m in after.players[foe].pokemon() if m.uid == defender.uid), None)
        if hit:
            return f"{text}: {hit.damage - defender.damage} damage."
        taken = len(me.prizes) - len(after.players[action.player].prizes)
        return f"{text}: Knocked Out! {who} takes {taken} Prize card(s)."
    return f"{who}: {action.kind.value}"


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

    def show(mon) -> str:
        hp = max_hp(state, mon)
        energy = "".join(catalog().energy[state.card_ids[u]][0] for u in mon.energy)
        return f"{name(mon.card)} {hp - mon.damage}/{hp}" + (f" [{energy}]" if energy else "")

    title = "setup" if state.phase == Phase.SETUP else f"turn {state.turn}: {names[state.current]}"
    if state.phase == Phase.OVER:
        winner = "nobody" if state.winner is None else names[state.winner]
        title = f"GAME OVER: {winner} wins ({state.end_reason})"
    out = [f"=== {title} ==="]
    for i, p in enumerate(state.players):
        out += ["",
                f"{names[i]} (mulligans: {p.mulligans})",
                f"  active: {show(p.active) if p.active else '-'}",
                f"  bench:  {', '.join(show(m) for m in p.bench) or '-'}",
                f"  hand:   {[name(u) for u in p.hand]}",
                f"  deck {len(p.deck)} cards, prizes {len(p.prizes)}, discard {len(p.discard)}"]
    return out


def show_board(state: State, names: tuple[str, str], path: Path = BOARD) -> None:
    lines = board_lines(state, names)
    write([CLEAR + lines[0], *lines[1:]], path)


def demo_choice(state: State, pick: random.Random) -> Action:
    """A simple stand-in player so a game can be watched. Not the model: it only knows a few habits."""
    actions = legal_actions(state)
    if state.phase == Phase.SETUP:
        return actions[-1]
    if state.pending:
        return pick.choice(actions)
    p = state.players[state.current]

    def fights(uid: int) -> bool:          # has an attack the engine can already play
        return any(m.kind == "attack" and m.text is None for m in card_of(state, uid).moves)

    mons = {m.uid: m for m in p.pokemon()}
    of_kind = {k: [a for a in actions if a.kind == k] for k in ActionKind}
    stuck = not fights(p.active.card)
    ready = [a for a in of_kind[ActionKind.RETREAT] if stuck and fights(mons[a.target].card)]
    evolve = [a for a in of_kind[ActionKind.EVOLVE] if fights(a.card) or a.target != p.active.uid]
    attach = of_kind[ActionKind.ATTACH_ENERGY]
    attach = ([a for a in attach if a.target == p.active.uid]                       # to attack, or to retreat
              or [a for a in attach if fights(mons[a.target].card)] or attach)
    for choices in (evolve, of_kind[ActionKind.BENCH], attach, ready, of_kind[ActionKind.ATTACK]):
        if choices:
            return pick.choice(choices)
    return Action(ActionKind.END_TURN, state.current)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a game and log it.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between steps")
    args = parser.parse_args()

    decks = ("lopunny", "mega_lucario")
    pick = random.Random(args.seed)
    state = new_game(decks, args.seed)
    write(opening_lines(state, decks))
    show_board(state, decks)
    while state.phase != Phase.OVER:
        time.sleep(args.delay)
        action = demo_choice(state, pick)
        after = step(state, action)
        write(describe(state, action, after, decks))
        show_board(after, decks)
        state = after
    print(f"Done after {state.turn} turns. The game was written to {LOG} and {BOARD}.")


if __name__ == "__main__":
    main()

