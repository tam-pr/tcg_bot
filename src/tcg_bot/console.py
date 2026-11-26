"""Play against the computer in the terminal.

    uv run python -m tcg_bot.console                      # you play lopunny
    uv run python -m tcg_bot.console --deck mega_lucario

At each of your decisions, type the number of a move, or words from it ("attach water buneary").
Type  help  for the other commands.

The computer is the simple stand-in from narrate.py, not a trained model. You are shown only your own view.
"""
import argparse
import random
import unicodedata

from tcg_bot.engine.board import card_of, max_hp
from tcg_bot.engine.catalog import Card, catalog
from tcg_bot.engine.game import legal_actions, new_game, step
from tcg_bot.engine.observe import Observation, observe
from tcg_bot.engine.types import Action, ActionKind, Phase, PokemonInPlay, State
from tcg_bot.narrate import _works, demo_choice, describe

DECKS = ("lopunny", "mega_lucario")
HELP = """  12            play move number 12
  attach water  play the move containing these words (it must be the only one)
  board         show the table again
  info hilda    read a card's text
  discard       list both discard piles
  help          this list
  quit          stop the game"""


def plain(text: str) -> str:
    """Lower case without accents, so "poke pad" finds "Poké Pad"."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))


# ---------------------------------------------------------------- describing things
def pokemon_label(state: State, uid: int) -> str:
    """Name, place and HP of a Pokémon in play, so that two copies can be told apart."""
    for player in state.players:
        for place, mon in enumerate(player.pokemon()):
            if mon.uid == uid:
                where = "Active" if mon is player.active else f"Bench {place if player.active else place + 1}"
                hp = max_hp(state, mon)
                energy = "".join(catalog().energy[state.card_ids[u]][0] for u in mon.energy)
                extra = f", {energy}" if energy else ""
                return f"{card_of(state, mon.card).name} ({where}, {max(hp - mon.damage, 0)}/{hp}{extra})"
    raise ValueError(f"no Pokémon {uid} in play")


def move_text(state: State, action: Action) -> str:
    """One legal move of the player to act, in words."""
    def card(uid: int) -> str:
        return card_of(state, uid).name

    kind = action.kind
    if kind == ActionKind.CHOOSE:
        return _choice_text(state, action.option)
    if kind == ActionKind.ATTACK:
        mon = state.players[action.player].active
        move = next(m for m in card_of(state, mon.card).moves if m.move_id == action.move_id)
        damage = f" ({move.damage})" if move.damage else ""
        nothing = "" if _works(state, move.move_id) else "  [would do nothing now]"
        return f"Attack: {move.name}{damage}{nothing}"
    if kind == ActionKind.USE_ABILITY:
        mon = next(m for m in state.players[action.player].pokemon() if m.uid == action.target)
        move = next(m for m in card_of(state, mon.card).moves if m.move_id == action.move_id)
        return f"Ability {move.name}: {pokemon_label(state, mon.uid)}"
    target = "" if action.target is None else pokemon_label(state, action.target)
    texts = {
        ActionKind.PLACE_ACTIVE: lambda: f"Put {card(action.card)} in the Active Spot",
        ActionKind.PLACE_BENCH: lambda: f"Put {card(action.card)} on the Bench",
        ActionKind.SETUP_DONE: lambda: "Done setting up",
        ActionKind.BENCH: lambda: f"Bench {card(action.card)}",
        ActionKind.EVOLVE: lambda: f"Evolve {target} into {card(action.card)}",
        ActionKind.ATTACH_ENERGY: lambda: f"Attach {card(action.card)} to {target}",
        ActionKind.ATTACH_TOOL: lambda: f"Attach {card(action.card)} to {target}",
        ActionKind.PLAY_ITEM: lambda: f"Play Item {card(action.card)}",
        ActionKind.PLAY_SUPPORTER: lambda: f"Play Supporter {card(action.card)}",
        ActionKind.PLAY_STADIUM: lambda: f"Play Stadium {card(action.card)}",
        ActionKind.RETREAT: lambda: f"Retreat, and bring up {target}",
        ActionKind.END_TURN: lambda: "End turn",
        ActionKind.CHOOSE_DONE: lambda: "Done choosing",
    }
    return texts[kind]()


def _choice_text(state: State, option: int) -> str:
    pending = state.pending
    kind = pending.kind
    if kind == "go_first":
        return "Go first" if option == 1 else "Go second"
    if kind == "mulligan_draw":
        return f"Draw {option} bonus card(s)"
    if kind == "use_ability":
        return "Yes, use the Ability" if option == 1 else "No, do not use it"
    if kind in ("new_active", "promote", "switch_to"):
        return f"Move {pokemon_label(state, option)} to the Active Spot"
    if kind == "gust":
        return f"Drag up their {pokemon_label(state, option)}"
    if kind == "heal":
        return f"Heal {pokemon_label(state, option)}"
    if kind == "attach_target":
        return f"Attach it to {pokemon_label(state, option)}"
    name = card_of(state, option).name
    if kind == "retreat_cost":
        return f"Discard {name} to pay the retreat"
    if kind == "energy_to_hand":
        return f"Put {name} back into your hand"
    if kind == "attach_pick":
        return f"Take {name} from the discard pile to attach"
    step_ = state.stack[-1].steps[0]                        # kind "pick"
    if step_.source == "attached_tool_any":
        holder = next(m for p in state.players for m in p.pokemon() if m.tool == option)
        return f"Discard the Tool {name} from {pokemon_label(state, holder.uid)}"
    if step_.source == "hand":
        return f"Discard {name}"
    if step_.destination == "bench":
        return f"Put {name} on the Bench"
    return f"Take {name}"


def menu(state: State) -> list[tuple[str, Action]]:
    """The legal moves with their text; two copies of the same card are listed once."""
    found: dict[str, Action] = {}
    stopping = (ActionKind.END_TURN, ActionKind.CHOOSE_DONE)
    for action in sorted(legal_actions(state), key=lambda a: a.kind in stopping):      # stopping goes last
        found.setdefault(move_text(state, action), action)
    return list(found.items())


def find_moves(moves: list[tuple[str, Action]], typed: str) -> list[tuple[str, Action]]:
    """The moves meant by what was typed: a number, or words that all appear in the move."""
    if typed.isdigit():
        return [moves[int(typed) - 1]] if 1 <= int(typed) <= len(moves) else []
    words = plain(typed).split()
    exact = [m for m in moves if plain(m[0]) == plain(typed)]
    return exact or [m for m in moves if all(w in plain(m[0]) for w in words)]


def card_text(card: Card) -> list[str]:
    head = f"{card.name} ({card.kind}" + (f", {card.hp} HP, {card.type}" if card.hp else "") + ")"
    out = [head]
    if card.weakness or card.resistance:
        out.append(f"  weakness {card.weakness or '-'}, resistance {card.resistance or '-'}")
    for move in card.moves:
        cost = " ".join(f"{n}x{t}" for t, n in move.typed_cost)
        if move.kind == "attack":
            out.append(f"  Attack {move.name}: {move.total_cost} Energy {cost} {move.damage or ''}".rstrip())
        elif move.name:
            out.append(f"  {move.kind.title()} {move.name}")
        if move.text:
            out.append(f"    {move.text}")
    return out


# ---------------------------------------------------------------- the table, from your side
def table(state: State, view: Observation, names: tuple[str, str]) -> list[str]:
    """Only what `view` holds is shown. `state` is used for HP, which Stadiums can change."""
    def name(uid: int) -> str:
        return catalog().cards[view.card_ids[uid]].name

    def mon_text(mon: PokemonInPlay) -> str:
        hp = max_hp(state, mon)
        energy = "".join(catalog().energy[view.card_ids[u]][0] for u in mon.energy)
        text = f"{name(mon.card)} {max(hp - mon.damage, 0)}/{hp}" + (f" [{energy}]" if energy else "")
        return text + (f" +{name(mon.tool)}" if mon.tool is not None else "") + (
            f" ({mon.rotation})" if mon.rotation else "")

    def side(title: str, p) -> list[str]:
        active = mon_text(p.active) if p.active else (f"{p.face_down} face down" if p.face_down else "-")
        counts = f"hand {p.hand_count}, deck {p.deck_count}, prizes {p.prize_count}, discard {len(p.discard)}"
        return [f"{title}: {counts}",
                f"  Active: {active}",
                f"  Bench:  {', '.join(mon_text(m) for m in p.bench) or '-'}"]

    turn = "Setup" if view.phase == Phase.SETUP else f"Turn {view.turn}"
    out = ["", f"================ {turn} ================"]
    out += side(f"OPPONENT {names[1 - view.player]}", view.opponent)
    if view.stadium is not None:
        out.append(f"Stadium: {name(view.stadium)}")
    out += side(f"YOU {names[view.player]}", view.me)
    out.append(f"  Hand:   {', '.join(sorted(name(u) for u in view.me.hand)) or '-'}")
    return out


# ---------------------------------------------------------------- the game loop
def play(deck: str, seed: int, ask=input, say=print) -> State:
    """You are player 0. `ask` and `say` can be replaced, which is how the tests play."""
    other = next(d for d in DECKS if d != deck)
    names = (f"{deck} (you)", f"{other} (computer)")
    state = new_game((deck, other), seed)
    pick = random.Random(seed)
    say(f"You play {deck} against {other}. Seed {seed}. Type help for the commands.")
    say(f"{names[state.pending.player]} won the coin flip.")

    while state.phase != Phase.OVER:
        view = observe(state, 0)
        if view.waiting_for == 0:
            action = _ask_move(state, view, names, ask, say)
            if action is None:
                say("Game stopped.")
                return state
        else:
            action = demo_choice(state, pick)
        after = step(state, action)
        for line in describe(state, action, after, names):
            if not line.startswith("Cards:"):
                say("  " + line)
        drawn = [u for u in after.players[0].hand if u not in state.players[0].hand]
        if drawn and after.phase != Phase.OVER:
            say("  New in your hand: " + ", ".join(card_of(after, u).name for u in drawn))
        state = after

    for line in table(state, observe(state, 0), names):
        say(line)
    result = "It is a draw." if state.winner is None else ("You win!" if state.winner == 0 else "You lose.")
    say(f"\n{result}  ({state.end_reason})")
    return state


def _ask_move(state: State, view: Observation, names, ask, say) -> Action | None:
    moves = menu(state)

    def show() -> None:
        for line in table(state, view, names):
            say(line)
        say("\nYour moves:")
        for number, (text, _) in enumerate(moves, 1):
            say(f"  {number:2d}. {text}")

    show()
    while True:
        try:
            typed = ask("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        command, _, rest = plain(typed).partition(" ")
        if command == "quit":
            return None
        if command == "help":
            say(HELP)
        elif command == "board":
            show()
        elif command == "discard":
            for title, p in (("Yours", view.me), ("Theirs", view.opponent)):
                say(f"{title}: " + (", ".join(catalog().cards[view.card_ids[u]].name for u in p.discard) or "-"))
        elif command == "info" and rest:
            deck_cards = {c for d in DECKS for c in catalog().decks[d]}
            found = [catalog().cards[c] for c in sorted(deck_cards) if rest in plain(catalog().cards[c].name)]
            for card in found:
                say("\n".join(card_text(card)))
            if not found:
                say("No card by that name in these two decks.")
        elif typed:
            chosen = find_moves(moves, typed)
            if len(chosen) == 1:
                return chosen[0][1]
            if chosen:
                say("That fits several moves:")
                for text, action in chosen:
                    say(f"  {moves.index((text, action)) + 1:2d}. {text}")
            else:
                say("No such move. Type a number from the list, or help.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against the computer.")
    parser.add_argument("--deck", choices=DECKS, default="lopunny")
    parser.add_argument("--seed", type=int, default=None, help="repeat a game; random if left out")
    args = parser.parse_args()
    seed = random.randrange(100_000) if args.seed is None else args.seed
    play(args.deck, seed)


if __name__ == "__main__":
    main()
