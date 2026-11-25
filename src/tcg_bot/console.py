"""Text console (task T9).

Type commands such as ``solrock to active``, ``basic energy to solrock``, ``lunatone to bench``,
``play ultra ball``, ``lucario "aura jab"``, ``use lunar cycle`` or ``end turn``. A command is matched
against the legal actions: every legal action has a canonical command (describe), and a typed command
matches when its words are a subset of a canonical command's words. Ambiguous or illegal commands
explain why and list the options. The opponent's moves are printed in the same style:
``model: lucario to active, lucario "aura jab"``.

H5 (console as simulator or advisor) is not decided: this is a simulator, plus a ``hint`` command
that prints what the advisor agent would play (the advisor mode).
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache

from tcg_bot.agents import Agent, HeuristicAgent, RandomAgent
from tcg_bot.engine import Action, acting_player, legal_actions, new_game, observe, step
from tcg_bot.engine import board as B
from tcg_bot.engine.catalog import tables
from tcg_bot.engine.types import State


class CommandError(ValueError):
    pass


@lru_cache(maxsize=None)
def move_name(move_id: str) -> str:
    m = tables()["moves"].set_index("move_id")
    return str(m.loc[move_id, "move_name"])


def norm(text: str) -> list[str]:
    t = text.lower().replace("’", "'").replace("{", " ").replace("}", " ").replace('"', ' " ')
    t = re.sub(r"[^a-z0-9@#'\" -]", " ", t)
    return [w for w in t.replace('"', " ").split() if w]


# ---------------------------------------------------------------- naming things on the board
def _name(state: State, uid: int) -> str:
    return B.card(state, uid).name.lower()


def pokemon_name(state: State, pid: int, me: int) -> str:
    """Name of a Pokémon in play, made unique on its side with @active / @bench2, prefixed 'opp' for the opponent."""
    p, s = B.find_slot(state, pid)
    base = _name(state, s.top)
    same = [x for x in state.players[p].in_play() if _name(state, x.top) == base]
    if len(same) > 1:
        pos = "active" if state.players[p].active is s else f"bench{state.players[p].bench.index(s) + 1}"
        base = f"{base}@{pos}"
    return base if p == me else f"opp {base}"


def describe(state: State, a: Action) -> str:
    """The canonical command for a legal action."""
    me = a.player
    k = a.kind
    if k == "choose_first":
        return a.choice
    if k == "set_active":
        return f"{_name(state, a.source)} to active"
    if k in ("set_bench", "bench_basic"):
        return f"{_name(state, a.source)} to bench"
    if k == "setup_done":
        return "done"
    if k in ("attach_energy", "attach_tool"):
        return f"{_name(state, a.source)} to {pokemon_name(state, a.target, me)}"
    if k == "evolve":
        return f"{_name(state, a.source)} on {pokemon_name(state, a.target, me)}"
    if k in ("play_item", "play_supporter", "play_stadium"):
        return f"play {_name(state, a.source)}"
    if k == "retreat":
        return f"{pokemon_name(state, a.target, me)} to active"
    if k == "use_ability":
        return f"use {move_name(a.choice).lower()}"
    if k == "attack":
        return f'{pokemon_name(state, a.source, me)} "{move_name(a.choice).lower()}"'
    if k == "end_turn":
        return "end turn"
    if k == "choose":
        pk = state.pending.kind
        c = a.choice
        if c == state.pending.stop:
            return "stop"
        if pk == "use_ability":
            return c
        if pk in ("promote", "switch_target"):
            return f"{pokemon_name(state, c, me)} to active"
        if pk == "heal_target":
            return f"heal {pokemon_name(state, c, me)}"
        if pk == "retreat_discard":
            return f"discard {_name(state, c)}"
        if isinstance(c, tuple):
            return f"{_name(state, c[0])} to {pokemon_name(state, c[1], me)}"
        if pk.endswith(">discard"):
            if pk.startswith("move:attached_tool_any"):
                _, s = next((p, s) for p in (0, 1) for s in state.players[p].in_play() if s.tool == c)
                return f"discard {_name(state, c)} from {pokemon_name(state, s.pid, me)}"
            return f"discard {_name(state, c)}"
        if pk.endswith(">bench"):
            return f"{_name(state, c)} to bench"
        return f"take {_name(state, c)}"
    raise CommandError(f"cannot describe {a}")


# ---------------------------------------------------------------- parsing
def candidates(state: State) -> dict[str, list[Action]]:
    out: dict[str, list[Action]] = {}
    for a in legal_actions(state):
        out.setdefault(describe(state, a), []).append(a)
    return out


def why_not(state: State, words: list[str]) -> str:
    p = acting_player(state)
    f = state.flags
    text = " ".join(words)
    if state.phase == "main" and state.pending is None:
        if "energy" in words and f.get("energy_attached"):
            return "you already attached an Energy from your hand this turn"
        if text.startswith("play") and f.get("supporter_played") and any(
                B.card(state, u).kind == "Supporter" and set(norm(B.card(state, u).name)) <= set(words) for u in state.players[p].hand):
            return "you already played a Supporter this turn"
        if '"' in text or any(w in ("attack",) for w in words):
            if p == state.first_player and state.players[p].turns_taken == 1:
                return "the player who goes first cannot attack on their first turn"
        if words and words[-1] == "active" and f.get("retreated"):
            return "you already retreated this turn"
    known = {w for u in state.catalog.refs for w in norm(B.card(state, u).name)}
    unknown = [w for w in words if w not in known and w not in ("to", "on", "active", "bench", "play", "use", "opp",
                                                                 "take", "discard", "heal", "end", "turn", "done", "stop")]
    if unknown:
        return f"unknown word(s): {', '.join(unknown)}"
    return "that is not a legal move right now"


def parse(state: State, line: str) -> Action:
    words = norm(line)
    if not words:
        raise CommandError("type a command, or 'help'")
    cands = candidates(state)
    exact = [c for c in cands if norm(c) == words]
    if exact:
        return cands[exact[0]][0]
    matches = [c for c in cands if set(words) <= set(norm(c))]
    if len(matches) == 1:
        return cands[matches[0]][0]
    if len(matches) > 1:
        raise CommandError("ambiguous; did you mean one of:\n  " + "\n  ".join(sorted(matches)))
    raise CommandError(f"{why_not(state, words)}. Legal commands:\n  " + "\n  ".join(sorted(cands)))


# ---------------------------------------------------------------- rendering the board
def render(state: State, me: int) -> str:
    o = observe(state, me)
    def slot(v, mine):
        if v is None:
            return "-"
        name = state.catalog.refs[v.cards[-1]].card.name
        en = ",".join(state.catalog.refs[u].card.name.replace("Basic ", "").replace(" Energy", "") for u in v.energy)
        extra = f" [{en}]" if en else ""
        tool = f" +{state.catalog.refs[v.tool].card.name}" if v.tool is not None else ""
        cond = f" ({','.join(v.conditions)})" if v.conditions else ""
        return f"{name} {v.hp - v.damage}/{v.hp}{extra}{tool}{cond}"
    lines = [f"turn {o.turn} | {'your' if o.current == me else 'opponent'} turn"]
    lines.append(f"opponent: hand {o.opp_hand_count} deck {o.opp_deck_count} prizes {o.opp_prize_count} "
                 f"discard {len(o.opp_discard)}")
    lines.append(f"  active: {slot(o.opp_active, False)}")
    lines.append("  bench: " + " | ".join(slot(v, False) for v in o.opp_bench))
    if o.stadium is not None:
        lines.append(f"stadium: {state.catalog.refs[o.stadium].card.name}")
    lines.append(f"you: deck {o.my_deck_count} prizes {o.my_prize_count} discard {len(o.my_discard)}")
    lines.append(f"  active: {slot(o.my_active, True)}")
    lines.append("  bench: " + " | ".join(slot(v, True) for v in o.my_bench))
    lines.append("  hand: " + ", ".join(state.catalog.refs[u].card.name for u in o.my_hand))
    return "\n".join(lines)


# ---------------------------------------------------------------- the console
class Console:
    """Simulator: a human plays seat `human` against `opponent`. Advisor: `hint` shows the advisor's move."""

    def __init__(self, decks=("mega_lucario", "lopunny"), seed: int = 0, human: int = 0,
                 opponent: Agent | None = None, advisor: Agent | None = None):
        self.state = new_game(decks, seed)
        self.human = human
        self.opponent = opponent or HeuristicAgent(seed)
        self.advisor = advisor or HeuristicAgent(seed + 1)
        self.opponent.reset(seed)

    def _opponent_moves(self) -> str:
        said = []
        while legal_actions(self.state) and acting_player(self.state) != self.human:
            p = acting_player(self.state)
            legal = legal_actions(self.state)
            a = self.opponent.act(observe(self.state, p), legal)
            said.append(describe(self.state, a))
            self.state = step(self.state, a)
        return ("model: " + ", ".join(said)) if said else ""

    def start(self) -> str:
        return self._opponent_moves()

    def over(self) -> bool:
        return not legal_actions(self.state)

    def result(self) -> str:
        s = self.state
        if s.winner is None:
            return f"game over: draw ({s.result})"
        return f"game over: {'you win' if s.winner == self.human else 'you lose'} ({s.result})"

    def handle(self, line: str) -> str:
        """Process one typed line and return the reply text."""
        cmd = line.strip().lower()
        if self.over():
            return self.result()
        if cmd in ("help", "?", "moves"):
            return "legal commands:\n  " + "\n  ".join(sorted(candidates(self.state)))
        if cmd in ("board", "show"):
            return render(self.state, self.human)
        if cmd in ("hint", "advise"):
            a = self.advisor.act(observe(self.state, self.human), legal_actions(self.state))
            return f"advisor: {describe(self.state, a)}"
        try:
            a = parse(self.state, line)
        except CommandError as e:
            return str(e)
        self.state = step(self.state, a)
        reply = []
        out = self._opponent_moves()
        if out:
            reply.append(out)
        if self.over():
            reply.append(self.result())
        return "\n".join(reply) if reply else "ok"


def _make_agent(name: str, seed: int) -> Agent:
    return {"random": RandomAgent, "heuristic": HeuristicAgent}[name](seed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="tcg-bot", description="Play the Pokémon TCG against a local agent.")
    ap.add_argument("--deck", default="mega_lucario")
    ap.add_argument("--opp-deck", default="lopunny")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opponent", default="heuristic", choices=["random", "heuristic"])
    args = ap.parse_args(argv)
    c = Console((args.deck, args.opp_deck), args.seed, 0, _make_agent(args.opponent, args.seed))
    print("Type 'help' for legal commands, 'board' to see the board, 'hint' for advice, 'quit' to leave.")
    first = c.start()
    if first:
        print(first)
    while not c.over():
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip().lower() in ("quit", "exit"):
            break
        print(c.handle(line))
    if c.over():
        print(c.result())
