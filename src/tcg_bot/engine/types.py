"""What a game state holds and what an action looks like. Data only: the rules live elsewhere.

A card in a game is a `uid`, unique across BOTH players: player 0 owns 0..59 and player 1 owns 60..119.
`State.card_ids[uid]` says which card it is. A Pokémon in play is identified by the uid at the bottom of
its evolution stack, so the id stays the same when it evolves.
"""
import random
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

DECK_SIZE = 60
PLAYERS = (0, 1)


def owner_of(uid: int) -> int:
    return uid // DECK_SIZE


class Phase(StrEnum):
    SETUP = "setup"
    MAIN = "main"
    CHECKUP = "checkup"
    OVER = "over"


# ---------------------------------------------------------------- actions
class ActionKind(StrEnum):
    PLACE_ACTIVE = "place_active"          # card
    PLACE_BENCH = "place_bench"            # card
    SETUP_DONE = "setup_done"
    BENCH = "bench"                        # card
    EVOLVE = "evolve"                      # card, target
    ATTACH_ENERGY = "attach_energy"        # card, target
    PLAY_ITEM = "play_item"                # card
    PLAY_SUPPORTER = "play_supporter"      # card
    PLAY_STADIUM = "play_stadium"          # card
    ATTACH_TOOL = "attach_tool"            # card, target
    RETREAT = "retreat"                    # target = the Benched Pokémon that comes up
    USE_ABILITY = "use_ability"            # target = the Pokémon using it, move_id
    ATTACK = "attack"                      # move_id
    END_TURN = "end_turn"                  # the player declares it
    CHOOSE = "choose"                      # option = one entry of the pending choice
    CHOOSE_DONE = "choose_done"            # stop choosing (allowed once the minimum is met)


@dataclass(frozen=True)
class Action:
    """Plain values only, so actions can be listed, compared, hashed and encoded for the model."""
    kind: ActionKind
    player: int
    card: int | None = None       # uid of the card being used
    target: int | None = None     # id of a Pokémon in play
    move_id: str | None = None    # attack or ability
    option: int | None = None     # answer to a pending choice


# every row of the `plays` table: the action that performs it, or None when the engine does it by itself
PLAYS: dict[str, ActionKind | None] = {
    "setup_coin_flip": None, "setup_draw_hand": None, "setup_mulligan": None,
    "setup_place_active": ActionKind.PLACE_ACTIVE, "setup_place_bench": ActionKind.PLACE_BENCH,
    "setup_set_prizes": None, "turn_draw": None,
    "main_bench_basic": ActionKind.BENCH, "main_evolve": ActionKind.EVOLVE,
    "main_attach_energy": ActionKind.ATTACH_ENERGY, "main_play_item": ActionKind.PLAY_ITEM,
    "main_play_supporter": ActionKind.PLAY_SUPPORTER, "main_play_stadium": ActionKind.PLAY_STADIUM,
    "main_attach_tool": ActionKind.ATTACH_TOOL, "main_retreat": ActionKind.RETREAT,
    "main_use_ability": ActionKind.USE_ABILITY,
    "main_play_fossil": None,                      # not applicable: neither deck has a Fossil
    "attack_use": ActionKind.ATTACK, "turn_end": ActionKind.END_TURN,
    "checkup": None, "knock_out": None, "win_check": None,
}


# ---------------------------------------------------------------- the board
@dataclass(slots=True)
class PokemonInPlay:
    stack: list[int]                          # evolution stack, Basic first, current card last
    in_play_since: int                        # turn it entered play or last evolved
    energy: list[int] = field(default_factory=list)
    tool: int | None = None
    damage: int = 0                           # damage points, not counters
    poisoned: bool = False
    burned: bool = False
    rotation: str | None = None               # "Asleep", "Paralyzed" or "Confused": one at a time
    paralyzed_on_turn: int | None = None
    moved_to_active_on_turn: int | None = None
    abilities_used: set[str] = field(default_factory=set)      # move_ids used this turn (per_pokemon)

    @property
    def uid(self) -> int:
        return self.stack[0]

    @property
    def card(self) -> int:
        return self.stack[-1]

    def clone(self) -> "PokemonInPlay":
        return replace(self, stack=self.stack.copy(), energy=self.energy.copy(),
                       abilities_used=self.abilities_used.copy())


@dataclass(slots=True)
class TurnFlags:
    """Action limits. Reset at the start of the player's turn. Card effects do not use these up."""
    energy_attached: bool = False
    supporter_played: bool = False
    stadium_played: bool = False
    retreated: bool = False
    names_used: set[str] = field(default_factory=set)          # per_name ability keys

    def clone(self) -> "TurnFlags":
        return replace(self, names_used=self.names_used.copy())


@dataclass(slots=True)
class PlayerState:
    deck: list[int] = field(default_factory=list)              # index 0 is the top
    hand: list[int] = field(default_factory=list)
    discard: list[int] = field(default_factory=list)
    prizes: list[int] = field(default_factory=list)            # face down, also to their owner
    playing: list[int] = field(default_factory=list)           # Trainer card being resolved
    active: PokemonInPlay | None = None
    bench: list[PokemonInPlay] = field(default_factory=list)
    flags: TurnFlags = field(default_factory=TurnFlags)
    revealed: set[int] = field(default_factory=set)            # hidden-zone cards the opponent has been shown
    mulligans: int = 0
    knocked_out_on_turn: int | None = None                     # last turn one of its Pokémon was Knocked Out

    def pokemon(self) -> list[PokemonInPlay]:
        return ([self.active] if self.active else []) + self.bench

    def clone(self) -> "PlayerState":
        return replace(self, deck=self.deck.copy(), hand=self.hand.copy(), discard=self.discard.copy(),
                       prizes=self.prizes.copy(), playing=self.playing.copy(),
                       active=self.active.clone() if self.active else None,
                       bench=[p.clone() for p in self.bench], flags=self.flags.clone(),
                       revealed=self.revealed.copy())


# ---------------------------------------------------------------- effects that are not finished yet
@dataclass(frozen=True)
class LastingEffect:
    """A lockout, prevention or stat change that outlives the card text that made it.
    Effects of cards that stay in play (Tools, Stadiums, Energy) are read from the board instead."""
    block: Any                       # a Lockout, Prevent or ModifyStat from tcg_bot.effects
    controller: int
    source: int                      # uid of the card that made it
    target: int | None = None        # id of a Pokémon, or None for a whole side
    first_turn: int = 0
    last_turn: int = 0


@dataclass(frozen=True)
class Frame:
    """The rest of an effect that is still being resolved."""
    controller: int
    source: int
    move_id: str
    steps: tuple                     # blocks from tcg_bot.effects still to run
    facts: frozenset[str] = frozenset()        # e.g. "drew_any", "healed_any"


@dataclass(frozen=True)
class PendingChoice:
    """The game is waiting for `player` to pick. Multiple picks are made one action at a time."""
    player: int
    kind: str                        # e.g. "search_deck", "new_active", "yes_no"
    options: tuple[int, ...]         # uids, Pokémon ids, or 0/1 for yes_no
    min_count: int = 1
    max_count: int = 1
    picked: tuple[int, ...] = ()


# ---------------------------------------------------------------- the whole game
@dataclass(slots=True)
class State:
    card_ids: tuple[int, ...]                       # card_id of each uid; never changes
    players: list[PlayerState]
    rng: random.Random
    phase: Phase = Phase.SETUP
    turn: int = 0                                   # 1 is the first player's first turn
    first_player: int = 0
    current: int = 0
    prize_count: int = 6                            # 1 in Sudden Death
    stadium: int | None = None
    lasting: list[LastingEffect] = field(default_factory=list)
    stack: list[Frame] = field(default_factory=list)
    pending: PendingChoice | None = None
    winner: int | None = None                       # None with phase OVER is a draw
    end_reason: str | None = None

    @classmethod
    def initial(cls, decks: tuple[list[int], list[int]], seed: int) -> "State":
        """All cards in their decks, unshuffled. `decks` holds the 60 card_ids of each player."""
        for p, d in enumerate(decks):
            if len(d) != DECK_SIZE:
                raise ValueError(f"player {p}: deck has {len(d)} cards, expected {DECK_SIZE}")
        players = [PlayerState(deck=list(range(p * DECK_SIZE, (p + 1) * DECK_SIZE))) for p in PLAYERS]
        return cls(card_ids=tuple(decks[0]) + tuple(decks[1]), players=players, rng=random.Random(seed))

    def is_first_turn_of(self, player: int) -> bool:
        return self.turn == (1 if player == self.first_player else 2)

    def all_uids(self) -> list[int]:
        """Every card in every zone, for checking that none is created or lost."""
        out = [] if self.stadium is None else [self.stadium]
        for p in self.players:
            out += p.deck + p.hand + p.discard + p.prizes + p.playing
            for mon in p.pokemon():
                out += mon.stack + mon.energy + ([] if mon.tool is None else [mon.tool])
        return out

    def clone(self) -> "State":
        """An independent copy, including the random generator."""
        rng = random.Random()
        rng.setstate(self.rng.getstate())
        return replace(self, players=[p.clone() for p in self.players], rng=rng,
                       lasting=self.lasting.copy(), stack=self.stack.copy())