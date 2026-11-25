"""Engine interface types (task T1).

Everything later tasks build on lives here: the game State, the Action an agent picks, the
Observation a player is allowed to see, the static card Catalog and the RuleSettings that hold
every rule value that is not yet confirmed.

Design rules (see reports/engine_interface.md):
- Cards are instances, never card IDs. A card instance has a uid that is unique across both
  players: player p's i-th deck card has uid ``p * UID_STRIDE + i``. A mirror match never collides.
- A Pokémon in play is identified by its ``pid``: the uid of the Basic card at the bottom of its
  evolution stack. The pid does not change when the Pokémon evolves.
- Actions are plain values (strings, ints, tuples), so they can be listed, compared, hashed and encoded.
- A multi-step choice is a Pending choice on the State, answered by further ``choose`` actions.
- State.copy() is cheap: it copies the mutable parts and shares the immutable catalog and settings.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

UID_STRIDE = 100          # player p owns uids p*100 .. p*100+59
NO_PLAYER = -1

# ---------------------------------------------------------------- action kinds
SETUP_KINDS = ("choose_first", "set_active", "set_bench", "setup_done")
MAIN_KINDS = ("bench_basic", "evolve", "attach_energy", "play_item", "play_supporter", "play_stadium",
              "attach_tool", "retreat", "use_ability", "attack", "end_turn")
CHOICE_KINDS = ("choose",)
ACTION_KINDS = SETUP_KINDS + MAIN_KINDS + CHOICE_KINDS

PHASES = ("setup", "main", "over")
RESULTS = ("prizes", "no_pokemon", "deck_out", "draw_turn_limit", "sudden_death_prizes",
           "sudden_death_no_pokemon", "sudden_death_deck_out", "both_lose_draw")


@dataclass(frozen=True, slots=True)
class Action:
    """One decision. ``source`` and ``target`` are card uids (a Pokémon in play is named by its pid);
    ``choice`` is a move_id, an option value of a Pending choice, or a short string such as "first"."""
    kind: str
    player: int
    source: int | None = None
    target: int | None = None
    choice: Any = None

    def __post_init__(self):
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}")
        if self.player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {self.player!r}")


# ---------------------------------------------------------------- static card data
@dataclass(frozen=True, slots=True)
class Attack:
    move_id: str
    name: str
    cost: tuple[tuple[str, int], ...]     # (("F", 1), ("C", 2)) ; "C" = any type
    damage: int                           # printed number, 0 when none
    damage_mod: str | None                # "flat", "times", "plus", "minus" or None


@dataclass(frozen=True, slots=True)
class CardStatic:
    """What every copy of a card shares. Built from the cards and moves tables."""
    card_id: int
    name: str
    family: str                 # pokemon | trainer | energy
    kind: str                   # Basic Pokémon, Stage 1 Pokémon, Item, Supporter, Basic Energy, ...
    evolves_from: str | None
    hp: int | None
    type: str | None            # "{F}" style, as in the cards table
    weakness: str | None
    resistance: str | None
    retreat_cost: int | None
    is_ex: bool
    is_mega: bool
    has_rule_box: bool
    is_ace_spec: bool
    attacks: tuple[Attack, ...]
    ability_ids: tuple[str, ...]
    effect_ids: tuple[str, ...]           # trainer or energy effect move_ids
    provides: tuple[str, ...]             # Energy types the card provides when attached ("F",) / ("C",)

    @property
    def is_basic_pokemon(self) -> bool:
        return self.kind == "Basic Pokémon"

    @property
    def stage(self) -> int:
        return {"Basic Pokémon": 0, "Stage 1 Pokémon": 1, "Stage 2 Pokémon": 2}.get(self.kind, -1)


@dataclass(frozen=True, slots=True)
class CardRef:
    """One card instance: its uid, its owner, a readable tag and its static data."""
    uid: int
    owner: int
    tag: str                    # p0_lopunny_pokemon_3
    card: CardStatic


@dataclass(frozen=True)
class Catalog:
    """Immutable per-game table uid -> CardRef, plus the two deck ids. Shared by every State copy."""
    decks: tuple[str, str]
    refs: dict[int, CardRef]
    effects: dict[str, Any]               # move_id -> built tcg_bot.effects.Effect
    constants: dict[str, int]             # verified numbers from the game_constants table
    deck_lists: tuple[tuple[int, ...], tuple[int, ...]]   # each player's 60 card_ids, in list order

    def card(self, uid: int) -> CardStatic:
        return self.refs[uid].card

    def owner(self, uid: int) -> int:
        return self.refs[uid].owner


# ---------------------------------------------------------------- rule settings
@dataclass(frozen=True)
class RuleSettings:
    """Rule values that are configurable. Every value marked ASSUMPTION is unconfirmed (task H1) and
    must be checked by a person; the engine reads it from here so it is never a buried number."""
    turn_limit: int = 200                        # global turns; reaching it ends the game as a draw
    # ASSUMPTION H1(a): evolving into a Mega Evolution Pokémon ex does not end the turn
    mega_evolution_ends_turn: bool = False
    # ASSUMPTION H1(b): Prize cards taken for Knocking Out a Mega Evolution Pokémon ex
    mega_ex_prize_value: int = 3
    # ASSUMPTION H1(c): Resistance subtracts this much damage
    resistance_amount: int = 30
    # ASSUMPTION H1(d): a player may end the turn without attacking
    may_end_turn_without_attack: bool = True
    # ASSUMPTION H1(e): Energy attached by an effect does not use the once-per-turn attach from hand
    effect_attach_uses_hand_attach: bool = False
    # ASSUMPTION H1(f): cards put on the bottom of the deck after being shuffled go in random order
    bottom_of_deck_order: str = "random"
    # ASSUMPTION H1(g): Hilda may find only one of its two cards (searching may come up short)
    search_may_find_fewer: bool = True
    # ASSUMPTION H1(h): Wild Press recoil damage to itself does not apply Weakness or Resistance
    recoil_applies_weakness_resistance: bool = False
    # ASSUMPTION (setup_mulligan says "may"): the opponent always takes the extra mulligan cards
    take_mulligan_bonus_cards: bool = True

    def assumptions(self) -> dict[str, Any]:
        """The values that still need confirmation, for reports and logs."""
        return {k: getattr(self, k) for k in (
            "mega_evolution_ends_turn", "mega_ex_prize_value", "resistance_amount",
            "may_end_turn_without_attack", "effect_attach_uses_hand_attach", "bottom_of_deck_order",
            "search_may_find_fewer", "recoil_applies_weakness_resistance", "take_mulligan_bonus_cards")}


# ---------------------------------------------------------------- mutable game state
@dataclass(slots=True)
class Slot:
    """A Pokémon in play (Active or Bench)."""
    pid: int                                   # uid of the bottom (Basic) card
    cards: list[int]                           # evolution stack, bottom first, top is the current card
    energy: list[int] = field(default_factory=list)
    tool: int | None = None
    damage: int = 0
    conditions: set[str] = field(default_factory=set)
    entered_turn: int = 0                      # turn it came into play
    evolved_turn: int = -1                     # turn it last evolved
    moved_to_active_turn: int = -1             # turn it moved from the Bench to the Active Spot
    paralyzed_turn: int = -1                   # turn it was Paralyzed (removal is tied to its owner's next turn)
    lockouts: dict[str, int] = field(default_factory=dict)    # move_id -> last turn it is locked
    used: dict[str, int] = field(default_factory=dict)        # ability move_id -> turn it was used

    @property
    def top(self) -> int:
        return self.cards[-1]

    def copy(self) -> "Slot":
        return Slot(self.pid, list(self.cards), list(self.energy), self.tool, self.damage, set(self.conditions),
                    self.entered_turn, self.evolved_turn, self.moved_to_active_turn, self.paralyzed_turn,
                    dict(self.lockouts), dict(self.used))


@dataclass(slots=True)
class PlayerState:
    deck: list[int] = field(default_factory=list)      # index -1 is the top of the deck
    hand: list[int] = field(default_factory=list)
    discard: list[int] = field(default_factory=list)
    prizes: list[int] = field(default_factory=list)
    active: Slot | None = None
    bench: list[Slot] = field(default_factory=list)
    mulligans: int = 0
    ko_turns: list[int] = field(default_factory=list)  # turns in which one of this player's Pokémon was KO'd
    setup_done: bool = False
    turns_taken: int = 0

    def in_play(self) -> list[Slot]:
        return ([self.active] if self.active else []) + self.bench

    def copy(self) -> "PlayerState":
        return PlayerState(list(self.deck), list(self.hand), list(self.discard), list(self.prizes),
                           self.active.copy() if self.active else None, [s.copy() for s in self.bench],
                           self.mulligans, list(self.ko_turns), self.setup_done, self.turns_taken)


@dataclass(slots=True)
class Pending:
    """A choice the engine is waiting for. Each option becomes Action("choose", player, choice=option).
    ``stop`` (if not None) is an extra option value meaning "done / take no more"."""
    player: int
    kind: str
    options: tuple
    label: str = ""
    stop: Any = None
    visible_to_opponent: bool = False          # e.g. promoting a new Active is public, a deck search is not

    def copy(self) -> "Pending":
        return Pending(self.player, self.kind, tuple(self.options), self.label, self.stop, self.visible_to_opponent)


@dataclass(slots=True)
class Frame:
    """One entry of the resumable interpreter stack. ``kind`` names who runs it ("effect", "core:...").
    ``data`` holds plain values only, so copying a State copies the program position too."""
    kind: str
    data: dict

    def copy(self) -> "Frame":
        return Frame(self.kind, _copy_plain(self.data))


def _copy_plain(x):
    if isinstance(x, dict):
        return {k: _copy_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_copy_plain(v) for v in x]
    if isinstance(x, set):
        return set(x)
    return x                                   # ints, strs, tuples, frozen dataclasses are shared


@dataclass(slots=True)
class State:
    catalog: Catalog
    settings: RuleSettings
    players: list[PlayerState]
    rng: random.Random
    phase: str = "setup"
    turn: int = 0                               # 0 during setup, then 1, 2, ... (one per player turn)
    current: int = 0                            # player whose turn it is (or who acts in setup)
    first_player: int = NO_PLAYER
    stadium: int | None = None                  # uid of the Stadium in play
    flags: dict = field(default_factory=dict)   # per-turn flags, cleared at the start of each turn
    lasting: list = field(default_factory=list) # lasting effects: dicts with "kind", "until_turn", ...
    pending: Pending | None = None
    frames: list[Frame] = field(default_factory=list)
    winner: int | None = None
    result: str | None = None
    sudden_death: bool = False
    action_count: int = 0
    public: list = field(default_factory=list)  # public events: (turn, player, what, card_id)

    def copy(self) -> "State":
        rng = random.Random()
        rng.setstate(self.rng.getstate())
        return State(self.catalog, self.settings, [p.copy() for p in self.players], rng, self.phase, self.turn,
                     self.current, self.first_player, self.stadium, _copy_plain(self.flags),
                     _copy_plain(self.lasting), self.pending.copy() if self.pending else None,
                     [f.copy() for f in self.frames], self.winner, self.result, self.sudden_death,
                     self.action_count, list(self.public))

    @property
    def is_over(self) -> bool:
        return self.phase == "over"


# ---------------------------------------------------------------- what a player may see
@dataclass(frozen=True, slots=True)
class SlotView:
    pid: int
    cards: tuple[int, ...]
    card_ids: tuple[int, ...]
    energy: tuple[int, ...]
    energy_card_ids: tuple[int, ...]
    tool: int | None
    tool_card_id: int | None
    damage: int
    hp: int                                   # current max HP after modifiers
    conditions: tuple[str, ...]
    entered_turn: int = 0
    evolved_turn: int = -1
    moved_to_active_turn: int = -1
    paralyzed_turn: int = -1
    lockouts: tuple[tuple[str, int], ...] = ()
    used: tuple[tuple[str, int], ...] = ()
    retreat_cost: int = 0                     # after modifiers


@dataclass(frozen=True, slots=True)
class Observation:
    """Only what ``player`` may see. Hidden: the order and content of both decks, both players'
    Prize cards, the opponent's hand, and the other player's private choices."""
    player: int
    phase: str
    turn: int
    current: int
    first_player: int
    my_hand: tuple[int, ...]
    my_hand_card_ids: tuple[int, ...]
    my_deck_count: int
    my_prize_count: int
    my_discard: tuple[int, ...]
    my_discard_card_ids: tuple[int, ...]
    my_active: SlotView | None
    my_bench: tuple[SlotView, ...]
    opp_hand_count: int
    opp_deck_count: int
    opp_prize_count: int
    opp_discard: tuple[int, ...]
    opp_discard_card_ids: tuple[int, ...]
    opp_active: SlotView | None
    opp_bench: tuple[SlotView, ...]
    stadium: int | None
    stadium_card_id: int | None
    flags: tuple[tuple[str, Any], ...]        # this turn's public flags
    pending_kind: str | None                  # kind of the pending choice, if any
    pending_player: int | None
    pending_options: tuple                    # only filled when the pending choice is this player's
    pending_option_card_ids: tuple            # card_id per option (top card for a Pokémon; a pair for (card, Pokémon))
    my_deck_list_card_ids: tuple[int, ...]    # the player's own 60-card list (known before the game)
    public_log: tuple[tuple[int, int, str, int], ...]   # (turn, player, what, card_id): cards revealed or played
    resolving: tuple[tuple[int, int], ...]    # (uid, card_id) of Trainer cards being played right now (public)
    lasting: tuple                            # public lasting effects (prevention, damage bonuses), as sorted item tuples
    turns_taken: tuple[int, int]              # turns each player has had (public)
    ko_turns: tuple[tuple[int, ...], tuple[int, ...]]   # turns in which each player had a Pokémon Knocked Out
    winner: int | None
    result: str | None
