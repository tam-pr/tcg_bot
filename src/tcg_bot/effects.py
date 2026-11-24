"""Effect vocabulary: reusable building blocks, validated when built from plain data.

An effect is DATA (a dict). build() turns it into frozen dataclass objects and rejects unknown
blocks, unknown parameters and values outside the allowed vocabulary, so a tagging mistake
fails loudly instead of being ignored.
"""
from dataclasses import dataclass, fields
from typing import Any

# ---------------------------------------------------------------- allowed vocabulary
SIDES = {"you", "opponent", "both", "that_player"}
SRC = {"deck", "discard", "hand", "top_of_deck", "prizes", "attached_to_this", "attached_to_target",   # <- "prizes" added
       "this_pokemon_and_attached", "stadium_in_play", "attached_tool_any"}
DST = {"hand", "bench", "deck_shuffle", "deck_bottom", "discard", "prizes", "attached_to_bench"}       # <- "prizes" added
TIMINGS = {"attack", "activated", "on_play_bench", "on_evolve", "on_attach_from_hand", "passive", "trainer_play"}
STATUSES = {"ok", "complex", "no_effect_in_scope"}
DURATIONS = {None, "this_turn", "opponent_next_turn", "your_next_turn"}
STATS = {"retreat_cost", "hp", "attack_damage"}
PREVENT_WHAT = {"damage_and_effects", "attack_effects", "damage_counters"}
PREVENT_TO = {"this_pokemon", "attached_pokemon", "all_benched"}
PREVENT_FROM = {"opponent_attacks", "opponent_attacks_and_abilities"}
COND_NAMES = {"has_in_play", "has_on_bench", "stadium_in_play", "moved_to_active_this_turn", "prizes_remaining_equals",
              "opponent_prizes_at_most", "own_pokemon_knocked_out_last_opponent_turn", "drew_any", "healed_any",
              "either_player_put_cards_on_bottom", "they_put_cards_on_bottom"}
COUNTERS = {"opp_hand_size"}
USAGE_SCOPES = {"none", "per_pokemon", "per_name", "per_name_contains"}
STATUS_CONDITIONS = {"Asleep", "Burned", "Confused", "Paralyzed", "Poisoned"}

def _check(value, allowed, what):
    if value not in allowed:
        raise ValueError(f"{what}: {value!r} is not one of {sorted(map(str, allowed))}")

# ---------------------------------------------------------------- amounts (numbers that can depend on the game)
@dataclass(frozen=True)
class Fixed:
    n: int
@dataclass(frozen=True)
class PerCount:                       # the multiplier
    each: int
    counter: str
    def __post_init__(self): _check(self.counter, COUNTERS, "counter")
@dataclass(frozen=True)
class Sum:                            # the sum
    parts: tuple
@dataclass(frozen=True)
class Cond:                           # a yes/no question about the state
    name: str
    arg: Any = None
    negate: bool = False
    def __post_init__(self): _check(self.name, COND_NAMES, "condition")
@dataclass(frozen=True)
class IfThen:                         # a number that depends on a condition
    condition: Cond
    then: Any
    otherwise: Any = Fixed(0)

# ---------------------------------------------------------------- card selectors and targets
@dataclass(frozen=True)
class Filter:
    card_kind: tuple = ()
    family: str | None = None
    type: str | None = None
    hp_max: int | None = None
    rule_box: bool | None = None
    is_mega: bool | None = None
    name: str | None = None
    any_of: tuple = ()
@dataclass(frozen=True)
class Target:                         # "1 of your Mega Evolution Pokémon ex", "this Pokémon", ...
    which: str = "this_pokemon"       # this_pokemon | your_active | opponent_active | chosen
    side: str = "you"
    filter: Filter | None = None
    def __post_init__(self): _check(self.side, SIDES, "side")

# ---------------------------------------------------------------- steps (the verbs)
@dataclass(frozen=True)
class MoveCards:                      # search, recover, shuffle hand, discard, return to hand, attach from discard...
    source: str
    destination: str
    filter: Filter | None = None
    count: Any = 1                    # a number, or "all"
    up_to: bool = False
    optional: bool = False
    reveal: bool = False
    shuffle_after: bool = False
    side: str = "you"
    look_n: int | None = None         # look at the top N cards only
    rest_to: str | None = None        # where the cards not taken go
    others_only: bool = False         # "2 OTHER cards" (not the card being played)
    distribution: str | None = None   # "any" = split between Pokémon however you like
    def __post_init__(self):
        _check(self.source, SRC, "source"); _check(self.destination, DST, "destination"); _check(self.side, SIDES, "side")
@dataclass(frozen=True)
class Draw:
    count: Any
    side: str = "you"
    def __post_init__(self): _check(self.side, SIDES, "side")
@dataclass(frozen=True)
class SwitchActive:
    side: str = "you"                 # whose Active Pokémon gets switched
    target: str = "your_active"       # this_pokemon | your_active
    chosen_by: str = "you"
    def __post_init__(self): _check(self.side, SIDES, "side")
@dataclass(frozen=True)
class Heal:
    target: Target
    amount: Any = "all"
@dataclass(frozen=True)
class ApplyStatus:
    status: str
    target: Target
    def __post_init__(self): _check(self.status, STATUS_CONDITIONS, "status")
@dataclass(frozen=True)
class DamageSelf:
    amount: int
@dataclass(frozen=True)
class Lockout:                        # "this Pokémon can't use X during your next turn"
    attack: str
    duration: str
    def __post_init__(self): _check(self.duration, DURATIONS, "duration")
@dataclass(frozen=True)
class Prevent:
    what: str
    to: str
    source: str
    duration: str | None = None       # None = lasts while the card is in play
    def __post_init__(self):
        _check(self.what, PREVENT_WHAT, "prevent.what"); _check(self.to, PREVENT_TO, "prevent.to")
        _check(self.source, PREVENT_FROM, "prevent.source"); _check(self.duration, DURATIONS, "duration")
@dataclass(frozen=True)
class ModifyStat:                     # standing or this-turn numeric changes
    stat: str
    amount: int
    applies_to: Filter | None = None
    side: str = "you"                 # you | opponent | both | attached
    duration: str | None = None       # None = while in play
    def __post_init__(self):
        _check(self.stat, STATS, "stat"); _check(self.duration, DURATIONS, "duration")
@dataclass(frozen=True)
class CoinFlip:
    heads: tuple = ()
    tails: tuple = ()
    side: str = "you"
@dataclass(frozen=True)
class If:
    condition: Cond
    then: tuple = ()
    otherwise: tuple = ()
@dataclass(frozen=True)
class ForEachPlayer:
    steps: tuple = ()

# ---------------------------------------------------------------- the effect wrapper
@dataclass(frozen=True)
class Usage:                          # the ability-limit tags from gap 3
    scope: str = "none"
    key: str | None = None
    first_turn_only: bool = False
    def __post_init__(self): _check(self.scope, USAGE_SCOPES, "usage scope")
@dataclass(frozen=True)
class Effect:
    move_id: str
    timing: str
    usage: Usage = Usage()
    optional: bool = False            # "you may"
    requires: tuple = ()              # Conds that must hold to play/use it
    cost: tuple = ()                  # steps paid before the effect
    damage: Any = None                # an amount, only when the text changes the printed number
    ignore_defender_effects: bool = False
    ignore_weakness_resistance: bool = False
    works_only_if: tuple = ()         # "...this attack does nothing" unless these hold
    steps: tuple = ()
    status: str = "ok"
    notes: str = ""
    def __post_init__(self): _check(self.timing, TIMINGS, "timing"); _check(self.status, STATUSES, "status")

REGISTRY = {c.__name__: c for c in (Fixed, PerCount, Sum, Cond, IfThen, Filter, Target, MoveCards, Draw, SwitchActive, Heal,
            ApplyStatus, DamageSelf, Lockout, Prevent, ModifyStat, CoinFlip, If, ForEachPlayer, Usage, Effect)}
NAMES = {"fixed": Fixed, "per_count": PerCount, "sum": Sum, "cond": Cond, "if_then": IfThen, "filter": Filter, "target": Target,
         "move_cards": MoveCards, "draw": Draw, "switch_active": SwitchActive, "heal": Heal, "apply_status": ApplyStatus,
         "damage_self": DamageSelf, "lockout": Lockout, "prevent": Prevent, "modify_stat": ModifyStat, "coin_flip": CoinFlip,
         "if": If, "for_each_player": ForEachPlayer, "usage": Usage, "effect": Effect}

def build(x):
    """Plain data -> block objects. Unknown block names or parameters raise."""
    if isinstance(x, dict):
        cls = NAMES.get(x.get("block"))
        if cls is None:
            raise ValueError(f"unknown block {x.get('block')!r}")
        allowed = {f.name for f in fields(cls)}
        extra = set(x) - allowed - {"block"}
        if extra:
            raise ValueError(f"{x['block']}: unknown parameter(s) {sorted(extra)}")
        return cls(**{k: build(v) for k, v in x.items() if k != "block"})
    if isinstance(x, list):
        return tuple(build(v) for v in x)
    return x