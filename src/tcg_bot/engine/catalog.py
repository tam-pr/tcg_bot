"""The facts the engine looks up: cards, deck lists and global numbers. Loaded once, never changed."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from tcg_bot.decks import expand_deck

PROC = Path(__file__).resolve().parents[3] / "data" / "processed"


@dataclass(frozen=True)
class Move:
    move_id: str
    name: str
    kind: str                        # "attack", "ability", "tera", ...
    text: str | None                 # None for an attack that only does its printed damage
    damage: int | None
    total_cost: int
    typed_cost: tuple[tuple[str, int], ...]     # (("F", 2),): the Energy that must be of one type


@dataclass(frozen=True)
class Card:
    card_id: int
    name: str
    family: str                      # "pokemon", "trainer" or "energy"
    kind: str                        # "Basic Pokémon", "Item", "Basic Energy", ...
    hp: int | None
    type: str | None
    weakness: str | None
    resistance: str | None
    retreat_cost: int | None
    previous_stage: str | None
    is_ex: bool
    is_mega: bool
    moves: tuple[Move, ...] = ()

    @property
    def is_basic_pokemon(self) -> bool:
        return self.kind == "Basic Pokémon"


@dataclass(frozen=True)
class Catalog:
    cards: dict[int, Card]
    constants: dict[str, int]
    decks: dict[str, tuple[int, ...]]          # deck_id -> 60 card_ids
    energy: dict[int, tuple[str, int]]         # Energy card_id -> (type letter, amount)


def _plain(value):
    return None if pd.isna(value) else value


def _moves(moves: pd.DataFrame) -> dict[int, tuple[Move, ...]]:
    typed = [col for col in moves.columns if col.startswith("cost_") and col not in ("cost_total", "cost_C")]
    out: dict[int, list[Move]] = {}
    for r in moves.sort_values(["card_id", "move_idx"]).to_dict("records"):
        out.setdefault(int(r["card_id"]), []).append(Move(
            move_id=r["move_id"], name=_plain(r["move_name"]), kind=r["move_kind"], text=_plain(r["effect_text"]),
            damage=None if pd.isna(r["damage_base"]) else int(r["damage_base"]),
            total_cost=0 if pd.isna(r["cost_total"]) else int(r["cost_total"]),
            typed_cost=tuple((col[5:], int(r[col])) for col in typed if not pd.isna(r[col]) and r[col] > 0)))
    return {card_id: tuple(found) for card_id, found in out.items()}


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    cards = pd.read_parquet(PROC / "cards.parquet")
    deck_cards = pd.read_parquet(PROC / "deck_cards.parquet")
    constants = pd.read_parquet(PROC / "game_constants.parquet")
    energy = pd.read_parquet(PROC / "energy_provides.parquet")
    moves = _moves(pd.read_parquet(PROC / "moves.parquet"))
    by_id = {}
    for r in cards.itertuples():
        by_id[int(r.card_id)] = Card(
            card_id=int(r.card_id), name=r.card_name, family=r.family, kind=r.card_kind,
            hp=None if pd.isna(r.hp) else int(r.hp), type=_plain(r.type), weakness=_plain(r.weakness),
            resistance=_plain(r.resistance),
            retreat_cost=None if pd.isna(r.retreat_cost) else int(r.retreat_cost),
            previous_stage=_plain(r.previous_stage), is_ex=bool(r.is_ex), is_mega=bool(r.is_mega),
            moves=moves.get(int(r.card_id), ()))
    missing = constants.loc[constants["value"].isna(), "key"].tolist()
    if missing:
        raise ValueError(f"game_constants without a value: {missing}")
    return Catalog(
        cards=by_id,
        constants={r.key: int(r.value) for r in constants.itertuples()},
        decks={d: tuple(c.card_id for c in expand_deck(deck_cards, cards, d))
               for d in deck_cards["deck_id"].unique()},
        energy={int(r.card_id): (r.provides, int(r.amount)) for r in energy.itertuples()})