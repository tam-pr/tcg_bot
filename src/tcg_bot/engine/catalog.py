"""The facts the engine looks up: cards, deck lists and global numbers. Loaded once, never changed."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from tcg_bot.decks import expand_deck

PROC = Path(__file__).resolve().parents[3] / "data" / "processed"


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

    @property
    def is_basic_pokemon(self) -> bool:
        return self.kind == "Basic Pokémon"


@dataclass(frozen=True)
class Catalog:
    cards: dict[int, Card]
    constants: dict[str, int]
    decks: dict[str, tuple[int, ...]]          # deck_id -> 60 card_ids


def _plain(value):
    return None if pd.isna(value) else value


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    cards = pd.read_parquet(PROC / "cards.parquet")
    deck_cards = pd.read_parquet(PROC / "deck_cards.parquet")
    constants = pd.read_parquet(PROC / "game_constants.parquet")
    by_id = {}
    for r in cards.itertuples():
        by_id[int(r.card_id)] = Card(
            card_id=int(r.card_id), name=r.card_name, family=r.family, kind=r.card_kind,
            hp=None if pd.isna(r.hp) else int(r.hp), type=_plain(r.type), weakness=_plain(r.weakness),
            resistance=_plain(r.resistance),
            retreat_cost=None if pd.isna(r.retreat_cost) else int(r.retreat_cost),
            previous_stage=_plain(r.previous_stage), is_ex=bool(r.is_ex), is_mega=bool(r.is_mega))
    missing = constants.loc[constants["value"].isna(), "key"].tolist()
    if missing:
        raise ValueError(f"game_constants without a value: {missing}")
    return Catalog(
        cards=by_id,
        constants={r.key: int(r.value) for r in constants.itertuples()},
        decks={d: tuple(c.card_id for c in expand_deck(deck_cards, cards, d))
               for d in deck_cards["deck_id"].unique()})