"""Turn a deck's rows in deck_cards into 60 individual, tagged card instances."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CardInstance:
    uid: int      # 0..59, position in the expanded deck (fast to index with)
    tag: str      # readable label: <deck>_<family>_<n>, numbered in .txt order
    card_id: int  # row in the cards table


def expand_deck(deck_cards: pd.DataFrame, cards: pd.DataFrame, deck_id: str) -> list[CardInstance]:
    rows = deck_cards[deck_cards["deck_id"] == deck_id].sort_values("line_order")
    if rows.empty:
        raise ValueError(f"unknown deck_id {deck_id!r}")
    family = cards.set_index("card_id")["family"]
    counters: dict[str, int] = {}
    out: list[CardInstance] = []
    for row in rows.itertuples():
        for _ in range(int(row.copies)):
            fam = family[int(row.card_id)]
            counters[fam] = counters.get(fam, 0) + 1
            out.append(CardInstance(uid=len(out), tag=f"{deck_id}_{fam}_{counters[fam]}", card_id=int(row.card_id)))
    return out