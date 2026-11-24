"""Checks that the hand-written effect data agrees with the card text it was written from,
plus guards for the two effects that are not plain data."""
import dataclasses
import re
from pathlib import Path

import pandas as pd
import pytest

from tcg_bot.effects import (CoinFlip, Cond, Draw, DamageSelf, Filter, Fixed, ForEachPlayer, Lockout,
                             ModifyStat, MoveCards, PerCount, Prevent, build)
from tcg_bot.effects_data import EFFECTS

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
cards = pd.read_parquet(PROC / "cards.parquet")
moves = pd.read_parquet(PROC / "moves.parquet").set_index("move_id")
built = {o.move_id: o for o in (build(e) for e in EFFECTS)}
WITH_TEXT = [mid for mid in built if pd.notna(moves.loc[mid, "effect_text"])]


def walk(x):
    if dataclasses.is_dataclass(x):
        yield x
        for f in dataclasses.fields(x):
            yield from walk(getattr(x, f.name))
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from walk(y)


def data_numbers(o):
    nums = set()
    for n in walk(o):
        if isinstance(n, Fixed): nums.add(abs(n.n))
        if isinstance(n, PerCount): nums.add(n.each)
        if isinstance(n, MoveCards):
            if isinstance(n.count, int): nums.add(n.count)
            if n.look_n: nums.add(n.look_n)
        if isinstance(n, Draw) and isinstance(n.count, int): nums.add(n.count)
        if isinstance(n, DamageSelf): nums.add(n.amount)
        if isinstance(n, ModifyStat): nums.add(abs(n.amount))
        if isinstance(n, Filter) and n.hp_max: nums.add(n.hp_max)
        if isinstance(n, Cond) and isinstance(n.arg, int): nums.add(n.arg)
    return nums


# effects where a check is allowed to differ, and why
NUMBERS_OK = {
    "849_0": "Gale Thrust repeats the printed 60 (checked separately) and a default 0",
    "1174_0": "Air Balloon writes the amount as two {C} symbols, the data says -2",
}


@pytest.mark.parametrize("mid", WITH_TEXT)
def test_numbers_in_the_text_match_the_numbers_in_the_data(mid):
    if mid in NUMBERS_OK:
        pytest.skip(NUMBERS_OK[mid])
    text = re.sub(r"Stage 2", "Stage", moves.loc[mid, "effect_text"])
    text_nums = {int(x) for x in re.findall(r"\b\d+\b", text)} - {1}
    assert text_nums == data_numbers(built[mid]) - {1}


@pytest.mark.parametrize("mid", WITH_TEXT)
def test_key_words_in_the_text_match_the_parameters(mid):
    o, low = built[mid], moves.loc[mid, "effect_text"].lower()
    steps = [n for n in walk(o) if isinstance(n, MoveCards)]
    assert ("reveal" in low) == any(n.reveal for n in steps), "reveal"
    assert bool(re.search(r"shuffles? .{0,60}into (your|their) deck|shuffle your deck|shuffle the other cards back into your deck", low)) == \
        any(n.shuffle_after or n.rest_to == "deck_shuffle" or n.destination == "deck_shuffle" for n in steps), "shuffle"
    assert ("up to" in low) == any(n.up_to for n in steps), "up to"
    assert bool(re.search(r"\byou may\b", low)) == (o.optional or any(n.optional for n in steps)), "you may"
    assert ("opponent's next turn" in low) == any(isinstance(n, Prevent) and n.duration == "opponent_next_turn" for n in walk(o)), "opponent's next turn"
    assert ("your next turn" in low) == any(isinstance(n, Lockout) and n.duration == "your_next_turn" for n in walk(o)), "your next turn"
    assert ("during this turn" in low) == any(isinstance(n, ModifyStat) and n.duration == "this_turn" for n in walk(o)), "this turn"
    assert ("flip" in low) == any(isinstance(n, CoinFlip) for n in walk(o)), "coin flip"


def test_gale_thrust_base_matches_the_printed_damage():
    assert built["849_0"].damage.parts[0].n == moves.loc["849_0", "damage_base"]


# ---- Psyduck's Damp switches off Abilities that make the Pokémon Knock itself Out.
KO_ITSELF = re.compile(r"this Pokémon is Knocked Out\.|Knock Out this Pokémon")


def test_damp_has_nothing_to_switch_off_in_these_decks():
    """If a deck is added whose ability Knocks its own Pokémon Out, this fails and Damp must be implemented."""
    in_scope = moves.loc[list(built)]
    abilities = in_scope[(in_scope["move_kind"] == "ability") & (in_scope.index != "858_0")]
    hits = abilities[abilities["effect_text"].str.contains(KO_ITSELF, regex=True, na=False)]
    assert hits.empty, hits[["move_name", "effect_text"]]
    assert built["858_0"].status == "no_effect_in_scope"


# ---- Lucian is plain data too: both hands go to the bottom, then each player flips for 6 or 3 cards.
def test_lucian_is_written_with_a_coin_flip_for_each_player():
    o = built["1237_0"]
    loops = [n for n in walk(o) if isinstance(n, ForEachPlayer)]
    assert len(loops) == 1
    flips = [n for n in walk(loops[0]) if isinstance(n, CoinFlip)]
    assert len(flips) == 1 and flips[0].side == "that_player"
    draws = lambda steps: [n.count.n for n in steps if isinstance(n, Draw)]
    assert draws(flips[0].heads) == [6] and draws(flips[0].tails) == [3]
    assert any(isinstance(n, MoveCards) and n.side == "both" and n.destination == "deck_bottom" for n in walk(o))