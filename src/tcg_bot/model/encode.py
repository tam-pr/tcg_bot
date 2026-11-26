"""Turning what a player sees into numbers for the network.

The only input is an Observation and the legal actions. This module never gets the full game State,
so nothing hidden can reach the network.

A state becomes a few fixed-size arrays; each legal action becomes one row, and the network will give
every row a score. Actions whose rows are identical (two copies of a card) are merged.

Card numbers ("idx") are lookup keys for an embedding layer: 0 is "no card", 1..N the cards of the two decks.
Next to them go the card's 30 numbers from the PCA (card_vectors.parquet).
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from tcg_bot.effects import Lockout, ModifyStat, Prevent
from tcg_bot.engine.catalog import PROC, catalog
from tcg_bot.engine.observe import Observation, PlayerView
from tcg_bot.engine.types import Action, ActionKind, Phase, PokemonInPlay

HP_SCALE = 340.0                                     # the most HP in the two decks
SLOTS = [(side, place) for side in ("me", "opponent") for place in range(6)]   # place 0 is the Active
ENERGY_TYPES = ("F", "W", "C")
NAME_LIMITS = ("Fan Call", "Lunar Cycle", "Last-Ditch")
CHOICE_KINDS = ("go_first", "mulligan_draw", "new_active", "promote", "switch_to", "gust", "heal", "pick",
                "energy_to_hand", "attach_pick", "attach_target", "retreat_cost", "use_ability")
SOURCES = ("deck", "top_of_deck", "hand", "discard", "attached", "attached_to_this", "attached_tool_any")
DESTINATIONS = ("hand", "bench", "discard", "attached_to_bench")
KINDS = tuple(ActionKind)

SLOT_FEATURES = ("present", "face_down", "hp_left", "damage", "max_hp", "retreat_cost",
                 *(f"energy_{t}" for t in ENERGY_TYPES), "tool", "asleep", "moved_up_this_turn",
                 "entered_this_turn", "ability_used", "locked_now", "locked_next", "protected_now", "protected_next")
GAME_FEATURES = ("turn", "setup", "my_turn", "i_went_first",
                 *(f"{who}_{what}" for who in ("my", "opp") for what in
                   ("hand", "deck", "prizes", "mulligans", "face_down", "lost_a_pokemon_last_turn", "damage_bonus")),
                 "energy_used", "supporter_used", "stadium_used", "retreat_used",
                 *(f"used_{n}" for n in NAME_LIMITS),
                 *(f"choice_{k}" for k in CHOICE_KINDS), *(f"from_{s}" for s in SOURCES),
                 *(f"to_{d}" for d in DESTINATIONS), "choice_min", "choice_max", "choice_picked")


@dataclass(frozen=True)
class Vocabulary:
    card_ids: tuple[int, ...]                        # idx 1.. in this order
    index: dict[int, int]                            # card_id -> idx
    vectors: np.ndarray                              # (N + 1, 30), row 0 all zeros


@lru_cache(maxsize=1)
def vocabulary() -> Vocabulary:
    ids = tuple(sorted({card for deck in catalog().decks.values() for card in deck}))
    table = pd.read_parquet(PROC / "card_vectors.parquet").set_index("card_id")
    missing = [i for i in ids if i not in table.index]
    if missing:
        raise ValueError(f"no card vector for {missing}")
    vectors = np.zeros((len(ids) + 1, table.shape[1]), dtype=np.float32)
    vectors[1:] = table.loc[list(ids)].to_numpy(dtype=np.float32)
    return Vocabulary(card_ids=ids, index={card: n for n, card in enumerate(ids, 1)}, vectors=vectors)


def card_idx(card_id: int | None) -> int:
    if card_id is None:
        return 0
    if card_id not in vocabulary().index:
        raise ValueError(f"card {card_id} is not in the two decks the encoder knows")
    return vocabulary().index[card_id]


@dataclass(frozen=True)
class Encoded:
    slots: np.ndarray                # (12, len(SLOT_FEATURES) + 30) the Pokémon on both sides
    slot_cards: np.ndarray           # (12,) idx of the card on top in each slot
    counts: np.ndarray               # (5, N) my hand, my discard, their discard, their shown hand, mine not seen
    game: np.ndarray                 # (len(GAME_FEATURES),)
    context_cards: np.ndarray        # (3,) idx of: Stadium, card asking for a choice, Trainer being played
    moves: np.ndarray                # (n, MOVE_SIZE) one row per different legal move
    move_cards: np.ndarray           # (n,) idx of the card each move uses or picks
    actions: tuple[Action, ...]      # the action to play for each row


def _slot_of(view: Observation) -> dict[int, int]:
    """Pokémon id -> row in `slots`."""
    out = {}
    for row, (side, place) in enumerate(SLOTS):
        player: PlayerView = getattr(view, side)
        mons = [player.active, *player.bench]
        if place < len(mons) and mons[place] is not None:
            out[mons[place].uid] = row
    return out


def _lasting(view: Observation, uid: int, block: type, turn: int) -> float:
    return float(any(isinstance(e.block, block) and e.target == uid and e.first_turn <= turn <= e.last_turn
                     for e in view.lasting))


def _pokemon_row(view: Observation, player: PlayerView, mon: PokemonInPlay) -> list[float]:
    letters = Counter(catalog().energy[view.card_ids[u]][0] for u in mon.energy)
    hp = player.max_hp[mon.uid]
    return [1.0, 0.0, max(hp - mon.damage, 0) / HP_SCALE, mon.damage / HP_SCALE, hp / HP_SCALE,
            player.retreat_cost[mon.uid] / 4, *(float(letters[t]) for t in ENERGY_TYPES),
            float(mon.tool is not None), float(mon.rotation == "Asleep"),
            float(mon.moved_to_active_on_turn == view.turn), float(mon.in_play_since == view.turn),
            float(bool(mon.abilities_used)),
            _lasting(view, mon.uid, Lockout, view.turn), _lasting(view, mon.uid, Lockout, view.turn + 2),
            _lasting(view, mon.uid, Prevent, view.turn), _lasting(view, mon.uid, Prevent, view.turn + 1)]


def _slots(view: Observation) -> tuple[np.ndarray, np.ndarray]:
    vocab = vocabulary()
    features = np.zeros((len(SLOTS), len(SLOT_FEATURES) + vocab.vectors.shape[1]), dtype=np.float32)
    cards = np.zeros(len(SLOTS), dtype=np.int64)
    for row, (side, place) in enumerate(SLOTS):
        player: PlayerView = getattr(view, side)
        mons = [player.active, *player.bench]
        if place < len(mons) and mons[place] is not None:
            mon = mons[place]
            cards[row] = card_idx(view.card_ids[mon.card])
            features[row] = np.concatenate([_pokemon_row(view, player, mon), vocab.vectors[cards[row]]])
        elif place < player.face_down:
            features[row, :2] = 1.0                  # something is there, face down
    return features, cards


def _counts(view: Observation) -> np.ndarray:
    n = len(vocabulary().card_ids)
    seen = list(view.me.hand) + list(view.me.discard) + list(view.me.playing)
    for mon in [m for m in [view.me.active, *view.me.bench] if m]:
        seen += mon.stack + mon.energy + ([] if mon.tool is None else [mon.tool])
    if view.stadium is not None and view.stadium // 60 == view.player:
        seen.append(view.stadium)
    unseen = Counter(view.my_deck_list) - Counter(view.card_ids[u] for u in seen)      # in my deck or Prizes
    groups = [Counter(view.card_ids[u] for u in cards)
              for cards in (view.me.hand, view.me.discard, view.opponent.discard, view.opponent.hand)] + [unseen]
    out = np.zeros((len(groups), n), dtype=np.float32)
    for row, group in enumerate(groups):
        for card, count in group.items():
            out[row, card_idx(card) - 1] = count / 4
    return out


def _damage_bonus(view: Observation, player: int) -> float:
    return sum(e.block.amount for e in view.lasting if isinstance(e.block, ModifyStat) and e.controller == player
               and e.block.stat == "attack_damage" and e.first_turn <= view.turn <= e.last_turn) / 100


def _one_hot(value, options) -> list[float]:
    return [float(value == option) for option in options]


def _game(view: Observation) -> np.ndarray:
    out = [view.turn / 50, float(view.phase == Phase.SETUP), float(view.current == view.player),
           float(view.first_player == view.player)]
    for player, number in ((view.me, view.player), (view.opponent, 1 - view.player)):
        out += [player.hand_count / 20, player.deck_count / 60, player.prize_count / 6, player.mulligans / 4,
                player.face_down / 6, float(player.knocked_out_on_turn == view.turn - 1), _damage_bonus(view, number)]
    flags = view.me.flags
    out += [float(flags.energy_attached), float(flags.supporter_played), float(flags.stadium_played),
            float(flags.retreated)]
    out += [float(name in flags.names_used) for name in NAME_LIMITS]
    pending, choice = view.pending, view.choice
    out += _one_hot(pending.kind if pending else None, CHOICE_KINDS)
    out += _one_hot(choice.source if choice else None, SOURCES)
    out += _one_hot(choice.destination if choice else None, DESTINATIONS)
    out += [pending.min_count / 3, pending.max_count / 3, len(pending.picked) / 3] if pending else [0.0, 0.0, 0.0]
    return np.array(out, dtype=np.float32)


def _context_cards(view: Observation) -> np.ndarray:
    playing = view.me.playing or view.opponent.playing
    return np.array([
        card_idx(None if view.stadium is None else view.card_ids[view.stadium]),
        card_idx(view.choice.card_id if view.choice else None),
        card_idx(view.card_ids[playing[0]] if playing else None)], dtype=np.int64)


# ---------------------------------------------------------------- moves
def _move(view: Observation, action: Action, slot_of: dict[int, int]) -> tuple[list[float], int]:
    """The numbers for one legal action, and the idx of the card it uses or picks."""
    card, target, move_number, damage, number = None, action.target, None, 0.0, 0.0
    mine = [m for m in [view.me.active, *view.me.bench] if m]
    if action.kind == ActionKind.CHOOSE:
        kind = view.pending.kind
        if kind in ("go_first", "mulligan_draw", "use_ability"):
            number = (action.option + 1) / 7                           # + 1, so that "no" is not all zeros
        elif kind in ("new_active", "promote", "switch_to", "gust", "heal", "attach_target"):
            target = action.option
        else:                                                          # a card is picked
            card = view.card_ids[action.option]
            if view.choice.source == "attached_tool_any":              # say whose Tool it is
                target = next(m.uid for p in (view.me, view.opponent) for m in [p.active, *p.bench]
                              if m and m.tool == action.option)
    elif action.card is not None:
        card = view.card_ids[action.card]
    elif action.kind in (ActionKind.ATTACK, ActionKind.USE_ABILITY):
        user = view.me.active if action.kind == ActionKind.ATTACK else next(m for m in mine if m.uid == action.target)
        card = view.card_ids[user.card]
        moves = catalog().cards[card].moves
        move_number = next(n for n, m in enumerate(moves) if m.move_id == action.move_id)
        damage = (moves[move_number].damage or 0) / HP_SCALE
    idx = card_idx(card)
    row = slot_of.get(target) if target is not None else None
    features = [*_one_hot(action.kind, KINDS), float(idx > 0), *vocabulary().vectors[idx],
                float(row is not None), *_one_hot(row, range(len(SLOTS))),
                *_one_hot(move_number, range(3)), damage, number]
    return features, idx


def encode(view: Observation, actions: list[Action]) -> Encoded:
    """`actions` are the legal actions of the player whose `view` this is."""
    if any(a.player != view.player for a in actions):
        raise ValueError("the actions belong to the other player")
    slots, slot_cards = _slots(view)
    slot_of = _slot_of(view)
    rows: dict[tuple, tuple[list[float], int, Action]] = {}
    for action in actions:
        features, idx = _move(view, action, slot_of)
        rows.setdefault((idx, tuple(features)), (features, idx, action))
    return Encoded(
        slots=slots, slot_cards=slot_cards, counts=_counts(view), game=_game(view),
        context_cards=_context_cards(view),
        moves=np.array([r[0] for r in rows.values()], dtype=np.float32),
        move_cards=np.array([r[1] for r in rows.values()], dtype=np.int64),
        actions=tuple(r[2] for r in rows.values()))


# ---------------------------------------------------------------- look at it: uv run python -m tcg_bot.model.encode
def _demo(seed: int = 7) -> None:
    import random

    from tcg_bot.engine.game import legal_actions, new_game, step
    from tcg_bot.engine.observe import observe
    from tcg_bot.narrate import demo_choice

    pick = random.Random(seed)
    state = new_game(("lopunny", "mega_lucario"), seed)
    while not (state.phase == Phase.MAIN and not state.pending and state.turn >= 7 and len(legal_actions(state)) >= 6):
        state = step(state, demo_choice(state, pick))
    view = observe(state, state.current)
    e = encode(view, legal_actions(state))
    columns = [*SLOT_FEATURES, *(f"card_{n}" for n in range(e.slots.shape[1] - len(SLOT_FEATURES)))]
    slots = pd.DataFrame(e.slots, columns=columns, index=[f"{side[:3]}_{place}" for side, place in SLOTS])
    slots.insert(0, "card_idx", e.slot_cards)
    print("slots", e.slots.shape)
    print(slots[["card_idx", "present", "hp_left", "energy_F", "energy_W", "tool", "card_0", "card_1"]].head(8).round(2))
    print("\ngame", e.game.shape, "(first 8)")
    print(pd.DataFrame([e.game[:8]], columns=GAME_FEATURES[:8]).round(2).to_string(index=False))
    print("\ncounts", e.counts.shape, "| moves", e.moves.shape)
    kinds = [KINDS[int(row[: len(KINDS)].argmax())].value for row in e.moves]
    print(pd.DataFrame({"kind": kinds, "card_idx": e.move_cards}).head(8))


if __name__ == "__main__":
    _demo()
