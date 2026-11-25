# Engine interface contract (T1)

_Status: **draft, waiting for the owner's approval** (TASKS.md T1). Written 2026-11-25._
_Code: `src/tcg_bot/engine/types.py` (types) and `src/tcg_bot/engine/__init__.py` (API)._

Every later task builds on this contract: the rules engine (T4, T5), belief trackers (T6), console (T9), agents and harness (T10, T11), and the model (T12).

## API

```python
from tcg_bot.engine import new_game, legal_actions, step, observe, winner

state = new_game(("lopunny", "mega_lucario"), seed=0)   # player 0 plays the first deck
acts  = legal_actions(state)        # list[Action]; empty only when the game is over
state = step(state, acts[0])        # returns a NEW State; the input is unchanged
obs   = observe(state, player=0)    # Observation: only what player 0 may see
w     = winner(state)               # 0, 1, or None (game running, or a draw)
```

Extra helpers:
- `step_inplace(state, action, validate=True)` is a faster version for self-play. It mutates `state` and skips the copy.
- `acting_player(state)` returns who must act now. This is the player of the pending choice if there is one, otherwise the player whose turn it is.
- `card_locations(state)` returns where every card instance is. Tests use it to prove that no card is created or lost.
- `IllegalActionError` is raised by `step` for any action not in `legal_actions(state)`.

## Cards are instances

- Each card in a game has a **uid**, unique across both players: `uid = player * 100 + i`, where `i` is the 0–59 index from `expand_deck`. A mirror match (`lopunny` vs `lopunny`) never collides: player 0 owns uids 0–59 and player 1 owns 100–159.
- `state.catalog.refs[uid]` is a `CardRef(uid, owner, tag, card)`. The tag is readable, for example `p1_lopunny_pokemon_3`. `card` is a `CardStatic` holding HP, type, attacks, ability IDs, effect IDs and the Energy it provides.
- A Pokémon in play is a `Slot`, identified by its **pid**: the uid of the Basic card at the bottom of its evolution stack. The pid does not change when the Pokémon evolves.

## State

`State` holds the following:
- **Per player:** deck (index −1 is the top), hand, discard pile, Prizes, one Active `Slot`, up to 5 Bench `Slot`s, mulligan count, the turns in which one of the player's Pokémon was Knocked Out, and the number of turns taken.
- **Per `Slot`:** the evolution stack, attached Energy, Tool, damage, Special Conditions, the turn it came into play, the turn it last evolved, the turn it moved to the Active Spot, attack lockouts, and Ability uses.
- **Shared:** Stadium, `turn` (0 during setup, then 1, 2, … with one per player turn), `current` player, `first_player`, per-turn `flags`, lasting effects (`lasting`), the pending choice, the interpreter frames, a seeded `random.Random`, `winner` and `result`.
- **Copying:** `State.copy()` is cheap. It copies the mutable lists and dicts and shares the immutable `Catalog` and `RuleSettings`. The random generator is copied by state, so a copy replays the same coin flips.

## Action

```python
Action(kind: str, player: int, source: int | None = None, target: int | None = None, choice = None)
```

Every field is a plain value, so actions can be listed, compared, hashed and encoded.

| Phase | kind | source | target | choice |
|---|---|---|---|---|
| setup | `choose_first` | | | `"first"` / `"second"` |
| setup | `set_active`, `set_bench` | card uid | | |
| setup | `setup_done` | | | |
| main | `bench_basic`, `play_item`, `play_supporter`, `play_stadium` | card uid | | |
| main | `evolve`, `attach_energy`, `attach_tool` | card uid | pid | |
| main | `retreat` | | Benched pid | |
| main | `use_ability` | pid | | ability move_id |
| main | `attack` | Active pid | | attack move_id |
| main | `end_turn` | | | |
| any | `choose` | | | option value of the pending choice |

When several copies of the same card are in a hand, `legal_actions` lists the card once, using the copy with the lowest uid. Identical cards are interchangeable, so this keeps the action list short.

## Multi-step choices

Some actions need further decisions: which card to search for, which Benched Pokémon to switch in, which Energy to discard when retreating, whether to use an optional Ability, or which Pokémon to promote after a Knock Out. For these, the engine sets `state.pending = Pending(player, kind, options, label, stop)` and `legal_actions` returns one `choose` action per option. If `stop` is set, it is an extra option meaning "take no more" (for example "up to 3", "you may", or a search that comes up short). A choice with exactly one possible answer and no stop option is resolved automatically.

The interpreter's position is stored as plain data in `state.frames`. A state copied in the middle of an effect therefore continues correctly. Determinized MCTS relies on this.

## Observation (hidden information)

`observe(state, p)` returns an `Observation` with:
- **The player's own cards:** their hand, both discard piles, all Pokémon in play (`SlotView`, with current HP after modifiers), the Stadium, and this turn's flags.
- **Counts only:** the player's own deck and Prize cards, and the opponent's hand, deck and Prize cards.
- **Public cards:** a log of cards that became public (played, benched, evolved, attached, revealed, Knocked Out), as `(turn, player, what, card_id)`.
- **Pending choices:** the player's own pending options. For the opponent's pending choice, only its kind, and only when it is public (promoting a new Active, switching).
- **The player's own deck list**, which a player knows before the game.

The observation hides deck order and content, all Prize cards, the opponent's hand, and the opponent's private choices. During setup, the opponent's face-down Active and Bench are hidden until both players have finished placing.

Tests (`tests/engine/test_games.py`):
- `test_observe_leaks_no_hidden_card` checks, at every step of 40 random games, that no hidden uid appears anywhere in either player's observation.
- `test_hidden_information_does_not_change_the_observation` changes only hidden information and checks that the observation does not change.

## Game end and draws

- `winner(state)` is 0 or 1 when someone won. `state.result` says how: `prizes`, `no_pokemon`, `deck_out`, or the same reasons prefixed with `sudden_death_`.
- **Draw by turn limit:** when `turn` would exceed `RuleSettings.turn_limit` (default 200), the game ends with `phase == "over"`, `winner is None` and `result == "draw_turn_limit"`.
- If both players win at once, the engine starts Sudden Death with `sudden_death_prize_count` Prize cards (rule `win_check`). The turn counter keeps running, so the turn limit still bounds the whole game.

## Rule values that are not confirmed

`RuleSettings` holds every value that is not yet confirmed (task H1) as a named setting, so none is a buried number. The defaults below are **assumptions**:

| Setting | Default | H1 item |
|---|---|---|
| `mega_evolution_ends_turn` | `False` | (a) |
| `mega_ex_prize_value` | `3` | (b) |
| `resistance_amount` | `30` | (c) |
| `may_end_turn_without_attack` | `True` | (d) |
| `effect_attach_uses_hand_attach` | `False` | (e) |
| `bottom_of_deck_order` | `"random"` | (f): the card texts say "shuffles … and puts it on the bottom" |
| `search_may_find_fewer` | `True` | (g) |
| `recoil_applies_weakness_resistance` | `False` | (h) |
| `take_mulligan_bonus_cards` | `True` | `setup_mulligan` says "may" |

`RuleSettings().assumptions()` returns this table so that it can be logged with every experiment.

## Speed

Random legal play, without validation, measured on the cloud workspace (2 vCPU, Python 3.13): **about 65 games per second**. That is about 11,700 actions per second, with an average of 180 actions per game.
