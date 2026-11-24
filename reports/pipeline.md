# Pokémon TCG deck-simulation pipeline

_Last updated: 2026-11-24_

## Goal

Give the system a 60-card deck. It plays simulated games against an opponent deck, tracks what it can't see (its own Prizes, the opponent's deck), and picks plays with a locally trained model.

**Constraints:** everything runs locally. No LLMs, no external AI services or APIs, and the agent uses no pretrained weights. Data is sourced by hand and cited.

## Status

| Step | Status |
|---|---|
| 1. Clean the data | Done |
| 2. Build the tables | Done (nine tables) |
| 3. Tag the effects | Done for the 54 in-scope effects; tagger for the rest of the pool next |
| 4. Feature matrix and PCA | Early test passed; full run after the tagger |
| 5. Decks and validator | Done, including card instance tags |
| 6. Belief trackers | Not started |
| 7. Game engine and text console | **Next** |
| 8. Baseline agents and evaluation harness | Not started |
| 9. Train the model | Not started |
| 10. Evaluate | Not started |

Tests: 24 passing (`uv run pytest`, run from the project folder).

## The pipeline

### 1. Clean the data (done)

Parse `EN_Card_Data_R2_full.csv`: strip `n/a` values and stray whitespace, turn costs like `{D}●●` into energy counts, split `30×` into a number plus a modifier, and mark moves as `attack`, `ability` or `tera` from the `[Ability]` / `[Tera]` prefix on the move name.

*Notebook:* `notebooks/data_processing.ipynb`.

### 2. Build the tables (done)

| Table | One row per | Source | Shape |
|---|---|---|---|
| `cards` | card | CSV | 1,431 × 26 |
| `moves` | attack, ability or effect | CSV | 2,265 × 19 |
| `plays` | generic turn action | `configs/plays.csv` | 23 × 10 |
| `game_constants` | global number | `configs/game_constants.csv` | 16 × 6 |
| `special_conditions` | status condition | `configs/special_conditions.csv` | 5 × 9 |
| `energy_provides` | Energy card | `configs/energy_provides.csv` plus `cards` | 25 × 10 |
| `decks`, `deck_cards`, `deck_profile` | deck, card in a deck, card frequency | `data/external/decks/*.txt` | 2, 48, 48 rows |

All tables live in `data/processed/` as Parquet and reproduce exactly from the notebooks.

**Rule layers.** `plays` and `game_constants` hold the baseline rules. Card text holds what a card adds or changes.
- *Action limits* (one Energy attach from hand, one Supporter, one retreat per turn) apply to the action itself. An effect that does the same job another way (Aura Jab attaches from the discard pile, effect switches skip retreat) doesn't use them up.
- *State rules* (Bench of at most 5, one Active Pokémon) always hold, however a card got there.
- *Values* (retreat cost, HP, damage) are read through a lookup that applies card modifiers (Air Balloon, Gravity Mountain).

*Notebooks:* `data_processing`, `plays`, `deck_building`. Run in that order.

### 3. Tag the effects (done for the in-scope effects)

Each effect is **data** built from a small vocabulary, not code per card. An effect has a timing, a usage limit, requirements and costs, and steps. The vocabulary (`src/tcg_bot/effects.py`):
- **12 step verbs:** `MoveCards` (search, recover, discard, shuffle hand, return to hand, attach from discard), `Draw`, `SwitchActive`, `Heal`, `ApplyStatus`, `DamageSelf`, `Lockout`, `Prevent`, `ModifyStat`, `CoinFlip`, `If`, `ForEachPlayer`.
- **Shared parts:** `Fixed`, `PerCount` (multiplier), `Sum`, `IfThen`, `Cond`, `Filter`, `Target`, `Usage`.
- **Usage scopes** for abilities: `per_pokemon`, `per_name`, `per_name_contains`, plus a first-turn-only flag.
- `build()` rejects unknown blocks, parameters and values, so a tagging mistake fails loudly.

The 54 effects of the two chosen decks are in `src/tcg_bot/effects_data.py`: 52 `ok`, 1 `complex` (Lucian), 1 `no_effect_in_scope` (Psyduck's Damp). Tests in `tests/test_effects.py` check that every in-scope move has exactly one effect, that usage tags match the card text, and that filters only use real card values.

**Next for this step:** tag the rest of the pool with the same vocabulary. Regex first, then a local classifier (TF-IDF plus logistic regression) for the leftovers, with hand labels. Anything unrecognized gets `unknown` and a parse status. The tags feed step 4.

### 4. Build the card feature matrix and apply PCA

PCA gives each card a compact vector for the model and a deck-level embedding. Plan:
- **Rows:** one per card (1,431). **Columns:** stats (HP, retreat, cost, damage), type one-hots, and block columns (which verbs a card uses, numeric parameters, filter attributes).
- **Fit on the bulk columns.** Leave the family and kind columns out (otherwise PC1 is just Pokémon vs. Trainer). Standardize. Fit once on the whole pool and freeze the fitted object.
- **Rare, nearly independent flags bypass the PCA** and are appended to the vector. A column bypasses if the chosen number of components reconstructs it poorly (for example R² below 0.5) and the engine uses it.
- **Choose the number of components** by parallel analysis and the variance curve, then check it with the ablation.
- **Check:** loadings, bootstrap stability, nearest neighbours, and the FAMD hybrid as a robustness comparison.
- **Ablation:** PCA vector only, PCA plus raw flags, raw columns only, and a learned embedding.
- **Report two sizes:** the PCA vector and the total model input per card.

**Early test (crude flags, not final):** about 52 to 79 columns reduced to 16 to 23 components (about 3.3×). Neighbours were sensible (Poké Pad near Ultra Ball, Boss's Orders near the other gust Supporters). The top two components were stable under resampling, and the two real decks were clearly more distinct than random 60-card piles. Weaknesses: several gameplay flags (attach Energy, switch, draw) were poorly reconstructed, and later components were less stable. Re-run on the final block features.

### 5. Decks and validator (done)

Deck text files in `data/external/decks/` become `decks`, `deck_cards` (with `line_order`) and `deck_profile`. Lines resolve by name, then set and number, then hand-checked print aliases. Whole-number lists are used exactly as written. Average lists are rounded to a legal 60. Deck rules (deck size, copies by name, ACE SPEC, Basic Pokémon) are read from `game_constants`.

**Card instances.** `expand_deck` (`src/tcg_bot/decks.py`) turns a deck into 60 cards, each with an integer `uid` and a readable tag `<deck>_<family>_<n>`, numbered in `.txt` order (for example `lopunny_pokemon_3`). At game time, prefix the player so a mirror match doesn't collide.

### 6. Build belief trackers

- **Prizes:** unseen pool = deck list − cards seen. Prize odds come from counting.
- **Opponent deck:** a Bayesian update over known archetypes, using frequencies from `deck_profile`. Keep the Limitless average lists for these priors.

### 7. Build the game engine and the text console (next)

Three layers: core rules (setup, turn phases, limits, attacks, Knock Outs, Checkup, win checks from `plays`), an effect interpreter that runs the step verbs, and the interfaces. It exposes `legal_actions(state)` and `step(state, action)`.
- **State:** one Active slot and up to 5 Benched Pokémon per player, zones, per-turn flags, lasting effects, and a seeded random number generator. Cards are tracked as instances, not card IDs.
- **Choices:** effects that need decisions (which card to search for, how to split Energy) expose their options, so they become multi-step actions.
- **Hidden information:** the engine holds full state; a separate `observe(state, player)` gives a player only what they may see, and is tested.
- **Unsupported effects** raise a clear error, never silently skip.
- **Text console:** parses typed commands against `legal_actions`, and renders the model's choice in the same command format.
- Measure games per second early.

### 8. Baseline agents and evaluation harness

Random, a hand-written heuristic, and determinized MCTS using the belief tracker. The harness runs fixed-seed games, both seats, with confidence intervals. Sanity checks: random vs. random is close to 50/50, no illegal actions, opening-without-a-Basic rate.

### 9. Train the model

A PyTorch network trained by self-play (PPO). It scores each legal action and masks illegal ones, with a value head for win probability. Start with a flat-slot MLP, then attention. Cards enter as the PCA vector plus the bypass flags. Start with one fixed opponent deck, then rotate 2 or 3, then hold one out.

### 10. Evaluate

Thousands of games per matchup against random, heuristic and MCTS players, a held-out deck, the PCA ablation, and play through the console. Set the success threshold before training.

## Scope

**2 decks now** (`mega_lucario`, `lopunny`): 42 distinct cards, 54 moves and effects, 120 card instances. A third deck is possible but means more cards to code and verify. Fallback if training doesn't pay off: a smaller learned component on top of the heuristic player.

Order of work: engine slice and console, then the tagger for the rest of the pool and the full PCA in parallel, then harness and baselines, then training.

## What the data looks like

- 2,273 raw rows (one per move) become 1,431 cards and 2,265 moves and effects.
- Move kinds: attack 1,754, ability 251, tera 32, trainer_effect 210, energy_effect 17, granted_attack 1.
- 1,133 distinct effect texts; 594 moves are plain damage with no text.
- Abilities are identified by the `[Ability]` prefix on the move name.
- The card CSV keeps one entry per functional card, so reprints are collapsed and some prints in decklists match by name.
- Weakness and Resistance amounts are not in the CSV; the multiplier is a constant.

## Main risks

1. **Engine bugs and design.** Effects with sub-choices, per-turn state and board-wide passives must fit the action space. Test the engine heavily before training.
2. **Hidden-information leaks.** The model must only see what the player sees.
3. **Training time.** Self-play needs many games on a laptop; measure speed early.
4. **Scope creep.** Each added deck adds effects, and possibly new verbs.
5. **Rules still unconfirmed.** Four rows are `needs_check` or `inferred`. The 2023 rulebook predates the new Mega Evolution ex cards, and the newer edition couldn't be fetched.
6. **Reproducibility.** Notebooks still use absolute paths; logic should keep moving into `src/tcg_bot/` (decks and effects already have).

## Decisions so far

- Bench max stays at 5; Buddy-Buddy Poffin puts as many as fit, up to 2.
- Energy attached by an effect (Aura Jab) doesn't use the once-per-turn hand attach (to confirm).
- Per-name ability limits are tagged by scope.
- `has_rule_box` is a column on `cards`.
- PCA is fit on the bulk columns; independent flags bypass it.

## Open questions

- Mega Evolution ex: does evolving end the turn, and how many Prize cards does a Knock Out give?
- What is the Resistance amount?
- Can a player end a turn without attacking?
- Do effect-attached Energy cards leave the hand attach available?
- When both hands go to the bottom of the deck, does the owner choose the order?
- Can Hilda take just one of its two cards?
- Does Wild Press recoil apply Weakness and Resistance?
- Is the console (a) a full simulator or (b) an advisor?
- What exactly does the PCA requirement say?
- Hours per week, and what must exist on the last day.
- Whether the optional MCTS player stays in scope.