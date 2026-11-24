# Pokémon TCG deck-simulation pipeline

_Last updated: 2026-11-23_

## Goal

Give the system a 60-card deck. It plays simulated games against an opponent deck, tracks what it can't see (its own Prizes, the opponent's deck), and picks plays with a locally trained model.

**Constraints:** everything runs locally. No LLMs, no external AI services or APIs, and the agent uses no pretrained weights. Data is sourced by hand and cited.

## Status

| Step | Status |
|---|---|
| 1. Clean the data | Done |
| 2. Build the tables | Done, except `energy_provides` and move tags |
| 3. Tag the effects | Next |
| 4. Feature matrix and PCA | After tagger v1 |
| 5. Decks and validator | Done (moved early) |
| 6. Belief trackers | Not started |
| 7. Game engine and text console | Not started |
| 8. Baseline agents and evaluation harness | Not started |
| 9. Train the model | Not started |
| 10. Evaluate | Not started |

## The pipeline

### 1. Clean the data (done)

Parse `EN_Card_Data_R2_full.csv`: strip `n/a` values and stray whitespace, turn costs like `{D}●●` into energy counts, split `30×` into a number plus a modifier, and mark moves as `attack`, `ability` or `tera` from the `[Ability]` / `[Tera]` prefix on the move name.

*Notebook:* `notebooks/data_processing.ipynb`.

### 2. Build the tables (done, except `energy_provides` and tags)

| Table | One row per | File |
|---|---|---|
| `cards` | card (1,431) | `data/processed/cards.parquet` |
| `moves` | attack, ability or effect (2,265) | `data/processed/moves.parquet` |
| `plays` | generic turn action (23) | `configs/plays.csv` to `plays.parquet` |
| `game_constants` | global number (14) | `configs/game_constants.csv` to parquet |
| `special_conditions` | status condition (5) | `configs/special_conditions.csv` to parquet |
| `energy_provides` | Energy card | not built yet |

Global rules (one Supporter per turn) live in `plays`. What a card's text adds lives in `moves`. The two are kept apart so the engine can check both and a rule change is made in one place.

*Notebooks:* `data_processing`, `plays`.

### 3. Tag the effects (next)

Tag each effect with the engine's **building blocks** plus parameters (for example `search_deck(filter, count, to)`), not loose flags. Facets: timing, effect verb, target, modifier, numeric parameters. The in-scope effects (54, from the two chosen decks) are verified by hand first. Regex covers the rest of the pool, then a local classifier (TF-IDF plus logistic regression) handles the leftovers. Anything unrecognized gets an `unknown` flag.

*Why:* tags become model features, and the same vocabulary is what the engine executes.

### 4. Build the card feature matrix and apply PCA

Combine the tags, stats (HP, retreat, cost counts, damage) and type one-hots. Standardize, then run PCA. Choose the number of components with the variance curve and parallel analysis. Fit per family where it helps, and compare with the FAMD hybrid. Keep the gameplay-critical raw tags next to the PCA vector. Run an ablation (PCA vs. learned embeddings vs. raw tags). KMeans on the components is an optional way to explore card groups. Re-run after the real tags exist.

### 5. Decks and validator (done)

Deck text files in `data/external/decks/` become `decks`, `deck_cards` and `deck_profile`. Lines resolve by name, set and number, and hand-checked print aliases. Whole-number lists are used exactly as written. Average lists are rounded to a legal 60. Deck rules (60 cards, 4 copies by name, 1 ACE SPEC, at least 1 Basic Pokémon) are read from `game_constants`.

*Notebook:* `deck_building`.

### 6. Build belief trackers

- **Prizes:** unseen pool = deck list − cards seen. Prize odds come from counting.
- **Opponent deck:** a Bayesian update over known archetypes, using the average-list frequencies in `deck_profile`.

### 7. Build the game engine and the text console

Setup, turn structure, attacks, status conditions, Knock Outs, Prizes and win conditions, for the cards in the chosen decks. It exposes `legal_actions(state)` and `step(state, action)`. The console parses typed commands against `legal_actions`, and the model's choice is rendered back to the same command format. Effects that need further choices (which card to search for, how to split Energy) are multi-step decisions.

### 8. Baseline agents and evaluation harness

Random, a hand-written heuristic, and determinized MCTS using the belief tracker. The harness runs fixed-seed games with both seats and reports confidence intervals.

### 9. Train the model

A PyTorch network trained by self-play (PPO). It scores each legal action and masks illegal ones, with a value head for win probability. Cards enter as PCA vectors plus raw tags. Start with one fixed opponent deck, then rotate 2 or 3.

### 10. Evaluate

Thousands of games per matchup against random, heuristic and MCTS players, plus a held-out deck, the PCA ablation, and play through the console. The success threshold is set before training.

## Scope

**2 or 3 decks**, not the whole pool. Current decks: `mega_lucario` and `lopunny` (42 distinct cards, 54 moves and effects).

Order of work: decks, tagging schema, tagger v1, PCA v1, engine slice, harness, training. Fallback if training doesn't pay off: a smaller learned component on top of the heuristic player.

## What the data looks like

- 2,273 raw rows (one per move) become 1,431 cards and 2,265 moves and effects.
- Move kinds: attack 1,754, ability 251, tera 32, trainer_effect 210, energy_effect 17, granted_attack 1.
- 1,133 distinct effect texts; 594 moves are plain damage with no text.
- Abilities are identified by the `[Ability]` prefix on the move name.
- The card CSV keeps one entry per functional card, so reprints are collapsed and some prints in decklists are matched by name.

## Main risks

1. **Engine bugs and design.** Effects with sub-choices, per-turn state, and board-wide passives (Psyduck, Gravity Mountain) must fit the action space.
2. **Training time.** Self-play needs many games on a laptop; measure engine speed early.
3. **Scope creep.** Each added deck means more cards to code and verify.
4. **Rulebook version.** The 2023 rulebook predates the new Mega Evolution ex cards. Three rules rows are marked `needs_check`.
5. **Reproducibility.** Notebooks use absolute paths and a generic kernel name; logic should move into `src/tcg_bot/`.

## Open questions

- Mega Evolution ex: does evolving end the turn, and how many Prize cards? Can a player end a turn without attacking?
- Is the console (a) a full simulator or (b) an advisor?
- What exactly does the PCA requirement say?
- Hours per week and what must exist on the last day.
- Whether the optional MCTS player stays in scope.