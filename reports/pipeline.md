# Pokémon TCG deck-simulation pipeline

_Last updated: 2026-11-26_

## Goal

Give the system a 60-card deck. It plays simulated games against an opponent deck, tracks what it can't see (its own Prizes, the opponent's deck), and picks plays with a locally trained model.

**Constraints:** everything runs locally. No LLMs, no external AI services or APIs, and the agent uses no pretrained weights. Data is sourced by hand and cited.

## Status

| Step | Status |
|---|---|
| 1. Clean the data | Done |
| 2. Build the tables | Done (nine tables) |
| 3. Tag the effects | Done for the 54 in-scope effects; the tagger for the rest of the pool is evaluated but not scored against hand labels |
| 4. Feature matrix and PCA | Done with effect verbs; ablation not run |
| 5. Decks and validator | Done, including card instance tags |
| 6. Belief trackers | Not started |
| 7. Game engine and text console | **Next** |
| 8. Baseline agents and evaluation harness | Not started |
| 9. Train the model | Not started |
| 10. Evaluate | Not started |

Tests: 119 passed, 2 skipped (`uv run pytest`, run from the project folder).

## The pipeline

### 1. Clean the data (done)

Parse `EN_Card_Data_R2_full.csv`: strip `n/a` values and stray whitespace, turn costs like `{D}●●` into energy counts, split `30×` into a number plus a modifier, and mark moves as `attack`, `ability` or `tera` from the `[Ability]` / `[Tera]` prefix on the move name.

*Notebook:* `notebooks/data_processing.ipynb`.

### 2. Build the tables (done)

| Table | One row per | Source | Shape |
|---|---|---|---|
| `cards` | card | CSV | 1,431 × 26 |
| `moves` | attack, ability or effect | CSV | 2,265 × 19 |
| `plays` | generic turn action | `configs/plays.csv` | 22 × 10 |
| `game_constants` | global number | `configs/game_constants.csv` | 16 × 6 |
| `special_conditions` | status condition | `configs/special_conditions.csv` | 5 × 9 |
| `energy_provides` | Energy card | `configs/energy_provides.csv` plus `cards` | 25 × 10 |
| `decks`, `deck_cards`, `deck_profile` | deck, card in a deck, card frequency | `data/external/decks/*.txt` | 2, 48, 48 rows |

All tables live in `data/processed/` as Parquet and are built by the notebooks.

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

The 54 effects of the two chosen decks are in `src/tcg_bot/effects_data.py`: 52 `ok`, 1 `complex` (Lucian), 1 `no_effect_in_scope` (Psyduck's Damp). Tests check that every in-scope move has exactly one effect, that usage tags match the card text, and that filters only use real card values.

**Rest of the pool** (`notebooks/edas/eda_tagging.ipynb`):
- 17 regex patterns label 1,368 of 1,671 effects with text (82%).
- A TF-IDF plus logistic regression classifier reproduces those labels with macro F1 0.96 on random folds, 0.96 on unseen wording and 0.91 on unseen wording families.
- Trained without their wordings, it matches the hand-written labels on 43 of 45 effects.
- A holdout of 100 effects (`configs/tag_holdout.csv`) is sampled; 0 are labelled by hand.

**Next for this step:** label the holdout, then build the full-pool tagger. Anything unrecognized gets `unknown` and a parse status.

### 4. Card feature matrix and PCA (done)

*Code:* `src/tcg_bot/features.py`. *Notebook:* `notebooks/edas/eda_pca.ipynb`. *Output:* `card_vectors.parquet` (1,431 × 31) and `card_pca.joblib`.

**Model vector (whole pool)**
- 1,431 cards × 56 columns: stats, flags, type / weakness / resistance indicators and 18 effect verbs (`switch` is split into self and opponent).
- Missing stats (Trainers, Energy) are neutral after scaling, so the first component isn't just "is this a Pokémon?".
- 47 columns become 21 components (parallel analysis), keeping 79.6% of their variance.
- 9 columns bypass the PCA (R² below 0.5): `is_ace_spec`, `is_ancient`, `has_owner`, `switch_opp`, `heal`, `recoil`, `modify_stat`, `coin_flip`, `status`.
- Model input per card: 21 + 9 = 30 numbers.
- Checks: subspace stability 0.94; 78% of nearest neighbours preserved; 97% agreement with a FAMD-style variant.

**Limits:** KMO is 0.34 over all columns; only the first three components are individually stable; verbs carry no amounts or filters; Dunsparce and Buneary get identical vectors.

**Report PCA (numeric Pokémon variables)**
- 1,196 Pokémon × 9 variables; KMO 0.68; Bartlett p < 0.001.
- 2 components keep 62.1% of the variance; 7 keep 96.5%.
- Classifying ex cards: accuracy 0.994 with the original variables, 0.971 with 7 components, against 0.860 for always answering "not ex". PCA does not improve the classifier.

**Still to do:** the ablation (PCA only, PCA plus bypass, raw columns, learned embedding), which needs the engine and training.

### 5. Decks and validator (done)

Deck text files in `data/external/decks/` become `decks`, `deck_cards` (with `line_order`) and `deck_profile`. Lines resolve by name, then set and number, then hand-checked print aliases. Whole-number lists are used exactly as written. Average lists are rounded to a legal 60. Deck rules (deck size, copies by name, ACE SPEC, Basic Pokémon) are read from `game_constants`.

**Card instances.** `expand_deck` (`src/tcg_bot/decks.py`) turns a deck into 60 cards, each with an integer `uid` and a readable tag `<deck>_<family>_<n>`, numbered in `.txt` order (for example `lopunny_pokemon_3`). At game time, prefix the player so a mirror match doesn't collide.

### 6. Build belief trackers

- **Prizes:** unseen pool = deck list − cards seen. Prize odds come from counting.
- **Opponent deck:** a Bayesian update over known archetypes, using frequencies from `deck_profile`. Keep the Limitless average lists for these priors.

### 7. Build the game engine and the text console (next)

Three layers: core rules (setup, turn phases, limits, attacks, Knock Outs, Checkup, win checks from `plays`), an effect interpreter that runs the step verbs, and the interfaces. It exposes `legal_actions(state)` and `step(state, action)`.
- **First:** define the state and action types (`src/tcg_bot/engine/types.py`).
- **State:** one Active slot and up to 5 Benched Pokémon per player, zones, per-turn flags, lasting effects, and a seeded random number generator. Cards are tracked as instances, not card IDs.
- **Choices:** effects that need decisions (which card to search for, how to split Energy) expose their options, so they become multi-step actions.
- **Ending the turn:** an explicit action the player chooses; the turn ends by itself only after an attack.
- **Hand to bottom of deck** (Lucian, Special Red Card): shuffle the hand first, then place it under the deck. The effect data does not say this, so the interpreter must.
- **Prizes:** a Knocked Out card with `is_mega` is worth `mega_ex_prize_value`.
- **Hidden information:** the engine holds full state; a separate `observe(state, player)` gives a player only what they may see, and is tested. It must include whether a Pokémon moved to Active this turn, lockouts, lasting effects, and which Energy and Stadium cards are in play.
- **Unsupported effects** raise a clear error, never silently skip.
- **Text console:** parses typed commands against `legal_actions`, and renders the model's choice in the same command format.
- Measure games per second early.

### 8. Baseline agents and evaluation harness

Random, a hand-written heuristic, and determinized MCTS using the belief tracker. The harness runs fixed-seed games, both seats, with confidence intervals. Sanity checks: random vs. random is close to 50/50, no illegal actions, opening-without-a-Basic rate.

### 9. Train the model

A PyTorch network trained by self-play (PPO). It scores each legal action and masks illegal ones, with a value head for win probability. Start with a flat-slot MLP, then attention. Cards enter as the PCA vector plus the bypass flags, with card identity so that twin vectors can be told apart. Start with one fixed opponent deck, then rotate 2 or 3, then hold one out.

### 10. Evaluate

Thousands of games per matchup against random, heuristic and MCTS players, a held-out deck, the PCA ablation, and play through the console. Set the success threshold before training.

## Scope

**2 decks now** (`mega_lucario`, `lopunny`): 42 distinct cards, 54 moves and effects, 120 card instances. A third deck is possible but means more cards to code and verify. Fallback if training doesn't pay off: a smaller learned component on top of the heuristic player.

Order of work: engine and console, then harness and baselines, then training. Belief trackers can start once the state types exist. The report and the holdout labelling don't depend on the engine.

## What the data looks like

- 2,273 raw rows (one per move) become 1,431 cards and 2,265 moves and effects.
- Move kinds: attack 1,754, ability 251, tera 32, trainer_effect 210, energy_effect 17, granted_attack 1.
- 1,133 distinct effect texts; 594 moves are plain damage with no text.
- Abilities are identified by the `[Ability]` prefix on the move name.
- The card CSV keeps one entry per functional card, so reprints are collapsed and some prints in decklists match by name.
- The CSV gives Weakness and Resistance types but no amounts; those are in `game_constants`.

## Main risks

1. **Engine bugs and design.** Effects with sub-choices, per-turn state and board-wide passives must fit the action space. Test the engine heavily before training.
2. **Hidden-information leaks.** The model must only see what the player sees.
3. **Training time.** Self-play needs many games on a laptop; measure speed early.
4. **Scope creep.** Each added deck adds effects, and possibly new verbs.
5. **Rule sources.** The 2023 rulebook predates the new Mega Evolution ex cards, and the newer edition couldn't be fetched. Several rules rest on a player's confirmation.
6. **Reproducibility.** The three build notebooks still use absolute paths; logic should keep moving into `src/tcg_bot/`.

## Decisions so far

- Bench max stays at 5; Buddy-Buddy Poffin puts as many as fit, up to 2.
- Mega evolving does not end the turn; a Knocked Out Mega Evolution ex gives 3 Prize cards.
- A player may end the turn without attacking, but must declare it.
- Resistance subtracts 30 (read from the Fan Rotom card).
- Energy attached by an effect (Aura Jab) doesn't use the once-per-turn hand attach.
- Lucian and Special Red Card: the hand is shuffled, then put on the bottom of the deck.
- Hilda: take both an Evolution Pokémon and an Energy card (rulebook reading; page still to be cited), then shuffle the deck.
- Wild Press recoil is a flat 70.
- Per-name ability limits are tagged by scope.
- `has_rule_box` is a column on `cards`.
- PCA is fit on the bulk columns; independent flags bypass it.

## Open questions

- Is the console (a) a full simulator or (b) an advisor?
- Hours per week, and what must exist on the last day.
- Whether the optional MCTS player stays in scope.
- What win rate counts as success?
