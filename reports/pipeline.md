# Pokémon TCG deck-simulation pipeline

_Last updated: 2026-11-25. Every number below comes from a test, a notebook or a file in `reports/results/`._

## Goal

Give the system a 60-card deck. It plays simulated games against an opponent deck, tracks what it can't see (its own Prizes, the opponent's deck), and picks plays with a locally trained model.

**Constraints:** everything runs locally. No LLMs, no external AI services or APIs, and the agent uses no pretrained weights. Data is sourced by hand and cited.

## Status

| Step | Status | Evidence |
|---|---|---|
| 1. Clean the data | Done | `data_processing.ipynb` |
| 2. Build the tables | Done (nine tables + `card_vectors`, `move_tags`) | the notebooks rebuild every table identically |
| 3. Tag the effects | Done: 54 hand-written effects + full-pool tagger | `tagger.py`, `eda_tagging.ipynb` §6b–6c; holdout scoring waits for H3 |
| 4. Feature matrix and PCA | Done (v1 and v2) | `eda_pca.ipynb` |
| 5. Decks and validator | Done | `tests/test_decks.py` |
| 6. Belief trackers | Done | `beliefs.py`, `tests/test_beliefs.py` |
| 7. Game engine and text console | Done; owner approval of the interface (T1) and rule confirmations (H1) pending | `tests/engine/`, `tests/test_console.py` |
| 8. Baseline agents and evaluation harness | Done | `reports/results/baselines.json` |
| 9. Train the model | Done: a first PPO run | `models/main/` |
| 10. Evaluate | Done: league and PCA ablation; no verdict until H5 sets the threshold | `notebooks/evaluation.ipynb` |

Tests: 245 passing, 2 skipped (`uv run pytest`, about 2.5 minutes, including 2,000 random games).

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

### 3. Tag the effects (done; holdout scoring waits for H3)

Each effect is **data** built from a small vocabulary, not code per card. An effect has a timing, a usage limit, requirements and costs, and steps. The vocabulary (`src/tcg_bot/effects.py`):
- **12 step verbs:** `MoveCards` (search, recover, discard, shuffle hand, return to hand, attach from discard), `Draw`, `SwitchActive`, `Heal`, `ApplyStatus`, `DamageSelf`, `Lockout`, `Prevent`, `ModifyStat`, `CoinFlip`, `If`, `ForEachPlayer`.
- **Shared parts:** `Fixed`, `PerCount` (multiplier), `Sum`, `IfThen`, `Cond`, `Filter`, `Target`, `Usage`.
- **Usage scopes** for abilities: `per_pokemon`, `per_name`, `per_name_contains`, plus a first-turn-only flag.
- `build()` rejects unknown blocks, parameters and values, so a tagging mistake fails loudly.

The 54 effects of the two chosen decks are in `src/tcg_bot/effects_data.py`: 52 `ok`, 1 `complex` (Lucian), 1 `no_effect_in_scope` (Psyduck's Damp). Tests in `tests/test_effects.py` check that every in-scope move has exactly one effect, that usage tags match the card text, and that filters only use real card values.

**Full pool (T3, T8).** `tagger.py` tags all 1,671 moves that have effect text, in three ways:
- **gold:** 45 hand-written effects that have text;
- **regex:** 1,324 effects matched by the patterns;
- **classifier:** the other 302, labelled by TF-IDF + logistic regression. It labels 97 of them. The remaining 205 are `unknown`, so none is silently skipped.

The patterns live only in `features.py`. Results from `eda_tagging.ipynb` §6b–6c, as macro F1 on the 15 labels with at least 15 examples:

| Split | Classifier | Nearest-neighbour baseline |
|---|---|---|
| Random | 0.96 | 0.88 |
| Unseen wording | 0.96 | 0.84 |
| Unseen family | 0.91 | 0.71 |

On unseen families, the weakest labels are `modify_stat` (0.42) and `look_top` (0.79). With the gold set held out of training, 43 of 45 gold effects are fully right.

The person-labelled holdout (H3) is still empty, so the tagger's real accuracy is not measured yet. The scoring code is in place and is tested on a synthetic file.

### 4. Card feature matrix and PCA (done)

PCA gives each card a compact vector for the model and a deck-level embedding. Plan:
- **Rows:** one per card (1,431). **Columns:** stats (HP, retreat, cost, damage), type one-hots, and block columns (which verbs a card uses, numeric parameters, filter attributes).
- **Fit on the bulk columns.** Leave the family and kind columns out (otherwise PC1 is just Pokémon vs. Trainer). Standardize. Fit once on the whole pool and freeze the fitted object.
- **Rare, nearly independent flags bypass the PCA** and are appended to the vector. A column bypasses if the chosen number of components reconstructs it poorly (for example R² below 0.5) and the engine uses it.
- **Choose the number of components** by parallel analysis and the variance curve, then check it with the ablation.
- **Check:** loadings, bootstrap stability, nearest neighbours, and the FAMD hybrid as a robustness comparison.
- **Ablation:** PCA vector only, PCA plus raw flags, raw columns only, and a learned embedding.
- **Report two sizes:** the PCA vector and the total model input per card.

**Measured (replaces the early estimate of about 3.3×).** Source: `notebooks/edas/eda_pca.ipynb`.

v1 features:
- 56 columns. 47 bulk columns are compressed to **17 components**, which keep 70.5% of the variance: a reduction of about **2.8×**.
- With the 9 bypass columns, each card has **26 inputs**.
- The cutoff is borderline: eigenvalue 17 is 1.110 against a random threshold of 1.085.
- Overall KMO is 0.21, low because the one-hot groups are included. Without them, stats and flags give 0.70 and the verbs 0.46.
- Subspace stability is 0.90, but only PC1 and PC2 are individually stable.
- 71% of 5-nearest neighbours are preserved. Agreement with the FAMD-style variant is 86%.
- 12 of the 42 deck cards share a vector with another deck card.

v2 features (T7: effect amounts, search filters, card identity):
- 66 columns. 57 bulk columns are compressed to **20 components** (70.2% of the variance). Each card has **29 inputs**, plus the identity index.
- With the identity input, **none** of the 42 deck cards share a model input.

Differences from the reference run are listed in §11 of the notebook.

### 5. Decks and validator (done)

Deck text files in `data/external/decks/` become `decks`, `deck_cards` (with `line_order`) and `deck_profile`. Lines resolve by name, then set and number, then hand-checked print aliases. Whole-number lists are used exactly as written. Average lists are rounded to a legal 60. Deck rules (deck size, copies by name, ACE SPEC, Basic Pokémon) are read from `game_constants`.

**Card instances.** `expand_deck` (`src/tcg_bot/decks.py`) turns a deck into 60 cards, each with an integer `uid` and a readable tag `<deck>_<family>_<n>`, numbered in `.txt` order (for example `lopunny_pokemon_3`). At game time, prefix the player so a mirror match doesn't collide.

### 6. Belief trackers (done)

- **Prize cards** (`PrizeBelief`): unseen cards = deck list minus every visible card. The number of copies of a card in the Prize cards is hypergeometric. Tests check hand-computed values and that the unseen pool matches the true one exactly.
- **Opponent deck** (`OpponentDeckBelief`): Bayes' rule over the known lists, with a multivariate hypergeometric likelihood of the cards the opponent has shown. With only two archetypes this model is small. It separates them after one distinctive card. Broader priors need more deck lists.

### 7. Game engine and text console (done)

- The engine runs the core rules and the 54 effects as data, with no code per card. A test cites every row of `plays.csv`, and a scenario test covers every effect.
- **Unconfirmed rules** are named settings (`RuleSettings`), listed in `reports/engine_interface.md` and `reports/decisions.md`.
- **Speed:** about 65 random games per second (11,700 actions per second) on a 2-vCPU cloud machine.
- **Robustness:** 1,000 random games per seat order finish without errors, and no card is ever created or lost.
- **Hidden information:** `observe` hides deck order, Prize cards, the opponent's hand and private choices. Tests check this at every step of 40 games.
- **Console:** typed commands are matched against the legal actions, for example `lunatone to bench`, `basic energy to solrock`, `lucario "aura jab"`. A scripted full game passes as a test.

### 8. Baseline agents and evaluation harness (done)

All matches use fixed seeds and rotate seats and decks. Intervals are 95% Wilson intervals. Source: `reports/results/baselines.json`.

| Match | Games | Win rate | 95% CI |
|---|---|---|---|
| random vs random | 400 | 0.535 | 0.486–0.583 (50% is inside) |
| heuristic vs random | 400 | 0.877 | 0.842–0.906 |
| MCTS vs random | 20 | 0.90 | 0.70–0.97 |
| MCTS vs heuristic | 20 | 0.80 | 0.58–0.92 |

MCTS used 4 determinizations, 24 iterations and play-outs of 40 actions, which took about 0.9–1.0 s per searched move. The observed opening-without-a-Basic rate matches the hypergeometric value (lopunny 0.259, mega_lucario 0.222, within 3 standard errors over 3,000 deals).

### 9. Train the model (first run done)

PPO self-play on the CPU, from random weights:
- The network is a flat-slot MLP. It scores each legal action and has a value head.
- Cards enter as the PCA vector plus the bypass columns, with a learned identity embedding.
- The run had 100 iterations of 32 games (3,200 games, about 9 minutes, seed 0).
- Opponents: random for the first 10 iterations, then a rotation of random, heuristic and snapshots of the policy.
- The run is reproducible: a test repeats a short run and gets the same parameters.

Results, from `models/main/eval.json`:

| Opponent | Games | Win rate | 95% CI |
|---|---|---|---|
| random | 200 | 0.96 | 0.92–0.98 |
| heuristic | 200 | 0.415 | 0.35–0.48 |

The model is clearly better than random. It does **not** reach the heuristic yet, and MCTS is stronger than both.

### 10. Evaluate (done, without a verdict)

`notebooks/evaluation.ipynb` puts the league, the main model and the PCA ablation side by side. The ablation compares PCA only, PCA plus bypass, PCA plus bypass plus identity, raw columns and a learned embedding, under the same training budget: 640 games per run, seeds 1–3, 100 evaluation games per opponent. Source: `reports/results/ablation.csv`.

| Card input | vs random | vs heuristic |
|---|---|---|
| PCA only | 0.903 | 0.240 |
| PCA + bypass | 0.913 | 0.227 |
| PCA + bypass + identity | 0.933 | 0.307 |
| raw columns | 0.943 | 0.303 |
| learned embedding only | 0.853 | 0.227 |

**At this budget the ablation does not show that the PCA helps.** All PCA variants have pooled intervals that overlap the raw columns. The learned embedding alone is worse against random (0.853 vs 0.943, intervals do not overlap). Two caveats:
- The pooled intervals ignore seed-to-seed variance (standard deviation 0.02–0.08).
- The budget is small, so larger budgets may change the picture.

No pass or fail verdict is given, because the success threshold has not been set (H5).

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

1. **Rules still unconfirmed.** Eight values are settings with assumed defaults (H1). The 2023 rulebook predates Mega Evolution ex, and the newer edition still has to be compared (H2).
2. **Training strength.** The first PPO run does not reach the heuristic. Longer runs, more opponents, or attention over slots are the next levers.
3. **Hidden-information leaks.** These are tested, but any new Observation field must keep the leak test passing.
4. **Scope creep.** Each added deck adds effects, and possibly new verbs.
5. **Holdout not labelled (H3).** The tagger's real accuracy on unseen text is not measured yet.

## Decisions so far

The full decision log, with sources and status, is in `reports/decisions.md`.

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