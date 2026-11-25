# Decision log

_One row per decision: date, decision, reason, source, status._

Status values:
- `decided`: made by the owner or forced by the card text or rulebook.
- `assumption`: an unconfirmed value, kept as a named setting (`RuleSettings`) or flagged in the data.
- `needs_check`: waiting on a person (tasks H1 to H5).
- `implementation`: an engineering choice that does not change the rules.

## Rules and engine

| Date | Decision | Reason | Source | Status |
|---|---|---|---|---|
| 2026-11-24 | The Bench maximum stays at 5. Buddy-Buddy Poffin puts as many Pokémon as fit, up to 2. | State rules always hold, whatever put the card there. | `effects_data.py` (1086_0 notes) | needs_check (confirm with a player) |
| 2026-11-24 | Per-name Ability limits are tagged by scope (`per_pokemon`, `per_name`, `per_name_contains`). | Card text: "You can't use more than 1 … Ability". | `effects.py`, `tests/test_effects.py` | decided |
| 2026-11-24 | `has_rule_box` is a column on `cards`. | Poké Pad filters on it. | `cards` table | decided |
| 2026-11-25 | Evolving into a Mega Evolution Pokémon ex does **not** end the turn (`mega_evolution_ends_turn=False`). | The 2023 rulebook rule is about the old Mega-EX cards. | `plays.csv` main_mega_turn_end | assumption (H1a) |
| 2026-11-25 | A Mega Evolution Pokémon ex gives **3** Prize cards (`mega_ex_prize_value=3`). | The `game_constants` value is empty (needs_check). | `game_constants.csv` | assumption (H1b) |
| 2026-11-25 | Resistance subtracts **30** (`resistance_amount=30`). | The 2023 rulebook does not state the number. | `game_constants.csv` | assumption (H1c) |
| 2026-11-25 | A player may end the turn without attacking. | `turn_end` is `inferred`. | `plays.csv` turn_end | assumption (H1d) |
| 2026-11-25 | Energy attached by an effect (Aura Jab) does not use the once-per-turn attach from hand. | Action limits apply to the action only (TASKS.md, rule 4). | TASKS.md §2 | assumption (H1e) |
| 2026-11-25 | Cards put on the bottom of the deck by Lucian or Special Red Card go in random order. | The card text says the hand is shuffled first. | card text 1237_0, 1413_0 | assumption (H1f) |
| 2026-11-25 | A search for cards with stated criteria may find fewer, so Hilda may take only one card. | The deck is hidden from the opponent, so a short search can't be checked. | — | assumption (H1g) |
| 2026-11-25 | Wild Press recoil ignores Weakness and Resistance. | Those apply to the opponent's Active Pokémon. | — | assumption (H1h) |
| 2026-11-25 | The opponent always takes the extra mulligan cards. | `setup_mulligan` says "may". This choice is a setting. | `plays.csv` setup_mulligan | assumption |
| 2026-11-25 | Every unconfirmed value is a named field of `RuleSettings`, and `assumptions()` returns them for logs. | Rule 4 of T4: no buried numbers. | `engine/types.py` | implementation |
| 2026-11-25 | Card uid = `player * 100 + i`. A Pokémon in play is named by the uid of its Basic card (pid). | IDs must be unique across both players (mirror matches). The pid survives evolution. | T1 | implementation |
| 2026-11-25 | Game turn limit 200. Reaching it is a draw (`result="draw_turn_limit"`). | T4: a turn limit prevents endless games. | T1/T4 | implementation |
| 2026-11-25 | Sudden Death restarts the deal with 1 Prize card and keeps the global turn counter running. | The turn limit still bounds the whole game. | `plays.csv` win_check | implementation |
| 2026-11-25 | A Prize card taken after a Knock Out is chosen at random. | Prize cards are face down, so a choice and a random pick are the same. | — | implementation |
| 2026-11-25 | Identical copies of a card in a hidden or public zone are listed once in `legal_actions`, using the lowest uid. | Identical cards are interchangeable, and this keeps the action space small. | T1 | implementation |
| 2026-11-25 | Knock Outs are checked after every main-phase action as well as after attacks and Checkup. | A Stadium that lowers HP (Gravity Mountain) can Knock Out at once. | — | implementation |
| 2026-11-25 | During setup, the opponent's face-down Active and Bench are hidden until both players finish. | Hidden information. | `setup_place_active` | implementation |

## Data, features and tagging

| Date | Decision | Reason | Source | Status |
|---|---|---|---|---|
| 2026-11-24 | PCA is fit on the bulk columns. Rare, independent flags bypass it. | Low R² flags lose their meaning when compressed. | `reports/pipeline.md` | decided |
| 2026-11-25 | Notebooks get paths from `tcg_bot.paths`. | Portability (T0). | T0 | decided |
| 2026-11-25 | Pokémon-only features are "neutral" (the column mean) for other cards, not 0. | With 0, PC1 measures Pokémon vs. non-Pokémon (\|corr\| 0.64 vs 0.08). | `eda_pca.ipynb` §1 | decided |
| 2026-11-25 | `is_ex` is dropped from the features (identical to `has_rule_box` in this pool). `n_effects` is added. | An exact duplicate makes the correlation matrix singular. | `eda_pca.ipynb` §11 | implementation |
| 2026-11-25 | Number of components from parallel analysis (50 permutations, 95th percentile): 17 for v1, 20 for v2. | A rule fixed in advance, not tuned. The cutoff is borderline. | `eda_pca.ipynb` §3, §9 | decided |
| 2026-11-25 | The 9 bypass columns are kept as designed. Each has R² < 0.5 in v1. 9 more columns also have R² < 0.5 but stay in the bulk. | Fixed design; reported honestly. | `eda_pca.ipynb` §4 | needs_check (revisit after ablation) |
| 2026-11-25 | Features v2 add effect amounts, search filters and a card identity input. "Heal all" counts as 300. | Without them, 12 of the 42 deck cards share a vector. | `eda_pca.ipynb` §9 | decided |
| 2026-11-25 | Wording families = connected components of masked wordings with TF-IDF cosine ≥ 0.7. | Needed for the unseen-family split. The reference definition was not available. | `eda_tagging.ipynb` 6b | implementation |
| 2026-11-25 | Verb patterns live only in `tcg_bot.features`. `switch` is split into `switch_self` / `switch_opp`, and `notebook_verbs()` maps them back to the single `switch` label. | One source of truth (T3). | `features.py` | decided |
| 2026-11-25 | The tagger's classifier labels a row only at probability ≥ 0.5. Otherwise the row is `unknown`. Holdout rows never train. | Never silently skip; never tune on the holdout. | `tagger.py` | implementation |

## Agents, model and evaluation

| Date | Decision | Reason | Source | Status |
|---|---|---|---|---|
| 2026-11-25 | The console is a simulator, with a `hint` command acting as the advisor. | H5 has not decided between the two. | `console.py` | needs_check (H5) |
| 2026-11-25 | MCTS hands choices inside effects and setup to the heuristic. | The Observation does not carry the interpreter position needed to rebuild them. | `agents/mcts.py` | implementation |
| 2026-11-25 | MCTS stays in scope as an optional baseline. | It works, but it is slow. | T11 | needs_check (H5) |
| 2026-11-25 | `torch` comes from the CPU-only PyTorch index. | Small download, runs on any laptop. | `pyproject.toml` | implementation |
| 2026-11-25 | PPO reward: +1/−1/0 at the end, plus 0.05 per Prize card taken (and −0.05 per Prize card lost). | Sparse wins alone learn slowly in short runs. | `model/ppo.py` | implementation |
| 2026-11-25 | Training opponents: random only for the first third of the run, then random, heuristic and snapshots of the policy. | T12: "one fixed opponent first, then rotation". | `model/ppo.py` | decided |
| 2026-11-25 | No success threshold is applied in the evaluation. | The threshold must be set by a person before training (H5). | T13 | needs_check (H5) |

## Process

| Date | Decision | Reason | Source | Status |
|---|---|---|---|---|
| 2026-11-25 | Task branches start from `main` (`b8d152e`). Each task branch is stacked on the previous one. | `origin/dev` does not exist yet (H6). The owner should rebase or retarget the PRs to `dev`. | TASKS.md §0 | needs_check (H6) |

## Data sources

| Source | Where | Notes |
|---|---|---|
| Card data | `data/raw/EN_Card_Data_R2_full.csv` | 2,273 rows (one per move) become 1,431 cards and 2,265 moves. The owner should add the citation (origin, date of download). |
| Deck lists | `data/external/decks/lopunny.txt`, `mega_lucario.txt` | Limitless deck lists (`decks.source = decklist`). The owner should add the URLs and dates. |
| Rules | Pokémon TCG rulebook, 2023 edition (page numbers in `configs/plays.csv`, `game_constants.csv`, `special_conditions.csv`) | The newer edition still has to be compared by hand (H2). |
