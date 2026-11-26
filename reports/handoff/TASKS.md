# Task list for delegation

_Written 2026-11-25 against commit `b8d152e`. Read this whole file before starting any task._

## 0. Branch rules (mandatory)

**Do ALL work outside `main`. Never commit to, push to, or merge into `main`.**

| Rule | Detail |
|---|---|
| Integration branch | `dev`, created from `main` by the repo owner |
| Task branches | `task/<id>-<short-name>`, created from the latest `origin/dev`. One task, one branch |
| Pull requests | Target **`dev`**, never `main` |
| Merging | A human reviews and merges. Agents never merge, including their own PR |
| `dev` to `main` | Only the repo owner, only when they decide |
| History | No force-push, no rebase of pushed branches, no branch deletion |
| Same machine | Each agent uses its own `git worktree`. Never switch branches under another agent |

Start every task like this:

```bash
git fetch origin
git switch -c task/<id>-<short-name> --no-track origin/dev
git branch --show-current
```

If the last command prints `main` or `dev`, stop.

Before opening a PR:

```bash
git fetch origin
git merge origin/dev
uv run pytest
git push -u origin task/<id>-<short-name>
```

The PR description states: the task ID, what changed, the commands run with their real output, what was **not** done, and any assumption made.

## 1. Project in brief

A system that plays the Pokémon Trading Card Game with a locally trained neural network. Give it a 60-card deck; it plays against an opponent deck, tracks what it cannot see (its own Prizes, the opponent's deck) and picks plays. The work also becomes a paper, so choices need evidence.

**Current training scope: two decks only**, `lopunny` and `mega_lucario` (42 distinct cards, 54 effects).

Overview: `reports/pipeline.md` (its status table and PCA numbers are out of date; this file is newer).

## 2. Hard requirements (every task)

1. **The product runs locally.** No LLMs, no external AI services or APIs, no pretrained weights in the playing agent. The model is a PyTorch network trained by self-play. (BERTopic is acceptable for analysis only.)
2. **Do not invent game rules.** A rule comes from the rulebook, a card's text, or a decision by the owner. Record the source. If unsure, mark it `needs_check` and ask.
3. **Do not hard-code behaviour per card.** Effects are data built from the vocabulary in `src/tcg_bot/effects.py`.
4. **Baseline rules versus cards.** `plays` and `game_constants` hold baseline rules; card text modifies them.
   - Action limits (one Energy attach from hand, one Supporter, one retreat per turn) apply to the action only. An effect doing the same job another way does not use them up.
   - State rules (Bench at most 5, one Active) always hold.
   - Values (HP, retreat cost, damage) are read through a lookup that applies modifiers.
5. **Never silently skip.** Unsupported input raises a clear error.
6. **Report results honestly.** Include weak numbers and failed checks.
7. **Agents must not fill in `configs/tag_holdout.csv`.** Only a human labels it. Machine labels would make the test worthless.
8. **Do not change `effects_data.py`, `configs/*.csv` or the deck lists** unless the task says so. If you find an error, report it.
9. **Do not bypass website bot protection** to fetch data. Ask the owner for the file.

## 3. Setup and baseline

Python 3.13, managed with `uv`. From the repo root:

```bash
uv sync --frozen
uv run pytest
```

Checked on a fresh clone on 2026-11-25: **115 passed, 2 skipped**. All nine tables in `data/processed/` rebuilt identically from `data_processing`, `plays`, `deck_building` (run in that order), with paths redirected to the clone.

## 4. What exists

| Area | State |
|---|---|
| Card data | `cards` (1,431), `moves` (2,265) built from `data/raw/EN_Card_Data_R2_full.csv` |
| Rules tables | `plays` (23), `game_constants` (16), `special_conditions` (5), `energy_provides` (25) |
| Decks | Two lists, validator, `expand_deck` (60 tagged instances per deck) |
| Effects | 54 effects as data: 52 `ok`, Lucian `complex`, Psyduck Damp `no_effect_in_scope` |
| Tagging EDA | `notebooks/edas/eda_tagging.ipynb` |
| PCA, tagging cross-validation | Written and tested, **not yet in the project**: see `reports/handoff/reference/` |
| Engine, console, beliefs, baselines, model | Not started |

## 5. Known problems

| Problem | Handled by |
|---|---|
| Notebooks use the author's absolute paths, so they fail on any other machine | T0 |
| No README; `pipeline_files/` (leftover render assets) is tracked; `packages.txt` repeats `pyproject.toml` | T0 |
| Verb patterns would live in two places (tagging notebook and `features.py`) | T3 |
| Dunsparce and Buneary get identical PCA vectors | T7 |
| Effect features have no amounts and no search filters | T7 |
| Four rules unconfirmed (see H1) | Humans |
| Holdout labelled 0 of 100 | Humans (H3) |

## 6. Human tasks (do not delegate to an agent)

| ID | Task | Blocks |
|---|---|---|
| H1 | Confirm with an experienced player or an official source: (a) does evolving into a Mega Evolution ex end the turn, (b) Prize cards for knocking one out, (c) the Resistance amount, (d) may a turn end without attacking, (e) does effect-attached Energy leave the hand attach available, (f) who orders cards put on the bottom of the deck, (g) can Hilda take only one of its two cards, (h) does Wild Press recoil use Weakness and Resistance. Then update the CSVs and `status` | Final T4, T5 |
| H2 | Download the newer official rulebook by hand and compare it with `plays.csv` | Final T4 |
| H3 | Label the 100 rows of `configs/tag_holdout.csv` from card text only (verb list in Appendix A). Type `none` when no verb applies | T8 |
| H4 | Review the meaning of the effects in `effects_data.py` against the cards | Final T5 |
| H5 | Decide: exact wording of the PCA requirement; console as simulator or advisor; deadline and minimum deliverable; whether MCTS stays; the success threshold (before training) | T9, T11, T12, T13 |
| H6 | Create `dev`; commit to it this folder, `configs/tag_holdout.csv` and the uncommitted tagging notebook; push it; turn on branch protection for `main` on GitHub (require a pull request); give teammates access | Everything |

## 7. Order of work

| Wave | Tasks that can run at the same time | Needs |
|---|---|---|
| 0 | T0 | H6 |
| 1 | T1, T2 | T0 |
| 2 | T4 and T6 (after T1 is approved); T3 and T7 (after T2) | |
| 3 | T5 (after T4); T8 (after T3 and H3) | |
| 4 | T9, T10 | T4, T5 |
| 5 | T11 (after T6, T10); T12 (after T7, T10) | |
| 6 | T13 | T12 |

T14 runs alongside all waves, with one owner.

**Files owned** below are the only files a task may create or edit. Anything else: ask first. One owner per notebook (notebooks merge badly). Do not commit regenerated files in `data/processed/` unless your task owns them.

## 8. Agent tasks

### T0. Make the repo portable
- **Branch:** `task/t0-portability` · **Depends:** H6
- **Owns:** `src/tcg_bot/paths.py`, the four notebooks (path lines only), `README.md`, `packages.txt`, `pipeline_files/`, `pyproject.toml` (description only)
- **How:** add `paths.py` with `ROOT = Path(__file__).resolve().parents[2]` and the data folders; replace every absolute path in the notebooks; write a README (setup, test command, notebook order, the branch rules above). Ask the owner before deleting `pipeline_files/` or `packages.txt`.
- **Done when:** no `/Users/` string remains in code cells; the three build notebooks run on a fresh clone; the rebuilt Parquet files equal the committed ones; tests pass.

### T1. Engine interface contract
- **Branch:** `task/t1-interfaces` · **Depends:** T0
- **Owns:** `src/tcg_bot/engine/__init__.py`, `engine/types.py`, `reports/engine_interface.md`
- **How:** define the types and signatures every later task builds on. Starting proposal:
  ```python
  new_game(deck_ids: tuple[str, str], seed: int) -> State
  legal_actions(state: State) -> list[Action]       # empty only when the game is over
  step(state: State, action: Action) -> State       # returns a new State; the input is unchanged
  observe(state: State, player: int) -> Observation # only what that player may see
  winner(state: State) -> int | None
  ```
  `State`: per player one Active slot, up to 5 Bench slots, deck, hand, discard, Prizes, Pokémon in play with damage, Energy, Tool, conditions and evolution history; Stadium; turn number and per-turn flags; lasting effects; pending choice; seeded random generator. `Action`: kind, player, source, target, choice, all as plain values so it can be listed, compared and encoded.
- **Requirements:** cards are instances, not card IDs, and IDs are unique **across both players** (`expand_deck` numbers each deck from 0, so a mirror match would collide). State must be cheap to copy. A multi-step choice is a pending choice answered by further actions. Define how a draw by turn limit is reported.
- **Done when:** types import, have tests for construction and copying, the document is written, and **the owner approves it**. T4 and T6 wait for that approval.

### T2. Card features and PCA
- **Branch:** `task/t2-pca` · **Depends:** T0
- **Owns:** `src/tcg_bot/features.py`, `notebooks/edas/eda_pca.ipynb`, `tests/test_features.py`, `data/processed/card_vectors.parquet`, `data/processed/card_pca.joblib`
- **How:** copy `reference/features.py` into `src/tcg_bot/`; build the notebook from `reference/eda_pca_cells.py`; add tests (one row per card; no missing values after `standardize`; Energy and non-fossil Trainers have no HP before it; `switch_self` and `switch_opp` agree with `SwitchActive.side` for the hand-written effects).
- **Expected results** (from the reference run): 56 columns; missing-as-zero gives PC1 correlation with `is_pokemon` 0.68, neutral gives 0.03; KMO 0.34 overall, 0.67 stats and flags, 0.46 verbs; 47 columns to 21 components, 79.6% variance; 9 bypass columns (`is_ace_spec`, `is_ancient`, `has_owner`, `switch_opp`, `heal`, `recoil`, `modify_stat`, `coin_flip`, `status`); 30 inputs per card; subspace stability 0.94; only PC1 to PC3 individually stable; 78% of 5 nearest neighbours preserved; 97% agreement with the FAMD-style variant.
- **Requirements:** keep the stated limits in the notebook (moderate reduction, low KMO, borderline cutoff at 21). Do not tune the method to get nicer numbers.
- **Done when:** the notebook runs top to bottom, the numbers match or every difference is explained, tests pass.

### T3. Tagging evaluation, one source of verb patterns
- **Branch:** `task/t3-tagging-eval` · **Depends:** T2
- **Owns:** `notebooks/edas/eda_tagging.ipynb`, `tests/test_tagging.py`
- **How:** add the 6b and 6c cells from `reference/tagging_cells.py` after section 6. Add the 6d cells but leave them unrun until H3 is done. If you import the verb patterns from `tcg_bot.features`, note that it splits `switch` into two verbs while the notebook, the gold labels and the holdout use one `switch`: keep a clear mapping.
- **Expected results:** 1,671 effects, 957 distinct masked wordings, 568 wording families. Logistic regression macro F1 0.96 random, 0.96 unseen wording, 0.91 unseen family; nearest-neighbour baseline 0.88, 0.84, 0.70; `modify_stat` 0.43 and `attach_energy` 0.79 on unseen families; 43 of 45 gold effects fully right.
- **Done when:** the notebook runs top to bottom and its text matches its output.

### T4. Engine: core rules
- **Branch:** `task/t4-engine-core` · **Depends:** T1 approved
- **Owns:** `src/tcg_bot/engine/` (except `effects_runtime.py` and `types.py`), `tests/engine/` (except `test_effects_runtime.py`)
- **How:** setup (coin flip, 7 cards, mulligan, Active, Bench, 6 Prizes), turn draw, bench, evolve, attach, Items, Supporter, Stadium, Tool, retreat, abilities hook, attack cost and damage, Weakness, Resistance, Knock Out, Prizes, promotion, Checkup and Special Conditions, win checks and Sudden Death. Read limits and constants from the tables.
- **Requirements:** every `plays` row is implemented or listed as not applicable. Unconfirmed values are named settings with a visible assumption, not buried numbers. Same seed and actions give the same game. A turn limit prevents endless games. `observe` leaks nothing hidden, with tests proving it. Measure games per second and report it.
- **Done when:** random-legal-action games between the two decks finish without errors across at least 1,000 seeds (card text may be stubbed with a clear marker), no card is created or lost, and each rule has a test citing its `play_id`.

### T5. Engine: effect interpreter
- **Branch:** `task/t5-effects-runtime` · **Depends:** T4 core merged
- **Owns:** `src/tcg_bot/engine/effects_runtime.py`, `tests/engine/test_effects_runtime.py`, small agreed hook points in T4 files
- **How:** execute the 12 step verbs, amounts, conditions, filters, usage limits and the seven timings for all 54 effects. Passive effects (Air Balloon, Gravity Mountain, Battle Cage, Mist Energy) act through modifier lookups. Choices appear as legal actions.
- **Requirements:** no per-card branches. Lucian needs a coin flip per player. Damp stays a no-op, with its guard test kept. Unknown blocks raise.
- **Done when:** each of the 54 effects has at least one scenario test, and 1,000 random games with effects enabled finish cleanly.

### T6. Belief trackers
- **Branch:** `task/t6-beliefs` · **Depends:** T1 approved
- **Owns:** `src/tcg_bot/beliefs.py`, `tests/test_beliefs.py`
- **How:** Prize beliefs by counting (unseen = deck list minus seen cards, hypergeometric odds); opponent deck by a Bayesian update over known lists using `deck_profile`.
- **Requirements:** input is `Observation` only, never `State`. With two archetypes the opponent model is small: say so. Broader priors need more lists from the owner.
- **Done when:** tests check exact probabilities on hand-computed cases and that probabilities sum correctly.

### T7. Features v2: amounts, filters, card identity
- **Branch:** `task/t7-features-v2` · **Depends:** T2
- **Owns:** same files as T2
- **How:** add effect amounts (cards drawn, damage healed, bonus damage, search count) and search filters; add evolution links or a card identity input next to the PCA vector. Rerun every T2 diagnostic.
- **Done when:** no two of the 42 deck cards share a model input, and the notebook compares v1 with v2 in one table.

### T8. Full-pool tagger
- **Branch:** `task/t8-tagger` · **Depends:** T3, H3
- **Owns:** `src/tcg_bot/tagger.py`, `tests/test_tagger.py`, `data/processed/move_tags.parquet`
- **How:** regex first, then a local TF-IDF plus logistic regression classifier for the rest. Each row records its source (`regex`, `classifier`, `gold`) and a status, with `unknown` allowed.
- **Requirements:** score on the holdout, report the two groups separately, and never tune patterns on holdout rows.
- **Done when:** every effect with text has tags or `unknown`, and holdout scores are reported.

### T9. Text console
- **Branch:** `task/t9-console` · **Depends:** T4, T5, H5
- **Owns:** `src/tcg_bot/console.py`, `tests/test_console.py`, the `tcg-bot` entry point in `src/tcg_bot/__init__.py`
- **How:** typed commands such as `solrock to active`, `basic energy to solrock`, `lunatone to bench`, matched against `legal_actions`. The reply uses the same style, such as `model: lucario to active, lucario "attack name"`.
- **Requirements:** ambiguous or illegal commands explain why and list options. Runs with a random or heuristic opponent until a model exists.
- **Done when:** a scripted full game passes as a test.

### T10. Baseline agents and evaluation harness
- **Branch:** `task/t10-harness` · **Depends:** T4, T5
- **Owns:** `src/tcg_bot/agents/` (random, heuristic), `src/tcg_bot/evaluate.py`, `tests/test_harness.py`
- **Requirements:** fixed seeds, both seats and both decks, confidence intervals, agents see `Observation` only.
- **Done when:** random against random is near 50% within its interval, no illegal actions occur, and the opening-without-a-Basic rate matches the calculated probability.

### T11. Determinized MCTS (optional)
- **Branch:** `task/t11-mcts` · **Depends:** T6, T10, H5
- **Owns:** `src/tcg_bot/agents/mcts.py`, `tests/test_mcts.py`
- **Requirements:** samples hidden cards from belief trackers; never reads the true hidden state.
- **Done when:** it beats random clearly, with games, time per move and interval reported.

### T12. Model and training
- **Branch:** `task/t12-training` · **Depends:** T7, T10, H5
- **Owns:** `src/tcg_bot/model/`, `scripts/train.py`, `models/` (small metadata only), `tests/test_model.py`, `pyproject.toml` and `uv.lock` (to add `torch`)
- **How:** encode the observation (card vector, bypass columns, family and kind, identity or evolution input); score each legal action with masking; value head; PPO self-play. Start with a flat-slot MLP, attention later.
- **Requirements:** trained from random weights, locally. Seeds and configs saved with every run. Keep large checkpoints out of git. One fixed opponent first, then rotation.
- **Done when:** a short run is reproducible, learning curves are saved, and results against random and heuristic are reported with intervals.

### T13. Evaluation and ablation
- **Branch:** `task/t13-evaluation` · **Depends:** T12, success threshold from H5
- **Owns:** `notebooks/evaluation.ipynb`, `reports/results/`
- **How:** compare PCA only, PCA plus bypass, raw columns, learned embedding under equal training budgets and several seeds; play every baseline.
- **Done when:** tables show win rates with intervals and the threshold verdict, including if PCA does not help.

### T14. Paper material
- **Branch:** `task/t14-paper` · **Depends:** none · **One owner for the whole project**
- **Owns:** `reports/pipeline.md`, `reports/decisions.md`, `reports/figures/`
- **How:** update the pipeline status and replace the early PCA estimate (about 3.3×) with measured results; keep a decision log (date, decision, reason, source, status); record data sources and citations (card CSV, deck lists, rulebook); collect figures.
- **Requirements:** claim only what a notebook or test shows.

## 9. Message to give each agent

```
Read reports/handoff/TASKS.md in full. Do task <ID> only.
Work OUTSIDE the main branch: branch task/<id>-<name> from origin/dev, and open the PR into dev.
Never commit, push or merge to main, and do not merge your own PR.
Edit only the files the task owns. If something is unclear or a rule is unconfirmed, ask; do not guess.
Finish by reporting what you ran, the real output, and what you did not do.
```

## Appendix A. Holdout verbs

| Verb | Use it when the effect... |
|---|---|
| `search_deck` | takes cards by searching your deck |
| `look_top` | looks at the top cards of a deck |
| `recover_discard` | puts cards from your discard pile into your hand |
| `return_to_hand` | puts cards into your hand from anywhere (often alongside search or recover) |
| `draw` | draws cards |
| `shuffle_hand` | sends a hand back into the deck |
| `discard` | puts any card into a discard pile |
| `attach_energy` | attaches Energy itself |
| `switch` | changes either player's Active Pokémon |
| `heal` | heals damage |
| `status` | applies Asleep, Burned, Confused, Paralyzed or Poisoned |
| `recoil` | damages the attacking Pokémon itself |
| `lockout` | stops something from being used or done |
| `prevent` | prevents damage or effects |
| `modify_stat` | changes HP, Retreat Cost, or damage done or taken while it applies |
| `coin_flip` | flips coins |
| `damage_modifier` | changes this attack's own damage, ignores effects on it, or can make it do nothing |
