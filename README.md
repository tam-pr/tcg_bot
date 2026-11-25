# tcg-bot

A system that plays the Pokémon Trading Card Game with a neural network trained locally by self-play. You give it a 60-card deck and it plays against an opponent deck. It tracks what it cannot see (its own Prize cards and the opponent's deck) and chooses plays.

Training currently covers two decks: `lopunny` and `mega_lucario`.

The product runs entirely locally. It uses no LLMs, no external AI services and no pretrained weights.

## Setup

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync --frozen
```

## Tests

```bash
uv run pytest
```

## Repository layout

| Path | Contents |
|---|---|
| `src/tcg_bot/` | Package code: deck tools (`decks.py`), the effect vocabulary (`effects.py`), effects as data (`effects_data.py`), project folders (`paths.py`) |
| `configs/` | Rules tables written by hand: `plays`, `game_constants`, `special_conditions`, `energy_provides` |
| `data/raw/` | Source card data (`EN_Card_Data_R2_full.csv`) |
| `data/external/decks/` | Deck lists |
| `data/processed/` | Parquet tables built by the notebooks |
| `notebooks/` | Build notebooks and EDAs |
| `reports/` | Pipeline overview and other reports |
| `tests/` | Tests |

Code and notebooks get folder paths from `tcg_bot.paths` (for example `from tcg_bot.paths import PROC, CONFIGS`). Do not type absolute paths.

## Notebook order

Each build notebook reads the tables written by the ones before it, so run them in this order:

1. `notebooks/data_processing.ipynb` builds `cards` and `moves` from the raw CSV.
2. `notebooks/plays.ipynb` builds `plays`, `game_constants` and `special_conditions` from `configs/`.
3. `notebooks/deck_building.ipynb` builds `decks`, `deck_cards`, `deck_profile` and `energy_provides`.

After that, `notebooks/edas/eda_tagging.ipynb` can be run at any time. It is analysis only and writes no tables.

To run a notebook from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/data_processing.ipynb --output-dir /tmp
```

## Using the system

| What | Command |
|---|---|
| Play against the heuristic in the text console | `uv run tcg-bot` (`--opponent random`, `--deck`, `--opp-deck`, `--seed`) |
| Train the model (PPO self-play, CPU) | `uv run python scripts/train.py --name main --iterations 100 --games-per-iter 32` |
| Baseline league (random, heuristic, MCTS) | `uv run python reports/results/run_baselines.py 400 20` |
| PCA ablation (five card inputs, three seeds) | `uv run python reports/results/run_ablation.py 40 16 100 1 2 3` |

In the console, type commands such as `lunatone to bench`, `basic energy to solrock`, `play ultra ball` or `lucario "aura jab"`. Then:
- `help` lists the legal commands.
- `board` shows the board.
- `hint` shows what the advisor would play.

Analysis notebooks:
- `notebooks/edas/eda_tagging.ipynb`: effect tagging.
- `notebooks/edas/eda_pca.ipynb`: card features and PCA (v1 and v2). It writes `card_vectors.parquet` and `card_pca.joblib`.
- `notebooks/evaluation.ipynb`: the baseline league, the main model and the PCA ablation. It reads `reports/results/` and `models/main/`.

Code map:

| Path | Contents |
|---|---|
| `src/tcg_bot/engine/` | The rules engine. API: `new_game`, `legal_actions`, `step`, `observe`, `winner` (see `reports/engine_interface.md`) |
| `src/tcg_bot/beliefs.py` | The Prize card and opponent-deck belief trackers |
| `src/tcg_bot/agents/` | Random, heuristic and determinized MCTS players |
| `src/tcg_bot/evaluate.py` | Matches with both seats, both decks and Wilson intervals |
| `src/tcg_bot/console.py` | The text console |
| `src/tcg_bot/features.py`, `tagger.py` | Card features and PCA, and the full-pool effect tagger |
| `src/tcg_bot/model/` | Encoder, policy-value network and PPO |

Rule values that are not confirmed yet are fields of `tcg_bot.engine.RuleSettings` (see `reports/decisions.md`).

## Branch rules

**Do all work outside `main`. Never commit to, push to or merge into `main`.**

| Rule | Detail |
|---|---|
| Integration branch | `dev`, created from `main` by the repo owner |
| Task branches | `task/<id>-<short-name>`, created from the latest `origin/dev`. One task per branch |
| Pull requests | Target **`dev`**, never `main` |
| Merging | A human reviews and merges. Agents never merge, including their own PR |
| `dev` to `main` | Only the repo owner, and only when they decide |
| History | No force-push, no rebase of pushed branches, no branch deletion |
| Same machine | Each agent uses its own `git worktree`. Never switch branches under another agent |

Start every task like this:

```bash
git fetch origin
git switch -c task/<id>-<short-name> --no-track origin/dev
git branch --show-current   # if this prints main or dev, stop
```

Before opening a PR:

```bash
git fetch origin
git merge origin/dev
uv run pytest
git push -u origin task/<id>-<short-name>
```

The PR description must give the task ID, what changed, the commands run with their real output, what was **not** done, and any assumptions made.
