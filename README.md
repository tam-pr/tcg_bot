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
