# tcg_bot

A Pokémon TCG engine for two decks (`lopunny`, `mega_lucario`), a small neural network trained on it by playing, and the PCA analysis of the card data. Everything runs locally; nothing is pretrained.

## Setup

You need [git](https://git-scm.com) and [uv](https://docs.astral.sh/uv/getting-started/installation/). uv fetches Python 3.13 by itself if needed.

```bash
git clone https://github.com/tam-pr/tcg_bot.git
cd tcg_bot
uv sync --frozen
```

The first `uv sync` downloads the libraries, including PyTorch, so it takes a while.

Check that it works:

```bash
uv run pytest
```

## Play

Against the trained network:

```bash
uv run python -m tcg_bot.console --deck mega_lucario --model models/first
```

Use `--deck lopunny` for the other side. Leave out `--model` to play the simple rule-following stand-in. Type `help` during a game for the commands.

Watch a game between two stand-ins:

```bash
uv run python -m tcg_bot.narrate --delay 0.5
```

## Train and evaluate

Win rate of the saved model:

```bash
uv run python -m tcg_bot.model.evaluate models/first --games 1000
```

Train a new one (about 2 seconds per iteration on an Apple M3):

```bash
uv run python -m tcg_bot.model.train --name mine --iterations 200
```

Continue it with `--resume`. Repeat any other options you used, because `--resume` does not remember them.

## Notebooks

Register the kernel once, then pick `tcg_bot` in the notebook:

```bash
uv run python -m ipykernel install --user --name tcg_bot
```

| Notebook | What it does |
|---|---|
| `notebooks/eval_model.ipynb` | evaluation of the trained network |
| `notebooks/edas/eda_pca.ipynb` | PCA of the card features |
| `notebooks/edas/eda_tagging.ipynb` | tagging card effects from their text |
| `notebooks/data_processing.ipynb`, `plays.ipynb`, `deck_building.ipynb` | build the tables in `data/processed/` |

The last three still contain paths from the original computer, so change them before running. You don't need to run them: the tables are included.

## Notes

- Tested only on a Mac with Apple silicon. The PyTorch version used here also has builds for 64-bit Windows and Linux (the Linux download is much larger), but not for Intel Macs. Macs need macOS 14 or newer.
- Only the 42 cards of the two decks are supported.
- `reports/pipeline.md` describes the design; parts of its status table are out of date. `reports/handoff/` holds older working notes.
