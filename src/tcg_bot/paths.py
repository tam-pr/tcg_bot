"""Project folders, found from this file's location so code runs from any clone.

Notebooks import these instead of typing absolute paths:

    from tcg_bot.paths import PROC, CONFIGS
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
EXTERNAL = DATA / "external"
DECK_DIR = EXTERNAL / "decks"
NOTEBOOKS = ROOT / "notebooks"
REPORTS = ROOT / "reports"

__all__ = [
    "ROOT",
    "CONFIGS",
    "DATA",
    "RAW",
    "PROC",
    "EXTERNAL",
    "DECK_DIR",
    "NOTEBOOKS",
    "REPORTS",
]
