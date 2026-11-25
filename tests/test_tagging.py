"""T3: one source of verb patterns, and a clear mapping from the split switch verbs to the single label."""
import json
import re

import pandas as pd

from tcg_bot.features import MERGE, SWITCH_ANY, VERBS, notebook_verbs, verb_flags
from tcg_bot.paths import NOTEBOOKS, PROC

moves = pd.read_parquet(PROC / "moves.parquet")
texts = moves["effect_text"].dropna()


def test_notebook_labels_are_the_17_holdout_verbs():
    nv = notebook_verbs()
    assert list(nv) == ["search_deck", "look_top", "recover_discard", "attach_energy", "discard", "shuffle_hand",
                        "return_to_hand", "draw", "switch", "heal", "status", "recoil", "lockout", "prevent",
                        "modify_stat", "coin_flip", "damage_modifier"]
    assert all(nv[k] == VERBS[k] for k in nv if k != "switch")


def test_split_switch_verbs_imply_the_single_switch_label():
    for t in texts:
        v = verb_flags(t)
        if v["switch_self"] or v["switch_opp"]:
            assert re.search(SWITCH_ANY, t), t
    assert set(MERGE) == {"switch_self", "switch_opp"}


def test_the_tagging_notebook_imports_the_patterns_instead_of_copying_them():
    nb = json.loads((NOTEBOOKS / "edas" / "eda_tagging.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "from tcg_bot.features import notebook_verbs" in code
    assert '"search_deck":' not in code


def test_holdout_cells_stay_unrun_until_h3():
    nb = json.loads((NOTEBOOKS / "edas" / "eda_tagging.ipynb").read_text(encoding="utf-8"))
    hold = [c for c in nb["cells"] if c["cell_type"] == "code" and "tag_holdout.csv" in "".join(c["source"])]
    assert len(hold) == 1 and "skip-execution" in hold[0]["metadata"].get("tags", [])
    assert not hold[0].get("outputs")
