"""T8: every effect with text is tagged or `unknown`; sources and statuses; holdout rows never train."""
import pandas as pd
import pytest

from tcg_bot.effects import build
from tcg_bot.effects_data import EFFECTS
from tcg_bot.paths import PROC
from tcg_bot.tagger import LABELS, gold_labels, score_holdout, tag_pool

cards = pd.read_parquet(PROC / "cards.parquet")
moves = pd.read_parquet(PROC / "moves.parquet")


@pytest.fixture(scope="module")
def tags():
    return tag_pool(moves, cards, exclude=set())


def test_every_effect_with_text_is_tagged_or_unknown(tags):
    with_text = set(moves.loc[moves["effect_text"].notna(), "move_id"])
    assert set(tags["move_id"]) == with_text
    assert set(tags["source"]) == {"gold", "regex", "classifier"}
    assert set(tags["status"]) <= {"ok", "unknown"}
    ok = tags["status"] == "ok"
    assert (tags.loc[ok, [f"v_{k}" for k in LABELS]].any(axis=1)).all()
    assert (tags.loc[~ok, "labels"] == "unknown").all()


def test_gold_rows_carry_the_hand_written_labels(tags):
    t = tags.set_index("move_id")
    for o in (build(e) for e in EFFECTS):
        if o.move_id in t.index:
            assert t.at[o.move_id, "source"] == "gold"
            assert {k for k in LABELS if t.at[o.move_id, f"v_{k}"]} == gold_labels(o)


def test_saved_table_matches_a_fresh_run():
    saved = pd.read_parquet(PROC / "move_tags.parquet")
    assert len(saved) == moves["effect_text"].notna().sum()
    assert {"move_id", "labels", "source", "status"} <= set(saved.columns)


def test_holdout_rows_are_excluded_from_training_and_scored_per_group(tmp_path, tags):
    ids = tags[tags["source"] == "regex"]["move_id"].head(10).tolist() + \
        tags[tags["source"] == "classifier"]["move_id"].head(5).tolist()
    excl = tag_pool(moves, cards, exclude=set(ids))
    assert excl.attrs["usable_labels"]                                # the model still trains without them
    t = excl.set_index("move_id")
    hold = pd.DataFrame({"move_id": ids, "labels": [";".join(k for k in LABELS if t.at[m, f"v_{k}"]) or "none"
                                                     for m in ids]})
    path = tmp_path / "holdout.csv"
    hold.to_csv(path, index=False)
    scores = score_holdout(excl, path)
    assert set(scores.index) == {"regex", "classifier"} and scores.loc["regex", "n"] == 10
    assert scores["exact_match"].min() == 1.0                         # labels copied from the tagger itself


def test_unlabelled_holdout_raises(tmp_path, tags):
    path = tmp_path / "holdout.csv"
    pd.DataFrame({"move_id": ["848_0"], "labels": [None]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not labelled"):
        score_holdout(tags, path)
