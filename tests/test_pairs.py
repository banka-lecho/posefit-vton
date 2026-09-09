import pandas as pd
import pytest

from posefit.pairs import CONFIGS, select_pairs


def _pairs():
    rows = []
    for i in range(20):
        rows.append({"pair_id": f"s{i}", "ветка": "studio", "split": "train",
                     "colour_rank": 0.0})
    for i in range(20):
        rows.append({"pair_id": f"w{i}", "ветка": "wild", "split": "train",
                     "colour_rank": i / 20})
    rows.append({"pair_id": "t0", "ветка": "wild", "split": "test", "colour_rank": 0.9})
    return pd.DataFrame(rows)


def test_studio_config_holds_no_review_pairs():
    got = select_pairs(_pairs(), "studio")
    assert len(got) == 20
    assert (got["ветка"] == "studio").all()


def test_wild_config_adds_review_pairs():
    got = select_pairs(_pairs(), "studio_wild")
    assert len(got) == 40


def test_clean_config_is_a_subset_of_the_wild_one():
    wild = set(select_pairs(_pairs(), "studio_wild")["pair_id"])
    clean = set(select_pairs(_pairs(), "studio_clean")["pair_id"])
    assert clean < wild
    assert {p for p in wild if p.startswith("s")} <= clean


def test_clean_threshold_ignores_the_split_being_selected():
    # Порог берётся по всей wild-ветке: добавление тестовых пар не должно
    # менять состав обучающей выборки, иначе прогоны несравнимы.
    frame = _pairs()
    base = select_pairs(frame, "studio_clean")
    extra = pd.concat([frame, frame[frame["split"] == "test"]], ignore_index=True)
    assert set(base["pair_id"]) == set(select_pairs(extra, "studio_clean")["pair_id"])


def test_only_the_requested_split_is_returned():
    assert (select_pairs(_pairs(), "studio_wild", split="train")["split"] == "train").all()
    assert len(select_pairs(_pairs(), "studio_wild", split="test")) == 1


def test_unknown_config_is_rejected():
    with pytest.raises(ValueError, match="studio"):
        select_pairs(_pairs(), "everything")


def test_config_list_matches_what_selection_accepts():
    for name in CONFIGS:
        assert len(select_pairs(_pairs(), name)) > 0
