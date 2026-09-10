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


def _wild(frame):
    return frame[frame["ветка"] == "wild"]


def test_random_control_matches_the_clean_config_in_size():
    # Ради этого контроль и существует: иначе clean отличается от wild и
    # чистотой, и объёмом сразу, и эффекты не развести.
    clean = _wild(select_pairs(_pairs(), "studio_clean"))
    random = _wild(select_pairs(_pairs(), "studio_random"))
    assert len(random) == len(clean) > 0


def test_random_control_keeps_all_studio_pairs():
    got = select_pairs(_pairs(), "studio_random")
    assert (got["ветка"] == "studio").sum() == 20


def test_random_control_is_not_filtered_by_colour():
    # Выборка без оглядки на цвет: в ней должны оказаться и пары ниже порога
    # clean, иначе контроль незаметно превратился бы во второй clean.
    frame = _pairs()
    threshold = _wild(frame[frame["split"] == "train"])["colour_rank"].quantile(0.65)
    random = _wild(select_pairs(frame, "studio_random"))
    assert (random["colour_rank"] < threshold).any()


def test_random_control_is_reproducible():
    a = set(select_pairs(_pairs(), "studio_random")["pair_id"])
    b = set(select_pairs(_pairs(), "studio_random")["pair_id"])
    assert a == b


def test_random_control_draws_only_from_the_wild_pool_of_its_split():
    got = _wild(select_pairs(_pairs(), "studio_random", split="train"))
    assert set(got["pair_id"]) <= {f"w{i}" for i in range(20)}
