import numpy as np
import pandas as pd

from posefit.arch import arch_flags, uses_condition
from posefit.reliability import (
    BUCKETS, N_TOKENS, NULL_TOKEN, STUDIO_TOKEN, assign_reliability, bucket_thresholds, describe,
)


def _pairs(n_wild=10):
    studio = pd.DataFrame({"ветка": ["studio"] * 3, "colour_rank": [np.nan] * 3})
    wild = pd.DataFrame({"ветка": ["wild"] * n_wild,
                         "colour_rank": np.linspace(0, 1, n_wild)})
    return pd.concat([studio, wild], ignore_index=True)


def test_studio_pairs_are_always_the_studio_bucket():
    bucket = assign_reliability(_pairs())
    assert (bucket[:3] == STUDIO_TOKEN).all()


def test_wild_pairs_split_by_colour_rank_quantiles():
    pairs = _pairs()
    bucket = assign_reliability(pairs, clean_quantile=0.65, noisy_quantile=0.3)
    upper, lower = bucket_thresholds(pairs, 0.65, 0.3)
    wild = pairs["ветка"] == "wild"
    rank = pairs["colour_rank"]
    assert (bucket[wild & (rank >= upper)] == BUCKETS.index("wild_clean")).all()
    assert (bucket[wild & (rank < lower)] == BUCKETS.index("wild_noisy")).all()
    assert (bucket[wild & (rank >= lower) & (rank < upper)] == BUCKETS.index("wild_mid")).all()


def test_clean_bucket_matches_the_baseline_selection():
    # Корзина wild_clean обязана совпадать с отбором studio_clean при той же
    # квантили: иначе якорная схема и базовая линия сравнивали бы разные вещи.
    from posefit.pairs import select_pairs

    pairs = _pairs(20)
    pairs["split"] = "train"
    clean = select_pairs(pairs, "studio_clean", "train", 0.65)
    n_clean_wild = int((clean["ветка"] == "wild").sum())
    bucket = assign_reliability(pairs, clean_quantile=0.65)
    assert int((bucket == BUCKETS.index("wild_clean")).sum()) == n_clean_wild


def test_wild_without_colour_rank_is_noisy():
    pairs = _pairs()
    pairs.loc[5, "colour_rank"] = np.nan
    assert assign_reliability(pairs)[5] == BUCKETS.index("wild_noisy")


def test_frame_without_wild_column_is_all_studio():
    frame = pd.DataFrame({"pair_id": ["a", "b"]})
    assert (assign_reliability(frame) == STUDIO_TOKEN).all()


def test_null_token_is_outside_the_buckets():
    assert NULL_TOKEN == len(BUCKETS)
    assert N_TOKENS == len(BUCKETS) + 1


def test_describe_counts_every_bucket():
    counts = describe(assign_reliability(_pairs()))
    assert set(counts) == set(BUCKETS)
    assert sum(counts.values()) == 13


def test_catvton_ignores_component_flags():
    flags = arch_flags({"arch": {"name": "catvton", "guide": True, "reliability": True}})
    assert not flags["guide"] and not flags["reliability"] and not flags["garment_embed"]
    assert flags["cond_dropout"] == 0.0
    assert not uses_condition(flags)


def test_anchor_defaults_to_all_components():
    flags = arch_flags({"arch": {"name": "anchor"}})
    assert flags["guide"] and flags["reliability"] and flags["garment_embed"]
    assert uses_condition(flags)


def test_anchor_components_can_be_ablated():
    flags = arch_flags({"arch": {"name": "anchor", "reliability": False, "garment_embed": False}})
    assert flags["guide"] and not uses_condition(flags)


def test_missing_arch_section_means_catvton():
    assert arch_flags({})["name"] == "catvton"


def test_unknown_arch_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="неизвестная архитектура"):
        arch_flags({"arch": {"name": "warpnet"}})


def test_empty_flags_mean_no_condition():
    # generate() по умолчанию получает пустой словарь флагов — это базовая схема.
    assert not uses_condition({})
