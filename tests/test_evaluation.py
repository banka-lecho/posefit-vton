import numpy as np

from posefit.evaluation import mask_bbox, masked_l1, mean_ci, pair_seed, paired_difference


def test_pair_seed_is_stable_and_pair_specific():
    assert pair_seed("a_b") == pair_seed("a_b")
    assert pair_seed("a_b") != pair_seed("a_c")
    assert pair_seed("a_b", base=1) != pair_seed("a_b", base=0)


def test_mask_bbox_pads_and_clips_to_the_frame():
    mask = np.zeros((20, 20), np.uint8)
    mask[5:10, 2:6] = 255
    assert mask_bbox(mask, pad=3) == (2, 13, 0, 9)


def test_mask_bbox_of_an_empty_mask_is_none():
    assert mask_bbox(np.zeros((8, 8), np.uint8)) is None


def test_masked_l1_ignores_pixels_outside_the_mask():
    pred = np.zeros((4, 4, 3), np.uint8)
    true = np.zeros((4, 4, 3), np.uint8)
    true[:2] = 255                     # отличие только в верхней половине
    mask = np.zeros((4, 4), np.uint8)
    mask[2:] = 255                     # маска в нижней
    assert masked_l1(pred, true, mask) == 0.0
    mask[:] = 255
    assert np.isclose(masked_l1(pred, true, mask), 0.5)


def test_masked_l1_of_an_empty_mask_is_nan_not_zero():
    # Ноль выглядел бы как идеальная примерка.
    assert np.isnan(masked_l1(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.zeros((2, 2))))


def test_mean_ci_contains_the_mean():
    values = np.random.default_rng(0).normal(1.0, 0.2, 200)
    mean, (lo, hi) = mean_ci(values)
    assert lo < mean < hi


def test_paired_difference_detects_a_consistent_shift():
    rng = np.random.default_rng(0)
    # Трудность пар сильно разная, но b стабильно чуть выше a на той же паре:
    # непарный тест утонул бы в разбросе трудности, парный — нет.
    difficulty = rng.normal(0.5, 0.3, 373)
    a = difficulty + rng.normal(0, 0.01, 373)
    b = difficulty + 0.02 + rng.normal(0, 0.01, 373)
    result = paired_difference(a, b)
    assert result["significant"] and result["mean_diff"] > 0
    assert result["share_b_higher"] > 0.8


def test_paired_difference_does_not_invent_an_effect():
    rng = np.random.default_rng(1)
    difficulty = rng.normal(0.5, 0.3, 373)
    a = difficulty + rng.normal(0, 0.02, 373)
    b = difficulty + rng.normal(0, 0.02, 373)
    assert not paired_difference(a, b)["significant"]


def test_paired_difference_skips_pairs_missing_in_either_run():
    a = np.array([0.1, np.nan, 0.3, 0.4])
    b = np.array([0.2, 0.2, np.nan, 0.5])
    assert paired_difference(a, b)["pairs"] == 2
