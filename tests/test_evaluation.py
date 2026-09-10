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


def _snapshot(tmp_path, **values):
    import yaml

    (tmp_path / "train_config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    return tmp_path


CONFIG = {"mask_stage": "agnostic", "height": 512, "width": 384, "lr": 1e-5}


def test_trained_run_takes_its_mask_from_the_snapshot_not_the_config(tmp_path):
    # Регрессия: прогон учился на уточнённой маске, конфиг задавал исходную,
    # и замер отказывался запускаться вместо того, чтобы взять маску прогона.
    from posefit.evaluation import resolve_eval_config

    run = _snapshot(tmp_path, mask_stage="agnostic_refined", height=512, width=384)
    hp, _ = resolve_eval_config(CONFIG, run, "final.pt")
    assert hp["mask_stage"] == "agnostic_refined"


def test_snapshot_resolution_wins_over_the_config(tmp_path):
    from posefit.evaluation import resolve_eval_config

    run = _snapshot(tmp_path, mask_stage="agnostic", height=1024, width=768)
    hp, _ = resolve_eval_config(CONFIG, run, "final.pt")
    assert (hp["height"], hp["width"]) == (1024, 768)


def test_keys_outside_the_trained_set_come_from_the_config(tmp_path):
    from posefit.evaluation import resolve_eval_config

    run = _snapshot(tmp_path, mask_stage="agnostic", lr=0.5)
    hp, _ = resolve_eval_config(CONFIG, run, "final.pt")
    assert hp["lr"] == 1e-5


def test_old_run_without_snapshot_is_treated_as_the_original_mask(tmp_path):
    from posefit.evaluation import resolve_eval_config

    hp, source = resolve_eval_config(dict(CONFIG, mask_stage="agnostic_refined"), tmp_path, "final.pt")
    assert hp["mask_stage"] == "agnostic"
    assert "старый" in source


def test_zero_shot_control_takes_the_mask_from_the_flag(tmp_path):
    from posefit.evaluation import resolve_eval_config

    hp, _ = resolve_eval_config(CONFIG, tmp_path, "none", mask_stage="agnostic_refined")
    assert hp["mask_stage"] == "agnostic_refined"


def test_mask_flag_is_refused_for_a_trained_run(tmp_path):
    import pytest

    from posefit.evaluation import resolve_eval_config

    with pytest.raises(ValueError, match="только с --checkpoint none"):
        resolve_eval_config(CONFIG, tmp_path, "final.pt", mask_stage="agnostic")


def test_config_passed_in_is_not_mutated(tmp_path):
    from posefit.evaluation import resolve_eval_config

    original = dict(CONFIG)
    run = _snapshot(tmp_path, mask_stage="agnostic_refined")
    resolve_eval_config(original, run, "final.pt")
    assert original == CONFIG


def test_trained_run_takes_its_arch_from_the_snapshot(tmp_path):
    import yaml

    from posefit.evaluation import resolve_eval_config

    (tmp_path / "train_config.yaml").write_text(
        yaml.safe_dump(dict(CONFIG, arch={"name": "anchor", "reliability": False})), encoding="utf-8")
    hp, _ = resolve_eval_config(dict(CONFIG, arch={"name": "catvton"}), tmp_path, "final.pt")
    assert hp["arch"] == {"name": "anchor", "reliability": False}


def test_snapshot_without_arch_means_the_baseline(tmp_path):
    import yaml

    from posefit.evaluation import resolve_eval_config

    (tmp_path / "train_config.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    hp, _ = resolve_eval_config(dict(CONFIG, arch={"name": "anchor"}), tmp_path, "final.pt")
    assert hp["arch"] == {"name": "catvton"}


def test_arch_flag_only_for_the_zero_shot_control(tmp_path):
    import pytest

    from posefit.evaluation import resolve_eval_config

    hp, _ = resolve_eval_config(CONFIG, tmp_path, "none", arch="anchor")
    assert hp["arch"]["name"] == "anchor"
    with pytest.raises(ValueError, match="--arch"):
        resolve_eval_config(CONFIG, tmp_path, "final.pt", arch="anchor")


def test_inference_variants_do_not_overwrite_the_run(tmp_path):
    from posefit.evaluation import eval_output_dir

    run = tmp_path / "studio_wild_anchor"
    assert eval_output_dir(run) == run
    assert eval_output_dir(run, "studio", 0.0) == run
    assert eval_output_dir(run, "studio", 2.0) == tmp_path / "studio_wild_anchor@g2"
    assert eval_output_dir(run, "wild_noisy", 0.0) == tmp_path / "studio_wild_anchor@wild_noisy"
    assert eval_output_dir(run, "wild_noisy", 1.5) == tmp_path / "studio_wild_anchor@wild_noisy_g1.5"
