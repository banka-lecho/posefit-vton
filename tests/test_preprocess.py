import numpy as np
import pandas as pd
import pytest

from posefit.preprocess import (
    ATR, STAGES, build_agnostic, output_path, pending, shard_of, torso_visible, working_set,
)


def _images(n=40):
    return pd.DataFrame({
        "image_id": [f"{i:012x}" for i in range(n)],
        "sku_id": [f"s{i % 5}" for i in range(n)],
        "rel_path": [f"cat/sku/{i}.webp" for i in range(n)],
        "read_ok": True, "is_template": False, "is_canonical": True, "label_ok": True,
    })


def _sku():
    return pd.DataFrame({"sku_id": [f"s{i}" for i in range(5)],
                         "category_group": ["upper", "lower", "dress", "outer", "drop"]})


def test_working_set_drops_unusable_frames():
    images = _images()
    images.loc[0, "read_ok"] = False
    images.loc[1, "is_template"] = True
    images.loc[2, "is_canonical"] = False
    images.loc[3, "label_ok"] = False
    kept = working_set(images, _sku())["image_id"].tolist()
    assert all(images.loc[i, "image_id"] not in kept for i in range(4))


def test_working_set_drops_excluded_categories():
    kept = working_set(_images(), _sku())
    assert "drop" not in set(kept["category_group"])


def test_shards_survive_non_random_identifiers():
    # Осмысленные имена без энтропии в префиксе не должны съезжать в один шард.
    ids = [f"sku-000-frame-{i}" for i in range(200)]
    counts = [sum(shard_of(i, 4) == s for i in ids) for s in range(4)]
    assert all(c > 0 for c in counts)


def test_shards_partition_every_row_exactly_once():
    ids = _images(200)["image_id"]
    assert len(set(ids)) == 200
    for n in (2, 3, 8):
        counts = [sum(shard_of(i, n) == s for i in ids) for s in range(n)]
        assert sum(counts) == len(ids)
        assert all(c > 0 for c in counts)


def test_pending_skips_finished_work(tmp_path):
    rows = working_set(_images(), _sku())
    first = rows.iloc[0]
    done = output_path(tmp_path, "parse", first["sku_id"], first["image_id"])
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_bytes(b"x")
    assert first["image_id"] not in pending(rows, tmp_path, "parse")["image_id"].tolist()
    assert first["image_id"] in pending(rows, tmp_path, "parse", overwrite=True)["image_id"].tolist()


def test_agnostic_covers_garment_and_arms_for_upper():
    parse = np.zeros((32, 32), np.uint8)
    parse[4:12, 4:12] = ATR["upper_clothes"]
    parse[14:18, 4:8] = ATR["left_arm"]
    parse[20:24, 20:24] = ATR["pants"]
    mask = build_agnostic(parse, "upper", dilate_px=0) > 0
    assert mask[6, 6] and mask[15, 5]
    # Низ остаётся видимым: примеряется верх, штаны трогать нельзя.
    assert not mask[22, 22]


def test_agnostic_for_lower_leaves_the_top_alone():
    parse = np.zeros((32, 32), np.uint8)
    parse[4:12, 4:12] = ATR["upper_clothes"]
    parse[20:28, 8:16] = ATR["pants"]
    mask = build_agnostic(parse, "lower", dilate_px=0) > 0
    assert mask[24, 12] and not mask[6, 6]


def test_dilation_grows_the_mask():
    parse = np.zeros((64, 64), np.uint8)
    parse[28:36, 28:36] = ATR["upper_clothes"]
    tight = build_agnostic(parse, "upper", dilate_px=0).sum()
    loose = build_agnostic(parse, "upper", dilate_px=12).sum()
    assert loose > tight


def test_torso_visibility_needs_shoulders_and_hips():
    full = {"scores": {n: 0.9 for n in
            ("left_shoulder", "right_shoulder", "left_hip", "right_hip")}}
    assert torso_visible(full)
    cropped = {"scores": dict(full["scores"], left_hip=0.05)}
    assert not torso_visible(cropped)


def test_stage_dependencies_reference_real_stages():
    for stage in STAGES.values():
        assert all(need in STAGES for need in stage.needs)
