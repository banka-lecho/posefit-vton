import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch", reason="posefit.dataset тянет torch")

from posefit.preprocess import TARGET_SIZE  # noqa: E402
from posefit.tryon import (  # noqa: E402
    GarmentInput, PersonInput, build_item, canonical_box, composite, open_image, uncrop,
)

ROOT = Path(__file__).resolve().parent.parent


def test_canonical_box_matches_load_canonical_padding():
    from posefit.preprocess import load_canonical

    # Широкое фото: поля сверху и снизу, ровно там, где их оставила load_canonical.
    image = Image.new("RGB", (1000, 750), (10, 20, 30))
    canon = np.asarray(load_canonical(image))
    x0, y0, x1, y1 = canonical_box(image.size)
    assert (x0, x1) == (0, TARGET_SIZE[0])
    assert (canon[y0:y1, x0:x1] == (10, 20, 30)).all()
    assert (canon[: y0] == 255).all() and (canon[y1:] == 255).all()


def test_uncrop_restores_the_original_aspect():
    x0, y0, x1, y1 = canonical_box((1000, 750))
    cropped = uncrop(np.zeros((512, 384, 3), np.uint8), (1000, 750))
    assert abs(cropped.width / cropped.height - 1000 / 750) < 0.02


def test_composite_keeps_pixels_outside_the_mask():
    item = {"person": np.zeros((3, 4, 4), np.float32), "mask": np.zeros((1, 4, 4), np.float32)}
    item["mask"][:, :2] = 1.0
    result = np.full((4, 4, 3), 200, np.uint8)
    out = composite(result, item)
    assert (out[:2] == 200).all()          # внутри маски — результат
    assert (out[2:] == 128).all()          # снаружи — исходные пиксели (0 в [-1,1] = 127.5)


def test_open_image_applies_exif_rotation(tmp_path):
    image = Image.new("RGB", (40, 20), (0, 0, 0))
    exif = image.getexif()
    exif[0x0112] = 6                        # «повернуть на 90°» — так пишет телефон
    path = tmp_path / "phone.jpg"
    image.save(path, exif=exif)
    assert open_image(path).size == (20, 40)


def _real_pair():
    import pandas as pd

    from posefit.paths import load_config, load_paths

    paths = load_paths(load_config(ROOT / "configs" / "data.yaml"))
    manifest = paths.cache_root / "manifest_pairs.parquet"
    if not manifest.exists() or not paths.raw_root.exists() or not paths.has_preproc \
            or not paths.preproc_root.exists():
        pytest.skip("нет выгрузки или результатов препроцессинга")
    rows = pd.read_parquet(manifest)
    return paths, rows[rows["ветка"] == "wild"].sample(3, random_state=11)


@pytest.mark.parametrize("mask_stage", ["agnostic", "agnostic_refined"])
def test_notebook_item_equals_the_training_item(mask_stage):
    """Пример для своих фото собирается теми же функциями, что обучающий:
    на кадре из выгрузки с сохранённым препроцессингом они обязаны совпасть."""
    from posefit.dataset import VTONPairs
    from posefit.preprocess import load_canonical, output_path

    paths, rows = _real_pair()
    dataset = VTONPairs(rows, paths.raw_root, paths.preproc_root, flip=False,
                        mask_stage=mask_stage, guide=True, garment_embed=True)
    flags = {"guide": True, "garment_embed": True}
    for index, row in enumerate(rows.itertuples()):
        def load(stage, image_id):
            return output_path(paths.preproc_root, stage, row.sku_id, image_id)

        masks = {stage: np.array(Image.open(load(stage, row.person_image_id)))
                 for stage in ("agnostic", "agnostic_refined")}
        person = PersonInput(
            canonical=load_canonical(paths.raw_root / row.person_rel_path), original_size=(0, 0),
            parse=np.array(Image.open(load("parse", row.person_image_id))),
            pose=json.loads(load("pose", row.person_image_id).read_text(encoding="utf-8")),
            n_persons=1, masks=masks)
        garment = GarmentInput(
            canonical=load_canonical(paths.raw_root / row.garment_rel_path), original_size=(0, 0),
            parse=np.array(Image.open(load("parse", row.garment_image_id))),
            embed=np.load(load("embed", row.garment_image_id)))
        mine = build_item(person, garment, row.category_group, row.pair_id, flags=flags,
                          mask_stage=mask_stage)
        theirs = dataset[index]
        for key in ("person", "garment", "mask", "guide_person", "guide_garment", "garment_embed"):
            assert np.array_equal(mine[key], theirs[key]), key


def test_monochrome_outfit_parsed_as_dress_is_flagged_for_upper():
    from posefit.preprocess import ATR
    from posefit.tryon import clothing_warning

    parse = np.zeros((100, 80), np.uint8)
    parse[20:90, 10:70] = ATR["dress"]           # образ целиком ушёл в «платье»
    parse[20:30, 10:20] = ATR["upper_clothes"]   # от верха остался кусок рукава
    warning = clothing_warning(parse, "upper", [0, 0, 80, 100])
    assert warning is not None and "dress" in warning
    assert clothing_warning(parse, "dress", [0, 0, 80, 100]) is None


def test_normal_top_is_not_flagged():
    from posefit.preprocess import ATR
    from posefit.tryon import clothing_warning

    parse = np.zeros((100, 80), np.uint8)
    parse[20:60, 10:70] = ATR["upper_clothes"]
    parse[60:95, 20:60] = ATR["pants"]
    assert clothing_warning(parse, "upper", [0, 0, 80, 100]) is None
    assert clothing_warning(parse, "lower", [0, 0, 80, 100]) is None
