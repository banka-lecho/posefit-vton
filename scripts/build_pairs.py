#!/usr/bin/env python
"""Финальный артефакт Ф0: cache/manifest_pairs.parquet.

Пара — это (изображение вещи, изображение человека в ней). Веток две:

  studio — плоская выкладка из карточки против студийного кадра на модели.
           Чисто, ровный свет, формат совпадает с VITON-HD.
  wild   — та же выкладка против кадра из отзыва: реальные тела, позы и свет.

Пороги фильтров подобраны по ручной разметке 250 кадров, а не назначены:
см. отчёт, который скрипт печатает в конце.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402

# Значения откалиброваны по labels_wild.csv: точность 0.95 / полнота 0.87
# для каскада и 0.76 / 0.86 для расцветки.
MIN_TORSO = 0.20
MIN_AREA = 0.05
MIN_GARMENT_SIM = 0.30
COLOUR_DROP_QUANTILE = 0.45
# Доля светлого фона отделяет выкладку целой вещи от макро-кропа ткани:
# у кропов она 0.00-0.04, у выкладок 0.37 и выше. Площадь маски parse для
# этого не годится — SegFormer обучен на людях и на выкладке без человека
# часто не находит одежду вовсе.
MIN_WHITE_FRAC = 0.25


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--colour-drop", type=float, default=COLOUR_DROP_QUANTILE)
    ap.add_argument("--min-white-frac", type=float, default=MIN_WHITE_FRAC)
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)

    signals = pd.read_parquet(paths.cache_root / "signals.parquet")
    colour = pd.read_parquet(paths.cache_root / "colour_sim.parquet")
    garment = pd.read_parquet(paths.cache_root / "garment_sim.parquet")[["image_id", "garment_sim"]]
    # split уже лежит в signals (пришёл из манифеста кадров), поэтому из
    # таблицы SKU берётся только то, чего там нет, — иначе merge разведёт
    # колонки в split_x / split_y.
    sku = pd.read_parquet(paths.manifest_sku)[["sku_id", "design_id"]]

    rank = lambda s: s.rank(pct=True)  # noqa: E731
    colour = colour.assign(
        colour_rank=rank(colour["colour_med"].fillna(-1e9)) + rank(colour["colour_hist"].fillna(-1e9))
    )
    frames = signals.merge(colour[["image_id", "colour_rank"]], on="image_id", how="left")
    frames = frames.merge(garment, on="image_id", how="left").merge(sku, on="sku_id")

    # Плоская выкладка: человека на кадре нет. Но отсутствия человека мало —
    # под то же правило попадает макро-кроп ткани, снятый на том же фоне.
    # Их разделяет доля светлого фона: вещь целиком лежит на белом, кроп
    # ткани занимает кадр без остатка.
    flat = frames[(frames["branch"] == "product") & (frames["n_persons"] == 0)].copy()
    flat = flat[flat["white_frac"] >= args.min_white_frac]
    garments = flat.sort_values("white_frac", ascending=False).drop_duplicates("sku_id")
    garments = garments.set_index("sku_id")[["image_id", "rel_path"]]
    print(f"кадров-кандидатов в выкладку: {len(flat)}   SKU с выкладкой: {len(garments)}\n")

    studio = frames[(frames["branch"] == "product") & (frames["n_persons"] > 0)].copy()
    studio["ветка"] = "studio"

    wild = frames[frames["branch"] == "review"].copy()
    threshold = wild["colour_rank"].quantile(args.colour_drop)
    steps = [
        ("кадров из отзывов", pd.Series(True, index=wild.index)),
        ("человек найден", wild["n_persons"] >= 1),
        ("занимает >=5% кадра", wild["area_frac"] >= MIN_AREA),
        ("торс виден", wild["torso_min"] >= MIN_TORSO),
        ("вещь та же", wild["garment_sim"] >= MIN_GARMENT_SIM),
        ("расцветка совпадает", wild["colour_rank"] >= threshold),
    ]
    print("=== каскад wild-ветки ===")
    keep = pd.Series(True, index=wild.index)
    for name, condition in steps:
        keep &= condition.fillna(False)
        print(f"  {name:<24} {int(keep.sum()):>6}")
    wild = wild[keep].copy()
    wild["ветка"] = "wild"

    pairs = pd.concat([studio, wild], ignore_index=True)
    pairs = pairs[pairs["sku_id"].isin(garments.index)].copy()
    pairs["garment_image_id"] = pairs["sku_id"].map(garments["image_id"])
    pairs["garment_rel_path"] = pairs["sku_id"].map(garments["rel_path"])
    pairs = pairs.rename(columns={"image_id": "person_image_id", "rel_path": "person_rel_path"})
    pairs["pair_id"] = pairs["garment_image_id"] + "_" + pairs["person_image_id"]

    columns = ["pair_id", "ветка", "sku_id", "design_id", "category_group", "split",
               "garment_image_id", "garment_rel_path", "person_image_id", "person_rel_path",
               "n_persons", "area_frac", "torso_min", "garment_sim", "colour_rank"]
    out = paths.cache_root / "manifest_pairs.parquet"
    pairs[columns].to_parquet(out, index=False)

    print(f"\n=== пар всего: {len(pairs)} -> {out} ===")
    print(pd.crosstab([pairs["ветка"], pairs["category_group"]], pairs["split"]).to_string())
    print("\nSKU задействовано:", pairs["sku_id"].nunique(), " кроёв:", pairs["design_id"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
