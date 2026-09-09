#!/usr/bin/env python
"""Цветовые признаки вещи по маске parse -> preproc_root/colour/.

DINOv2 для расцветки не годится: он обучен с цветовой аугментацией и
инвариантен к цвету (AUC 0.55 против 0.80 у гистограмм). Поэтому расцветка
сравнивается напрямую по пикселям вещи.

На кадр сохраняются медиана Lab и нормированная гистограмма Lab 8x8x8 —
515 чисел, из которых потом считается сходство без повторного чтения картинок.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.preprocess import (  # noqa: E402
    ATR, GARMENT_LABELS, load_canonical, output_path, working_set,
)

BINS = 8
MAX_PX = 20000
MIN_PX = 500


def features(image_path: Path, parse_path: Path, group: str) -> np.ndarray | None:
    parse = cv2.imread(str(parse_path), cv2.IMREAD_GRAYSCALE)
    if parse is None:
        return None
    mask = np.isin(parse, [ATR[name] for name in GARMENT_LABELS[group]])
    if mask.sum() < MIN_PX:
        return None

    pixels = cv2.cvtColor(np.asarray(load_canonical(image_path)), cv2.COLOR_RGB2LAB)[mask]
    if len(pixels) > MAX_PX:
        # Подвыборка фиксированным генератором: полная маска даёт до миллиона
        # пикселей, а гистограмма на 20 тысячах уже стабильна.
        idx = np.random.default_rng(0).choice(len(pixels), MAX_PX, replace=False)
        pixels = pixels[idx]

    hist, _ = np.histogramdd(pixels, bins=BINS, range=[(0, 256)] * 3)
    return np.concatenate([np.median(pixels, axis=0), (hist / hist.sum()).ravel()]).astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)
    root = paths.preproc_root
    rows = working_set(pd.read_parquet(paths.manifest_images),
                       pd.read_parquet(paths.manifest_sku))

    def run(row) -> bool:
        target = output_path(root, "colour", row.sku_id, row.image_id)
        if target.exists() and not args.overwrite:
            return True
        values = features(paths.raw_root / row.rel_path,
                          output_path(root, "parse", row.sku_id, row.image_id),
                          row.category_group)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Пустой массив, как и у эмбеддингов: «вещь не найдена» должно
        # отличаться от «признаки не посчитаны».
        np.save(target, np.zeros(0, np.float16) if values is None else values)
        return values is not None

    with ThreadPoolExecutor(args.workers) as pool:
        done = list(tqdm(pool.map(run, rows.itertuples(), chunksize=64),
                         total=len(rows), desc="colour", unit="img", mininterval=2.0))

    print(f"\nкадров: {len(rows)}   с найденной вещью: {sum(done)}   -> {root / 'colour'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
