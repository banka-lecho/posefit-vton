#!/usr/bin/env python
"""Сходство вещи в отзыве с карточкой + калибровка порога по разметке.

Пишет cache/garment_sim.parquet. Если рядом лежит labels_wild.csv, печатает
AUC и таблицу порогов, чтобы выбрать рабочую точку осознанно.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.matching import auc, load_embeddings, reference_bank, similarity, sweep  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.preprocess import output_path, working_set  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--signals", default=None, type=Path,
                    help="по умолчанию cache/signals.parquet из collect_signals.py")
    ap.add_argument("--labels", default=Path("cache/labeling/labels_wild.csv"), type=Path)
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)
    root = paths.preproc_root

    rows = working_set(pd.read_parquet(paths.manifest_images),
                       pd.read_parquet(paths.manifest_sku))
    signals_path = args.signals or (paths.cache_root / "signals.parquet")
    if not signals_path.exists():
        print(f"ОШИБКА: нет {signals_path}. Сначала: python scripts/collect_signals.py",
              file=sys.stderr)
        return 2
    signals = pd.read_parquet(signals_path)[["image_id", "n_persons"]]
    rows = rows.merge(signals, on="image_id", how="left")

    embeddings = load_embeddings(rows, root, output_path)
    print(f"эмбеддингов загружено: {len(embeddings)} из {len(rows)}")

    product = rows[(rows["branch"] == "product") & (rows["n_persons"] > 0)]
    review = rows[rows["branch"] == "review"].copy()
    bank = reference_bank(product, embeddings)
    print(f"SKU с эталонами: {len(bank)}")

    review["garment_sim"] = similarity(review, embeddings, bank)
    out = paths.cache_root / "garment_sim.parquet"
    review[["image_id", "sku_id", "category_group", "garment_sim"]].to_parquet(out, index=False)
    print(f"посчитано: {int(review.garment_sim.notna().sum())} -> {out}")

    if not args.labels.exists():
        return 0

    labels = pd.read_csv(args.labels)[["image_id", "label"]]
    marked = review.merge(labels, on="image_id").dropna(subset=["garment_sim"])
    marked = marked[marked.label.isin(["good", "color_mismatch"])]
    if marked.empty:
        return 0

    positive = (marked.label == "good").to_numpy()
    scores = marked.garment_sim.to_numpy()
    print(f"\nразмечено пар: {len(marked)}   доля совпавших: {positive.mean():.3f}")
    print(f"AUC: {auc(positive, scores):.3f}   (гистограммы Lab давали 0.799)")
    print("\n" + sweep(scores, positive).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
