#!/usr/bin/env python
"""Чистый тестовый набор из проверенных вручную пар -> cache/test_set.parquet.

Автоматический wild-набор для замера качества не годится: точность фильтра
расцветки 0.79, то есть каждая пятая пара в нём с неверным цветом, и метрика
мерила бы шум. Поэтому тест собирается только из пар, подтверждённых глазами.

Разметка живёт в нескольких файлах labels_verify*.csv — по одному на партию.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.matching import auc  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402


def load_labels(labeling_dir: Path) -> pd.DataFrame:
    files = sorted(labeling_dir.glob("labels_verify*.csv"))
    if not files:
        raise SystemExit(f"нет размеченных партий в {labeling_dir}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"партий: {len(files)}   строк: {len(frame)}")
    return frame.drop_duplicates("image_id")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)

    labels = load_labels(paths.cache_root / "labeling")
    signals = pd.read_parquet(paths.cache_root / "signals.parquet")
    pairs = pd.read_parquet(paths.cache_root / "manifest_pairs.parquet")
    garments = pairs.drop_duplicates("sku_id").set_index("sku_id")

    frame = signals.merge(labels[labels["label"] == "ok"][["image_id"]], on="image_id")
    frame = frame[frame["sku_id"].isin(garments.index) & (frame["split"] == "test")].copy()
    frame["garment_image_id"] = frame["sku_id"].map(garments["garment_image_id"])
    frame["garment_rel_path"] = frame["sku_id"].map(garments["garment_rel_path"])
    frame = frame.rename(columns={"image_id": "person_image_id", "rel_path": "person_rel_path"})
    frame["pair_id"] = frame["garment_image_id"] + "_" + frame["person_image_id"]

    columns = ["pair_id", "sku_id", "category_group", "garment_image_id",
               "garment_rel_path", "person_image_id", "person_rel_path"]
    out = paths.cache_root / "test_set.parquet"
    frame[columns].to_parquet(out, index=False)

    print(f"\nтестовый набор: {len(frame)} пар, {frame['sku_id'].nunique()} SKU -> {out}")
    print(frame.groupby("category_group")
               .agg(пар=("pair_id", "size"), SKU=("sku_id", "nunique")).to_string())

    train_skus = set(pairs.loc[pairs["split"] != "test", "sku_id"])
    overlap = len(set(frame["sku_id"]) & train_skus)
    print(f"\nпересечение SKU с обучением: {overlap}")
    if overlap:
        raise SystemExit("тестовые вещи встречаются в обучении — разбиение сломано")

    # Разметка партий verify относится к сплиту test и в подборе порогов не
    # участвовала (те калибровались на labels_wild.csv из train), поэтому здесь
    # получается непредвзятая оценка фильтра расцветки.
    colour = pd.read_parquet(paths.cache_root / "colour_sim.parquet")
    rank = colour[["colour_med", "colour_hist"]].rank(pct=True).sum(axis=1)
    colour = colour.assign(colour_rank=rank)
    marked = labels.merge(colour[["image_id", "colour_rank"]], on="image_id")
    marked = marked[marked["label"].isin(["ok", "wrong_colour"])].dropna(subset=["colour_rank"])
    positive = (marked["label"] == "ok").to_numpy()
    scores = marked["colour_rank"].to_numpy()
    keep = scores >= colour["colour_rank"].quantile(0.45)
    tp, fp, fn = int((keep & positive).sum()), int((keep & ~positive).sum()), int((~keep & positive).sum())
    print(f"\nфильтр расцветки на независимой выборке ({len(marked)} пар):")
    print(f"  AUC {auc(positive, scores):.3f}   точность {tp / (tp + fp):.3f}   полнота {tp / (tp + fn):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
