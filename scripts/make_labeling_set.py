#!/usr/bin/env python
"""Блок E: собирает калибровочные наборы для ручной разметки.

Пишет cache/labeling/<task>.html — самодостаточный файл с миниатюрами внутри.
Открывается двойным кликом, размечается с клавиатуры, выгружает CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.labeling import TASKS, build_items, render, sample  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402

TARGET_GROUPS = ("upper", "lower", "dress", "outer")


def _pools(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    sku = pd.read_parquet(paths.manifest_sku)
    images = pd.read_parquet(paths.manifest_images)
    frame = images.merge(sku[["sku_id", "category_group"]], on="sku_id")
    frame = frame[frame["category_group"].isin(TARGET_GROUPS) & frame["read_ok"]]
    # Размечается только train: val и test должны остаться нетронутыми,
    # иначе пороги окажутся подогнаны под тестовую выборку.
    frame = frame[frame["split"] == "train"]

    garment = frame[(frame["branch"] == "product") & ~frame["is_template"]]

    wild = frame[(frame["branch"] == "review") & frame["label_ok"] & frame["is_canonical"]]
    # Эталон расцветки — первый кадр карточки: он всегда съёмка на модели.
    refs = (
        garment.sort_values("order_idx")
        .drop_duplicates("sku_id")
        .set_index("sku_id")["rel_path"]
    )
    wild = wild.assign(ref_path=wild["sku_id"].map(refs))
    return garment, wild.dropna(subset=["ref_path"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--task", choices=[*TASKS, "all"], default="all")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)
    garment_pool, wild_pool = _pools(paths)
    pools = {"garment": garment_pool, "wild": wild_pool}

    out_dir = paths.cache_root / "labeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(TASKS) if args.task == "all" else [args.task]
    for name in names:
        task = TASKS[name]
        rows = sample(pools[name], args.n, args.seed)
        items = build_items(rows, paths.raw_root, task)
        target = out_dir / f"{name}.html"
        target.write_text(render(task, items), encoding="utf-8")
        size_mb = target.stat().st_size / 1e6
        print(f"{name:<8} {len(items):>4} кадров  {size_mb:>5.1f} МБ  -> {target}")
        print("         " + rows["category_group"].value_counts().to_dict().__str__())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
