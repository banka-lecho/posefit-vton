#!/usr/bin/env python
"""Блок F: дедупликация, группировка кроёв и фиксация train/val/test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.dedup import annotate, design_components  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.splits import assign, check_leakage, write  # noqa: E402

TARGET_GROUPS = ("upper", "lower", "dress", "outer")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)
    sku = pd.read_parquet(paths.manifest_sku)
    images = pd.read_parquet(paths.manifest_images)
    # Скрипт идемпотентен: колонки прошлого прогона снимаются, иначе merge
    # развёл бы их в split_x / split_y.
    derived = ["split", "split_group", "is_template", "dup_designs", "label_ok", "is_canonical"]
    sku = sku.drop(columns=[c for c in derived if c in sku.columns])
    images = images.drop(columns=[c for c in derived if c in images.columns])

    merged = images[images["read_ok"]].merge(
        sku[["sku_id", "design_id", "category_raw", "category_group"]], on="sku_id"
    )
    annotated = annotate(merged)
    components = design_components(annotated)

    groups = assign(sku, components, seed=args.seed)
    sku = sku.assign(split_group=sku["design_id"].map(components))
    sku = sku.merge(groups[["split_group", "split"]], on="split_group", how="left")

    annotated = annotated.assign(split_group=annotated["design_id"].map(components))
    annotated = annotated.merge(groups[["split_group", "split"]], on="split_group", how="left")
    leakage = check_leakage(annotated[annotated["is_canonical"] & annotated["label_ok"]])

    sku.to_parquet(paths.manifest_sku, index=False)
    keep = [c for c in annotated.columns if c not in {"design_id", "category_raw", "category_group"}]
    annotated[keep].to_parquet(paths.manifest_images, index=False)
    write(paths.splits, groups, args.seed, leakage)

    print(f"групп кроёв: {len(groups)}  seed={args.seed}  -> {paths.splits}")
    print("\nSKU по сплитам и группам одежды:")
    table = sku[sku["category_group"].isin(TARGET_GROUPS)]
    print(pd.crosstab(table["category_group"], table["split"]).to_string())

    print("\nкадры, пригодные в пары (уникальные, метка надёжна, не шаблон):")
    pool = annotated[annotated["is_canonical"] & annotated["label_ok"]]
    pool = pool[pool["category_group"].isin(TARGET_GROUPS)]
    print(pd.crosstab([pool["branch"], pool["category_group"]], pool["split"]).to_string())

    print(f"\nпроверка утечки: {leakage}")
    return 0 if leakage["content_hashes_in_multiple_splits"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
