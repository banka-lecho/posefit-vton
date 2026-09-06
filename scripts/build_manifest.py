#!/usr/bin/env python
"""Блок B: строит cache/manifest_sku.parquet и cache/manifest_images.parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.manifest import build  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)
    paths.cache_root.mkdir(parents=True, exist_ok=True)

    sku, images = build(paths.raw_root, config, workers=args.workers)
    sku.to_parquet(paths.manifest_sku, index=False)
    images.to_parquet(paths.manifest_images, index=False)

    print(f"\nSKU:    {len(sku):>6}  -> {paths.manifest_sku}")
    print(f"images: {len(images):>6}  -> {paths.manifest_images}")

    unmapped = sku.loc[sku["category_group"] == "unmapped", "category_raw"].unique()
    if len(unmapped):
        print(f"\nВНИМАНИЕ: категории вне конфига: {sorted(unmapped)}")

    print("\nSKU по группам:")
    for group, n in sku["category_group"].value_counts().items():
        n_img = images.loc[images["sku_id"].isin(sku.loc[sku["category_group"] == group, "sku_id"])]
        print(f"  {group:<10} {n:>5} SKU   {len(n_img):>6} фото")

    corrupt = int(images["corrupt"].sum())
    print(f"\nбитых файлов: {corrupt}")
    print(f"кроёв (design_id): {sku['design_id'].nunique()} на {len(sku)} папок")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
