#!/usr/bin/env python
"""Блок C: досчитывает CPU-признаки качества в cache/manifest_images.parquet."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.qc import features  # noqa: E402

_RAW_ROOT: Path | None = None


def _init(raw_root: str) -> None:
    global _RAW_ROOT
    _RAW_ROOT = Path(raw_root)


def _work(rel_path: str) -> dict:
    return features(_RAW_ROOT / rel_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="обработать только N кадров (отладка)")
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)
    images = pd.read_parquet(paths.manifest_images)
    todo = images if not args.limit else images.head(args.limit)

    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init, initargs=(str(paths.raw_root),)
    ) as pool:
        rows = list(
            tqdm(
                pool.map(_work, todo["rel_path"].tolist(), chunksize=64),
                total=len(todo), desc="qc", unit="img", mininterval=2.0,
            )
        )

    feats = pd.DataFrame(rows, index=todo.index)
    out = images.drop(columns=[c for c in feats.columns if c in images.columns])
    out = out.join(feats)
    if args.limit:
        # Иначе отладочный прогон затёр бы манифест столбцами из одних NaN.
        print(f"--limit={args.limit}: манифест не перезаписан")
    else:
        out.to_parquet(paths.manifest_images, index=False)

    ok = feats["read_ok"]
    print(f"\nпрочитано: {int(ok.sum())} / {len(feats)}   -> {paths.manifest_images}")
    print("\nрезкость по ветке (перцентили):")
    joined = out.loc[ok[ok].index]
    for branch, grp in joined.groupby("branch"):
        q = grp["sharpness"].quantile([0.05, 0.10, 0.50, 0.90]).round(1)
        print(f"  {branch:<8} p05={q.iloc[0]:>8}  p10={q.iloc[1]:>8}  "
              f"p50={q.iloc[2]:>8}  p90={q.iloc[3]:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
