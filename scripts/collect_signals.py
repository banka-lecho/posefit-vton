#!/usr/bin/env python
"""Сводит результаты стадий detect и pose в одну таблицу cache/signals.parquet.

Это входные данные фильтра: сколько людей в кадре, какую долю занимает
крупнейший, видны ли плечи и бёдра. Держать их россыпью из 86 тысяч json
неудобно, а пороги по ним перебираются десятками — отсюда отдельный шаг.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.preprocess import output_path, working_set  # noqa: E402

TORSO = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)
    root = paths.preproc_root
    rows = working_set(pd.read_parquet(paths.manifest_images),
                       pd.read_parquet(paths.manifest_sku))

    def read(row) -> tuple:
        detect_path = output_path(root, "detect", row.sku_id, row.image_id)
        pose_path = output_path(root, "pose", row.sku_id, row.image_id)
        if not detect_path.exists() or not pose_path.exists():
            return (np.nan, np.nan, np.nan)
        detect = json.loads(detect_path.read_text(encoding="utf-8"))
        scores = json.loads(pose_path.read_text(encoding="utf-8")).get("scores", {})
        # Минимум по четырём точкам, а не среднее: торс считается видимым,
        # только когда видны все опорные точки, иначе кроп по грудь прошёл бы.
        torso = min((scores.get(k, 0.0) for k in TORSO), default=0.0)
        return (detect["n_persons"], detect["max_area_frac"], torso)

    with ThreadPoolExecutor(args.workers) as pool:
        collected = list(tqdm(pool.map(read, rows.itertuples(), chunksize=256),
                              total=len(rows), desc="signals", unit="img", mininterval=2.0))

    rows[["n_persons", "area_frac", "torso_min"]] = pd.DataFrame(collected, index=rows.index)
    missing = int(rows["n_persons"].isna().sum())
    out = paths.cache_root / "signals.parquet"
    rows.to_parquet(out, index=False)

    print(f"\nкадров: {len(rows)}   без результатов стадий: {missing}   -> {out}")
    if missing:
        print("Незаполненные строки означают, что detect или pose отработали не для всех кадров.")
    ready = rows.dropna(subset=["n_persons"])
    print("\nлюдей в кадре, по ветке:")
    print(pd.crosstab(ready["branch"], ready["n_persons"].clip(upper=3).astype(int)).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
