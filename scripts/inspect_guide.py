#!/usr/bin/env python
"""Контактный лист каналов-подсказок якорной схемы.

Карты координат считаются аналитически и не падают никогда — а значит, ошибка
в них (перепутанные оси, ворот на бёдрах, скелет от отражения в зеркале)
всплывёт только через часы обучения. Лист показывает по каждой паре: человек
с маской | вещь | карта u | карта v | скелет. У человека и вещи карта одной
и той же координаты должна выглядеть одним градиентом.

    python scripts/inspect_guide.py --n 6 --config-name studio_wild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.dataset import VTONPairs  # noqa: E402
from posefit.pairs import CONFIGS, select_pairs  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.reliability import BUCKETS, assign_reliability  # noqa: E402


def to_u8(chw: np.ndarray) -> np.ndarray:
    return ((chw.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def heat(values: np.ndarray) -> np.ndarray:
    """[-1, 1] -> от синего к красному."""
    red = ((values + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return np.stack([red, np.zeros_like(red), 255 - red], axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--config-name", default="studio_wild", choices=list(CONFIGS))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="по умолчанию cache/inspect_guide.png")
    args = ap.parse_args()

    paths = load_paths(load_config(args.config))
    pairs = pd.read_parquet(paths.cache_root / "manifest_pairs.parquet")
    selected = select_pairs(pairs, args.config_name, "train", 0.65)
    selected["reliability"] = assign_reliability(selected).to_numpy()
    # Поровну студии и отзывов, чтобы были видны обе ветки.
    parts = [frame.sample(min(len(frame), max(1, args.n // 2)), random_state=args.seed)
             for _, frame in selected.groupby("ветка")]
    rows = pd.concat(parts).head(args.n)

    dataset = VTONPairs(rows, paths.raw_root, paths.preproc_root, flip=False, guide=True)
    tiles = []
    for index in range(len(dataset)):
        item = dataset[index]
        person, garment = to_u8(item["person"]), to_u8(item["garment"])
        inside = (item["mask"][0] > 0)[..., None]
        person = (person * (1.0 - 0.5 * inside)).astype(np.uint8)
        gp, gg = item["guide_person"], item["guide_garment"]
        skeleton = np.repeat((gp[2] * 255).astype(np.uint8)[..., None], 3, axis=-1)
        tiles.append(np.concatenate([
            person, garment,
            heat(np.concatenate([gp[0], gg[0]], axis=1)),
            heat(np.concatenate([gp[1], gg[1]], axis=1)),
            skeleton,
        ], axis=1))
        print(f"{item['pair_id']}  {item['category_group']:6}  "
              f"{BUCKETS[item['reliability']]:11}  скелет {'есть' if gp[2].max() > 0 else 'нет'}")

    sheet = Image.fromarray(np.concatenate(tiles, axis=0))
    sheet = sheet.resize((sheet.width // 2, sheet.height // 2))
    out = args.out or paths.cache_root / "inspect_guide.png"
    sheet.save(out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
