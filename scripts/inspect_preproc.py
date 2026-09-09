#!/usr/bin/env python
"""Визуальная проверка результатов препроцессинга.

Стадия может отработать без единой ошибки и выдать мусор: сегментация
съедет на фон, поза прилипнет к отражению в зеркале, маска закроет не ту
половину человека. Скрипт собирает контактный лист, на котором это видно
сразу, и его можно приложить к отчёту.

    python scripts/inspect_preproc.py --n 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.preprocess import load_canonical, output_path, working_set  # noqa: E402


SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def palette(n: int = 18) -> np.ndarray:
    """Различимые цвета для 18 классов ATR."""
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, size=(n, 3), dtype=np.uint8)
    colors[0] = (30, 30, 30)  # фон приглушён, чтобы не спорил с одеждой
    return colors


def colorize(parse: np.ndarray) -> Image.Image:
    return Image.fromarray(palette()[np.clip(parse, 0, 17)])


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    tint = Image.new("RGB", image.size, (255, 40, 90))
    alpha = Image.fromarray((mask > 0).astype(np.uint8) * 140)
    out = image.copy()
    out.paste(tint, (0, 0), alpha)
    return out


def draw_pose(image: Image.Image, pose: dict, min_score: float = 0.3) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if pose.get("box"):
        draw.rectangle(pose["box"], outline=(0, 200, 255), width=3)
    points, scores = pose.get("keypoints", {}), pose.get("scores", {})
    for a, b in SKELETON:
        if scores.get(a, 0) >= min_score and scores.get(b, 0) >= min_score:
            draw.line([tuple(points[a][:2]), tuple(points[b][:2])], fill=(255, 220, 0), width=4)
    for name, xy in points.items():
        if scores.get(name, 0) >= min_score:
            x, y = xy[0], xy[1]
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 60, 60))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cell", type=int, default=260)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    paths = load_paths(config)
    root = paths.preproc_root
    rows = working_set(pd.read_parquet(paths.manifest_images),
                       pd.read_parquet(paths.manifest_sku))

    picked = []
    for row in rows.itertuples():
        parse_path = output_path(root, "parse", row.sku_id, row.image_id)
        if parse_path.exists():
            picked.append(row)
        if len(picked) >= args.n:
            break
    if not picked:
        print("нет результатов стадии parse — сначала прогони препроцессинг")
        return 1

    columns = ["исходник", "parse", "agnostic", "pose"]
    cell = args.cell
    header = 22
    sheet = Image.new("RGB", (len(columns) * cell, header + len(picked) * cell), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    # Подписи на отдельной полосе: поверх кадров они тонули то в светлом
    # фоне студии, то в тёмной карте сегментации.
    draw.rectangle([0, 0, sheet.width, header], fill=(24, 24, 24))
    for c, name in enumerate(columns):
        draw.text((c * cell + 8, 6), name, fill=(255, 255, 255))

    for r, row in enumerate(picked):
        base = load_canonical(paths.raw_root / row.rel_path)
        parse = np.array(Image.open(output_path(root, "parse", row.sku_id, row.image_id)))
        panels = [base, colorize(parse)]

        mask_path = output_path(root, "agnostic", row.sku_id, row.image_id)
        panels.append(overlay_mask(base, np.array(Image.open(mask_path)))
                      if mask_path.exists() else Image.new("RGB", base.size, (235, 235, 235)))

        pose_path = output_path(root, "pose", row.sku_id, row.image_id)
        panels.append(draw_pose(base, json.loads(pose_path.read_text(encoding="utf-8")))
                      if pose_path.exists() else Image.new("RGB", base.size, (235, 235, 235)))

        for c, panel in enumerate(panels):
            thumb = panel.copy()
            thumb.thumbnail((cell, cell))
            sheet.paste(thumb, (c * cell + (cell - thumb.width) // 2, header + r * cell))
        draw.text((4, header + r * cell + 4), row.category_group, fill=(90, 90, 90))

    target = args.out or (paths.cache_root / "inspect_preproc.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    print(f"{len(picked)} кадров -> {target}")
    print("Открой файл и посмотри: сегментация на человеке, маска закрывает "
          "примеряемую вещь, скелет не уехал в отражение.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
