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

from posefit.labeling import TASKS, batch_id, build_items, render, sample  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402

TARGET_GROUPS = ("upper", "lower", "dress", "outer")


def _verify_pool(paths) -> pd.DataFrame:
    """Кандидаты в чистый тестовый набор.

    Берутся только кадры из сплита test, прошедшие каскад, но БЕЗ фильтра
    расцветки: именно его точность (0.76) и делает автоматический набор
    непригодным для замера качества. Проверенные вручную пары становятся
    эталоном, на котором сравниваются прогоны обучения.
    """
    signals = pd.read_parquet(paths.cache_root / "signals.parquet")
    garment_sim = pd.read_parquet(paths.cache_root / "garment_sim.parquet")[
        ["image_id", "garment_sim"]
    ]
    pairs = pd.read_parquet(paths.cache_root / "manifest_pairs.parquet")
    # Выкладка берётся из всех пар, а не только из wild: wild-ветка уже прошла
    # фильтр расцветки, и SKU, у которых все отзывы отфильтровались, потеряли бы
    # эталон — хотя сама выкладка у них есть и годится.
    garments = pairs.drop_duplicates("sku_id").set_index("sku_id")["garment_rel_path"]

    frame = signals.merge(garment_sim, on="image_id", how="left")
    frame = frame[
        (frame["branch"] == "review")
        & (frame["split"] == "test")
        & (frame["n_persons"] >= 1)
        & (frame["area_frac"] >= 0.05)
        & (frame["torso_min"] >= 0.20)
        & (frame["garment_sim"] >= 0.30)
    ].copy()
    frame["ref_path"] = frame["sku_id"].map(garments)
    frame = frame.dropna(subset=["ref_path"])

    # Уже размеченное исключается, а внутри SKU берётся не больше cap кадров:
    # разнообразие теста определяется числом разных вещей, а не числом пар,
    # и без ограничения одна популярная вещь съедает всю выборку.
    # Собираются все выгрузки задачи, а не один файл: партий несколько,
    # и у каждой своё имя.
    done = sorted((paths.cache_root / "labeling").glob("labels_verify*.csv"))
    covered: set[str] = set()
    if done:
        seen = pd.concat([pd.read_csv(f) for f in done], ignore_index=True)
        frame = frame[~frame["image_id"].isin(seen["image_id"])]
        test_set = paths.cache_root / "test_set.parquet"
        if test_set.exists():
            covered = set(pd.read_parquet(test_set)["sku_id"])

    # Вещи, которых в тесте ещё нет, идут первыми: пул ограничен 117 SKU,
    # и разнообразие теста упирается именно в них, а не в число пар.
    frame = frame.sample(frac=1.0, random_state=0)
    frame["_new_sku"] = ~frame["sku_id"].isin(covered)
    frame = frame.sort_values("_new_sku", ascending=False, kind="stable")
    return frame.groupby("sku_id", group_keys=False).head(4).drop(columns="_new_sku")


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
    pools = {"garment": garment_pool, "wild": wild_pool, "verify": _verify_pool(paths)}

    out_dir = paths.cache_root / "labeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(TASKS) if args.task == "all" else [args.task]
    for name in names:
        task = TASKS[name]
        rows = sample(pools[name], args.n, args.seed)
        items = build_items(rows, paths.raw_root, task)
        body = render(task, items)
        # Имя файла тоже несёт отпечаток партии: открыть вчерашнюю копию,
        # думая, что размечаешь новую, становится невозможно.
        target = out_dir / f"{name}_{batch_id(items)}.html"
        target.write_text(body, encoding="utf-8")
        size_mb = target.stat().st_size / 1e6
        print(f"{name:<8} {len(items):>4} кадров  {size_mb:>5.1f} МБ  -> {target}")
        print("         " + rows["category_group"].value_counts().to_dict().__str__())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
