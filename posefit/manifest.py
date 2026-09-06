"""Блок B: инвентаризация выгрузки в две таблицы.

manifest_sku.parquet    — строка на SKU (карточку товара)
manifest_images.parquet — строка на изображение

Размеры читаются из заголовка файла (PIL не декодирует пиксели при обращении
к .size), поэтому обход 78k webp занимает секунды, а не минуты.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .catalog import (
    attrs_json,
    build_category_map,
    design_id,
    image_sort_key,
    parse_info_csv,
    split_variant,
)

Image.MAX_IMAGE_PIXELS = 200_000_000

PRODUCT = "product"
REVIEW = "review"


def _short_id(text: str, n: int = 12) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()[:n]


def iter_sku_dirs(raw_root: Path, layout: dict) -> list[Path]:
    info_file = layout["info_file"]
    return sorted(
        p.parent for p in raw_root.glob(f"*/*/{info_file}") if p.parent.is_dir()
    )


def _probe(args: tuple[Path, Path]) -> dict:
    path, raw_root = args
    row = {
        "rel_path": path.relative_to(raw_root).as_posix(),
        "width": -1,
        "height": -1,
        "file_size": -1,
        "corrupt": True,
    }
    try:
        row["file_size"] = path.stat().st_size
        with Image.open(path) as im:
            row["width"], row["height"] = im.size
        row["corrupt"] = False
    except Exception:
        pass
    return row


def build(raw_root: Path, config: dict, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    layout = config["layout"]
    category_map = build_category_map(config)
    typed_keys = config["info_columns"]

    sku_rows: list[dict] = []
    image_tasks: list[tuple[Path, Path]] = []
    image_meta: dict[str, dict] = {}

    for sku_dir in tqdm(iter_sku_dirs(raw_root, layout), desc="SKU", unit="sku"):
        folder = sku_dir.name
        category_raw = sku_dir.parent.name
        rel = sku_dir.relative_to(raw_root).as_posix()
        sku_id = _short_id(rel)
        base, variant_no = split_variant(folder)
        info = parse_info_csv(sku_dir / layout["info_file"])

        counts = {}
        for branch, subdir in ((PRODUCT, layout["product_dir"]), (REVIEW, layout["review_dir"])):
            files = sorted(
                (p for p in (sku_dir / subdir).glob("*.webp") if p.is_file()),
                key=image_sort_key,
            )
            counts[branch] = len(files)
            for order_idx, path in enumerate(files):
                key = path.relative_to(raw_root).as_posix()
                image_meta[key] = {
                    "image_id": _short_id(key),
                    "sku_id": sku_id,
                    "branch": branch,
                    "order_idx": order_idx,
                    # Приор для поиска плоской выкладки: она обычно последним кадром.
                    "is_last_product": branch == PRODUCT and order_idx == len(files) - 1,
                }
                image_tasks.append((path, raw_root))

        row = {
            "sku_id": sku_id,
            "rel_dir": rel,
            "folder": folder,
            "title_base": base,
            "variant_no": variant_no,
            "design_id": design_id(folder),
            "category_raw": category_raw,
            "category_group": category_map.get(category_raw, "unmapped"),
            "n_product": counts[PRODUCT],
            "n_review": counts[REVIEW],
        }
        for key in typed_keys:
            row[key] = info.get(key)
        row["attrs_json"] = attrs_json(info, typed_keys)
        sku_rows.append(row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        probed = list(
            tqdm(
                pool.map(_probe, image_tasks),
                total=len(image_tasks),
                desc="images",
                unit="img",
            )
        )

    images = pd.DataFrame(probed)
    images = images.join(
        pd.DataFrame([image_meta[p] for p in images["rel_path"]]),
    )
    images["aspect"] = images["width"] / images["height"].where(images["height"] > 0)

    sku = pd.DataFrame(sku_rows)
    # design_no_variant считает, сколько цветовых вариантов у одного кроя —
    # это размер группы, которая обязана целиком уехать в один сплит.
    sku["design_variants"] = sku.groupby("design_id")["sku_id"].transform("size")

    ordered = [
        "image_id", "sku_id", "branch", "order_idx", "is_last_product",
        "rel_path", "width", "height", "aspect", "file_size", "corrupt",
    ]
    return sku, images[ordered]
