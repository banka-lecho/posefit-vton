"""Блок F: разбиение на train/val/test по группам кроёв.

Единица разбиения — не папка и не SKU, а группа связанных кроёв из
dedup.design_components. Разбиение по папкам увело бы разные расцветки одной
вещи в разные сплиты и завысило бы метрики на тесте.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

# 70/15/15, а не привычные 80/10/10: узкое место работы — не объём обучающей
# выборки, а разнообразие вещей в тесте. При 80/10/10 в тест попадало 69 SKU
# с годными отзывами, при 70/15/15 — 102, ценой 13% обучающих SKU.
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _rank(key: str, seed: int) -> float:
    """Детерминированное положение группы в [0,1) — не зависит от порядка строк."""
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def assign(
    sku: pd.DataFrame,
    components: dict[str, str],
    seed: int = 20260906,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Возвращает таблицу split_group -> split, стратифицированную по группе одежды."""
    ratios = ratios or RATIOS
    frame = sku.assign(split_group=sku["design_id"].map(components))

    # Категория группы — самая частая среди её SKU: связанные кроя
    # (обычная и plus-size линейка) изредка лежат в разных разделах WB.
    groups = (
        frame.groupby("split_group")
        .agg(
            category_group=("category_group", lambda s: s.value_counts().index[0]),
            n_sku=("sku_id", "size"),
        )
        .reset_index()
    )
    groups["rank"] = [_rank(g, seed) for g in groups["split_group"]]

    parts = []
    for _, stratum in groups.groupby("category_group"):
        stratum = stratum.sort_values("rank").reset_index(drop=True)
        position = (stratum.index + 0.5) / len(stratum)
        bounds = {}
        acc = 0.0
        for name, share in ratios.items():
            bounds[name] = (acc, acc + share)
            acc += share
        split = pd.Series("train", index=stratum.index)
        for name, (lo, hi) in bounds.items():
            split[(position >= lo) & (position < hi)] = name
        parts.append(stratum.assign(split=split.to_numpy()))

    return pd.concat(parts, ignore_index=True)[
        ["split_group", "category_group", "n_sku", "split"]
    ]


def check_leakage(images: pd.DataFrame) -> dict[str, int]:
    """Ни один кадр не должен встречаться побайтово в двух разных сплитах."""
    spread = images.groupby("content_md5")["split"].nunique()
    offenders = spread[spread > 1]
    return {
        "content_hashes_in_multiple_splits": int(len(offenders)),
        "images_affected": int(images["content_md5"].isin(offenders.index).sum()),
    }


def write(path: Path, groups: pd.DataFrame, seed: int, leakage: dict[str, int]) -> None:
    payload = {
        "seed": seed,
        "ratios": RATIOS,
        "n_groups": int(len(groups)),
        "counts": groups["split"].value_counts().to_dict(),
        "leakage_check": leakage,
        "assignment": dict(zip(groups["split_group"], groups["split"])),
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    payload["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
