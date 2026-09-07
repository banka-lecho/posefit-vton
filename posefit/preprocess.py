"""GPU-этап: детекция, поза, human parsing, agnostic-маски, VAE-латенты.

Стек намеренно собран целиком на `transformers` и `diffusers`, а не на
каноническом для VTON наборе (mmpose + detectron2). Тот набор требует сборки
mmcv под конкретную пару torch+CUDA и detectron2 из исходников; на машине,
к которой нет доступа, это самый дорогой способ отлаживаться. Здесь все веса
тянутся с HuggingFace одной строкой и не требуют компиляции.

Каждая стадия возобновляема: готовый файл не пересчитывается. Падение на
сорокатысячном кадре не означает пересчёт с нуля.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# ATR-18 — разметка, в которой обучен segformer_b2_clothes.
ATR = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
    "face": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
    "bag": 16, "scarf": 17,
}

# Что закрывается agnostic-маской для каждой группы одежды: сама вещь плюс
# то, что модель обязана дорисовать заново (руки под рукавами, шея под
# воротником). Оставить их видимыми — значит подсказать ответ.
AGNOSTIC_LABELS = {
    "upper": ["upper_clothes", "left_arm", "right_arm", "scarf", "belt"],
    "outer": ["upper_clothes", "dress", "left_arm", "right_arm", "scarf", "belt"],
    "dress": ["dress", "upper_clothes", "skirt", "left_arm", "right_arm", "belt"],
    "lower": ["pants", "skirt", "left_leg", "right_leg", "belt"],
}

MODELS = {
    "detect": "PekingU/rtdetr_r50vd_coco_o365",
    "pose": "usyd-community/vitpose-base-simple",
    "parse": "mattmdjaga/segformer_b2_clothes",
    "vae": "stabilityai/sd-vae-ft-mse",
}

TARGET_SIZE = (768, 1024)  # ширина, высота — формат VITON-HD


@dataclass(frozen=True)
class Stage:
    name: str
    ext: str
    needs: tuple[str, ...] = ()


STAGES = {
    s.name: s
    for s in (
        Stage("detect", "json"),
        Stage("parse", "png"),
        Stage("pose", "json", needs=("detect",)),
        Stage("agnostic", "png", needs=("parse", "pose")),
        Stage("latents", "npy", needs=("agnostic",)),
    )
}


def output_path(root: Path, stage: str, sku_id: str, image_id: str) -> Path:
    """Раскладка по SKU: каталог на десятки тысяч файлов тормозит на любой ФС."""
    return root / stage / sku_id / f"{image_id}.{STAGES[stage].ext}"


def shard_of(image_id: str, n_shards: int) -> int:
    """Устойчивое распределение по шардам: не зависит от порядка строк.

    Хэшируется весь идентификатор, а не его префикс: id не обязан быть
    равномерным hex-дайджестом, и на осмысленных именах префикс вырождается.
    """
    digest = hashlib.blake2b(image_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_shards


def working_set(
    images: pd.DataFrame,
    sku: pd.DataFrame,
    target_groups: Iterable[str] = ("upper", "lower", "dress", "outer"),
) -> pd.DataFrame:
    """Кадры, которые вообще имеет смысл препроцессить.

    Отбрасываются битые, шаблонные баннеры, повторы по содержимому и кадры
    отзывов с недостоверной привязкой к товару.
    """
    frame = images.merge(sku[["sku_id", "category_group"]], on="sku_id")
    keep = (
        frame["read_ok"]
        & ~frame["is_template"]
        & frame["is_canonical"]
        & frame["label_ok"]
        & frame["category_group"].isin(list(target_groups))
    )
    return frame.loc[keep].reset_index(drop=True)


def pending(
    rows: pd.DataFrame,
    root: Path,
    stage: str,
    shard: int = 0,
    n_shards: int = 1,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Строки этого шарда, для которых результат ещё не посчитан."""
    if n_shards > 1:
        mine = rows["image_id"].map(lambda i: shard_of(i, n_shards)) == shard
        rows = rows.loc[mine]
    if overwrite:
        return rows.reset_index(drop=True)
    done = rows.apply(
        lambda r: output_path(root, stage, r["sku_id"], r["image_id"]).exists(), axis=1
    )
    return rows.loc[~done].reset_index(drop=True) if len(rows) else rows


def build_agnostic(parse: np.ndarray, group: str, dilate_px: int = 12) -> np.ndarray:
    """Бинарная маска области, которую модель должна перерисовать."""
    import cv2

    labels = AGNOSTIC_LABELS.get(group, AGNOSTIC_LABELS["upper"])
    mask = np.isin(parse, [ATR[name] for name in labels]).astype(np.uint8) * 255
    if dilate_px:
        # Края сегментации неточны на пару пикселей; без запаса на границе
        # остаётся полоска исходной вещи, и модель учится её копировать.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel)
    return mask


def torso_visible(keypoints: dict, min_score: float = 0.3) -> bool:
    """Плечи и бёдра видимы — кадр годится в пару.

    Отсекает кропы по пояс и макро ткани, которых в отзывах очень много.
    """
    needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    scores = keypoints.get("scores", {})
    return all(scores.get(name, 0.0) >= min_score for name in needed)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
