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
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

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
    "embed": "facebook/dinov2-base",
    "vae": "stabilityai/sd-vae-ft-mse",
}

# Классы ATR, образующие саму примеряемую вещь (без рук и шеи, в отличие от
# AGNOSTIC_LABELS): по ним вырезается кроп для эмбеддинга.
GARMENT_LABELS = {
    "upper": ["upper_clothes"],
    "outer": ["upper_clothes", "dress"],
    "dress": ["dress", "upper_clothes"],
    "lower": ["pants", "skirt"],
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
        Stage("agnostic", "png", needs=("parse",)),
        # Та же маска, но с вырезанными лицом и кистями — см. refine_agnostic.
        Stage("agnostic_refined", "png", needs=("agnostic", "parse", "pose")),
        Stage("embed", "npy", needs=("parse",)),
        Stage("colour", "npy", needs=("parse",)),
        # Латенты обычного кадра. Латенты замаскированного кадра зависят от
        # соглашения о маскировании в выбранной архитектуре, поэтому считаются
        # позже, когда бэкбон определён.
        Stage("latents", "npy"),
    )
}


def load_canonical(path: Path | Image.Image, size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """Кадр, приведённый к целевому размеру с сохранением пропорций и паддингом.

    Пропорции у отзывов гуляют от 450x1000 до 1000x750, и растягивание исказило
    бы силуэт — а именно его модель и учится воспроизводить. Все стадии
    препроцессинга считались через эту же функцию, поэтому маски и позы
    совмещаются с кадром пиксель в пиксель. Принимает путь или уже открытое
    изображение (ноутбук со своими примерами).
    """
    image = (path if isinstance(path, Image.Image) else Image.open(path)).convert("RGB")
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((max(1, round(image.width * scale)),
                            max(1, round(image.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return canvas


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


def _components_near(region: np.ndarray, anchors: list[np.ndarray], radius: float) -> np.ndarray:
    """Связные куски region, у которых есть пиксель ближе radius к одной из опор."""
    import cv2

    if not region.any() or not anchors:
        return np.zeros_like(region, dtype=bool)
    count, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(region)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    near = np.zeros(len(points), dtype=bool)
    for anchor in anchors:
        near |= np.linalg.norm(points - anchor, axis=1) < radius
    chosen = np.unique(labels[ys[near], xs[near]])
    chosen = chosen[chosen != 0]
    return np.isin(labels, chosen)


def hand_region(parse: np.ndarray, keypoints: dict, scores: dict,
                min_score: float = 0.3) -> np.ndarray:
    """Кисти: связная часть руки за запястьем.

    В разметке ATR кисть — часть класса «рука», поэтому граница проводится по
    позе: берётся всё за запястьем по оси локоть→запястье, а из этого — только
    куски, касающиеся самого запястья. Радиусом от запястья ограничивать нельзя:
    когда рука вытянута к камере, предплечье в кадре укорочено, и пальцы
    выходят за любой радиус, пропорциональный его длине.

    Проверяются оба запястья — соглашения лево/право у ATR и COCO расходятся.
    """
    arm = np.isin(parse, [ATR["left_arm"], ATR["right_arm"]])
    hand = np.zeros_like(arm)
    ys, xs = np.nonzero(arm)
    if not len(ys):
        return hand
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    for side in ("left", "right"):
        elbow, wrist = f"{side}_elbow", f"{side}_wrist"
        if scores.get(elbow, 0.0) < min_score or scores.get(wrist, 0.0) < min_score:
            continue
        e = np.asarray(keypoints[elbow][:2], dtype=np.float64)
        w = np.asarray(keypoints[wrist][:2], dtype=np.float64)
        axis = w - e
        length = float(np.linalg.norm(axis))
        if length < 5:
            continue
        # 0 у локтя, 1 у запястья. Порог чуть раньше запястья: точка запястья
        # у позы стоит на суставе, и кромка ладони иначе осталась бы в маске.
        beyond = np.zeros_like(arm)
        along = (points - e) @ axis / length**2
        beyond[ys[along > 0.9], xs[along > 0.9]] = True
        hand |= _components_near(beyond, [w], radius=max(0.35 * length, 6.0))
    return hand


def face_region(parse: np.ndarray, keypoints: dict, scores: dict,
                min_score: float = 0.3) -> np.ndarray:
    """Лицо — только там, где по позе действительно голова.

    Класс «лицо» у parse иногда ложно срабатывает на одежде: на тестовом наборе
    он нашёлся на брюках. Вырезать его вслепую значит пробить дыру в маске
    вещи, где осталась бы старая одежда. Поэтому берутся только куски,
    касающиеся носа или глаз; не найдена голова — не вырезается ничего.
    """
    face = parse == ATR["face"]
    heads = [np.asarray(keypoints[name][:2], dtype=np.float64)
             for name in ("nose", "left_eye", "right_eye")
             if scores.get(name, 0.0) >= min_score and name in keypoints]
    if not heads:
        return np.zeros_like(face)
    # Масштаб головы — расстояние между плечами; без плеч берётся запас по кадру.
    if scores.get("left_shoulder", 0) >= min_score and scores.get("right_shoulder", 0) >= min_score:
        scale = float(np.linalg.norm(np.subtract(keypoints["left_shoulder"][:2],
                                                 keypoints["right_shoulder"][:2])))
    else:
        scale = 0.1 * parse.shape[0]
    return _components_near(face, heads, radius=max(0.5 * scale, 8.0))


def refine_agnostic(mask: np.ndarray, parse: np.ndarray, keypoints: dict, scores: dict,
                    carve_px: int = 4) -> np.ndarray:
    """Вырезает из маски лицо и кисти: они никогда не бывают частью одежды.

    На тестовом наборе кисть попадала в маску в 77% пар, край лица — в 82%:
    модель рисовала их с нуля, и SD 1.5 на 384x512 делает это плохо. Вырез
    небольшой — кисти занимают 3.3% маски, — а выигрыш виден на каждом кадре.

    Вырезается уже расширенная маска, иначе расширение снова залезло бы на лицо.
    Лицо и кисти берутся с небольшим запасом carve_px, чтобы край не остался
    внутри маски полоской.
    """
    import cv2

    keep = face_region(parse, keypoints, scores) | hand_region(parse, keypoints, scores)
    if carve_px and keep.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (carve_px, carve_px))
        keep = cv2.dilate(keep.astype(np.uint8), kernel) > 0
    refined = mask.copy()
    refined[keep] = 0
    return refined


def garment_crop(
    image: np.ndarray,
    parse: np.ndarray,
    group: str,
    min_px: int = 500,
    pad: int = 8,
    fill: int = 128,
):
    """Кроп по маске вещи; всё, что не вещь, заливается нейтральным серым.

    Заливка обязательна: без неё эмбеддинг вбирает кожу, фон и обстановку
    комнаты, и сравнение отзыва со студийной карточкой меряет разницу
    интерьеров, а не расцветки.
    """
    mask = np.isin(parse, [ATR[name] for name in GARMENT_LABELS[group]])
    if mask.sum() < min_px:
        return None

    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - pad), min(parse.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(parse.shape[1], xs.max() + pad + 1)

    crop = image[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = fill
    return crop


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
