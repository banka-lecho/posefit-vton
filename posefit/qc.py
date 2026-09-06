"""Блок C: признаки качества изображений, считаемые на CPU.

Всё содержимое кадра сводится к 64-битному perceptual hash и десятку скаляров,
поэтому таблица на 78k строк весит единицы мегабайт и переживает любые
переборы порогов без повторного декодирования webp.

phash здесь нужен не только для отсева дублей внутри SKU: 3220 папок дают
1869 кроёв, и студийные кадры переиспользуются между цветовыми вариантами.
Связные компоненты по phash — единственный способ поймать такую утечку
до того, как она завысит метрики на тесте.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

# Сторона, к которой приводится кадр перед подсчётом резкости: без этого
# дисперсия лапласиана зависела бы от исходного разрешения, а у отзывов оно
# гуляет от 184 до 1000 пикселей.
STAT_SIDE = 256
HASH_SIDE = 32
BORDER_PX = 12
NEAR_WHITE = 235


def phash64(gray: np.ndarray) -> np.uint64:
    """64-битный DCT-хэш (реализация Zauner) поверх уже обесцвеченного кадра."""
    small = cv2.resize(gray, (HASH_SIDE, HASH_SIDE), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))
    block = dct[:8, :8].flatten()
    # DC-коэффициент несёт среднюю яркость, а не структуру, и его исключают.
    bits = block > np.median(block[1:])
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Побитовое расстояние между массивами uint64."""
    x = np.bitwise_xor(a, b)
    out = np.zeros_like(x, dtype=np.uint8)
    for _ in range(64):
        out += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return out


def features(path: Path) -> dict:
    """Признаки одного кадра. При ошибке чтения возвращает read_ok=False."""
    row = {
        "read_ok": False, "sharpness": np.nan, "brightness": np.nan,
        "contrast": np.nan, "border_mean": np.nan, "border_std": np.nan,
        "white_frac": np.nan, "phash": np.uint64(0), "content_md5": "",
    }
    try:
        blob = path.read_bytes()
    except OSError:
        return row
    # Один проход по файлу даёт и хэш содержимого, и пиксели: побайтовое
    # равенство отличает настоящий дубликат от совпадения phash на двух
    # разных, но одинаково снятых вещах.
    row["content_md5"] = hashlib.md5(blob).hexdigest()
    if not blob:
        # В выгрузке есть файлы нулевой длины; imdecode на них падает ассертом.
        return row
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return row

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = STAT_SIDE / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)

    border = np.concatenate([
        small[:BORDER_PX].ravel(), small[-BORDER_PX:].ravel(),
        small[:, :BORDER_PX].ravel(), small[:, -BORDER_PX:].ravel(),
    ])

    row.update(
        read_ok=True,
        sharpness=float(cv2.Laplacian(small, cv2.CV_64F).var()),
        brightness=float(small.mean()),
        contrast=float(small.std()),
        # Плоская выкладка снята на равномерном светлом фоне: у неё высокий
        # border_mean при низком border_std. У кадра на модели фон тот же,
        # но силуэт заходит в рамку и std растёт.
        border_mean=float(border.mean()),
        border_std=float(border.std()),
        white_frac=float((small > NEAR_WHITE).mean()),
        phash=phash64(gray),
    )
    return row
