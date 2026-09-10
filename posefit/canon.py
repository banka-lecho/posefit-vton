"""Канонические координаты тела: общий пространственный код для двух половин склейки.

В схеме CatVTON человек и вещь лежат рядом в одном тензоре, и self-attention
должен сам найти, какой участок вещи соответствует какому участку тела. Ничего,
кроме содержимого, ему в этом не помогает: позиционные коды у половин разные
(это просто разные координаты внутри широкой картинки), а поза человека в
отзывах гуляет. Здесь каждой половине приписывается карта координат в одной и
той же системе — «системе торса»:

* у человека она строится аффинно по плечам и бёдрам из позы: середина плеч —
  начало, плечи — ось u (левое плечо u=+0.5, правое u=-0.5), плечи→бёдра —
  ось v (плечи v=0, бёдра v=1). Поворот, наклон и ракурс торса учитываются
  самой аффинностью, ничего не нужно выпрямлять;
* у плоской выкладки — по габаритам вещи и приору её пропорций для группы
  одежды: ворот футболки кладётся на v=0, подол — на v≈1.3, ширина выкладки
  с рукавами считается за 1.6 ширины плеч.

Точка на плече человека и точка на плече футболки получают близкие (u, v), и
внимание может опираться на это как на подсказку вместо того, чтобы выводить
соответствие из одной фактуры. Приоры грубые — и это нормально: сеть получает
карты входными каналами с нулевой инициализацией и сама решает, насколько им
верить. Третий канал — скелет по позе: положение рук под рукавами и ног под
штанинами, которых в agnostic-маске не видно.

Модуль без torch: проверяется на машине без GPU.
"""

from __future__ import annotations

import numpy as np

GUIDE_CHANNELS = 3  # u, v, скелет

TORSO_POINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

# Пары точек COCO, между которыми рисуется скелет. Лицо не рисуется: оно не
# одежда и в уточнённой маске всё равно вырезано.
LIMBS = (
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)

# Приоры плоской выкладки в системе торса: (ширина выкладки в ширинах плеч,
# v верхнего края, длина вещи в длинах торса). Ширина с рукавами у футболки
# заметно больше плеч; юбка и брюки начинаются чуть выше бёдер (v=1) и тянутся
# к щиколоткам (v≈2.7 при обычных пропорциях тела).
GARMENT_FRAME = {
    "upper": (1.6, 0.0, 1.3),
    "outer": (1.7, 0.0, 1.6),
    "dress": (1.3, 0.0, 2.4),
    "lower": (1.0, 0.9, 1.8),
}

# Диапазон, в котором координаты информативны; дальше они обрезаются, иначе
# фон в углах кадра получал бы огромные значения и ломал бы масштаб канала.
U_RANGE = (-2.0, 2.0)
V_RANGE = (-1.0, 3.0)


def scale_keypoints(keypoints: dict, src: tuple[int, int], dst: tuple[int, int]) -> dict:
    """Переводит точки из размера кадра src=(w, h) в dst=(w, h)."""
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    return {name: (float(xy[0]) * sx, float(xy[1]) * sy) for name, xy in keypoints.items()}


def torso_frame(keypoints: dict, scores: dict, size: tuple[int, int],
                min_score: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Аффинная система торса: (начало, ось u, ось v) в пикселях.

    None — когда торс не виден или вырожден (профиль: плечи в одной точке,
    оси коллинеарны). Тогда вызывающий код берёт запасную систему по маске.
    """
    for name in TORSO_POINTS:
        if name not in keypoints or scores.get(name, 0.0) < min_score:
            return None
    ls = np.asarray(keypoints["left_shoulder"][:2], dtype=np.float64)
    rs = np.asarray(keypoints["right_shoulder"][:2], dtype=np.float64)
    lh = np.asarray(keypoints["left_hip"][:2], dtype=np.float64)
    rh = np.asarray(keypoints["right_hip"][:2], dtype=np.float64)

    origin = (ls + rs) / 2.0
    u_axis = ls - rs                      # длина = ширина плеч
    v_axis = (lh + rh) / 2.0 - origin     # длина = длина торса
    width, height = size
    shoulder = float(np.linalg.norm(u_axis))
    torso = float(np.linalg.norm(v_axis))
    if shoulder < 0.04 * width or torso < 0.06 * height:
        return None
    det = u_axis[0] * v_axis[1] - u_axis[1] * v_axis[0]
    # sin угла между осями меньше ~6° — система почти вырождена.
    if abs(det) < 0.1 * shoulder * torso:
        return None
    return origin, u_axis, v_axis


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w]
    return xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5


def affine_uv(h: int, w: int, frame) -> np.ndarray:
    """Координаты (u, v) каждого пикселя в аффинной системе frame — (2, h, w)."""
    origin, u_axis, v_axis = frame
    xs, ys = _grid(h, w)
    px, py = xs - origin[0], ys - origin[1]
    det = u_axis[0] * v_axis[1] - u_axis[1] * v_axis[0]
    # Решение [u_axis v_axis] @ (a, b) = p через обратную матрицу 2x2.
    u = (px * v_axis[1] - py * v_axis[0]) / det
    v = (py * u_axis[0] - px * u_axis[1]) / det
    return np.stack([u, v]).astype(np.float32)


def box_uv(h: int, w: int, box: tuple[float, float, float, float], group: str) -> np.ndarray:
    """(u, v) по прямоугольнику вещи (x0, y0, x1, y1) и приору группы — (2, h, w).

    Используется для плоской выкладки и как запасной вариант для человека,
    когда торс по позе не построить: тогда область вещи на нём трактуется
    как та же выкладка.
    """
    u_span, v_top, v_len = GARMENT_FRAME.get(group, GARMENT_FRAME["upper"])
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    xs, ys = _grid(h, w)
    u = (xs - (x0 + x1) / 2.0) / bw * u_span
    v = v_top + (ys - y0) / bh * v_len
    return np.stack([u, v]).astype(np.float32)


def encode_uv(uv: np.ndarray) -> np.ndarray:
    """Обрезает координаты рабочим диапазоном и приводит к [-1, 1]."""
    u = np.clip(uv[0], *U_RANGE) / U_RANGE[1]
    v = (np.clip(uv[1], *V_RANGE) - (V_RANGE[0] + V_RANGE[1]) / 2.0) / ((V_RANGE[1] - V_RANGE[0]) / 2.0)
    return np.stack([u, v]).astype(np.float32)


def skeleton_map(h: int, w: int, keypoints: dict, scores: dict,
                 min_score: float = 0.3, thickness: float = 0.015) -> np.ndarray:
    """Скелет по позе линиями толщиной thickness от ширины кадра — (h, w) в [0, 1]."""
    import cv2

    canvas = np.zeros((h, w), dtype=np.uint8)
    px = max(1, int(round(thickness * w)))
    for a, b in LIMBS:
        if scores.get(a, 0.0) < min_score or scores.get(b, 0.0) < min_score:
            continue
        if a not in keypoints or b not in keypoints:
            continue
        pa = (int(round(keypoints[a][0])), int(round(keypoints[a][1])))
        pb = (int(round(keypoints[b][0])), int(round(keypoints[b][1])))
        cv2.line(canvas, pa, pb, 255, px, lineType=cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Прямоугольник (x0, y0, x1, y1) вокруг ненулевых пикселей маски."""
    ys, xs = np.nonzero(mask > 0)
    if not len(ys):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def garment_box(parse: np.ndarray, group: str, image: np.ndarray | None = None,
                min_px: int = 500) -> tuple[float, float, float, float] | None:
    """Прямоугольник вещи на плоской выкладке.

    Сначала по классам parse, как в garment_crop; когда parsing на выкладке
    не сработал — по всему, что не белый фон студийной съёмки.
    """
    from .preprocess import ATR, GARMENT_LABELS

    labels = GARMENT_LABELS.get(group, GARMENT_LABELS["upper"])
    mask = np.isin(parse, [ATR[name] for name in labels])
    if mask.sum() >= min_px:
        return mask_box(mask)
    if image is not None:
        dark = image.min(axis=-1) < 240
        if dark.sum() >= min_px:
            return mask_box(dark)
    return None


def person_guide(h: int, w: int, keypoints: dict, scores: dict, group: str,
                 fallback_mask: np.ndarray | None = None) -> np.ndarray:
    """Три канала для половины с человеком: u, v по торсу и скелет — (3, h, w).

    keypoints уже в пикселях кадра (h, w). Без торса координаты берутся по
    прямоугольнику fallback_mask (agnostic-маска) с приором группы; без него —
    по всему кадру.
    """
    frame = torso_frame(keypoints, scores, (w, h))
    if frame is not None:
        uv = affine_uv(h, w, frame)
    else:
        box = mask_box(fallback_mask) if fallback_mask is not None else None
        uv = box_uv(h, w, box or (0.0, 0.0, float(w), float(h)), group)
    return np.concatenate([encode_uv(uv), skeleton_map(h, w, keypoints, scores)[None]])


def garment_guide(h: int, w: int, box: tuple[float, float, float, float] | None,
                  group: str) -> np.ndarray:
    """Три канала для половины с вещью: u, v по габаритам и пустой скелет — (3, h, w)."""
    uv = box_uv(h, w, box or (0.0, 0.0, float(w), float(h)), group)
    return np.concatenate([encode_uv(uv), np.zeros((1, h, w), dtype=np.float32)])
