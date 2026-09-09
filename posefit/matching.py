"""Сходство вещи в отзыве с вещью из карточки товара.

Карточка WB склеивает цветовые варианты, и отзыв под чёрной курткой может
показывать бежевую. По разметке таких кадров 31%, а в группе upper — 41%:
это крупнейший источник шума в wild-ветке, и обучение на нём учит модель
перекрашивать вещь наугад.

Сравниваются эмбеддинги DINOv2 кропа вещи, вырезанного по маске parse.
Цветовые гистограммы на той же задаче дают AUC около 0.80; эмбеддинг видит
ещё принт и фактуру.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_embeddings(rows: pd.DataFrame, root, output_path) -> dict[str, np.ndarray]:
    """image_id -> нормированный вектор. Пустые (вещь не найдена) пропускаются."""
    out: dict[str, np.ndarray] = {}
    for row in rows.itertuples():
        path = output_path(root, "embed", row.sku_id, row.image_id)
        if not path.exists():
            continue
        vector = np.load(path)
        if vector.size:
            out[row.image_id] = vector.astype(np.float32)
    return out


def reference_bank(
    product: pd.DataFrame, embeddings: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Матрица эталонных векторов на каждый SKU.

    Берутся кадры карточки, где найден человек: на них вещь надета, и её вид
    сопоставим с отзывом. Один кадр эталоном делать нельзя — у части карточек
    сегментация на отдельном ракурсе срывается и даёт почти пустой кроп.
    """
    bank: dict[str, list[np.ndarray]] = {}
    for row in product.itertuples():
        vector = embeddings.get(row.image_id)
        if vector is not None:
            bank.setdefault(row.sku_id, []).append(vector)
    return {sku: np.stack(vectors) for sku, vectors in bank.items()}


def similarity(
    review: pd.DataFrame, embeddings: dict[str, np.ndarray], bank: dict[str, np.ndarray]
) -> pd.Series:
    """Максимум косинусной близости к эталонам своего SKU.

    Максимум, а не среднее: карточка снята с нескольких ракурсов, и совпадение
    хотя бы с одним из них — уже свидетельство той же вещи, тогда как среднее
    штрафовало бы за развороты и détail-кадры.
    """
    scores = []
    for row in review.itertuples():
        vector = embeddings.get(row.image_id)
        matrix = bank.get(row.sku_id)
        if vector is None or matrix is None:
            scores.append(np.nan)
        else:
            scores.append(float((matrix @ vector).max()))
    return pd.Series(scores, index=review.index, name="garment_sim")


def sweep(scores: np.ndarray, positive: np.ndarray, steps: int = 40) -> pd.DataFrame:
    """Точность и полнота по порогам — для выбора рабочей точки."""
    rows = []
    for threshold in np.quantile(scores, np.linspace(0, 0.9, steps)):
        keep = scores >= threshold
        tp, fp, fn = int((keep & positive).sum()), int((keep & ~positive).sum()), int((~keep & positive).sum())
        if tp:
            rows.append({
                "порог": float(threshold), "оставлено": int(keep.sum()),
                "точность": tp / (tp + fp), "полнота": tp / (tp + fn),
            })
    return pd.DataFrame(rows)


def auc(positive: np.ndarray, scores: np.ndarray) -> float:
    """Ранговая AUC (статистика Манна—Уитни), без зависимости от sklearn."""
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
