"""Замер качества: общий шум по парам, область маски, парная статистика.

Центральное утверждение работы — «конфигурация B лучше A» — требует парного
сравнения на одних и тех же тестовых парах. Отсюда три требования:

1. Одинаковый стартовый шум для одной пары во всех прогонах. Иначе разница
   между моделями смешивается с разницей в случайном шуме, и на 373 парах
   её не отделить.
2. Метрики по каждой паре, а не только среднее: по среднему нельзя сказать,
   значима ли разница или это разброс.
3. Метрики внутри маски. Вне маски модель копирует вход, и метрика по целому
   кадру на четыре пятых состоит из совпадений, которые от модели не зависят.

Модуль без torch — его арифметика проверяется на машине без GPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml


def pair_seed(pair_id: str, base: int = 0) -> int:
    """Сид стартового шума для пары — один и тот же во всех прогонах."""
    digest = hashlib.blake2b(f"{base}:{pair_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def mask_bbox(mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int] | None:
    """Прямоугольник вокруг маски (y0, y1, x0, x1) с запасом. None — маска пуста."""
    ys, xs = np.nonzero(mask > 0)
    if not len(ys):
        return None
    h, w = mask.shape[:2]
    return (max(0, ys.min() - pad), min(h, ys.max() + pad + 1),
            max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def masked_l1(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    """Средняя абсолютная ошибка по пикселям внутри маски, в шкале [0, 1]."""
    inside = mask > 0
    if not inside.any():
        return float("nan")
    diff = np.abs(pred.astype(np.float64) - true.astype(np.float64)) / 255.0
    return float(diff[inside].mean())


def mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0, level: float = 0.95):
    """Среднее и бутстреп-интервал."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float(values.mean()) if len(values) else float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return float(values.mean()), (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def paired_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Парная разница b - a на общих парах: среднее, интервал, доля пар, где b выше.

    Бутстреп по парам, а не по двум выборкам отдельно: одна и та же тестовая
    пара трудна для всех моделей, и учёт этого сужает интервал в разы.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    diff = b[keep] - a[keep]
    mean, (lo, hi) = mean_ci(diff, n_boot=n_boot, seed=seed)
    result = {
        "pairs": int(keep.sum()),
        "mean_diff": mean,
        "ci_low": lo,
        "ci_high": hi,
        "share_b_higher": float((diff > 0).mean()) if len(diff) else float("nan"),
        # Интервал не накрывает ноль — разница устойчива при выбранном уровне.
        "significant": bool(len(diff) >= 2 and (lo > 0 or hi < 0)),
    }
    try:
        from scipy.stats import wilcoxon

        nonzero = diff[diff != 0]
        result["wilcoxon_p"] = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 10 else float("nan")
    except ImportError:
        result["wilcoxon_p"] = float("nan")
    return result


# Параметры, которые обязаны совпадать с обучением: модель, обученная на одной
# маске или разрешении, на другой меряет не себя.
TRAINED_KEYS = ("mask_stage", "height", "width")


def resolve_eval_config(hp: dict, run: Path, checkpoint: str,
                        mask_stage: str | None = None) -> tuple[dict, str]:
    """Гиперпараметры для замера и пояснение, откуда они взяты.

    Обученный прогон — единственный источник правды о том, на чём он учился:
    маска и разрешение берутся из его снимка train_config.yaml, а текущий
    configs/train.yaml к моменту замера мог смениться и голоса не имеет.
    Раньше расхождение с конфигом считалось ошибкой, и замер прогона с
    уточнённой маской отказывался запускаться.
    """
    hp = dict(hp)
    snapshot = Path(run) / "train_config.yaml"
    if checkpoint == "none":
        if mask_stage:
            hp["mask_stage"] = mask_stage
        return hp, "контроль без обучения"
    if mask_stage:
        raise ValueError("--mask-stage допустим только с --checkpoint none: "
                         "обученный прогон сам помнит свою маску")
    if snapshot.exists():
        trained = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
        for key in TRAINED_KEYS:
            if key in trained:
                hp[key] = trained[key]
        return hp, f"из снимка {snapshot}"
    # Прогоны до появления снимков учились на исходной маске.
    hp["mask_stage"] = "agnostic"
    return hp, "снимка нет (старый прогон) — считаю mask_stage=agnostic"
