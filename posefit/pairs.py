"""Отбор обучающих пар под конфигурацию эксперимента.

Отдельно от dataset.py намеренно: здесь нет torch, поэтому логику отбора можно
проверять на машине без GPU — а именно она определяет, чем три прогона
отличаются друг от друга.
"""

from __future__ import annotations

import pandas as pd

CONFIGS = ("studio", "studio_wild", "studio_clean")


def select_pairs(
    pairs: pd.DataFrame,
    config: str,
    split: str = "train",
    clean_quantile: float = 0.65,
) -> pd.DataFrame:
    """Подвыборка пар под конфигурацию обучения.

    studio       только студийные пары — базовая линия
    studio_wild  плюс кадры из отзывов, как их отобрал рабочий фильтр
    studio_clean плюс кадры из отзывов с более жёстким порогом расцветки
    """
    if config not in CONFIGS:
        raise ValueError(f"неизвестная конфигурация {config!r}, ожидается одна из {CONFIGS}")

    frame = pairs[pairs["split"] == split]
    studio = frame[frame["ветка"] == "studio"]
    if config == "studio":
        return studio.reset_index(drop=True)

    wild = frame[frame["ветка"] == "wild"]
    if config == "studio_clean":
        # Порог считается по wild-парам этого же сплита, а не по всему
        # манифесту: иначе состав обучающей выборки зависел бы от того, сколько
        # строк других сплитов лежит во входном кадре, и прогоны, запущенные
        # в разное время, перестали бы быть сравнимыми.
        threshold = wild["colour_rank"].quantile(clean_quantile)
        wild = wild[wild["colour_rank"] >= threshold]
    return pd.concat([studio, wild], ignore_index=True)
