"""Отбор обучающих пар под конфигурацию эксперимента.

Отдельно от dataset.py намеренно: здесь нет torch, поэтому логику отбора можно
проверять на машине без GPU — а именно она определяет, чем три прогона
отличаются друг от друга.
"""

from __future__ import annotations

import pandas as pd

CONFIGS = ("studio", "studio_wild", "studio_clean", "studio_random")

# Сид случайной подвыборки studio_random: фиксирован, чтобы прогон повторялся.
RANDOM_SUBSET_SEED = 20260910


def select_pairs(
    pairs: pd.DataFrame,
    config: str,
    split: str = "train",
    clean_quantile: float = 0.65,
) -> pd.DataFrame:
    """Подвыборка пар под конфигурацию обучения.

    studio        только студийные пары — базовая линия
    studio_wild   плюс кадры из отзывов, как их отобрал рабочий фильтр
    studio_clean  плюс кадры из отзывов с более жёстким порогом расцветки
    studio_random плюс СЛУЧАЙНЫЕ кадры из отзывов в том же числе, что у clean

    studio_random — контроль, без которого wild и clean несравнимы: clean
    отличается от wild не только чистотой, но и объёмом (913 пар против 2609).
    Контроль того же объёма, что clean, но без фильтра расцветки, разводит
    эффекты: clean против random — чистота при равном объёме, wild против
    random — объём при равном уровне шума.
    """
    if config not in CONFIGS:
        raise ValueError(f"неизвестная конфигурация {config!r}, ожидается одна из {CONFIGS}")

    frame = pairs[pairs["split"] == split]
    studio = frame[frame["ветка"] == "studio"]
    if config == "studio":
        return studio.reset_index(drop=True)

    wild = frame[frame["ветка"] == "wild"]
    # Порог считается по wild-парам этого же сплита, а не по всему манифесту:
    # иначе состав обучающей выборки зависел бы от того, сколько строк других
    # сплитов лежит во входном кадре, и прогоны перестали бы быть сравнимыми.
    threshold = wild["colour_rank"].quantile(clean_quantile)
    clean = wild[wild["colour_rank"] >= threshold]

    if config == "studio_clean":
        wild = clean
    elif config == "studio_random":
        # Ровно столько же, сколько у clean, но выбранных без оглядки на цвет.
        wild = wild.sample(n=len(clean), random_state=RANDOM_SUBSET_SEED)
    return pd.concat([studio, wild], ignore_index=True)
