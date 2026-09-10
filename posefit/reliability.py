"""Надёжность пары как условие модели, а не как фильтр данных.

Три конфигурации базовой линии (studio / studio_wild / studio_clean) отвечают
на вопрос «какие пары оставить». Здесь вопрос ставится иначе: пары оставляются
все, а модели сообщается, насколько каждой можно верить. Токен надёжности
идёт в UNet через временной эмбеддинг — тот же путь, которым модель узнаёт
шаг диффузии, — и на выводе всегда подаётся токен «студия». Так шумные
отзывы учат модель телам и позам, но не учат её ошибаться в расцветке: ошибка
расцветки объясняется токеном «шумная пара», а не переносится на чистые.

Часть примеров получает пустой токен — это даёт на выводе guidance по
надёжности по образцу classifier-free guidance: шаг в сторону «чистой» пары
от «пары неизвестного качества».

Модуль без torch: разбиение по корзинам проверяется на машине без GPU.
"""

from __future__ import annotations

import pandas as pd

# Порядок фиксирован: индекс корзины хранится в чекпоинте.
BUCKETS = ("studio", "wild_clean", "wild_mid", "wild_noisy")
NULL_TOKEN = len(BUCKETS)       # «качество неизвестно» — для guidance
N_TOKENS = len(BUCKETS) + 1
STUDIO_TOKEN = BUCKETS.index("studio")


def bucket_thresholds(pairs: pd.DataFrame, clean_quantile: float = 0.65,
                      noisy_quantile: float = 0.3) -> tuple[float, float]:
    """Пороги colour_rank по wild-парам переданного кадра (верхний, нижний).

    Считаются по тому же кадру, что и select_pairs: по обучающему сплиту,
    а не по всему манифесту, иначе состав корзин зависел бы от других сплитов.
    """
    if "ветка" not in pairs or "colour_rank" not in pairs:
        return float("nan"), float("nan")
    wild = pairs.loc[pairs["ветка"] == "wild", "colour_rank"].dropna()
    if wild.empty:
        return float("nan"), float("nan")
    return float(wild.quantile(clean_quantile)), float(wild.quantile(noisy_quantile))


def assign_reliability(pairs: pd.DataFrame, clean_quantile: float = 0.65,
                       noisy_quantile: float = 0.3) -> pd.Series:
    """Индекс корзины для каждой пары.

    studio — студийная съёмка, wild_clean — отзыв с расцветкой не хуже порога
    studio_clean (та же квантиль, что у базовой линии, чтобы корзина совпадала
    с её отбором), wild_noisy — нижняя часть, wild_mid — между ними. Отзыв без
    оценки расцветки попадает в шумные: неизвестное качество нельзя считать
    хорошим.
    """
    upper, lower = bucket_thresholds(pairs, clean_quantile, noisy_quantile)
    rank = pairs["colour_rank"] if "colour_rank" in pairs else pd.Series(float("nan"), index=pairs.index)
    wild = pairs["ветка"] == "wild" if "ветка" in pairs else pd.Series(False, index=pairs.index)

    bucket = pd.Series(BUCKETS.index("studio"), index=pairs.index, dtype="int64")
    bucket[wild] = BUCKETS.index("wild_noisy")
    bucket[wild & (rank >= lower)] = BUCKETS.index("wild_mid")
    bucket[wild & (rank >= upper)] = BUCKETS.index("wild_clean")
    return bucket


def describe(bucket: pd.Series) -> dict[str, int]:
    """Сколько пар в каждой корзине — для лога запуска."""
    counts = bucket.value_counts()
    return {name: int(counts.get(i, 0)) for i, name in enumerate(BUCKETS)}
