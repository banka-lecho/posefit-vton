"""Дедупликация и проверка надёжности привязки кадра к товару.

В выгрузке три разных вида повторов, и смешивать их нельзя:

1. Шаблонные ассеты — баннеры размерной сетки, вшитые в карточки. Это
   инфографика, а не съёмка вещи. Часть из них общая для всех категорий,
   часть своя у каждой ('Брюки', 'Юбки'), поэтому одного признака мало.
2. Повтор внутри одного кроя — кадр переиспользован в карточках разных
   расцветок. Метка верна, кадр засчитывается один раз.
3. Повтор между разными кроями — фото отзыва лежит под несколькими
   непохожими вещами. Метка недостоверна: верной может быть максимум одна
   карточка, а какая — из данных не видно. Такие кадры в пары не идут.
"""

from __future__ import annotations

import pandas as pd

# Студийный кадр может быть общим у обычной и plus-size карточки одного кроя,
# изредка у трёх расцветок. Разброс в четыре и более кроя означает баннер.
TEMPLATE_MIN_DESIGNS = 4


def annotate(images: pd.DataFrame, template_min_designs: int = TEMPLATE_MIN_DESIGNS) -> pd.DataFrame:
    """Добавляет is_template, dup_designs, label_ok, is_canonical."""
    out = images.copy()
    by_md5 = out.groupby("content_md5")
    out["dup_designs"] = by_md5["design_id"].transform("nunique")
    dup_cats = by_md5["category_raw"].transform("nunique")

    # Товар принадлежит ровно одной категории, поэтому файл, лежащий и в
    # 'Юбках', и в 'Блузках', служебный. Внутрикатегорийные баннеры ловятся
    # по разбросу: студия снимается под конкретную вещь и так не расходится.
    out["is_template"] = (dup_cats > 1) | (
        (out["branch"] == "product") & (out["dup_designs"] >= template_min_designs)
    )
    out["label_ok"] = (out["dup_designs"] == 1) & ~out["is_template"]

    order = out.sort_values(["content_md5", "sku_id", "branch", "order_idx"])
    out["is_canonical"] = out.index.isin(order.drop_duplicates("content_md5").index)
    return out


def design_components(images: pd.DataFrame) -> dict[str, str]:
    """Объединяет кроя, связанные общим студийным кадром, в группу для сплита.

    Ловит то, что не ловит нормализация названия: 'Блузка на шнуровке' и
    'Блузка на шнуровке больших размеров' — разные строки, но плоская выкладка
    у них побайтово одна, то есть крой один и в разные сплиты им нельзя.

    Связываются только кадры товара: фото отзывов повторяются под непохожими
    вещами по ошибке привязки, и склейка по ним схлопнула бы полкаталога.
    """
    parent: dict[str, str] = {design: design for design in images["design_id"].unique()}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    links = images[(images["branch"] == "product") & ~images["is_template"]]
    for _, designs in links.groupby("content_md5")["design_id"]:
        unique = designs.unique()
        for other in unique[1:]:
            union(unique[0], other)

    return {design: find(design) for design in parent}
