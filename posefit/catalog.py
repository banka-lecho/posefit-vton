"""Разбор выгрузки WB: info.csv, нормализация id, отображение категорий.

Структура выгрузки:
    data/<Категория>/<Название SKU>/info.csv
                                   /Фото товара/<N>.webp
                                   /Фото с отзывов/<N>_fs.webp

Имена файлов: у товара `1.webp`..`N.webp`, у отзывов `fs.webp` и `N_fs.webp`.
Порядок фото товара значим — последний кадр обычно плоская выкладка вещи.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

# Хвост " (2)", " (17)" отличает цветовые варианты одного и того же кроя.
_VARIANT_SUFFIX = re.compile(r"\s*\((\d+)\)\s*$")
_LEADING_INT = re.compile(r"^(\d+)")
_WS = re.compile(r"\s+")


def rel_posix(path: Path, root: Path) -> str:
    """Путь относительно корня выгрузки, нормализованный в NFC.

    Без нормализации манифест непереносим между машинами: macOS отдаёт имена
    файлов в NFD (кириллические 'й' и 'ё' разложены на букву и диакритику),
    Linux — в NFC. Идентификаторы считаются как хэш от пути, поэтому одна и та
    же папка получала бы разные sku_id на разных системах, и результаты
    препроцессинга, посчитанные на сервере, не находились бы локально.
    """
    return unicodedata.normalize("NFC", path.relative_to(root).as_posix())


def parse_info_csv(path: Path) -> dict[str, str]:
    """Читает info.csv в словарь. Повторяющиеся ключи склеиваются через '; '.

    Повторы реальны: у части SKU 'Сезон' указан несколько раз.
    """
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or row[0] == "key":
                continue
            key = row[0].strip()
            value = ",".join(row[1:]).strip()
            if key and value:
                out.setdefault(key, []).append(value)
    return {k: "; ".join(dict.fromkeys(v)) for k, v in out.items()}


def split_variant(folder_name: str) -> tuple[str, int]:
    """'Куртка стеганая (3)' -> ('Куртка стеганая', 3). Без хвоста -> вариант 1."""
    match = _VARIANT_SUFFIX.search(folder_name)
    if not match:
        return folder_name.strip(), 1
    return folder_name[: match.start()].strip(), int(match.group(1))


def design_id(folder_name: str) -> str:
    """Идентификатор кроя, общий для всех цветовых вариантов.

    Используется как единица разбиения на train/val/test: 3220 папок дают
    ~1869 кроёв, и разбиение по папкам увело бы разные расцветки одной вещи
    в разные сплиты.
    """
    base, _ = split_variant(folder_name)
    base = unicodedata.normalize("NFKC", base).casefold()
    base = base.replace("ё", "е")
    return _WS.sub(" ", base).strip()


def image_sort_key(path: Path) -> tuple[int, str]:
    """Порядок кадров внутри папки: по числовому префиксу, затем по имени."""
    match = _LEADING_INT.match(path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def build_category_map(config: dict) -> dict[str, str]:
    """Сырая категория WB -> укрупнённая группа ('drop' для исключённых)."""
    mapping: dict[str, str] = {}
    for group, names in config["categories"].items():
        for name in names:
            if name in mapping:
                raise ValueError(f"Категория {name!r} указана в конфиге дважды")
            mapping[name] = group
    return mapping


def attrs_json(info: dict[str, str], typed_keys: list[str]) -> str:
    """Ключи info.csv, не вынесенные в типизированные колонки."""
    rest = {k: v for k, v in info.items() if k not in typed_keys}
    return json.dumps(rest, ensure_ascii=False, sort_keys=True)
