"""Разрешение путей проекта.

raw_root и cache_root живут локально. preproc_root (поза, parsing, DensePose,
латенты) на порядок тяжелее и предназначен для GPU-машины, поэтому может быть
не задан: обращение к нему до настройки падает с внятной ошибкой, а не пишет
120 ГБ на системный диск.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"


@dataclass(frozen=True)
class Paths:
    raw_root: Path
    cache_root: Path
    _preproc_root: Path | None

    @property
    def preproc_root(self) -> Path:
        if self._preproc_root is None:
            raise RuntimeError(
                "preproc_root не задан. Укажи paths.preproc_root в configs/data.yaml "
                "или переменную окружения POSEFIT_PREPROC_ROOT — это каталог на "
                "80–120 ГБ, ему не место на системном диске."
            )
        return self._preproc_root

    @property
    def has_preproc(self) -> bool:
        return self._preproc_root is not None

    @property
    def manifest_sku(self) -> Path:
        return self.cache_root / "manifest_sku.parquet"

    @property
    def manifest_images(self) -> Path:
        return self.cache_root / "manifest_images.parquet"

    @property
    def splits(self) -> Path:
        return self.cache_root / "splits.json"


def _resolve(value: str | None, env: str) -> Path | None:
    value = os.environ.get(env) or value
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_paths(config: dict) -> Paths:
    raw = config["paths"]
    return Paths(
        raw_root=_resolve(raw["raw_root"], "POSEFIT_RAW_ROOT"),
        cache_root=_resolve(raw["cache_root"], "POSEFIT_CACHE_ROOT"),
        _preproc_root=_resolve(raw.get("preproc_root"), "POSEFIT_PREPROC_ROOT"),
    )
