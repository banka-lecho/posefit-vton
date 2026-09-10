"""Датасет пар для виртуальной примерки.

Каждый пример — вещь, человек в ней и маска области, которую модель должна
перерисовать. Маски посчитаны стадией agnostic на кадре, приведённом через
load_canonical, поэтому изображение и маска совмещаются пиксель в пиксель,
если применять то же преобразование.

Конфигурации обучения различаются только составом пар, не кодом; сам отбор
живёт в pairs.py, чтобы его можно было проверять без torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .pairs import CONFIGS, select_pairs  # noqa: F401  (реэкспорт для скриптов)
from .preprocess import load_canonical, output_path

@dataclass(frozen=True)
class Sample:
    person: np.ndarray      # (H, W, 3) float32 в [-1, 1]
    garment: np.ndarray     # (H, W, 3) float32 в [-1, 1]
    mask: np.ndarray        # (H, W, 1) float32, 1 = перерисовать
    pair_id: str


class VTONPairs(Dataset):
    """Пары (вещь, человек) с маской области примерки."""

    def __init__(
        self,
        pairs: pd.DataFrame,
        raw_root: Path,
        preproc_root: Path,
        height: int = 512,
        width: int = 384,
        flip: bool = True,
        seed: int = 0,
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.raw_root = Path(raw_root)
        self.preproc_root = Path(preproc_root)
        self.size = (width, height)
        self.flip = flip
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def _image(self, rel_path: str) -> np.ndarray:
        image = load_canonical(self.raw_root / rel_path).resize(self.size, Image.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 127.5 - 1.0

    def _mask(self, sku_id: str, image_id: str) -> np.ndarray:
        path = output_path(self.preproc_root, "agnostic", sku_id, image_id)
        mask = Image.open(path).resize(self.size, Image.NEAREST)
        return (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)[..., None]

    def __getitem__(self, index: int) -> dict:
        row = self.pairs.iloc[index]
        person = self._image(row["person_rel_path"])
        garment = self._image(row["garment_rel_path"])
        mask = self._mask(row["sku_id"], row["person_image_id"])

        if self.flip:
            # Отражается вся пара целиком: перевернуть вещь отдельно от человека
            # значило бы учить модель на несуществующем соответствии.
            rng = np.random.default_rng(self.seed + index)
            if rng.random() < 0.5:
                person, garment, mask = person[:, ::-1], garment[:, ::-1], mask[:, ::-1]

        return {
            "person": np.ascontiguousarray(person.transpose(2, 0, 1)),
            "garment": np.ascontiguousarray(garment.transpose(2, 0, 1)),
            "mask": np.ascontiguousarray(mask.transpose(2, 0, 1)),
            "pair_id": row["pair_id"],
            # Для разбивки метрик по группам одежды при замере.
            "category_group": str(row.get("category_group", "")),
        }
