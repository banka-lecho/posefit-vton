"""Датасет пар для виртуальной примерки.

Каждый пример — вещь, человек в ней и маска области, которую модель должна
перерисовать. Маски посчитаны стадией agnostic на кадре, приведённом через
load_canonical, поэтому изображение и маска совмещаются пиксель в пиксель,
если применять то же преобразование.

Конфигурации обучения различаются только составом пар, не кодом; сам отбор
живёт в pairs.py, чтобы его можно было проверять без torch.

Для якорной схемы (см. model.py) датасет дополнительно отдаёт:
  guide_person, guide_garment — каналы-подсказки из canon.py по позе человека
                                и parse плоской выкладки;
  garment_embed               — DINOv2-вектор вещи со стадии embed;
  reliability                 — корзина надёжности пары (столбец reliability
                                в таблице пар; без него — «студия»).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .canon import GUIDE_CHANNELS, garment_box, garment_guide, person_guide, scale_keypoints
from .preprocess import TARGET_SIZE, load_canonical, output_path
from .reliability import STUDIO_TOKEN

GARMENT_EMBED_DIM = 768


def to_model_image(canonical: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Канонический кадр 768x1024 -> (H, W, 3) float32 в [-1, 1] рабочего размера size=(w, h)."""
    image = canonical.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 127.5 - 1.0


def to_model_mask(mask: Image.Image | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Маска 0/255 канонического размера -> (H, W, 1) float32 из {0, 1}."""
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask = mask.resize(size, Image.NEAREST)
    return (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)[..., None]


def make_guides(pose: dict, garment_parse: np.ndarray | Image.Image, mask: np.ndarray,
                garment: np.ndarray, group: str,
                size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Каналы-подсказки обеих половин в рабочем размере: два массива (3, H, W).

    pose — результат стадии pose на каноническом кадре человека; garment_parse —
    карта parse канонического кадра вещи; mask и garment — уже в рабочем
    размере (как их отдают to_model_mask и to_model_image).
    """
    width, height = size
    # Поза считалась на каноническом кадре 768x1024; рабочий размер другой.
    keypoints = scale_keypoints(pose.get("keypoints", {}), TARGET_SIZE, size)
    person = person_guide(height, width, keypoints, pose.get("scores", {}), group,
                          fallback_mask=mask[..., 0])

    if not isinstance(garment_parse, Image.Image):
        garment_parse = Image.fromarray(np.asarray(garment_parse, dtype=np.uint8))
    parse = np.asarray(garment_parse.resize(size, Image.NEAREST))
    garment_u8 = ((garment + 1.0) * 127.5).astype(np.uint8)
    box = garment_box(parse, group, image=garment_u8)
    return person, garment_guide(height, width, box, group)


def garment_vector(vector: np.ndarray) -> np.ndarray:
    """Вектор стадии embed -> (768,) float32; пустой вектор -> нули.

    Стадия embed пишет пустой вектор, когда вещь на кадре не найдена: модель
    получает нули, как при выключенном условии.
    """
    vector = np.asarray(vector)
    if vector.shape != (GARMENT_EMBED_DIM,):
        return np.zeros(GARMENT_EMBED_DIM, dtype=np.float32)
    return vector.astype(np.float32)


def pack_item(person: np.ndarray, garment: np.ndarray, mask: np.ndarray, pair_id: str,
              category_group: str = "", reliability: int = STUDIO_TOKEN,
              guides: tuple[np.ndarray, np.ndarray] | None = None,
              garment_embed: np.ndarray | None = None) -> dict:
    """Пример в том виде, в каком его ждут обучение, замер и генерация."""
    item = {
        "person": np.ascontiguousarray(person.transpose(2, 0, 1)),
        "garment": np.ascontiguousarray(garment.transpose(2, 0, 1)),
        "mask": np.ascontiguousarray(mask.transpose(2, 0, 1)),
        "pair_id": pair_id,
        # Для разбивки метрик по группам одежды при замере.
        "category_group": category_group,
        "reliability": int(reliability),
    }
    if guides is not None:
        item["guide_person"] = np.ascontiguousarray(guides[0])
        item["guide_garment"] = np.ascontiguousarray(guides[1])
        assert item["guide_person"].shape[0] == GUIDE_CHANNELS
    if garment_embed is not None:
        item["garment_embed"] = garment_vector(garment_embed)
    return item


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
        mask_stage: str = "agnostic",
        guide: bool = False,
        garment_embed: bool = False,
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        # Какую маску брать: "agnostic" — исходную, "agnostic_refined" — без
        # лица и кистей. Задаётся конфигом обучения и обязана совпадать при
        # замере, иначе модель получает не ту область, на которой училась.
        self.mask_stage = mask_stage
        self.raw_root = Path(raw_root)
        self.preproc_root = Path(preproc_root)
        self.size = (width, height)
        self.flip = flip
        self.seed = seed
        self.guide = guide
        self.garment_embed = garment_embed

    def __len__(self) -> int:
        return len(self.pairs)

    def _image(self, rel_path: str) -> np.ndarray:
        return to_model_image(load_canonical(self.raw_root / rel_path), self.size)

    def _mask(self, sku_id: str, image_id: str) -> np.ndarray:
        path = output_path(self.preproc_root, self.mask_stage, sku_id, image_id)
        return to_model_mask(Image.open(path), self.size)

    def _guides(self, row, mask: np.ndarray, garment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Каналы-подсказки обеих половин в пикселях рабочего размера (3, H, W)."""
        pose = json.loads(output_path(self.preproc_root, "pose", row["sku_id"],
                                      row["person_image_id"]).read_text(encoding="utf-8"))
        parse = Image.open(output_path(self.preproc_root, "parse", row["sku_id"],
                                       row["garment_image_id"]))
        return make_guides(pose, parse, mask, garment,
                           str(row.get("category_group", "upper")), self.size)

    def _embed(self, sku_id: str, image_id: str) -> np.ndarray:
        return garment_vector(np.load(output_path(self.preproc_root, "embed", sku_id, image_id)))

    def __getitem__(self, index: int) -> dict:
        row = self.pairs.iloc[index]
        person = self._image(row["person_rel_path"])
        garment = self._image(row["garment_rel_path"])
        mask = self._mask(row["sku_id"], row["person_image_id"])
        guides = self._guides(row, mask, garment) if self.guide else None

        if self.flip:
            # Отражается вся пара целиком: перевернуть вещь отдельно от человека
            # значило бы учить модель на несуществующем соответствии.
            rng = np.random.default_rng(self.seed + index)
            if rng.random() < 0.5:
                person, garment, mask = person[:, ::-1], garment[:, ::-1], mask[:, ::-1]
                if guides is not None:
                    # Подсказки отражаются вместе с кадром. Знак u менять не нужно:
                    # обе половины отражены одинаково, соответствие сохраняется.
                    guides = (guides[0][:, :, ::-1], guides[1][:, :, ::-1])

        return pack_item(
            person, garment, mask, row["pair_id"],
            category_group=str(row.get("category_group", "")),
            reliability=int(row["reliability"]) if "reliability" in row else STUDIO_TOKEN,
            guides=guides,
            garment_embed=(self._embed(row["sku_id"], row["garment_image_id"])
                           if self.garment_embed else None),
        )
