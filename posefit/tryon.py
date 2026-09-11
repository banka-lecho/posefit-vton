"""Примерка на своих фотографиях: одна пара «человек — вещь» от файла до результата.

Путь обязан совпадать с обучением, иначе модель получит вход, которого не
видела. Поэтому здесь нет своих преобразований:

* кадр приводится к канону той же load_canonical, что и в препроцессинге;
* детекция, parsing, поза и эмбеддинг считаются теми же моделями и с теми же
  настройками, что в scripts/preprocess.py (функции ниже повторяют его стадии
  для одной картинки);
* маска, ресайз, карты координат и упаковка примера — функции posefit.dataset,
  через которые идёт и обучающий датасет;
* генерация — posefit.generation, общая с замером.

Отличие от обучающих данных одно: фото с телефона несут поворот в EXIF, и он
применяется до всего остального. В выгрузке WB поворотов нет.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .canon import torso_frame
from .dataset import make_guides, pack_item, to_model_image, to_model_mask
from .preprocess import (
    AGNOSTIC_LABELS, ATR, GARMENT_LABELS, MODELS, TARGET_SIZE, build_agnostic, garment_crop,
    load_canonical, refine_agnostic,
)

GROUPS = ("upper", "lower", "dress", "outer")

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# Классы ATR, которые вообще бывают одеждой, и порог «вещи группы подозрительно
# мало»: доля её пикселей в рамке человека. У верха в кадре по пояс это обычно
# 20–40%, у брюк в полный рост — 15–25%.
CLOTHING = ("upper_clothes", "skirt", "pants", "dress")
MIN_GARMENT_SHARE = 0.08


def clothing_warning(parse: np.ndarray, group: str, box: list[float]) -> str | None:
    """Предупреждение, если разметка почти не нашла одежду группы на человеке.

    Частый случай — однотонный образ, который SegFormer целиком относит к
    «платью»: маска группы upper этот класс не закрывает, и перерисовываются
    только рукава. Код тут ни при чём, так размечены и обучающие кадры, но
    результат будет неожиданным, и об этом нужно сказать.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    region = parse[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if not region.size:
        return None
    own = [name for name in CLOTHING if name in AGNOSTIC_LABELS[group]]
    share = np.isin(region, [ATR[n] for n in own]).mean()
    if share >= MIN_GARMENT_SHARE:
        return None
    others = {n: float((region == ATR[n]).mean()) for n in CLOTHING if n not in own}
    dominant = max(others, key=others.get) if others else None
    hint = (f"; разметка отнесла одежду к классу {dominant} ({others[dominant]:.0%} фигуры) — "
            f"попробуй другую группу или другое фото"
            if dominant and others[dominant] > share else "")
    return (f"одежды группы {group!r} на человеке всего {share:.0%} фигуры — "
            f"маска закроет лишь её часть{hint}")


def open_image(source) -> Image.Image:
    """Путь или PIL -> RGB с применённым поворотом из EXIF."""
    image = source if isinstance(source, Image.Image) else Image.open(source)
    return ImageOps.exif_transpose(image).convert("RGB")


def canonical_box(original: tuple[int, int], size: tuple[int, int] = TARGET_SIZE
                  ) -> tuple[int, int, int, int]:
    """Где в каноническом кадре лежит исходное фото: (x0, y0, x1, y1).

    Та же арифметика, что в load_canonical: по ней результат обрезается обратно
    до пропорций исходного фото, без белых полей паддинга.
    """
    width, height = original
    target_w, target_h = size
    scale = min(target_w / width, target_h / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    x0, y0 = (target_w - w) // 2, (target_h - h) // 2
    return x0, y0, x0 + w, y0 + h


@dataclass
class PersonInput:
    canonical: Image.Image          # 768x1024
    original_size: tuple[int, int]
    parse: np.ndarray               # (1024, 768) uint8, ATR-18
    pose: dict                      # {"box", "keypoints", "scores"} в пикселях канона
    n_persons: int
    # Область примерки (1024, 768) uint8 0/255 для обеих масок обучения:
    # "agnostic" и "agnostic_refined" (без лица и кистей). Какую брать, решает
    # снимок прогона — у двух сравниваемых моделей они могут различаться.
    masks: dict[str, np.ndarray]
    warnings: list[str] = field(default_factory=list)


@dataclass
class GarmentInput:
    canonical: Image.Image
    original_size: tuple[int, int]
    parse: np.ndarray
    embed: np.ndarray               # (768,) или пустой, если вещь не найдена
    warnings: list[str] = field(default_factory=list)


class Preprocessor:
    """Модели препроцессинга, загружаемые по первому требованию."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._models: dict = {}

    def _get(self, name: str):
        if name in self._models:
            return self._models[name]
        from transformers import (
            AutoImageProcessor, AutoModel, AutoModelForObjectDetection, AutoProcessor,
            SegformerForSemanticSegmentation, SegformerImageProcessor, VitPoseForPoseEstimation,
        )

        repo = MODELS[name]
        if name == "detect":
            pair = (AutoImageProcessor.from_pretrained(repo),
                    AutoModelForObjectDetection.from_pretrained(repo))
        elif name == "parse":
            pair = (SegformerImageProcessor.from_pretrained(repo),
                    SegformerForSemanticSegmentation.from_pretrained(repo))
        elif name == "pose":
            pair = (AutoProcessor.from_pretrained(repo), VitPoseForPoseEstimation.from_pretrained(repo))
        elif name == "embed":
            pair = (AutoImageProcessor.from_pretrained(repo), AutoModel.from_pretrained(repo))
        else:
            raise KeyError(name)
        pair = (pair[0], pair[1].to(self.device).eval())
        self._models[name] = pair
        return pair

    # Стадии ниже повторяют scripts/preprocess.py для одной картинки.

    def detect(self, image: Image.Image) -> tuple[list, list]:
        import torch

        proc, model = self._get("detect")
        inputs = proc(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model(**inputs)
        sizes = torch.tensor([TARGET_SIZE[::-1]]).to(self.device)
        res = proc.post_process_object_detection(out, threshold=0.5, target_sizes=sizes)[0]
        keep = [i for i, label in enumerate(res["labels"].tolist())
                if model.config.id2label[label].lower() == "person"]
        return res["boxes"][keep].cpu().tolist(), res["scores"][keep].cpu().tolist()

    def parse(self, image: Image.Image) -> np.ndarray:
        import torch

        proc, model = self._get("parse")
        inputs = proc(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=TARGET_SIZE[::-1], mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def pose(self, image: Image.Image, box: list[float]) -> dict:
        import torch

        proc, model = self._get("pose")
        xyxy = np.array([[box[0], box[1], box[2] - box[0], box[3] - box[1]]])
        inputs = proc(image, boxes=[xyxy], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = proc.post_process_pose_estimation(outputs, boxes=[xyxy])[0][0]
        points = result["keypoints"].cpu().tolist()
        confidences = result["scores"].cpu().tolist()
        return {
            "box": box,
            "keypoints": {n: points[i] for i, n in enumerate(COCO_KEYPOINTS) if i < len(points)},
            "scores": {n: confidences[i] for i, n in enumerate(COCO_KEYPOINTS) if i < len(confidences)},
        }

    def embed(self, image: Image.Image, parse: np.ndarray, group: str) -> np.ndarray:
        import torch

        crop = garment_crop(np.asarray(image), parse, group)
        if crop is None:
            return np.zeros(0, dtype=np.float32)
        proc, model = self._get("embed")
        inputs = proc(images=[Image.fromarray(crop)], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state[:, 0]  # CLS-токен
        return torch.nn.functional.normalize(out, dim=-1)[0].cpu().numpy().astype(np.float32)

    # Сборка входов.

    def person(self, source, group: str, dilate: int = 12) -> PersonInput:
        if group not in GROUPS:
            raise ValueError(f"группа одежды {group!r}, ожидается одна из {GROUPS}")
        original = open_image(source)
        canonical = load_canonical(original)
        warnings: list[str] = []

        boxes, _ = self.detect(canonical)
        if not boxes:
            warnings.append("человек не найден детектором — поза считается по всему кадру")
            box = [0.0, 0.0, float(TARGET_SIZE[0]), float(TARGET_SIZE[1])]
        else:
            if len(boxes) > 1:
                warnings.append(f"людей в кадре: {len(boxes)} — берётся самый крупный")
            # Самый крупный человек, как в стадии pose.
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        pose = self.pose(canonical, box)
        if torso_frame(pose["keypoints"], pose["scores"], TARGET_SIZE) is None:
            warnings.append("торс по позе не виден — координаты строятся по маске, "
                            "примерка будет хуже")

        parse = self.parse(canonical)
        warning = clothing_warning(parse, group, box)
        if warning:
            warnings.append(warning)
        mask = build_agnostic(parse, group, dilate_px=dilate)
        if not mask.any():
            raise ValueError(f"на фото человека не найдено одежды группы {group!r}: "
                             "маска пуста — проверь группу или выбери другое фото")
        masks = {"agnostic": mask,
                 "agnostic_refined": refine_agnostic(mask, parse, pose["keypoints"], pose["scores"])}
        return PersonInput(canonical, original.size, parse, pose, len(boxes), masks, warnings)

    def garment(self, source, group: str, with_embed: bool = True) -> GarmentInput:
        if group not in GROUPS:
            raise ValueError(f"группа одежды {group!r}, ожидается одна из {GROUPS}")
        original = open_image(source)
        canonical = load_canonical(original)
        parse = self.parse(canonical)
        warnings: list[str] = []
        labels = [ATR[name] for name in GARMENT_LABELS[group]]
        if np.isin(parse, labels).sum() < 500:
            warnings.append("parsing не распознал вещь на выкладке — рамка вещи берётся "
                            "по не-белым пикселям; лучше фото на светлом однотонном фоне")
        embed = self.embed(canonical, parse, group) if with_embed else np.zeros(0, np.float32)
        return GarmentInput(canonical, original.size, parse, embed, warnings)


def build_item(person: PersonInput, garment: GarmentInput, group: str, pair_id: str,
               size: tuple[int, int] = (384, 512), flags: dict | None = None,
               mask_stage: str = "agnostic", reliability: int = 0) -> dict:
    """Пример в формате датасета — через те же функции, что и при обучении."""
    flags = flags or {}
    if mask_stage not in person.masks:
        raise ValueError(f"неизвестная маска {mask_stage!r}, есть {sorted(person.masks)}")
    person_img = to_model_image(person.canonical, size)
    garment_img = to_model_image(garment.canonical, size)
    mask = to_model_mask(person.masks[mask_stage], size)
    guides = (make_guides(person.pose, garment.parse, mask, garment_img, group, size)
              if flags.get("guide") else None)
    return pack_item(person_img, garment_img, mask, pair_id, category_group=group,
                     reliability=reliability, guides=guides,
                     garment_embed=garment.embed if flags.get("garment_embed") else None)


def collate(items: list[dict]):
    """Список примеров -> батч, как его собрал бы DataLoader."""
    from torch.utils.data import default_collate

    return default_collate(items)


def composite(result: np.ndarray, item: dict) -> np.ndarray:
    """Результат внутри маски, исходное фото снаружи — (H, W, 3) uint8.

    VAE слегка меняет весь кадр, включая лицо и фон. Метрики считаются по
    сырому результату; для показа честнее вернуть исходные пиксели вне маски.
    """
    person = ((item["person"].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255)
    mask = item["mask"].transpose(1, 2, 0)
    return (result * mask + person * (1.0 - mask)).round().astype(np.uint8)


def uncrop(image: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
    """Убирает белые поля паддинга: результат в пропорциях исходного фото."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = canonical_box(original_size)
    sx, sy = width / TARGET_SIZE[0], height / TARGET_SIZE[1]
    box = (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))
    return Image.fromarray(image).crop(box)


class TryOn:
    """Модель для примерки, собранная так же, как при замере.

    run + checkpoint — свой обученный прогон (схема из его снимка);
    checkpoint="none" — исходный SD inpainting без обучения;
    official="dresscode" | "vitonhd" | "mix" — опубликованный CatVTON с весами
    авторов и их настройками вывода (posefit/official.py).
    """

    def __init__(self, run: str | Path | None = None, checkpoint: str = "final.pt",
                 device: str = "cuda", train_config: str | Path = "configs/train.yaml",
                 arch: str | None = None, mask_stage: str | None = None,
                 official: str | None = None) -> None:
        import torch
        import yaml

        from .arch import arch_flags
        from .evaluation import resolve_eval_config
        from .generation import load_models
        from .official import VARIANTS, load_official

        hp = yaml.safe_load(Path(train_config).read_text(encoding="utf-8"))
        self.device = device
        self.official = official
        if official:
            self.hp, _ = resolve_eval_config(hp, Path("runs/official"), "none", mask_stage)
            self.hp["arch"] = {"name": "catvton"}
            self.source = f"опубликованный CatVTON, веса {VARIANTS[official]}"
            self.flags = arch_flags(self.hp)
            self.name, self.step = f"CatVTON {official}", 0
            self.dtype = torch.float16 if self.hp.get("fp16", True) else torch.float32
            self.vae_dtype = torch.float16 if self.hp.get("vae_fp16", False) else torch.float32
            self.unet, self.vae, self.scheduler = load_official(official, device, self.dtype,
                                                                self.vae_dtype)
            return
        if run is None and checkpoint != "none":
            raise ValueError("для обученной модели нужен каталог прогона run")
        run = Path(run) if run is not None else Path("runs/zero_shot")
        self.hp, self.source = resolve_eval_config(hp, run, checkpoint, mask_stage, arch)
        self.flags = arch_flags(self.hp)
        self.name = run.name if checkpoint != "none" else f"{run.name} (без обучения)"
        (self.unet, self.vae, self.scheduler, self.step,
         self.dtype, self.vae_dtype) = load_models(self.hp, self.flags, run, checkpoint, device)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.hp["width"]), int(self.hp["height"])

    @property
    def mask_stage(self) -> str:
        return self.hp.get("mask_stage", "agnostic")

    def item(self, person: PersonInput, garment: GarmentInput, group: str, pair_id: str) -> dict:
        """Пример для этой модели: её разрешение, её маска, её подсказки."""
        return build_item(person, garment, group, pair_id, size=self.size, flags=self.flags,
                          mask_stage=self.mask_stage)

    def __call__(self, items: list[dict], steps: int = 50, seed: int = 0,
                 token: str = "studio", guidance: float = 0.0) -> list[np.ndarray]:
        """Результаты примерки, (H, W, 3) uint8 на каждый пример."""
        from .generation import generate, to_uint8
        from .official import generate_official
        from .reliability import BUCKETS

        batch = collate(items)
        if self.official:
            # Их настройки вывода: guidance 2.5 и eta 1, как в замере авторов.
            out = generate_official(self.unet, self.vae, self.scheduler, batch, self.device,
                                    self.dtype, self.vae_dtype, steps, base_seed=seed)
        else:
            out = generate(self.unet, self.vae, self.scheduler, batch, self.device, self.dtype,
                           self.vae_dtype, steps, base_seed=seed, flags=self.flags,
                           token=BUCKETS.index(token), guidance=guidance)
        return list(to_uint8(out).cpu().numpy().transpose(0, 2, 3, 1))
