#!/usr/bin/env python
"""GPU-этап Ф0: детекция, поза, human parsing, agnostic-маски, VAE-латенты.

Стадии запускаются по одной и возобновляемы: уже посчитанные файлы
пропускаются. Для нескольких карт или нескольких заходов есть --shard.

    python scripts/preprocess.py --stage parse
    python scripts/preprocess.py --stage pose --shard 0/2

Импорты моделей ленивые: --help и --dry-run работают на машине без torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.preprocess import (  # noqa: E402
    MODELS, STAGES, TARGET_SIZE, build_agnostic, output_path, pending,
    working_set, write_json,
)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def load_canonical(path: Path) -> Image.Image:
    """Кадр, приведённый к 768x1024 с сохранением пропорций и паддингом.

    Пропорции у отзывов гуляют (от 450x1000 до 1000x750), и растягивание
    исказило бы силуэт — а именно его модель и учится воспроизводить.
    """
    image = Image.open(path).convert("RGB")
    target_w, target_h = TARGET_SIZE
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((max(1, round(image.width * scale)),
                            max(1, round(image.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
    canvas.paste(resized, ((target_w - resized.width) // 2,
                           (target_h - resized.height) // 2))
    return canvas


def _batches(rows: pd.DataFrame, size: int):
    for start in range(0, len(rows), size):
        yield rows.iloc[start:start + size]


def run_detect(rows, paths, args) -> None:
    import torch
    from transformers import AutoModelForObjectDetection, AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(MODELS["detect"])
    model = AutoModelForObjectDetection.from_pretrained(MODELS["detect"]).to(args.device).eval()

    for batch in tqdm(list(_batches(rows, args.batch_size)), desc="detect", unit="batch"):
        images = [load_canonical(paths.raw_root / p) for p in batch["rel_path"]]
        inputs = proc(images=images, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**inputs)
        sizes = torch.tensor([TARGET_SIZE[::-1]] * len(images)).to(args.device)
        results = proc.post_process_object_detection(out, threshold=0.5, target_sizes=sizes)

        for row, res in zip(batch.itertuples(), results):
            keep = [i for i, label in enumerate(res["labels"].tolist())
                    if model.config.id2label[label].lower() == "person"]
            boxes = res["boxes"][keep].cpu().tolist()
            scores = res["scores"][keep].cpu().tolist()
            area = TARGET_SIZE[0] * TARGET_SIZE[1]
            write_json(
                output_path(paths.preproc_root, "detect", row.sku_id, row.image_id),
                {
                    "boxes": boxes,
                    "scores": scores,
                    "n_persons": len(boxes),
                    # Доля кадра, занятая крупнейшим человеком: по ней
                    # отсекаются групповые фото и дальние планы.
                    "max_area_frac": max(
                        [((b[2] - b[0]) * (b[3] - b[1])) / area for b in boxes], default=0.0
                    ),
                },
            )


def run_parse(rows, paths, args) -> None:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    proc = SegformerImageProcessor.from_pretrained(MODELS["parse"])
    model = SegformerForSemanticSegmentation.from_pretrained(MODELS["parse"]).to(args.device).eval()

    for batch in tqdm(list(_batches(rows, args.batch_size)), desc="parse", unit="batch"):
        images = [load_canonical(paths.raw_root / p) for p in batch["rel_path"]]
        inputs = proc(images=images, return_tensors="pt").to(args.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=TARGET_SIZE[::-1], mode="bilinear", align_corners=False
        )
        parsed = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for row, seg in zip(batch.itertuples(), parsed):
            target = output_path(paths.preproc_root, "parse", row.sku_id, row.image_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg).save(target, optimize=True)


def run_pose(rows, paths, args) -> None:
    import json

    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    proc = AutoProcessor.from_pretrained(MODELS["pose"])
    model = VitPoseForPoseEstimation.from_pretrained(MODELS["pose"]).to(args.device).eval()

    for row in tqdm(list(rows.itertuples()), desc="pose", unit="img"):
        det_path = output_path(paths.preproc_root, "detect", row.sku_id, row.image_id)
        if not det_path.exists():
            continue
        boxes = json.loads(det_path.read_text(encoding="utf-8"))["boxes"]
        target = output_path(paths.preproc_root, "pose", row.sku_id, row.image_id)
        if not boxes:
            write_json(target, {"keypoints": {}, "scores": {}, "box": None})
            continue

        # Самый крупный человек: в зеркальных селфи в кадр часто попадают
        # люди на заднем плане и отражения.
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        image = load_canonical(paths.raw_root / row.rel_path)
        xyxy = np.array([[box[0], box[1], box[2] - box[0], box[3] - box[1]]])
        inputs = proc(image, boxes=[xyxy], return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = proc.post_process_pose_estimation(outputs, boxes=[xyxy])[0][0]

        points = result["keypoints"].cpu().tolist()
        confidences = result["scores"].cpu().tolist()
        write_json(target, {
            "box": box,
            "keypoints": {n: points[i] for i, n in enumerate(COCO_KEYPOINTS) if i < len(points)},
            "scores": {n: confidences[i] for i, n in enumerate(COCO_KEYPOINTS) if i < len(confidences)},
        })


def run_agnostic(rows, paths, args) -> None:
    for row in tqdm(list(rows.itertuples()), desc="agnostic", unit="img"):
        parse_path = output_path(paths.preproc_root, "parse", row.sku_id, row.image_id)
        if not parse_path.exists():
            continue
        parse = np.array(Image.open(parse_path))
        mask = build_agnostic(parse, row.category_group, dilate_px=args.dilate)
        target = output_path(paths.preproc_root, "agnostic", row.sku_id, row.image_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(target, optimize=True)


def run_latents(rows, paths, args) -> None:
    import torch
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(MODELS["vae"]).to(args.device).eval()
    if args.fp16:
        vae = vae.half()

    for batch in tqdm(list(_batches(rows, args.batch_size)), desc="latents", unit="batch"):
        images = [load_canonical(paths.raw_root / p) for p in batch["rel_path"]]
        array = np.stack([np.asarray(im, dtype=np.float32) / 127.5 - 1.0 for im in images])
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(args.device)
        if args.fp16:
            tensor = tensor.half()
        with torch.no_grad():
            latents = vae.encode(tensor).latent_dist.mean * vae.config.scaling_factor

        for row, latent in zip(batch.itertuples(), latents.cpu().numpy().astype(np.float16)):
            target = output_path(paths.preproc_root, "latents", row.sku_id, row.image_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, latent)


RUNNERS = {
    "detect": run_detect, "parse": run_parse, "pose": run_pose,
    "agnostic": run_agnostic, "latents": run_latents,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--shard", default="0/1", help="i/N — часть работы для этого процесса")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dilate", type=int, default=12, help="запас маски agnostic, пикселей")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="показать объём работы и выйти")
    args = ap.parse_args()

    shard, n_shards = (int(x) for x in args.shard.split("/"))
    config = load_config(args.config)
    paths = load_paths(config)

    try:
        root = paths.preproc_root
    except RuntimeError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)

    images = pd.read_parquet(paths.manifest_images)
    sku = pd.read_parquet(paths.manifest_sku)
    rows = working_set(images, sku)

    for required in STAGES[args.stage].needs:
        print(f"стадия {args.stage} использует результаты стадии {required}")

    todo = pending(rows, root, args.stage, shard, n_shards, args.overwrite)
    if args.limit:
        todo = todo.head(args.limit)

    print(f"пригодных кадров: {len(rows)}   шард {shard}/{n_shards}   к обработке: {len(todo)}")
    if args.dry_run or todo.empty:
        return 0

    RUNNERS[args.stage](todo, paths, args)
    print(f"готово: {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
