#!/usr/bin/env python
"""Генерация примерки на тестовом наборе и метрики.

    python scripts/evaluate.py --run runs/studio_wild

Замер парный: для каждой пары известно, как эта вещь выглядит на этом человеке
на самом деле, поэтому считаются и попиксельные метрики (SSIM), и перцептивная
(LPIPS), и распределенческая (FID).

Тестовый набор — только вручную подтверждённые пары (cache/test_set.parquet):
автоматически отобранные содержат около 20% неверных расцветок, и метрика на
них мерила бы шум фильтра, а не качество модели.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.dataset import VTONPairs  # noqa: E402
from posefit.model import (  # noqa: E402
    BACKBONE, build_inputs, decode, empty_conditioning, encode,
    freeze_except_self_attention, take_person_half, unet_input,
)
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Из [-1,1] в целые [0,255], как ждут метрики."""
    return ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def generate(unet, vae, scheduler, batch, device, dtype, vae_dtype, steps: int) -> torch.Tensor:
    person = batch["person"].to(device, vae_dtype)
    garment = batch["garment"].to(device, vae_dtype)
    mask = batch["mask"].to(device, dtype)

    person_latents = encode(vae, person).to(dtype)
    garment_latents = encode(vae, garment).to(dtype)
    _, mask_latent, masked_latent = build_inputs(person_latents, garment_latents, mask)

    latents = torch.randn_like(masked_latent)
    scheduler.set_timesteps(steps, device=device)
    latents = latents * scheduler.init_noise_sigma
    conditioning = empty_conditioning(latents.shape[0], device, dtype)

    for t in scheduler.timesteps:
        model_input = scheduler.scale_model_input(latents, t)
        noise_pred = unet(
            unet_input(model_input, mask_latent, masked_latent), t,
            encoder_hidden_states=conditioning,
        ).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return decode(vae, take_person_half(latents).to(vae_dtype))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--train-config", default=Path("configs/train.yaml"), type=Path)
    ap.add_argument("--run", required=True, type=Path, help="каталог прогона, например runs/studio")
    ap.add_argument("--checkpoint", default="final.pt")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    hp = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    paths = load_paths(load_config(args.config))
    device = args.device or hp.get("device", "cuda")
    dtype = torch.float16 if hp.get("fp16", True) else torch.float32

    test = pd.read_parquet(paths.cache_root / "test_set.parquet")
    if args.limit:
        test = test.head(args.limit)
    # Разбиение уже гарантирует, что тестовых вещей нет в обучении, но проверка
    # стоит миллисекунды, а цена незамеченной утечки — вся работа.
    train_skus = set(pd.read_parquet(paths.cache_root / "manifest_pairs.parquet")
                     .query("split != 'test'")["sku_id"])
    leaked = set(test["sku_id"]) & train_skus
    if leaked:
        raise SystemExit(f"тестовые вещи встречаются в обучении: {sorted(leaked)[:5]}")

    dataset = VTONPairs(test, paths.raw_root, paths.preproc_root,
                        height=hp["height"], width=hp["width"], flip=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=hp.get("workers", 4))

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from torchmetrics.image import (
        FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity,
        StructuralSimilarityIndexMeasure,
    )

    vae_dtype = torch.float16 if hp.get("vae_fp16", False) else torch.float32
    vae = AutoencoderKL.from_pretrained(BACKBONE, subfolder="vae").to(device, vae_dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(BACKBONE, subfolder="unet").to(device, dtype).eval()
    freeze_except_self_attention(unet)
    state = torch.load(args.run / args.checkpoint, map_location=device)
    unet.load_state_dict({k: v.to(dtype) for k, v in state["unet"].items()}, strict=False)
    print(f"веса с шага {state['step']} из {args.run / args.checkpoint}")
    scheduler = DDIMScheduler.from_pretrained(BACKBONE, subfolder="scheduler")

    ssim = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    out_dir = args.run / "preds"
    if args.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="генерация", unit="batch"):
        predicted = generate(unet, vae, scheduler, batch, device, dtype, vae_dtype, args.steps)
        truth = batch["person"].to(device, predicted.dtype)

        pred_u8, true_u8 = to_uint8(predicted), to_uint8(truth)
        ssim.update(pred_u8.float(), true_u8.float())
        # LPIPS и FID ждут [0,1] при normalize=True.
        lpips.update(pred_u8.float() / 255.0, true_u8.float() / 255.0)
        fid.update(true_u8.float() / 255.0, real=True)
        fid.update(pred_u8.float() / 255.0, real=False)

        if args.save_images:
            for pair_id, image in zip(batch["pair_id"], pred_u8.cpu().numpy()):
                Image.fromarray(image.transpose(1, 2, 0)).save(out_dir / f"{pair_id}.jpg", quality=95)

    metrics = {
        "run": str(args.run), "step": int(state["step"]), "pairs": len(dataset),
        "steps": args.steps,
        "ssim": float(ssim.compute()), "lpips": float(lpips.compute()), "fid": float(fid.compute()),
    }
    (args.run / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
