#!/usr/bin/env python
"""Генерация примерки на тестовом наборе и метрики.

    python scripts/evaluate.py --run runs/studio_wild

Замер парный: для каждой пары известно, как эта вещь выглядит на этом человеке
на самом деле, поэтому считаются и попиксельные метрики (SSIM), и перцептивная
(LPIPS), и распределенческая (FID).

Тестовый набор — только вручную подтверждённые пары (cache/test_set.parquet):
автоматически отобранные содержат около 20% неверных расцветок, и метрика на
них мерила бы шум фильтра, а не качество модели.

Стартовый шум для каждой пары фиксирован и одинаков во всех прогонах: разница
между моделями не должна смешиваться с разницей в случайном шуме. Метрики
пишутся по каждой паре в metrics_pairs.csv — по ним compare_runs.py считает
парную значимость.

    python scripts/evaluate.py --run runs/zero_shot --checkpoint none   # без обучения

Архитектура берётся из снимка train_config.yaml прогона: расширенный conv_in и
глобальное условие якорной схемы должны быть собраны до загрузки весов.
Для якорной схемы на выводе подаётся токен «студия»; --reliability-guidance
включает второй проход с токеном «неизвестно» и шаг от него к «студии».
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.arch import ARCH_NAMES, arch_flags, uses_condition  # noqa: E402
from posefit.dataset import VTONPairs  # noqa: E402
from posefit.model import (  # noqa: E402
    BACKBONE, build_guide, build_inputs, build_unet, decode, empty_conditioning, encode,
    load_component, pack_condition, take_person_half, unet_input,
)
from posefit.evaluation import mask_bbox, masked_l1, pair_seed  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.reliability import BUCKETS, NULL_TOKEN  # noqa: E402


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Из [-1,1] в целые [0,255], как ждут метрики."""
    return ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)


def initial_noise(shape, pair_ids, base_seed: int, device, dtype) -> torch.Tensor:
    """Стартовый шум, одинаковый для одной пары во всех прогонах.

    Генерируется на CPU: генераторы CUDA дают разные последовательности на
    разном железе, и повторить замер на другой карте было бы нельзя.
    """
    noise = [
        torch.randn(shape[1:], generator=torch.Generator("cpu").manual_seed(pair_seed(pid, base_seed)))
        for pid in pair_ids
    ]
    return torch.stack(noise).to(device, dtype)


@torch.no_grad()
def generate(unet, vae, scheduler, batch, device, dtype, vae_dtype, steps: int,
             base_seed: int = 0, flags: dict | None = None, token: int = 0,
             guidance: float = 0.0) -> torch.Tensor:
    flags = flags or {}
    person = batch["person"].to(device, vae_dtype)
    garment = batch["garment"].to(device, vae_dtype)
    mask = batch["mask"].to(device, dtype)

    person_latents = encode(vae, person).to(dtype)
    garment_latents = encode(vae, garment).to(dtype)
    _, mask_latent, masked_latent = build_inputs(person_latents, garment_latents, mask)

    latents = initial_noise(masked_latent.shape, batch["pair_id"], base_seed, device, dtype)
    scheduler.set_timesteps(steps, device=device)
    latents = latents * scheduler.init_noise_sigma
    conditioning = empty_conditioning(latents.shape[0], device, dtype)

    # Дополнения якорной схемы; для базовой всё остаётся None.
    guide = None
    if flags.get("guide"):
        guide = build_guide(batch["guide_person"].to(device), batch["guide_garment"].to(device),
                            masked_latent.shape[-2:]).to(dtype)
    condition = null_condition = None
    if uses_condition(flags):
        ids = torch.full((latents.shape[0],), token, device=device, dtype=torch.long)
        embed = batch["garment_embed"].to(device) if flags.get("garment_embed") else None
        condition = pack_condition(ids, embed)
        if guidance > 0 and flags.get("reliability"):
            # Второй проход с токеном «неизвестно»: guidance по надёжности —
            # шаг от пары неизвестного качества к студийной, по образцу CFG.
            null_condition = pack_condition(torch.full_like(ids, NULL_TOKEN), embed)

    for t in scheduler.timesteps:
        model_input = unet_input(scheduler.scale_model_input(latents, t),
                                 mask_latent, masked_latent, guide)
        noise_pred = unet(model_input, t, encoder_hidden_states=conditioning,
                          class_labels=condition).sample
        if null_condition is not None:
            noise_null = unet(model_input, t, encoder_hidden_states=conditioning,
                              class_labels=null_condition).sample
            noise_pred = noise_null + guidance * (noise_pred - noise_null)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return decode(vae, take_person_half(latents).to(vae_dtype))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--train-config", default=Path("configs/train.yaml"), type=Path)
    ap.add_argument("--run", required=True, type=Path, help="каталог прогона, например runs/studio")
    ap.add_argument("--mask-stage", default=None, choices=["agnostic", "agnostic_refined"],
                    help="только для --checkpoint none: с какой маской мерить исходную модель")
    ap.add_argument("--checkpoint", default="final.pt",
                    help="none — исходные веса без обучения (контроль: помогло ли обучение вообще)")
    ap.add_argument("--arch", default=None, choices=list(ARCH_NAMES),
                    help="только для --checkpoint none; обученный прогон сам помнит схему")
    ap.add_argument("--reliability-token", default="studio", choices=list(BUCKETS),
                    help="токен надёжности на выводе (якорная схема); по умолчанию «студия»")
    ap.add_argument("--reliability-guidance", type=float, default=0.0,
                    help="сила guidance по надёжности: 0 — выключено (один проход), "
                         "1 — то же, что без guidance, больше 1 — шаг от «неизвестно» к токену")
    ap.add_argument("--seed", type=int, default=0, help="база сидов стартового шума")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    # Замер не обучает ничего. Глобальный выключатель — третий, независимый
    # рубеж: даже если декоратор с generate снова уедет при правке, граф
    # градиентов не построится.
    torch.set_grad_enabled(False)

    hp = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    # Прогон помнит, на чём учился. Расхождение с текущим конфигом — ошибка,
    # а не повод молча взять один из вариантов: маска и разрешение при замере
    # обязаны совпадать с обучением.
    if args.mask_stage:
        if args.checkpoint != "none":
            raise SystemExit("--mask-stage допустим только с --checkpoint none: "
                             "обученный прогон сам помнит свою маску")
        hp["mask_stage"] = args.mask_stage
    if args.arch:
        if args.checkpoint != "none":
            raise SystemExit("--arch допустим только с --checkpoint none: "
                             "обученный прогон сам помнит свою схему")
        hp["arch"] = {**(hp.get("arch") or {}), "name": args.arch}
    snapshot = args.run / "train_config.yaml"
    if snapshot.exists():
        trained = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        for key in ("mask_stage", "height", "width"):
            if trained.get(key, hp.get(key)) != hp.get(key):
                raise SystemExit(
                    f"{key}: прогон учился с {trained.get(key)!r}, а конфиг задаёт {hp.get(key)!r}. "
                    f"Замер с другой маской или разрешением бессмыслен.")
        # Схема берётся из снимка целиком: чекпоинт содержит веса ровно тех
        # модулей, которые были собраны при обучении.
        hp["arch"] = trained.get("arch") or {"name": "catvton"}
    elif args.checkpoint != "none":
        print("ВНИМАНИЕ: в прогоне нет train_config.yaml (старый прогон) — "
              "считаю, что учился на mask_stage=agnostic по базовой схеме")
        hp["mask_stage"] = "agnostic"
        hp["arch"] = {"name": "catvton"}
    flags = arch_flags(hp)
    token = BUCKETS.index(args.reliability_token)
    print(f"архитектура: {flags['name']}"
          + (f"   токен «{args.reliability_token}»   guidance {args.reliability_guidance}"
             if uses_condition(flags) else ""))
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
                        height=hp["height"], width=hp["width"], flip=False,
                        mask_stage=hp.get("mask_stage", "agnostic"),
                        guide=flags["guide"], garment_embed=flags["garment_embed"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=hp.get("workers", 4))

    from diffusers import AutoencoderKL, DDIMScheduler
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image import (
        FrechetInceptionDistance, KernelInceptionDistance,
        LearnedPerceptualImagePatchSimilarity,
    )

    vae_dtype = torch.float16 if hp.get("vae_fp16", False) else torch.float32
    vae = load_component(AutoencoderKL, "vae").to(device, vae_dtype).eval()
    unet = build_unet(flags).to(device, dtype).eval()
    # Замер не обучает: замораживается всё. freeze_except_self_attention здесь
    # была ошибкой — она размораживает слои внимания, и 50 шагов диффузии
    # строили граф градиентов через все шаги сразу: 23 ГБ на первом батче.
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    if args.checkpoint == "none":
        # Контроль: исходный SD inpainting, склейку «человек | вещь» не видевший.
        # Если обученные модели его не обгонят — обучение ничего не дало.
        step = 0
        print("веса исходные, без обучения")
    else:
        state = torch.load(args.run / args.checkpoint, map_location=device)
        unet.load_state_dict({k: v.to(dtype) for k, v in state["unet"].items()}, strict=False)
        step = int(state["step"])
        print(f"веса с шага {step} из {args.run / args.checkpoint}")
    args.run.mkdir(parents=True, exist_ok=True)
    scheduler = DDIMScheduler.from_pretrained(BACKBONE, subfolder="scheduler")

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    # FID на нескольких сотнях картинок смещён и шумен; KID несмещён и для
    # малых выборок надёжнее. Считаются оба, в выводах опираться на KID.
    kid = KernelInceptionDistance(subset_size=min(100, len(dataset)), normalize=True).to(device)
    rows: list[dict] = []

    out_dir = args.run / "preds"
    if args.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="генерация", unit="batch"):
        predicted = generate(unet, vae, scheduler, batch, device, dtype, vae_dtype,
                             args.steps, base_seed=args.seed, flags=flags, token=token,
                             guidance=args.reliability_guidance)
        truth = batch["person"].to(device, predicted.dtype)

        pred_u8, true_u8 = to_uint8(predicted), to_uint8(truth)
        pred01, true01 = pred_u8.float() / 255.0, true_u8.float() / 255.0
        fid.update(true01, real=True)
        fid.update(pred01, real=False)
        kid.update(true01, real=True)
        kid.update(pred01, real=False)

        ssim_each = structural_similarity_index_measure(
            pred_u8.float(), true_u8.float(), data_range=255.0, reduction="none")
        masks = batch["mask"].cpu().numpy()[:, 0]
        pred_np = pred_u8.cpu().numpy().transpose(0, 2, 3, 1)
        true_np = true_u8.cpu().numpy().transpose(0, 2, 3, 1)

        for i, pair_id in enumerate(batch["pair_id"]):
            row = {
                "pair_id": pair_id,
                "category_group": batch["category_group"][i],
                "ssim": float(ssim_each[i]),
                "lpips": float(lpips(pred01[i:i + 1], true01[i:i + 1])),
                "l1_mask": masked_l1(pred_np[i], true_np[i], masks[i]),
                "lpips_mask": float("nan"),
            }
            # Вне маски модель копирует вход, и метрика по целому кадру на
            # четыре пятых мерит совпадения, от модели не зависящие.
            box = mask_bbox(masks[i])
            if box is not None and (box[1] - box[0]) >= 32 and (box[3] - box[2]) >= 32:
                y0, y1, x0, x1 = box
                row["lpips_mask"] = float(lpips(pred01[i:i + 1, :, y0:y1, x0:x1],
                                                true01[i:i + 1, :, y0:y1, x0:x1]))
            rows.append(row)

        if args.save_images:
            for pair_id, image in zip(batch["pair_id"], pred_u8.cpu().numpy()):
                Image.fromarray(image.transpose(1, 2, 0)).save(out_dir / f"{pair_id}.jpg", quality=95)

    per_pair = pd.DataFrame(rows)
    per_pair.to_csv(args.run / "metrics_pairs.csv", index=False)
    kid_mean, kid_std = kid.compute()
    metrics = {
        "run": str(args.run), "step": step, "pairs": len(per_pair),
        "steps": args.steps, "seed": args.seed, "arch": flags["name"],
        "reliability_token": args.reliability_token if uses_condition(flags) else None,
        "reliability_guidance": args.reliability_guidance if flags["reliability"] else None,
        "ssim": float(per_pair.ssim.mean()),
        "lpips": float(per_pair.lpips.mean()),
        "l1_mask": float(per_pair.l1_mask.mean()),
        "lpips_mask": float(per_pair.lpips_mask.mean()),
        "kid": float(kid_mean), "kid_std": float(kid_std),
        "fid": float(fid.compute()),
    }
    (args.run / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
