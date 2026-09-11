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
    python scripts/evaluate.py --run runs/catvton_dresscode --official dresscode

--official — опубликованный CatVTON с весами авторов (posefit/official.py):
существующая модель, с которой сравниваются обученные. --run тогда задаёт
только каталог для метрик.

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
from posefit.generation import generate, load_models, to_uint8  # noqa: E402
from posefit.official import (  # noqa: E402
    DEFAULT_ETA, DEFAULT_GUIDANCE, VARIANTS, generate_official, load_official,
)
from posefit.evaluation import (  # noqa: E402
    eval_output_dir, mask_bbox, masked_l1, resolve_eval_config,
)
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.reliability import BUCKETS  # noqa: E402


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
    ap.add_argument("--official", default=None, choices=list(VARIANTS),
                    help="опубликованный CatVTON с весами авторов вместо своего прогона: "
                         "dresscode — верх, низ, платья в 512x384 (основной вариант), "
                         "vitonhd — только верх, mix — обучен в 1024x768")
    ap.add_argument("--official-guidance", type=float, default=DEFAULT_GUIDANCE,
                    help="classifier-free guidance CatVTON; 2.5 — как в их замере")
    ap.add_argument("--eta", type=float, default=DEFAULT_ETA,
                    help="стохастичность DDIM у CatVTON; 1.0 — как в их замере")
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
    # VAE кодирует выборкой из апостериорного распределения; без сида повторный
    # замер отличался бы в последних знаках.
    torch.manual_seed(args.seed)

    hp = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    if args.official:
        # Своего прогона нет: маска и разрешение — из конфига или --mask-stage,
        # как у контроля без обучения, схема — базовая, без подсказок.
        if args.arch:
            raise SystemExit("--arch и --official несовместимы: у CatVTON своя схема")
        args.checkpoint = "none"
    try:
        hp, source = resolve_eval_config(hp, args.run, args.checkpoint,
                                         args.mask_stage, args.arch)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.official:
        hp["arch"] = {"name": "catvton"}
        source = f"опубликованный CatVTON, веса {VARIANTS[args.official]}"
        if args.official == "mix" and (hp["height"], hp["width"]) != (1024, 768):
            print(f"ВНИМАНИЕ: mix обучен в 1024x768, замер идёт в {hp['width']}x{hp['height']}")
    print(f"параметры обучения: {source}")
    print(f"маска: {hp.get('mask_stage', 'agnostic')}   разрешение {hp['width']}x{hp['height']}")
    flags = arch_flags(hp)
    if not uses_condition(flags):
        # Базовой схеме токен и guidance подать некуда: замер под другим
        # именем был бы копией обычного.
        args.reliability_token, args.reliability_guidance = "studio", 0.0
    elif not flags["reliability"]:
        args.reliability_guidance = 0.0
    token = BUCKETS.index(args.reliability_token)
    out_run = eval_output_dir(args.run, args.reliability_token, args.reliability_guidance)
    print(f"архитектура: {flags['name']}"
          + (f"   токен «{args.reliability_token}»   guidance {args.reliability_guidance}"
             if uses_condition(flags) else ""))
    if out_run != args.run:
        print(f"метрики -> {out_run} (вариант вывода, каталог прогона не трогаю)")
    paths = load_paths(load_config(args.config))
    device = args.device or hp.get("device", "cuda")

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

    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image import (
        FrechetInceptionDistance, KernelInceptionDistance,
        LearnedPerceptualImagePatchSimilarity,
    )

    # checkpoint="none" — контроль: исходный SD inpainting, склейку «человек |
    # вещь» не видевший. Если обученные модели его не обгонят — обучение
    # ничего не дало.
    if args.official:
        dtype = torch.float16 if hp.get("fp16", True) else torch.float32
        vae_dtype = torch.float16 if hp.get("vae_fp16", False) else torch.float32
        unet, vae, scheduler = load_official(args.official, device, dtype, vae_dtype)
        step = 0
        print(f"опубликованный CatVTON {VARIANTS[args.official]}: guidance "
              f"{args.official_guidance}, eta {args.eta}")
    else:
        unet, vae, scheduler, step, dtype, vae_dtype = load_models(
            hp, flags, args.run, args.checkpoint, device)
        print("веса исходные, без обучения" if args.checkpoint == "none"
              else f"веса с шага {step} из {args.run / args.checkpoint}")
    out_run.mkdir(parents=True, exist_ok=True)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    # FID на нескольких сотнях картинок смещён и шумен; KID несмещён и для
    # малых выборок надёжнее. Считаются оба, в выводах опираться на KID.
    kid = KernelInceptionDistance(subset_size=min(100, len(dataset)), normalize=True).to(device)
    rows: list[dict] = []

    out_dir = out_run / "preds"
    if args.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="генерация", unit="batch"):
        if args.official:
            predicted = generate_official(unet, vae, scheduler, batch, device, dtype, vae_dtype,
                                          args.steps, guidance=args.official_guidance,
                                          eta=args.eta, base_seed=args.seed)
        else:
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
    per_pair.to_csv(out_run / "metrics_pairs.csv", index=False)
    kid_mean, kid_std = kid.compute()
    metrics = {
        "run": str(out_run), "weights": str(args.run), "step": step, "pairs": len(per_pair),
        "steps": args.steps, "seed": args.seed, "arch": flags["name"],
        "official": VARIANTS[args.official] if args.official else None,
        "official_guidance": args.official_guidance if args.official else None,
        "eta": args.eta if args.official else None,
        "reliability_token": args.reliability_token if uses_condition(flags) else None,
        "reliability_guidance": args.reliability_guidance if flags["reliability"] else None,
        "ssim": float(per_pair.ssim.mean()),
        "lpips": float(per_pair.lpips.mean()),
        "l1_mask": float(per_pair.l1_mask.mean()),
        "lpips_mask": float(per_pair.lpips_mask.mean()),
        "kid": float(kid_mean), "kid_std": float(kid_std),
        "fid": float(fid.compute()),
    }
    (out_run / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
