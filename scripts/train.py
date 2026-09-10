#!/usr/bin/env python
"""Обучение примерки по схеме CatVTON.

    python scripts/train.py --config-name studio
    python scripts/train.py --config-name studio_wild   # продолжит с последней точки
    python scripts/train.py --config-name studio_wild --arch anchor   # якорная схема

Три конфигурации отличаются только составом обучающих пар (см. posefit.pairs.CONFIGS);
всё остальное — модель, гиперпараметры, seed — держится одинаковым, иначе
сравнение прогонов ничего не покажет.

Обучение возобновляемо и продолжается по умолчанию: при падении запусти ту же
команду, состояние поднимется из последней контрольной точки. Начать заново —
явным флагом --restart.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch и всё, что его тянет, импортируются внутри main(): --dry-run обязан
# работать на машине без GPU, иначе проверить состав данных можно только там,
# где идёт обучение.
from posefit.arch import ARCH_NAMES, arch_flags, uses_condition  # noqa: E402
from posefit.pairs import CONFIGS, select_pairs  # noqa: E402
from posefit.paths import DEFAULT_CONFIG, load_config, load_paths  # noqa: E402
from posefit.preprocess import output_path  # noqa: E402
from posefit.reliability import NULL_TOKEN, assign_reliability, describe  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--train-config", default=Path("configs/train.yaml"), type=Path)
    ap.add_argument("--config-name", required=True, choices=list(CONFIGS))
    ap.add_argument("--out", type=Path, default=None, help="по умолчанию runs/<config-name>")
    ap.add_argument("--mask-stage", default=None, choices=["agnostic", "agnostic_refined"],
                    help="перебивает mask_stage из конфига; по умолчанию каталог "
                         "прогона получает суффикс _refined")
    ap.add_argument("--arch", default=None, choices=list(ARCH_NAMES),
                    help="перебивает arch.name из конфига: catvton — базовая схема, "
                         "anchor — якорная; каталог прогона получает суффикс _anchor")
    ap.add_argument("--restart", action="store_true",
                    help="начать с нуля, затерев прошлый прогон "
                         "(по умолчанию обучение продолжается с последней точки)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="проверить состав данных и выйти, не трогая GPU")
    return ap.parse_args()


def check_inputs(selected: pd.DataFrame, paths, probe: int = 200,
                 flags: dict | None = None) -> None:
    """Проверяет, что файлы, которые запросит загрузчик, существуют.

    Отсутствующая маска всплыла бы через десять минут после старта, посреди
    первой эпохи, и уронила бы прогон. Выборочная проверка стоит секунды.
    Якорная схема тянет ещё позу, parse выкладки и вектор вещи — они
    проверяются по тем же строкам, когда включены.
    """
    if selected.empty:
        raise SystemExit("под эту конфигурацию не отобрано ни одной пары")
    flags = flags or {}

    sample = selected.sample(min(probe, len(selected)), random_state=0)
    missing: list[str] = []
    checked = 0
    for row in sample.itertuples():
        wanted = [paths.raw_root / row.person_rel_path, paths.raw_root / row.garment_rel_path,
                  output_path(paths.preproc_root, "agnostic", row.sku_id, row.person_image_id)]
        if flags.get("guide"):
            wanted.append(output_path(paths.preproc_root, "pose", row.sku_id, row.person_image_id))
            wanted.append(output_path(paths.preproc_root, "parse", row.sku_id, row.garment_image_id))
        if flags.get("garment_embed"):
            wanted.append(output_path(paths.preproc_root, "embed", row.sku_id, row.garment_image_id))
        for path in wanted:
            checked += 1
            if not path.exists():
                missing.append(f"нет файла: {path}")

    if missing:
        for line in missing[:5]:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(f"не хватает {len(missing)} файлов из {checked} проверенных")
    print(f"  проверено файлов: {checked}, все на месте")


def save_checkpoint(path: Path, step: int, unet, optimizer, scheduler, scaler) -> None:
    """Атомарная запись контрольной точки.

    Запись идёт во временный файл рядом и завершается переименованием: оно
    атомарно на POSIX, поэтому прерывание посреди сохранения оставляет
    предыдущую точку целой. Прямая запись в latest.pt при неудачном моменте
    убила бы единственный чекпоинт вместе с прогоном.
    """
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = {n: p for n, p in unet.named_parameters() if p.requires_grad}
    state = {
        "step": step,
        # Сохраняются только обучаемые веса: замороженный бэкбон и так
        # восстанавливается из репозитория, а чекпоинт весит 200 МБ вместо 4 ГБ.
        "unet": {n: p.detach().cpu() for n, p in trainable.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        # Масштаб потерь в fp16 подбирается адаптивно; без него возобновление
        # начинает с дефолта и впустую пропускает первые шаги.
        "scaler": scaler.state_dict(),
        # Без состояния генератора шум и таймстепы после перезапуска пошли бы
        # той же последовательностью, что в начале обучения.
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, unet, optimizer, scheduler, scaler, device) -> int:
    import torch

    state = torch.load(path, map_location=device)
    missing = unet.load_state_dict(state["unet"], strict=False)
    if missing.unexpected_keys:
        raise RuntimeError(f"чужие веса в чекпоинте: {missing.unexpected_keys[:3]}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if state.get("rng") is not None:
        torch.set_rng_state(state["rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"])


def main() -> int:
    args = parse_args()
    hp = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    paths = load_paths(load_config(args.config))
    if args.mask_stage:
        hp["mask_stage"] = args.mask_stage
    if args.arch:
        hp["arch"] = {**(hp.get("arch") or {}), "name": args.arch}
    # Флаги считаются до импорта torch: --dry-run обязан показать, что именно
    # будет обучаться, и упасть на опечатке в имени схемы.
    flags = arch_flags(hp)
    suffix = "_refined" if hp.get("mask_stage") == "agnostic_refined" else ""
    if flags["name"] != "catvton":
        suffix += f"_{flags['name']}"
    out_dir = args.out or Path("runs") / f"{args.config_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Снимок гиперпараметров рядом с весами. Замер читает отсюда, а не из
    # configs/train.yaml: конфиг к тому времени мог смениться, а модель обязана
    # получить ту же маску и то же разрешение, на которых училась.
    (out_dir / "train_config.yaml").write_text(yaml.safe_dump(hp, allow_unicode=True),
                                               encoding="utf-8")

    device = args.device or hp.get("device", "cuda")

    pairs = pd.read_parquet(paths.cache_root / "manifest_pairs.parquet")
    selected = select_pairs(pairs, args.config_name, "train", hp.get("clean_quantile", 0.65))

    print(f"конфигурация: {args.config_name}")
    print(f"  пар: {len(selected)}   "
          f"студия {int((selected['ветка'] == 'studio').sum())} / "
          f"wild {int((selected['ветка'] == 'wild').sum())}")
    print(f"  разрешение {hp['width']}x{hp['height']}   батч {hp['batch_size']}")
    print(f"  архитектура: {flags['name']}   "
          + "   ".join(f"{k}={'да' if flags[k] else 'нет'}"
                       for k in ("guide", "reliability", "garment_embed")))
    if flags["reliability"]:
        # Корзины считаются по отобранным парам этого прогона: пороги зависят
        # от состава, и у studio_clean шумных корзин не будет вовсе.
        selected = selected.copy()
        selected["reliability"] = assign_reliability(
            selected, hp.get("clean_quantile", 0.65),
            (hp.get("arch") or {}).get("noisy_quantile", 0.3)).to_numpy()
        print(f"  корзины надёжности: {describe(selected['reliability'])}")
    check_inputs(selected, paths, flags=flags)
    if args.dry_run:
        return 0

    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from torch.utils.data import DataLoader

    from posefit.dataset import VTONPairs
    from posefit.model import (
        BACKBONE, build_guide, build_inputs, build_unet, count_parameters,
        drop_condition, empty_conditioning, encode, load_component, masked_mse,
        pack_condition, trainable_parameters, unet_input,
    )

    dtype = torch.float16 if hp.get("fp16", True) else torch.float32
    torch.manual_seed(hp.get("seed", 0))

    dataset = VTONPairs(selected, paths.raw_root, paths.preproc_root,
                        height=hp["height"], width=hp["width"], flip=hp.get("flip", True),
                        mask_stage=hp.get("mask_stage", "agnostic"),
                        guide=flags["guide"], garment_embed=flags["garment_embed"])
    loader = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True,
                        num_workers=hp.get("workers", 8), pin_memory=True, drop_last=True)

    # VAE от SD 1.5 известен переполнениями в fp16: латенты уходят в NaN, и
    # обучение молча портится. Держим его в fp32, память это позволяет —
    # он замороженный и без состояний оптимизатора.
    vae_dtype = torch.float16 if hp.get("vae_fp16", False) else torch.float32
    vae = load_component(AutoencoderKL, "vae").to(device, vae_dtype).eval()
    vae.requires_grad_(False)
    # Один и тот же сборщик для обучения и замера: расширенный conv_in и
    # глобальное условие должны существовать до загрузки чекпоинта.
    unet = build_unet(flags).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(BACKBONE, subfolder="scheduler")

    trainable = trainable_parameters(unet, flags)
    print(f"  обучаемых параметров: {count_parameters(trainable) / 1e6:.1f} млн "
          f"из {count_parameters(unet.parameters()) / 1e6:.0f} млн")
    if hp.get("gradient_checkpointing", True):
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(trainable, lr=hp["lr"], weight_decay=hp.get("weight_decay", 1e-2))
    max_steps = args.max_steps or hp["max_steps"]
    accumulate = hp.get("grad_accum", 1)
    # Планировщик шагает раз в accumulate микробатчей, поэтому его горизонт
    # считается в шагах оптимизатора, а не в проходах по данным.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=hp["lr"], total_steps=math.ceil(max_steps / accumulate),
        pct_start=hp.get("warmup_frac", 0.05), anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    step = 0
    latest = out_dir / "latest.pt"
    if latest.exists() and not args.restart:
        # Возобновление по умолчанию: забытый флаг не должен стоить десяти
        # часов обучения, затёртых стартом с нуля.
        step = load_checkpoint(latest, unet, optimizer, scheduler, scaler, device)
        print(f"  продолжаю с шага {step} (--restart начнёт заново)")
    elif latest.exists():
        print("  --restart: прошлый прогон будет затёрт")

    log_path = out_dir / "log.jsonl"
    progress = tqdm(total=max_steps, initial=step, desc=args.config_name, unit="step")
    started = time.time()
    unet.train()

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            person = batch["person"].to(device, vae_dtype, non_blocking=True)
            garment = batch["garment"].to(device, vae_dtype, non_blocking=True)
            mask = batch["mask"].to(device, dtype, non_blocking=True)

            with torch.no_grad():
                person_latents = encode(vae, person).to(dtype)
                garment_latents = encode(vae, garment).to(dtype)
            target, mask_latent, masked_latent = build_inputs(person_latents, garment_latents, mask)

            guide = None
            if flags["guide"]:
                guide = build_guide(batch["guide_person"].to(device, non_blocking=True),
                                    batch["guide_garment"].to(device, non_blocking=True),
                                    target.shape[-2:])
            condition = None
            if uses_condition(flags):
                reliability = batch["reliability"].to(device)
                if flags["reliability"]:
                    # Часть токенов заменяется на «неизвестно»: без этого на
                    # выводе не из чего строить guidance по надёжности.
                    reliability = drop_condition(reliability, flags["cond_dropout"], NULL_TOKEN)
                condition = pack_condition(
                    reliability,
                    batch["garment_embed"].to(device) if flags["garment_embed"] else None)

            noise = torch.randn_like(target)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (target.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(target, noise, timesteps)

            with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.float16):
                predicted = unet(
                    unet_input(noisy, mask_latent, masked_latent, guide),
                    timesteps,
                    encoder_hidden_states=empty_conditioning(target.shape[0], device, dtype),
                    class_labels=condition,
                ).sample
                # Потери считаются только на половине с человеком: правая
                # половина — вход-эталон, восстанавливать её незачем.
                half = target.shape[-1] // 2
                error = (predicted[..., :half].float() - noise[..., :half].float()) ** 2
                loss = error.mean() / accumulate

            # Диагностика, в обучении не участвует. Вне маски модель видит
            # чистый латент во входных каналах и восстанавливает шум точно,
            # поэтому общая ошибка на четыре пятых состоит из нулей и почти
            # не меняется. Учится модель ровно внутри маски — и смотреть надо
            # на эту величину.
            with torch.no_grad():
                masked_loss = masked_mse(error.detach(), mask_latent)

            scaler.scale(loss).backward()
            if (step + 1) % accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, hp.get("clip_grad", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1
            progress.update(1)
            if step % hp.get("log_every", 50) == 0:
                record = {"step": step, "loss": loss.detach().item() * accumulate,
                          "loss_masked": masked_loss.item(),
                          "lr": scheduler.get_last_lr()[0],
                          "hours": round((time.time() - started) / 3600, 3)}
                progress.set_postfix(loss=f"{record['loss']:.4f}",
                                     masked=f"{record['loss_masked']:.4f}")
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            if step % hp.get("save_every", 1000) == 0:
                save_checkpoint(latest, step, unet, optimizer, scheduler, scaler)

    save_checkpoint(out_dir / "final.pt", step, unet, optimizer, scheduler, scaler)
    save_checkpoint(latest, step, unet, optimizer, scheduler, scaler)
    print(f"\nготово: {step} шагов за {(time.time() - started) / 3600:.1f} ч -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
