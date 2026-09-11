"""Генерация примерки обученной моделью: общая часть замера и ноутбука.

Вынесено из scripts/evaluate.py, чтобы ноутбук со своими примерами собирал
модель и шёл по шагам диффузии ровно так же, как замер на тестовом наборе:
та же схема из снимка прогона, тот же стартовый шум для пары, тот же guidance.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .arch import uses_condition
from .evaluation import pair_seed
from .model import (
    BACKBONE, build_guide, build_inputs, build_unet, decode, empty_conditioning, encode,
    load_component, pack_condition, take_person_half, unet_input,
)
from .reliability import NULL_TOKEN


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


def load_models(hp: dict, flags: dict, run: Path | None, checkpoint: str, device):
    """VAE, UNet с дополнениями схемы и планировщик DDIM, всё замороженное.

    checkpoint="none" — исходные веса SD inpainting без обучения. Возвращает
    (unet, vae, scheduler, step, dtype, vae_dtype).
    """
    from diffusers import AutoencoderKL, DDIMScheduler

    dtype = torch.float16 if hp.get("fp16", True) else torch.float32
    vae_dtype = torch.float16 if hp.get("vae_fp16", False) else torch.float32
    vae = load_component(AutoencoderKL, "vae").to(device, vae_dtype).eval()
    unet = build_unet(flags).to(device, dtype).eval()
    # Вывод не обучает: замораживается всё. freeze_except_self_attention здесь
    # была ошибкой — она размораживает слои внимания, и 50 шагов диффузии
    # строили граф градиентов через все шаги сразу: 23 ГБ на первом батче.
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    step = 0
    if checkpoint != "none":
        if run is None:
            raise ValueError("для обученных весов нужен каталог прогона")
        state = torch.load(Path(run) / checkpoint, map_location=device)
        missing = unet.load_state_dict({k: v.to(dtype) for k, v in state["unet"].items()},
                                       strict=False)
        if missing.unexpected_keys:
            raise RuntimeError(f"веса чекпоинта не подходят к схеме: {missing.unexpected_keys[:3]}")
        step = int(state["step"])
    scheduler = DDIMScheduler.from_pretrained(BACKBONE, subfolder="scheduler")
    return unet, vae, scheduler, step, dtype, vae_dtype


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
