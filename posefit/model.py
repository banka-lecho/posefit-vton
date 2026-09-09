"""Схема CatVTON поверх SD 1.5 inpainting.

Вещь и человек склеиваются по ширине в один тензор и подаются в UNet как одна
картинка: латенты становятся (4, h, 2w), маска закрывает только левую половину.
Внимание внутри UNet само переносит фактуру и цвет из правой половины в левую,
поэтому не нужны ни отдельный энкодер вещи, ни warping-модуль.

Обучаются только проекции self-attention — порядка 50 млн параметров вместо
860 млн у полного UNet. Это и делает обучение возможным на одной 24-гигабайтной
карте: остальные веса живут замороженными и не требуют состояний оптимизатора.

Кросс-внимание получает нули вместо текстовых эмбеддингов: задача обусловлена
изображением, текстовый энкодер не нужен и в память не грузится.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BACKBONE = "runwayml/stable-diffusion-inpainting"
VAE_SCALE = 8
CROSS_ATTENTION_DIM = 768
TEXT_TOKENS = 77


def freeze_except_self_attention(unet) -> list[torch.nn.Parameter]:
    """Размораживает только q/k/v/out слоёв self-attention, остальное замораживает."""
    for parameter in unet.parameters():
        parameter.requires_grad_(False)

    trainable: list[torch.nn.Parameter] = []
    for name, module in unet.named_modules():
        # attn1 — self-attention; attn2 — кросс-внимание к тексту, оно здесь
        # не используется и обучать его нечему.
        if name.endswith("attn1"):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
    if not trainable:
        raise RuntimeError("не найдено ни одного слоя attn1 — проверь версию diffusers")
    return trainable


def count_parameters(parameters) -> int:
    return sum(p.numel() for p in parameters)


def encode(vae, images: torch.Tensor) -> torch.Tensor:
    """Изображения из [-1,1] в латенты."""
    latents = vae.encode(images).latent_dist.sample()
    return latents * vae.config.scaling_factor


def decode(vae, latents: torch.Tensor) -> torch.Tensor:
    images = vae.decode(latents / vae.config.scaling_factor).sample
    return images.clamp(-1, 1)


def build_inputs(
    person_latents: torch.Tensor,
    garment_latents: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Готовит склеенные латенты, маску и латенты закрытой картинки.

    Возвращает (target, mask_latent, masked_latent) — всё шириной 2w.
    """
    target = torch.cat([person_latents, garment_latents], dim=-1)

    latent_size = person_latents.shape[-2:]
    mask_small = F.interpolate(mask, size=latent_size, mode="nearest")
    # Правая половина — вещь-эталон, её модель не перерисовывает, поэтому там нули.
    mask_latent = torch.cat([mask_small, torch.zeros_like(mask_small)], dim=-1)

    masked_person = person_latents * (1.0 - mask_small)
    masked_latent = torch.cat([masked_person, garment_latents], dim=-1)
    return target, mask_latent, masked_latent


def unet_input(noisy: torch.Tensor, mask_latent: torch.Tensor,
               masked_latent: torch.Tensor) -> torch.Tensor:
    """Девять каналов, которых ждёт inpainting-UNet: 4 шума + 1 маска + 4 контекста."""
    return torch.cat([noisy, mask_latent, masked_latent], dim=1)


def empty_conditioning(batch: int, device, dtype) -> torch.Tensor:
    return torch.zeros(batch, TEXT_TOKENS, CROSS_ATTENTION_DIM, device=device, dtype=dtype)


def take_person_half(latents: torch.Tensor) -> torch.Tensor:
    """Левая половина склейки — то, ради чего всё считалось."""
    return latents[..., : latents.shape[-1] // 2]
