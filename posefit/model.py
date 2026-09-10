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

# Официальное зеркало SD 1.5 inpainting. Исходный runwayml/... остался без
# файлов safetensors, и diffusers молча откатывался на pickle-сериализацию.
BACKBONE = "stable-diffusion-v1-5/stable-diffusion-inpainting"
VAE_SCALE = 8
CROSS_ATTENTION_DIM = 768
TEXT_TOKENS = 77


def load_component(cls, subfolder: str, **kwargs):
    """Загружает часть бэкбона, предпочитая safetensors.

    У зеркала SD 1.5 inpainting файлы safetensors выложены только в fp16-варианте,
    и без явного variant diffusers не находит их и молча откатывается на pickle.
    Веса при этом те же: dtype задаётся отдельно при переносе на устройство.
    """
    try:
        return cls.from_pretrained(BACKBONE, subfolder=subfolder, variant="fp16", **kwargs)
    except (OSError, ValueError):
        # Зеркало без fp16-варианта: пусть diffusers выбирает файл сам.
        return cls.from_pretrained(BACKBONE, subfolder=subfolder, **kwargs)


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


def masked_mse(error: torch.Tensor, mask_latent: torch.Tensor) -> torch.Tensor:
    """Средняя ошибка внутри маски — то, чему модель действительно учится.

    Вне маски UNet видит чистый латент во входных каналах и восстанавливает шум
    точно, поэтому общая ошибка на четыре пятых состоит из нулей и почти не
    двигается за прогон. Диагностическая величина, в обратном проходе не участвует.
    """
    half = error.shape[-1]
    inside = mask_latent[..., :half].to(error.dtype)
    weight = inside.sum() * error.shape[1]
    return (error * inside).sum() / weight.clamp(min=1.0)


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
               masked_latent: torch.Tensor, guide: torch.Tensor | None = None) -> torch.Tensor:
    """Девять каналов, которых ждёт inpainting-UNet: 4 шума + 1 маска + 4 контекста.

    В якорной схеме к ним добавляются каналы-подсказки guide (см. build_guide);
    conv_in под них расширен в build_unet.
    """
    parts = [noisy, mask_latent, masked_latent]
    if guide is not None:
        parts.append(guide.to(noisy.dtype))
    return torch.cat(parts, dim=1)


def empty_conditioning(batch: int, device, dtype) -> torch.Tensor:
    return torch.zeros(batch, TEXT_TOKENS, CROSS_ATTENTION_DIM, device=device, dtype=dtype)


def take_person_half(latents: torch.Tensor) -> torch.Tensor:
    """Левая половина склейки — то, ради чего всё считалось."""
    return latents[..., : latents.shape[-1] // 2]


# ---------------------------------------------------------------------------
# Якорная схема: канонические координаты и глобальное условие поверх CatVTON.
#
# Три дополнения, все с нулевой инициализацией — на нулевом шаге модель
# тождественна базовой, и любой выигрыш относится на счёт дополнений:
#   guide         три входных канала (u, v, скелет) из canon.py — общий
#                 пространственный код для половин склейки; conv_in расширяется;
#   reliability   токен надёжности пары из reliability.py через временной
#                 эмбеддинг; на выводе — токен «студия»;
#   garment_embed DINOv2-вектор вещи через тот же временной эмбеддинг.
# ---------------------------------------------------------------------------

GARMENT_EMBED_DIM = 768   # DINOv2-base, стадия embed


def extend_conv_in(unet, extra: int) -> None:
    """Добавляет conv_in входные каналы с нулевыми весами.

    Старые девять каналов сохраняют свои веса, новые начинают с нуля: пока
    сеть не научилась ими пользоваться, они не вносят ничего. Иначе случайные
    веса на старте сломали бы предобученный вход.
    """
    old = unet.conv_in
    new = torch.nn.Conv2d(old.in_channels + extra, old.out_channels,
                          kernel_size=old.kernel_size, padding=old.padding,
                          bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new.to(old.weight.device, old.weight.dtype)
    unet.conv_in = new
    unet.register_to_config(in_channels=new.in_channels)


class GlobalCondition(torch.nn.Module):
    """Глобальное условие, прибавляемое к временному эмбеддингу UNet.

    Принимает один тензор (B, 1 + embed_dim): первый столбец — индекс токена
    надёжности, остальное — вектор вещи. Один тензор вместо двух аргументов,
    потому что diffusers пропускает в class_embedding ровно один class_labels.
    Обе таблицы с нуля: на старте условие ничего не меняет.
    """

    def __init__(self, time_dim: int, n_tokens: int = 0, embed_dim: int = 0) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.token = torch.nn.Embedding(n_tokens, time_dim) if n_tokens else None
        self.garment = torch.nn.Linear(embed_dim, time_dim) if embed_dim else None
        with torch.no_grad():
            if self.token is not None:
                self.token.weight.zero_()
            if self.garment is not None:
                self.garment.weight.zero_()
                self.garment.bias.zero_()

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        out = None
        if self.token is not None:
            ids = cond[:, 0].round().long().clamp(0, self.n_tokens - 1)
            out = self.token(ids)
        if self.garment is not None:
            vec = cond[:, 1: 1 + self.embed_dim].to(self.garment.weight.dtype)
            part = self.garment(vec)
            out = part if out is None else out + part
        if out is None:
            raise RuntimeError("GlobalCondition без единой таблицы — нечего вычислять")
        return out


def pack_condition(reliability: torch.Tensor, garment_embed: torch.Tensor | None,
                   embed_dim: int = GARMENT_EMBED_DIM) -> torch.Tensor:
    """(B,) индексов и (B, embed_dim) векторов -> один тензор (B, 1 + embed_dim)."""
    ids = reliability.to(torch.float32).reshape(-1, 1)
    if garment_embed is None:
        garment_embed = torch.zeros(ids.shape[0], embed_dim, device=ids.device)
    return torch.cat([ids, garment_embed.to(ids.device, torch.float32)], dim=1)


def drop_condition(reliability: torch.Tensor, p: float, null_token: int,
                   generator=None) -> torch.Tensor:
    """Заменяет долю p токенов на «неизвестно» — для guidance по надёжности."""
    if p <= 0:
        return reliability
    keep = torch.rand(reliability.shape, device=reliability.device, generator=generator) >= p
    return torch.where(keep, reliability, torch.full_like(reliability, null_token))


def attach_global_condition(unet, n_tokens: int, embed_dim: int) -> None:
    """Вешает GlobalCondition на штатный путь class_embedding UNet.

    diffusers прибавляет class_embedding(class_labels) к временному эмбеддингу,
    если class_embedding задан; тип условия не проверяется, поэтому вместо
    таблицы классов туда встаёт наш модуль.
    """
    time_dim = unet.time_embedding.linear_2.out_features
    module = GlobalCondition(time_dim, n_tokens=n_tokens, embed_dim=embed_dim)
    module.to(unet.conv_in.weight.device, unet.conv_in.weight.dtype)
    unet.class_embedding = module
    unet.register_to_config(class_embed_type=None, class_embeddings_concat=False)


def build_unet(flags: dict, **kwargs):
    """UNet бэкбона с дополнениями по флагам. Одинаков для обучения и замера."""
    from diffusers import UNet2DConditionModel

    from .canon import GUIDE_CHANNELS
    from .reliability import N_TOKENS

    unet = load_component(UNet2DConditionModel, "unet", **kwargs)
    if flags["guide"]:
        extend_conv_in(unet, GUIDE_CHANNELS)
    if flags["reliability"] or flags["garment_embed"]:
        attach_global_condition(
            unet,
            n_tokens=N_TOKENS if flags["reliability"] else 0,
            embed_dim=GARMENT_EMBED_DIM if flags["garment_embed"] else 0,
        )
    return unet


def trainable_parameters(unet, flags: dict) -> list[torch.nn.Parameter]:
    """self-attention плюс то, что добавила схема: conv_in и глобальное условие."""
    trainable = freeze_except_self_attention(unet)
    extra: list[torch.nn.Module] = []
    if flags["guide"]:
        extra.append(unet.conv_in)
    if flags["reliability"] or flags["garment_embed"]:
        extra.append(unet.class_embedding)
    for module in extra:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable


def build_guide(guide_person: torch.Tensor, guide_garment: torch.Tensor,
                latent_size: tuple[int, int]) -> torch.Tensor:
    """Каналы-подсказки обеих половин, сведённые к размеру латента: (B, 3, h, 2w).

    Усреднение по площади: координаты гладкие, а скелет после усреднения
    становится мягкой линией вместо рваной.
    """
    person = F.interpolate(guide_person, size=latent_size, mode="area")
    garment = F.interpolate(guide_garment, size=latent_size, mode="area")
    return torch.cat([person, garment], dim=-1)
