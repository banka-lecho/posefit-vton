"""Опубликованный CatVTON как существующая модель для сравнения.

Chong Z. et al. CatVTON: Concatenation Is All You Need for Virtual Try-On with
Diffusion Models. ICLR 2025. Код: github.com/Zheng-Chong/CatVTON, веса:
huggingface.co/zhengchong/CatVTON. Лицензия весов и кода — CC BY-NC-SA 4.0:
только некоммерческое исследование.

Модуль воспроизводит их CatVTONPipeline (model/pipeline.py) без его
зависимостей, чтобы замер шёл тем же кодом, что у наших моделей, и на тех же
373 парах. Отличия от нашей схемы — все на их стороне и перенесены как есть:

* склейка «человек / вещь» по высоте, а не по ширине;
* кросс-внимание выключено SkipAttnProcessor: слой возвращает свой вход,
  и блок прибавляет его к остаточной ветви — так модель и обучалась;
* маска накладывается на пиксели до VAE: кодируется image * (mask < 0.5);
* VAE — stabilityai/sd-vae-ft-mse, а не VAE из SD inpainting;
* classifier-free guidance 2.5: безусловная ветвь получает нулевой латент
  вещи; DDIM стохастический, eta = 1.

Вход (кадры, маски) и эталон — наши, одинаковые для всех моделей замера.
Стартовый шум и шум шагов DDIM зависят только от пары и seed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .evaluation import pair_seed
from .model import BACKBONE, decode, empty_conditioning, encode, load_component

REPO = "zhengchong/CatVTON"
VARIANTS = {
    "dresscode": "dresscode-16k-512",   # верх, низ, платья; обучен в 512x384
    "vitonhd": "vitonhd-16k-512",       # только верх; обучен в 512x384
    "mix": "mix-48k-1024",              # смесь датасетов; обучен в 1024x768
}
VAE_REPO = "stabilityai/sd-vae-ft-mse"
CONCAT_DIM = -2          # по высоте, как в оригинале
DEFAULT_GUIDANCE = 2.5
DEFAULT_ETA = 1.0
# Шаг индексов в их файле весов. Они сохраняли ModuleList всех модулей, в имени
# которых есть "attn1": сам слой, to_q, to_k, to_v, to_out, to_out.0,
# to_out.1 и процессор — восемь на каждый слой внимания.
INDEX_STRIDE = 8
PARTS = ("to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight", "to_out.0.bias")


class SkipAttnProcessor:
    """Кросс-внимание CatVTON: слой возвращает вход без изменений."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        return hidden_states


def skip_cross_attention(unet) -> None:
    """attn2 — SkipAttnProcessor, attn1 — обычное внимание, как в их init_adapter."""
    from diffusers.models.attention_processor import AttnProcessor2_0

    processors = {name: SkipAttnProcessor() if ".attn2." in name else AttnProcessor2_0()
                  for name in unet.attn_processors}
    unet.set_attn_processor(processors)


def self_attention_layers(unet) -> list:
    """Слои attn1 в порядке named_modules — в том же, в каком их сохраняли авторы."""
    return [module for name, module in unet.named_modules() if name.endswith("attn1")]


def load_attention_weights(unet, state: dict) -> int:
    """Раскладывает их 80 тензоров по нашим слоям attn1. Возвращает число тензоров.

    Проверяется всё, что может разойтись молча: индексы идут ровно с шагом 8,
    слоёв столько же, сколько в файле, у каждого слоя есть все пять тензоров,
    формы совпадают, лишних ключей нет.
    """
    layers = self_attention_layers(unet)
    indices = sorted({int(key.split(".", 1)[0]) for key in state})
    expected = [INDEX_STRIDE * i for i in range(len(layers))]
    if indices != expected:
        raise RuntimeError(f"индексы весов {indices[:5]}… не совпадают с {len(layers)} "
                           f"слоями attn1 с шагом {INDEX_STRIDE}")
    used = 0
    with torch.no_grad():
        for i, layer in enumerate(layers):
            params = dict(layer.named_parameters())
            for part in PARTS:
                key = f"{INDEX_STRIDE * i}.{part}"
                if key not in state:
                    raise RuntimeError(f"в весах нет {key}")
                if params[part].shape != state[key].shape:
                    raise RuntimeError(f"{key}: форма {tuple(state[key].shape)}, "
                                       f"у слоя {tuple(params[part].shape)}")
                params[part].copy_(state[key].to(params[part].dtype))
                used += 1
    if used != len(state):
        raise RuntimeError(f"использовано {used} тензоров из {len(state)}")
    return used


def load_official(variant: str, device, dtype=torch.float16, vae_dtype=torch.float32):
    """UNet SD inpainting с весами внимания CatVTON, их VAE и DDIM."""
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if variant not in VARIANTS:
        raise ValueError(f"вариант {variant!r}, ожидается один из {tuple(VARIANTS)}")
    unet = load_component(UNet2DConditionModel, "unet")
    skip_cross_attention(unet)
    path = hf_hub_download(REPO, f"{VARIANTS[variant]}/attention/model.safetensors")
    load_attention_weights(unet, load_file(path))
    unet = unet.to(device, dtype).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(VAE_REPO).to(device, vae_dtype).eval().requires_grad_(False)
    scheduler = DDIMScheduler.from_pretrained(BACKBONE, subfolder="scheduler")
    return unet, vae, scheduler


def pair_generators(pair_ids, base_seed: int) -> list[torch.Generator]:
    """Генераторы шума шагов DDIM, по одному на пару: eta > 0 добавляет шум на
    каждом шаге, и без своего генератора он зависел бы от соседей по батчу."""
    return [torch.Generator("cpu").manual_seed(pair_seed(pid, base_seed + 1)) for pid in pair_ids]


@torch.no_grad()
def generate_official(unet, vae, scheduler, batch, device, dtype, vae_dtype, steps: int = 50,
                      guidance: float = DEFAULT_GUIDANCE, eta: float = DEFAULT_ETA,
                      base_seed: int = 0) -> torch.Tensor:
    """Их CatVTONPipeline.__call__ на нашем батче. Возвращает кадры в [-1, 1]."""
    from .generation import initial_noise

    person = batch["person"].to(device, vae_dtype)
    garment = batch["garment"].to(device, vae_dtype)
    mask = batch["mask"].to(device, vae_dtype)

    masked_latent = encode(vae, person * (mask < 0.5)).to(dtype)
    condition_latent = encode(vae, garment).to(dtype)
    mask_latent = F.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest").to(dtype)

    masked_concat = torch.cat([masked_latent, condition_latent], dim=CONCAT_DIM)
    mask_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=CONCAT_DIM)

    latents = initial_noise(masked_concat.shape, batch["pair_id"], base_seed, device, dtype)
    scheduler.set_timesteps(steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    cfg = guidance > 1.0
    if cfg:
        unconditional = torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=CONCAT_DIM)
        masked_concat = torch.cat([unconditional, masked_concat])
        mask_concat = torch.cat([mask_concat] * 2)
    # Кросс-внимание пропускается, значение не используется; нули вместо None —
    # чтобы не зависеть от того, как версия diffusers обходится с None.
    conditioning = empty_conditioning(masked_concat.shape[0], device, dtype)
    generators = pair_generators(batch["pair_id"], base_seed)

    for t in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if cfg else latents
        model_input = scheduler.scale_model_input(model_input, t)
        model_input = torch.cat([model_input, mask_concat, masked_concat], dim=1)
        noise_pred = unet(model_input, t, encoder_hidden_states=conditioning).sample
        if cfg:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance * (cond - uncond)
        latents = scheduler.step(noise_pred, t, latents, eta=eta, generator=generators).prev_sample

    person_half = latents.split(latents.shape[CONCAT_DIM] // 2, dim=CONCAT_DIM)[0]
    return decode(vae, person_half.to(vae_dtype))
