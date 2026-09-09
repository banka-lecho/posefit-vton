import pytest

torch = pytest.importorskip("torch", reason="torch есть только на машине с GPU")

from posefit.model import (  # noqa: E402
    build_inputs, count_parameters, empty_conditioning, freeze_except_self_attention,
    take_person_half, unet_input,
)


def _latents(batch=2, channels=4, h=8, w=6):
    return torch.arange(batch * channels * h * w, dtype=torch.float32).reshape(batch, channels, h, w)


def test_inputs_are_concatenated_along_width():
    person, garment = _latents(), _latents()
    mask = torch.zeros(2, 1, 64, 48)
    target, mask_latent, masked = build_inputs(person, garment, mask)
    assert target.shape == (2, 4, 8, 12)
    assert mask_latent.shape == (2, 1, 8, 12)
    assert masked.shape == (2, 4, 8, 12)


def test_garment_half_is_never_masked():
    # Правая половина — эталон вещи; закрыв её, модель лишилась бы того,
    # что должна перенести на человека.
    mask = torch.ones(2, 1, 64, 48)
    _, mask_latent, masked = build_inputs(_latents(), _latents(), mask)
    assert (mask_latent[..., 6:] == 0).all()
    assert (masked[..., 6:] != 0).any()


def test_masked_region_is_zeroed_on_the_person_half():
    person = torch.ones(1, 4, 8, 6)
    mask = torch.ones(1, 1, 64, 48)
    _, _, masked = build_inputs(person, torch.ones(1, 4, 8, 6), mask)
    assert (masked[..., :6] == 0).all()


def test_unmasked_person_survives_untouched():
    person = torch.full((1, 4, 8, 6), 3.0)
    mask = torch.zeros(1, 1, 64, 48)
    _, _, masked = build_inputs(person, torch.zeros(1, 4, 8, 6), mask)
    assert (masked[..., :6] == 3.0).all()


def test_unet_input_has_the_nine_channels_inpainting_expects():
    person, garment = _latents(), _latents()
    mask = torch.zeros(2, 1, 64, 48)
    target, mask_latent, masked = build_inputs(person, garment, mask)
    assert unet_input(torch.randn_like(target), mask_latent, masked).shape[1] == 9


def test_person_half_is_the_left_one():
    person, garment = torch.zeros(1, 4, 8, 6), torch.ones(1, 4, 8, 6)
    target, _, _ = build_inputs(person, garment, torch.zeros(1, 1, 64, 48))
    assert (take_person_half(target) == 0).all()


def test_empty_conditioning_matches_the_batch():
    assert empty_conditioning(3, "cpu", torch.float32).shape == (3, 77, 768)


def test_only_self_attention_is_trainable():
    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(4, 4)

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = Attn()
            self.attn2 = Attn()
            self.conv = torch.nn.Conv2d(4, 4, 1)

    unet = Fake()
    trainable = freeze_except_self_attention(unet)
    assert count_parameters(trainable) == count_parameters(unet.attn1.parameters())
    assert all(not p.requires_grad for p in unet.attn2.parameters())
    assert all(not p.requires_grad for p in unet.conv.parameters())


def test_missing_attention_layers_are_reported():
    with pytest.raises(RuntimeError, match="attn1"):
        freeze_except_self_attention(torch.nn.Linear(4, 4))
