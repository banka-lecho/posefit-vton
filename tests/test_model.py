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


def test_masked_mse_ignores_everything_outside_the_mask():
    from posefit.model import masked_mse

    error = torch.zeros(1, 4, 8, 6)
    error[..., 0:2] = 9.0      # внутри маски
    error[..., 2:] = 100.0     # снаружи — учитываться не должно
    mask_latent = torch.zeros(1, 1, 8, 12)
    mask_latent[..., 0:2] = 1.0
    assert torch.allclose(masked_mse(error, mask_latent), torch.tensor(9.0))


def test_masked_mse_averages_over_masked_cells_and_channels():
    from posefit.model import masked_mse

    error = torch.ones(1, 4, 8, 6) * 2.0
    mask_latent = torch.zeros(1, 1, 8, 12)
    mask_latent[..., 0:3] = 1.0
    assert torch.allclose(masked_mse(error, mask_latent), torch.tensor(2.0))


def test_masked_mse_survives_an_empty_mask():
    from posefit.model import masked_mse

    # Кадр, где human parsing не нашёл вещь: делить на ноль нельзя.
    error = torch.ones(1, 4, 8, 6)
    assert torch.isfinite(masked_mse(error, torch.zeros(1, 1, 8, 12)))


def test_masked_mse_uses_only_the_person_half_of_the_mask():
    from posefit.model import masked_mse

    error = torch.ones(1, 4, 8, 6)
    mask_latent = torch.zeros(1, 1, 8, 12)
    mask_latent[..., 6:] = 1.0   # правая половина — вещь-эталон
    assert torch.allclose(masked_mse(error, mask_latent), torch.tensor(0.0))


# ---------------------------------------------------------------------------
# Якорная схема.
# ---------------------------------------------------------------------------

class _TimeEmbedding(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.linear_2 = torch.nn.Linear(dim, dim)


class _FakeUNet(torch.nn.Module):
    """Ровно те части UNet, которых касается схема: conv_in, временной эмбеддинг, attn1."""

    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(9, 4, 3, padding=1)
        self.time_embedding = _TimeEmbedding()
        self.attn1 = torch.nn.Linear(4, 4)
        self.attn2 = torch.nn.Linear(4, 4)
        self.class_embedding = None
        self.config = {}

    def register_to_config(self, **kwargs):
        self.config.update(kwargs)


def test_extended_conv_in_ignores_the_new_channels_at_start():
    from posefit.model import extend_conv_in

    unet = _FakeUNet()
    x = torch.randn(1, 9, 8, 6)
    before = unet.conv_in(x)
    extend_conv_in(unet, 3)
    assert unet.conv_in.in_channels == 12
    assert unet.config["in_channels"] == 12
    after = unet.conv_in(torch.cat([x, torch.randn(1, 3, 8, 6)], dim=1))
    # Нулевые веса новых каналов: выход не зависит от них, старые веса целы.
    assert torch.allclose(before, after, atol=1e-6)


def test_global_condition_starts_as_zero_and_is_trainable():
    from posefit.model import GlobalCondition, pack_condition

    module = GlobalCondition(time_dim=8, n_tokens=5, embed_dim=6)
    cond = pack_condition(torch.tensor([0, 4]), torch.randn(2, 6))
    assert cond.shape == (2, 7)
    out = module(cond)
    assert out.shape == (2, 8)
    assert torch.allclose(out, torch.zeros_like(out))
    out.sum().backward()   # граф строится — параметры обучаемы
    assert module.token.weight.grad is not None


def test_global_condition_without_garment_vector_uses_zeros():
    from posefit.model import GlobalCondition, pack_condition

    module = GlobalCondition(time_dim=8, n_tokens=5, embed_dim=0)
    with torch.no_grad():
        module.token.weight[2] = 1.0
    out = module(pack_condition(torch.tensor([2, 0]), None))
    assert torch.allclose(out[0], torch.ones(8)) and torch.allclose(out[1], torch.zeros(8))


def test_attached_condition_lands_on_the_class_embedding_path():
    from posefit.model import attach_global_condition

    unet = _FakeUNet()
    attach_global_condition(unet, n_tokens=5, embed_dim=768)
    assert unet.class_embedding is not None
    assert unet.config["class_embed_type"] is None
    assert unet.config["class_embeddings_concat"] is False


def test_condition_dropout_replaces_tokens_with_null():
    from posefit.model import drop_condition

    ids = torch.zeros(1000, dtype=torch.long)
    dropped = drop_condition(ids, 0.3, null_token=4, generator=torch.Generator().manual_seed(0))
    share = (dropped == 4).float().mean().item()
    assert 0.25 < share < 0.35
    assert torch.equal(drop_condition(ids, 0.0, null_token=4), ids)


def test_trainable_parameters_include_the_additions():
    from posefit.model import attach_global_condition, extend_conv_in, trainable_parameters

    unet = _FakeUNet()
    extend_conv_in(unet, 3)
    attach_global_condition(unet, n_tokens=5, embed_dim=0)
    flags = {"guide": True, "reliability": True, "garment_embed": False}
    trainable = {id(p) for p in trainable_parameters(unet, flags)}
    assert {id(p) for p in unet.conv_in.parameters()} <= trainable
    assert {id(p) for p in unet.class_embedding.parameters()} <= trainable
    assert all(not p.requires_grad for p in unet.attn2.parameters())

    flags = {"guide": False, "reliability": False, "garment_embed": False}
    trainable = {id(p) for p in trainable_parameters(unet, flags)}
    assert trainable == {id(p) for p in unet.attn1.parameters()}


def test_guide_is_downsampled_and_concatenated_like_the_latents():
    from posefit.model import build_guide

    person = torch.ones(2, 3, 64, 48)
    garment = torch.zeros(2, 3, 64, 48)
    guide = build_guide(person, garment, (8, 6))
    assert guide.shape == (2, 3, 8, 12)
    assert (guide[..., :6] == 1).all() and (guide[..., 6:] == 0).all()


def test_unet_input_appends_guide_channels():
    person, garment = _latents(), _latents()
    target, mask_latent, masked = build_inputs(person, garment, torch.zeros(2, 1, 64, 48))
    guide = torch.zeros(2, 3, 8, 12)
    assert unet_input(torch.randn_like(target), mask_latent, masked, guide).shape[1] == 12
    assert unet_input(torch.randn_like(target), mask_latent, masked).shape[1] == 9
