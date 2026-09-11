import pytest

torch = pytest.importorskip("torch", reason="torch есть только на машине с GPU")

from posefit.official import (  # noqa: E402
    INDEX_STRIDE, SkipAttnProcessor, load_attention_weights, self_attention_layers,
)


class _Attention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim), torch.nn.Dropout(0.0)])


class _Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn1 = _Attention(dim)
        self.attn2 = _Attention(dim)


class _UNet(torch.nn.Module):
    """Три блока с разной шириной: перепутанный порядок слоёв дал бы ошибку формы."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(4), _Block(6), _Block(8)])


def _state(unet, fill=lambda i: float(i + 1)):
    state = {}
    for i, layer in enumerate(self_attention_layers(unet)):
        for part, param in layer.named_parameters():
            state[f"{INDEX_STRIDE * i}.{part}"] = torch.full_like(param, fill(i))
    return state


def test_weights_land_on_self_attention_in_order():
    unet = _UNet()
    assert load_attention_weights(unet, _state(unet)) == 3 * 5
    for i, block in enumerate(unet.blocks):
        assert (block.attn1.to_q.weight == i + 1).all()
        assert (block.attn1.to_out[0].bias == i + 1).all()
        # Кросс-внимание их весами не трогается.
        assert not (block.attn2.to_q.weight == i + 1).all()


def test_wrong_index_stride_is_rejected():
    unet = _UNet()
    state = {k.replace(f"{INDEX_STRIDE}.", "7.", 1) if k.startswith(f"{INDEX_STRIDE}.") else k: v
             for k, v in _state(unet).items()}
    with pytest.raises(RuntimeError, match="шагом"):
        load_attention_weights(unet, state)


def test_missing_tensor_is_rejected():
    unet = _UNet()
    state = _state(unet)
    del state[f"{INDEX_STRIDE}.to_v.weight"]
    with pytest.raises(RuntimeError, match="нет"):
        load_attention_weights(unet, state)


def test_shape_mismatch_is_rejected():
    unet = _UNet()
    state = _state(unet)
    state["0.to_q.weight"] = torch.zeros(6, 6)
    with pytest.raises(RuntimeError, match="форма"):
        load_attention_weights(unet, state)


def test_extra_tensor_is_rejected():
    unet = _UNet()
    state = _state(unet)
    state["0.extra.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="использовано"):
        load_attention_weights(unet, state)


def test_skip_processor_returns_its_input():
    x = torch.randn(2, 5, 4)
    assert SkipAttnProcessor()(None, x, encoder_hidden_states=torch.randn(2, 77, 768)) is x
