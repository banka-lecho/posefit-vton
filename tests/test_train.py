import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch есть только на машине с GPU")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train import load_checkpoint, save_checkpoint  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trained = torch.nn.Linear(4, 4)
        self.frozen = torch.nn.Linear(4, 4)
        self.frozen.requires_grad_(False)


def _setup():
    model = _Tiny()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def test_checkpoint_round_trips_the_step(tmp_path):
    model, opt, sch, scaler = _setup()
    save_checkpoint(tmp_path / "latest.pt", 1234, model, opt, sch, scaler)
    assert load_checkpoint(tmp_path / "latest.pt", *_setup()[:3], scaler, "cpu") == 1234


def test_only_trainable_weights_are_stored(tmp_path):
    model, opt, sch, scaler = _setup()
    save_checkpoint(tmp_path / "latest.pt", 1, model, opt, sch, scaler)
    stored = torch.load(tmp_path / "latest.pt", map_location="cpu")["unet"]
    # Замороженный бэкбон восстанавливается из репозитория; хранить его —
    # это 4 ГБ на точку вместо 200 МБ.
    assert set(stored) == {"trained.weight", "trained.bias"}


def test_saving_leaves_no_temporary_file(tmp_path):
    model, opt, sch, scaler = _setup()
    save_checkpoint(tmp_path / "latest.pt", 1, model, opt, sch, scaler)
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_broken_write_does_not_destroy_the_previous_point(tmp_path):
    target = tmp_path / "latest.pt"
    model, opt, sch, scaler = _setup()
    save_checkpoint(target, 100, model, opt, sch, scaler)

    # Имитируем гибель процесса посреди записи: временный файл остался,
    # переименования не было. Прошлая точка обязана уцелеть.
    (tmp_path / "latest.pt.tmp").write_bytes(b"\x00\x01\x02")
    assert load_checkpoint(target, *_setup()[:3], scaler, "cpu") == 100


def test_weights_survive_the_round_trip(tmp_path):
    model, opt, sch, scaler = _setup()
    with torch.no_grad():
        model.trained.weight.fill_(0.5)
    save_checkpoint(tmp_path / "latest.pt", 7, model, opt, sch, scaler)

    restored, opt2, sch2, _ = _setup()
    load_checkpoint(tmp_path / "latest.pt", restored, opt2, sch2, scaler, "cpu")
    assert torch.allclose(restored.trained.weight, torch.full((4, 4), 0.5))


def test_foreign_checkpoint_is_rejected(tmp_path):
    model, opt, sch, scaler = _setup()
    save_checkpoint(tmp_path / "latest.pt", 1, model, opt, sch, scaler)
    state = torch.load(tmp_path / "latest.pt", map_location="cpu")
    state["unet"]["someone_elses.weight"] = torch.zeros(2)
    torch.save(state, tmp_path / "latest.pt")
    with pytest.raises(RuntimeError, match="чужие веса"):
        load_checkpoint(tmp_path / "latest.pt", *_setup()[:3], scaler, "cpu")
