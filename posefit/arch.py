"""Флаги архитектуры из конфига обучения.

Отдельно от model.py, потому что тот импортирует torch, а --dry-run обязан
работать на машине без GPU и показывать, что именно будет обучаться.
"""

from __future__ import annotations

ARCH_NAMES = ("catvton", "anchor")


def arch_flags(hp: dict) -> dict:
    """Какие дополнения включены. Для catvton — никакие, что бы ни стояло в конфиге."""
    arch = hp.get("arch") or {}
    name = arch.get("name", "catvton")
    if name not in ARCH_NAMES:
        raise ValueError(f"неизвестная архитектура {name!r}, ожидается одна из {ARCH_NAMES}")
    on = name == "anchor"
    return {
        "name": name,
        "guide": on and bool(arch.get("guide", True)),
        "reliability": on and bool(arch.get("reliability", True)),
        "garment_embed": on and bool(arch.get("garment_embed", True)),
        "cond_dropout": float(arch.get("cond_dropout", 0.1)) if on else 0.0,
    }


def uses_condition(flags: dict) -> bool:
    """Нужно ли глобальное условие (токен надёжности или вектор вещи)."""
    return bool(flags["reliability"] or flags["garment_embed"])
