#!/usr/bin/env python
"""Диагностика машины перед GPU-этапом.

Запускается первой на целевой машине и печатает всё, что нужно, чтобы
зафиксировать версии зависимостей и размер шардов: питон, torch и CUDA,
модель и объём видеопамяти, свободное место, готовность манифестов.

Ничего не импортирует жёстко: на чистой машине скрипт должен не падать,
а показать, чего не хватает.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import site
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PACKAGES = [
    "numpy", "pandas", "pyarrow", "cv2", "PIL", "yaml", "tqdm",
    "torch", "torchvision", "onnxruntime", "transformers", "diffusers", "accelerate", "scipy",
]


def line(key: str, value: object) -> None:
    print(f"  {key:<26} {value}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_interpreter() -> None:
    """Venv и ~/.local: пакет, найденный вне venv, — источник трудноуловимых
    конфликтов версий, поэтому расположение важнее самого факта установки."""
    section("интерпретатор")
    in_venv = sys.prefix != sys.base_prefix
    line("внутри venv", in_venv)
    line("sys.prefix", sys.prefix)
    if not in_venv:
        line("ВНИМАНИЕ", "venv не активирован")
    line("user site включён", site.ENABLE_USER_SITE)
    try:
        user_site = site.getusersitepackages()
        line("user site", f"{user_site}  {'есть' if Path(user_site).exists() else 'нет'}")
        if site.ENABLE_USER_SITE and Path(user_site).exists():
            line("ВНИМАНИЕ", "пакеты из ~/.local перекрывают venv; PYTHONNOUSERSITE=1 отключает")
    except Exception:
        pass


def check_packages() -> None:
    section("пакеты")
    prefix = Path(sys.prefix)
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
        except Exception:
            line(name, "— НЕТ")
            continue
        version = getattr(module, "__version__", "установлен")
        path = getattr(module, "__file__", None)
        where = ""
        if path:
            location = Path(path).resolve().parent
            try:
                location.relative_to(prefix)
            except ValueError:
                where = f"  ВНЕ VENV: {location}"
        line(name, f"{version}{where}")


def check_torch() -> None:
    section("GPU")
    try:
        import torch
    except Exception as exc:
        line("torch", f"не импортируется: {exc}")
        return

    line("torch", torch.__version__)
    line("собран под CUDA", torch.version.cuda or "нет (CPU-сборка)")
    line("cuda.is_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        # Колесо torch, собранное под CUDA новее драйвера, не видит карту
        # вовсе. На общем сервере драйвер не обновить, поэтому подбирается
        # колесо: --index-url .../whl/cuXYZ под версию из nvidia-smi.
        line("ПОЧЕМУ", "сверь 'собран под CUDA' с версией CUDA из nvidia-smi ниже")
        return
    line("устройств", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        line(f"[{i}] имя", props.name)
        line(f"[{i}] видеопамять", f"{props.total_memory / 1024**3:.1f} ГБ")
        line(f"[{i}] compute capability", f"{props.major}.{props.minor}")
    try:
        free, total = torch.cuda.mem_get_info()
        line("свободно сейчас", f"{free / 1024**3:.1f} из {total / 1024**3:.1f} ГБ")
    except Exception:
        pass


def check_driver() -> None:
    section("драйвер")
    exe = shutil.which("nvidia-smi")
    if not exe:
        line("nvidia-smi", "— не найден")
        return
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        for row in out.stdout.strip().splitlines():
            line("nvidia-smi", row.strip())
    except Exception as exc:
        line("nvidia-smi", f"ошибка запуска: {exc}")
        return
    try:
        # Максимальная версия CUDA, которую поддерживает драйвер: колесо
        # torch должно быть собрано под неё или более раннюю.
        head = subprocess.run([exe], capture_output=True, text=True, timeout=20, check=False)
        for row in head.stdout.splitlines():
            if "CUDA Version" in row:
                line("драйвер поддерживает", row.split("CUDA Version:")[-1].strip(" |"))
    except Exception:
        pass


def check_data(config_path: Path) -> None:
    section("данные")
    try:
        from posefit.paths import load_config, load_paths
        config = load_config(config_path)
        paths = load_paths(config)
    except Exception as exc:
        line("конфиг", f"не читается: {exc}")
        return

    raw = paths.raw_root
    line("raw_root", f"{raw}  {'есть' if raw.exists() else 'НЕТ'}")
    if raw.exists():
        line("кадров в выгрузке", sum(1 for _ in raw.glob("*/*/*/*.webp")))
    for name, path in (("manifest_sku", paths.manifest_sku),
                       ("manifest_images", paths.manifest_images),
                       ("splits.json", paths.splits)):
        line(name, "есть" if path.exists() else "НЕТ — сначала make data")

    line("preproc_root", paths._preproc_root or "не задан (POSEFIT_PREPROC_ROOT)")
    target = paths._preproc_root or Path.cwd()
    probe = target if target.exists() else Path.cwd()
    usage = shutil.disk_usage(probe)
    line("свободно на диске", f"{usage.free / 1024**3:.0f} ГБ  (препроцессингу нужно 5–10 ГБ)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=Path("configs/data.yaml"), type=Path)
    args = ap.parse_args()

    section("система")
    line("платформа", platform.platform())
    line("процессор", platform.processor() or "—")
    line("ядер", os.cpu_count())
    line("python", sys.version.split()[0])
    line("исполняемый файл", sys.executable)

    check_interpreter()
    check_packages()
    check_torch()
    check_driver()
    check_data(args.config)

    print("\nСкопируй весь вывод целиком и пришли мне.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
