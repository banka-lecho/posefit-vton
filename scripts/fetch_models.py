#!/usr/bin/env python
"""Предзагрузка весов с HuggingFace.

Отдельным шагом, чтобы проблемы с сетью, прокси или доступом вылезли за
минуту, а не на середине многочасового прогона.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.preprocess import MODELS  # noqa: E402


def fetch(kind: str, repo: str) -> bool:
    print(f"\n--- {kind}: {repo}")
    try:
        if kind == "vae":
            from diffusers import AutoencoderKL
            AutoencoderKL.from_pretrained(repo)
        elif kind == "detect":
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            AutoImageProcessor.from_pretrained(repo)
            AutoModelForObjectDetection.from_pretrained(repo)
        elif kind == "pose":
            from transformers import AutoProcessor, VitPoseForPoseEstimation
            AutoProcessor.from_pretrained(repo)
            VitPoseForPoseEstimation.from_pretrained(repo)
        elif kind == "embed":
            from transformers import AutoImageProcessor, AutoModel
            AutoImageProcessor.from_pretrained(repo)
            AutoModel.from_pretrained(repo)
        elif kind == "parse":
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            SegformerImageProcessor.from_pretrained(repo)
            SegformerForSemanticSegmentation.from_pretrained(repo)
        else:
            raise ValueError(f"неизвестная модель: {kind}")
    except Exception as exc:
        print(f"    ОШИБКА: {type(exc).__name__}: {exc}")
        return False
    print("    ок")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=list(MODELS), help="скачать одну модель")
    args = ap.parse_args()

    items = {args.only: MODELS[args.only]} if args.only else MODELS
    failed = [kind for kind, repo in items.items() if not fetch(kind, repo)]

    if failed:
        print(f"\nне скачалось: {', '.join(failed)}")
        print("Пришли мне текст ошибки целиком.")
        return 1
    print("\nвсе веса на месте")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
