#!/usr/bin/env python
"""Парное сравнение прогонов на общих тестовых парах.

    python scripts/compare_runs.py runs/zero_shot runs/studio runs/studio_wild runs/studio_clean

Точка отсчёта задаётся --reference (по умолчанию первый прогон); для остальных
печатается парная разница с ней. Для главного вопроса работы отсчёт — studio:

    python scripts/compare_runs.py runs/studio runs/studio_wild runs/studio_clean runs/zero_shot
Парность здесь главное: одна и та же тестовая пара трудна для всех моделей,
и сравнение «пара к паре» отделяет эффект модели от разброса трудности.
Непарное сравнение средних на 373 парах в этом разбросе бы утонуло.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posefit.evaluation import mean_ci, paired_difference  # noqa: E402

# Направление: у SSIM больше — лучше, у остальных меньше — лучше.
METRICS = {
    "lpips_mask": ("LPIPS внутри маски", -1),
    "l1_mask": ("L1 внутри маски", -1),
    "lpips": ("LPIPS по кадру", -1),
    "ssim": ("SSIM по кадру", +1),
}


def load(run: Path) -> pd.DataFrame:
    path = run / "metrics_pairs.csv"
    if not path.exists():
        raise SystemExit(f"нет {path}: сначала python scripts/evaluate.py --run {run}")
    return pd.read_csv(path).set_index("pair_id")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--reference", default=None, help="имя прогона-точки отсчёта (по умолчанию первый)")
    ap.add_argument("--by-category", action="store_true", help="разбивка по группам одежды")
    ap.add_argument("--out", type=Path, default=Path("runs/comparison.json"))
    args = ap.parse_args()

    frames = {run.name: load(run) for run in args.runs}
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    if not common:
        raise SystemExit("у прогонов нет общих тестовых пар")
    lost = {name: len(f) - len(common) for name, f in frames.items() if len(f) != len(common)}
    print(f"общих пар: {len(common)}" + (f"   (не у всех: {lost})" if lost else ""))

    # Сид замера обязан совпадать, иначе стартовый шум разный и парность теряется.
    seeds = {}
    for run in args.runs:
        meta = run / "metrics.json"
        if meta.exists():
            seeds[run.name] = json.loads(meta.read_text(encoding="utf-8")).get("seed")
    if len(set(seeds.values())) > 1:
        print(f"ВНИМАНИЕ: разные сиды замера {seeds} — стартовый шум не совпадает")

    reference = args.reference or args.runs[0].name
    if reference not in frames:
        raise SystemExit(f"точки отсчёта {reference!r} нет среди прогонов {list(frames)}")
    report: dict = {"pairs": len(common), "reference": reference, "metrics": {}}

    for metric, (title, direction) in METRICS.items():
        print(f"\n=== {title} ({'больше' if direction > 0 else 'меньше'} — лучше) ===")
        values = {name: f.loc[common, metric].to_numpy() for name, f in frames.items()}
        report["metrics"][metric] = {}
        for name, v in values.items():
            mean, (lo, hi) = mean_ci(v)
            line = f"  {name:<14} {mean:.4f}  [{lo:.4f}, {hi:.4f}]"
            entry = {"mean": mean, "ci": [lo, hi]}
            if name != reference:
                d = paired_difference(values[reference], v)
                better = d["mean_diff"] * direction > 0
                if d["significant"]:
                    verdict = "ЛУЧШЕ" if better else "ХУЖЕ"
                else:
                    verdict = "не отличим"
                line += (f"   vs {reference}: {d['mean_diff']:+.4f} "
                         f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  {verdict}")
                entry["vs_reference"] = d
            print(line)
            report["metrics"][metric][name] = entry

        if args.by_category:
            category = frames[reference].loc[common, "category_group"]
            for group in sorted(category.unique()):
                idx = (category == group).to_numpy()
                cells = "  ".join(f"{name}={values[name][idx].mean():.4f}" for name in values)
                print(f"    {group:<7} n={int(idx.sum()):>3}  {cells}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
