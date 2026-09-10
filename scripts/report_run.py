#!/usr/bin/env python
"""Разбор log.jsonl: сглаженный тренд и проверка, что модель училась.

Мгновенные записи в логе почти бесполезны: ошибка сильно зависит от того,
какой таймстеп выпал батчу, и разброс между соседними записями больше
возможного эффекта обучения. Смотреть надо на скользящее среднее и на
сравнение начала прогона с концом.

    python scripts/report_run.py runs/studio
    python scripts/report_run.py runs/*/ --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(run: Path) -> pd.DataFrame:
    lines = (run / "log.jsonl").read_text(encoding="utf-8").splitlines()
    return pd.DataFrame([json.loads(line) for line in lines if line.strip()])


def trend(values: np.ndarray, steps: np.ndarray) -> dict:
    """Наклон прямой и сравнение первой десятой прогона с последней."""
    head = values[: max(1, len(values) // 10)]
    tail = values[-max(1, len(values) // 10) :]
    slope = float(np.polyfit(steps, values, 1)[0]) if len(values) > 2 else 0.0
    # Стандартная ошибка среднего: без неё «упало на 8%» нельзя отличить от шума.
    # На совсем коротком логе дисперсия по одному замеру не определена — тогда
    # честнее вернуть бесконечность, и вердикт гарантированно будет «шум»,
    # чем NaN, который тихо даёт тот же ответ, но выглядит как результат.
    if len(head) < 2 or len(tail) < 2:
        spread = float("inf")
    else:
        spread = float(np.sqrt(head.var(ddof=1) / len(head) + tail.var(ddof=1) / len(tail)))
    return {
        "начало": float(head.mean()),
        "конец": float(tail.mean()),
        "разница": float(tail.mean() - head.mean()),
        "ошибка_разницы": spread,
        "наклон_на_10к_шагов": slope * 10000,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--window", type=int, default=40, help="окно скользящего среднего, записей")
    ap.add_argument("--plot", action="store_true", help="сохранить график рядом с логом")
    args = ap.parse_args()

    for run in args.runs:
        if not (run / "log.jsonl").exists():
            print(f"{run}: нет log.jsonl")
            continue
        frame = load(run)
        print(f"\n=== {run.name}: {len(frame)} записей, {frame.hours.iloc[-1]:.2f} ч, "
              f"{int(frame.step.iloc[-1])} шагов ===")

        for column in ("loss", "loss_masked"):
            if column not in frame:
                continue
            stats = trend(frame[column].to_numpy(), frame["step"].to_numpy())
            significant = abs(stats["разница"]) > 2 * stats["ошибка_разницы"]
            verdict = "значимо" if significant else "в пределах шума"
            print(f"  {column:<12} {stats['начало']:.4f} -> {stats['конец']:.4f}  "
                  f"({stats['разница']:+.4f} ± {stats['ошибка_разницы']:.4f}, {verdict})")
            print(f"  {'':<12} наклон {stats['наклон_на_10к_шагов']:+.4f} на 10к шагов")

        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(9, 4))
            for column, colour in (("loss", "#9aa3ad"), ("loss_masked", "#2563eb")):
                if column not in frame:
                    continue
                axis.plot(frame.step, frame[column], color=colour, alpha=0.18, linewidth=0.8)
                axis.plot(frame.step, frame[column].rolling(args.window, min_periods=1).mean(),
                          color=colour, linewidth=1.8, label=f"{column} (скольз. {args.window})")
            axis.set_xlabel("шаг"); axis.set_ylabel("ошибка"); axis.legend()
            axis.set_title(run.name); axis.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(run / "loss.png", dpi=140)
            print(f"  график -> {run / 'loss.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
