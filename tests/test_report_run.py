import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from report_run import trend  # noqa: E402


def _steps(n=600):
    return np.arange(1, n + 1) * 50


def test_a_real_decline_is_called_significant():
    rng = np.random.default_rng(0)
    steps = _steps()
    values = 0.16 - 5e-6 * steps + rng.normal(0, 0.01, len(steps))
    stats = trend(values, steps)
    assert stats["разница"] < 0
    assert abs(stats["разница"]) > 2 * stats["ошибка_разницы"]


def test_pure_noise_is_not_called_significant():
    # Ради этого сравнения и нужна ошибка разницы: без неё случайные ±8%
    # выглядели бы как обучение.
    rng = np.random.default_rng(1)
    steps = _steps()
    values = 0.13 + rng.normal(0, 0.05, len(steps))
    stats = trend(values, steps)
    assert abs(stats["разница"]) <= 2 * stats["ошибка_разницы"]


def test_slope_is_reported_per_ten_thousand_steps():
    steps = _steps()
    values = 0.2 - 1e-5 * steps
    assert np.isclose(trend(values, steps)["наклон_на_10к_шагов"], -0.1, atol=1e-6)


def test_head_and_tail_use_a_tenth_of_the_run():
    steps = _steps(100)
    values = np.concatenate([np.full(10, 1.0), np.full(80, 0.5), np.full(10, 0.0)])
    stats = trend(values, steps)
    assert np.isclose(stats["начало"], 1.0)
    assert np.isclose(stats["конец"], 0.0)


def test_short_logs_report_infinite_uncertainty_instead_of_nan():
    stats = trend(np.array([0.2, 0.1]), np.array([50, 100]))
    assert np.isfinite(stats["разница"])
    # NaN дал бы тот же вердикт «шум», но выглядел бы как посчитанная величина.
    assert stats["ошибка_разницы"] == float("inf")
