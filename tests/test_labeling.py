import pandas as pd

from posefit.labeling import TASKS, sample


def _pool(n=400):
    cats = ["upper"] * (n // 2) + ["lower"] * (n // 4) + ["dress"] * (n // 4)
    return pd.DataFrame({"image_id": [f"i{i}" for i in range(n)], "category_group": cats})


def test_sample_is_reproducible():
    a = sample(_pool(), 100, seed=5)["image_id"].tolist()
    b = sample(_pool(), 100, seed=5)["image_id"].tolist()
    assert a == b


def test_sample_keeps_category_proportions():
    got = sample(_pool(), 100, seed=5)["category_group"].value_counts(normalize=True)
    assert abs(got["upper"] - 0.5) < 0.05
    assert abs(got["lower"] - 0.25) < 0.05


def test_sample_never_exceeds_a_thin_stratum():
    pool = pd.DataFrame({
        "image_id": [f"i{i}" for i in range(12)],
        "category_group": ["upper"] * 10 + ["outer"] * 2,
    })
    got = sample(pool, 100, seed=1)
    assert (got["category_group"] == "outer").sum() <= 2
    assert not got["image_id"].duplicated().any()


def test_class_codes_are_unique_per_task():
    for task in TASKS.values():
        codes = [c for c, _ in task.classes]
        assert len(codes) == len(set(codes))
