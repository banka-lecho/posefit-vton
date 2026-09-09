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


def test_batch_id_depends_only_on_the_frames():
    from posefit.labeling import batch_id

    a = [{"image_id": "x"}, {"image_id": "y"}]
    assert batch_id(a) == batch_id(list(reversed(a)))
    assert batch_id(a) != batch_id([{"image_id": "x"}, {"image_id": "z"}])


def test_render_isolates_storage_and_download_per_batch():
    from posefit.labeling import TASKS, batch_id, render

    task = TASKS["verify"]
    first = [{"image_id": "a", "rel_path": "p/a.webp", "category": "upper",
              "thumb": "", "ref_path": "p/r.webp", "ref_thumb": ""}]
    second = [dict(first[0], image_id="b")]
    html_a, html_b = render(task, first), render(task, second)
    assert batch_id(first) in html_a and batch_id(second) in html_b
    # Разные партии не должны делить ни ключ хранилища, ни имя выгрузки.
    assert batch_id(first) not in html_b
