import pandas as pd

from posefit.dedup import annotate, design_components


def _frame(rows):
    cols = ["content_md5", "design_id", "category_raw", "branch", "sku_id", "order_idx"]
    return pd.DataFrame(rows, columns=cols)


def test_cross_category_asset_is_template():
    df = _frame([
        ("banner", "юбка а", "Юбки", "product", "s1", 2),
        ("banner", "блузка б", "Блузки", "product", "s2", 2),
    ])
    assert annotate(df)["is_template"].all()


def test_wide_product_asset_is_template_within_one_category():
    rows = [("chart", f"брюки {i}", "Брюки", "product", f"s{i}", 2) for i in range(5)]
    assert annotate(_frame(rows))["is_template"].all()


def test_shared_photo_of_two_designs_is_not_a_template():
    df = _frame([
        ("flat", "блузка на шнуровке", "Блузки", "product", "s1", 7),
        ("flat", "блузка на шнуровке больших размеров", "Блузки", "product", "s2", 8),
    ])
    assert not annotate(df)["is_template"].any()


def test_review_under_several_designs_loses_its_label():
    df = _frame([
        ("shot", "боди а", "Блузки", "review", "s1", 3),
        ("shot", "боди б", "Блузки", "review", "s2", 3),
    ])
    out = annotate(df)
    assert not out["label_ok"].any()
    # Ровно один представитель на уникальное содержимое.
    assert out["is_canonical"].sum() == 1


def test_components_merge_designs_sharing_a_product_photo():
    df = annotate(_frame([
        ("flat", "блузка", "Блузки", "product", "s1", 7),
        ("flat", "блузка больших размеров", "Блузки", "product", "s2", 8),
        ("solo", "юбка", "Юбки", "product", "s3", 1),
    ]))
    comp = design_components(df)
    assert comp["блузка"] == comp["блузка больших размеров"]
    assert comp["юбка"] != comp["блузка"]


def test_components_ignore_review_duplicates():
    # Фото отзыва под двумя вещами — ошибка привязки, а не признак одного кроя.
    df = annotate(_frame([
        ("shot", "боди а", "Блузки", "review", "s1", 3),
        ("shot", "боди б", "Блузки", "review", "s2", 3),
    ]))
    comp = design_components(df)
    assert comp["боди а"] != comp["боди б"]
