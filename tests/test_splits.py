import pandas as pd

from posefit.splits import assign, check_leakage


def _sku(n_per_group=1, n_groups=100):
    rows = []
    for i in range(n_groups):
        for j in range(n_per_group):
            rows.append({
                "sku_id": f"s{i}_{j}",
                "design_id": f"d{i}_{j}",
                "category_group": "upper" if i % 2 else "lower",
            })
    return pd.DataFrame(rows)


def test_assignment_is_deterministic():
    sku = _sku()
    comp = {d: d for d in sku["design_id"]}
    a = assign(sku, comp, seed=1).set_index("split_group")["split"]
    b = assign(sku, comp, seed=1).set_index("split_group")["split"]
    assert a.equals(b)


def test_seed_changes_assignment():
    sku = _sku()
    comp = {d: d for d in sku["design_id"]}
    a = assign(sku, comp, seed=1).set_index("split_group")["split"]
    b = assign(sku, comp, seed=2).set_index("split_group")["split"]
    assert not a.equals(b)


def test_linked_designs_land_in_one_split():
    sku = _sku(n_per_group=3, n_groups=40)
    comp = {d: d.split("_")[0] for d in sku["design_id"]}
    groups = assign(sku, comp, seed=7)
    merged = sku.assign(g=sku["design_id"].map(comp)).merge(groups, left_on="g", right_on="split_group")
    assert (merged.groupby("g")["split"].nunique() == 1).all()


def test_every_stratum_gets_all_three_splits():
    groups = assign(_sku(), {d: d for d in _sku()["design_id"]}, seed=3)
    per_cat = groups.groupby("category_group")["split"].nunique()
    assert (per_cat == 3).all()


def test_leakage_check_spots_a_shared_hash():
    images = pd.DataFrame({
        "content_md5": ["a", "a", "b"],
        "split": ["train", "test", "train"],
    })
    assert check_leakage(images)["content_hashes_in_multiple_splits"] == 1
