import numpy as np
import pandas as pd

from posefit.matching import auc, reference_bank, similarity, sweep


def _unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_reference_bank_groups_vectors_per_sku():
    product = pd.DataFrame({"sku_id": ["a", "a", "b"], "image_id": ["1", "2", "3"]})
    emb = {"1": _unit(1, 0), "2": _unit(0, 1), "3": _unit(1, 1)}
    bank = reference_bank(product, emb)
    assert bank["a"].shape == (2, 2)
    assert bank["b"].shape == (1, 2)


def test_reference_bank_skips_frames_without_embeddings():
    product = pd.DataFrame({"sku_id": ["a", "a"], "image_id": ["1", "missing"]})
    bank = reference_bank(product, {"1": _unit(1, 0)})
    assert bank["a"].shape == (1, 2)


def test_similarity_takes_the_best_matching_view():
    # Карточка снята с нескольких ракурсов; совпадение хотя бы с одним из них
    # уже говорит, что вещь та же.
    review = pd.DataFrame({"sku_id": ["a"], "image_id": ["r"]})
    emb = {"r": _unit(1, 0), "p1": _unit(0, 1), "p2": _unit(1, 0)}
    bank = reference_bank(pd.DataFrame({"sku_id": ["a", "a"], "image_id": ["p1", "p2"]}), emb)
    assert similarity(review, emb, bank).iloc[0] == 1.0


def test_similarity_is_nan_without_a_reference():
    review = pd.DataFrame({"sku_id": ["zzz"], "image_id": ["r"]})
    got = similarity(review, {"r": _unit(1, 0)}, {})
    assert np.isnan(got.iloc[0])


def test_auc_is_one_for_perfect_separation():
    assert auc(np.array([True, True, False, False]), np.array([0.9, 0.8, 0.2, 0.1])) == 1.0


def test_auc_is_half_for_no_signal():
    assert auc(np.array([True, False, True, False]), np.array([1.0, 1.0, 1.0, 1.0])) == 0.5


def test_sweep_trades_recall_for_precision():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    table = sweep(scores, np.array([False, False, True, True]), steps=4)
    assert table["полнота"].is_monotonic_decreasing
    assert table["точность"].iloc[-1] >= table["точность"].iloc[0]
