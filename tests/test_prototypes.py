"""
Full-rigor unit tests for prototypes.py. Everything here is pure vector math
and JSON I/O — no camera, no DINOv3, no torch needed. Runs anywhere.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.prototypes import PrototypeStore, cosine_similarity, MatchResult


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def jittered(v: np.ndarray, amount: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = v + rng.normal(scale=amount, size=v.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy)


# --------------------------------------------------------------------------
# cosine_similarity
# --------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_similarity_is_one(self):
        v = unit_vector(1)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_similarity_is_minus_one(self):
        v = unit_vector(1)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors_similarity_is_zero(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_is_scale_invariant(self):
        v = unit_vector(2)
        assert cosine_similarity(v, v * 5.0) == pytest.approx(1.0, abs=1e-6)

    def test_handles_zero_vector_without_crashing(self):
        z = np.zeros(4, dtype=np.float32)
        v = unit_vector(3, dim=4)
        result = cosine_similarity(z, v)
        assert math.isfinite(result)


# --------------------------------------------------------------------------
# PrototypeStore — building & inspecting
# --------------------------------------------------------------------------

class TestStoreBuilding:
    def test_starts_empty(self):
        store = PrototypeStore()
        assert store.is_empty()
        assert store.labels() == []

    def test_add_example_creates_class(self):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        assert "mug" in store.labels()
        assert store.example_count("mug") == 1
        assert not store.is_empty()

    def test_add_multiple_examples_same_class(self):
        store = PrototypeStore()
        for i in range(5):
            store.add_example("bottle", unit_vector(i))
        assert store.example_count("bottle") == 5

    def test_add_examples_bulk(self):
        store = PrototypeStore()
        vecs = [unit_vector(i) for i in range(3)]
        store.add_examples("keys", vecs)
        assert store.example_count("keys") == 3

    def test_embed_dim_locked_on_first_add(self):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1, dim=8))
        assert store.embed_dim == 8

    def test_mismatched_embed_dim_raises(self):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1, dim=8))
        with pytest.raises(ValueError):
            store.add_example("mug", unit_vector(2, dim=16))

    def test_remove_class(self):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        store.remove_class("mug")
        assert "mug" not in store.labels()
        assert store.is_empty()

    def test_remove_nonexistent_class_is_a_noop(self):
        store = PrototypeStore()
        store.remove_class("ghost")  # should not raise

    def test_clear(self):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        store.add_example("bottle", unit_vector(2))
        store.clear()
        assert store.is_empty()


# --------------------------------------------------------------------------
# PrototypeStore — prototype (centroid) computation
# --------------------------------------------------------------------------

class TestClassPrototype:
    def test_single_example_prototype_equals_that_example(self):
        store = PrototypeStore()
        v = unit_vector(1)
        store.add_example("mug", v)
        proto = store.class_prototype("mug")
        np.testing.assert_allclose(proto, v, atol=1e-6)

    def test_prototype_is_unit_norm(self):
        store = PrototypeStore()
        for i in range(4):
            store.add_example("mug", jittered(unit_vector(0), 0.3, seed=i))
        proto = store.class_prototype("mug")
        assert np.linalg.norm(proto) == pytest.approx(1.0, abs=1e-6)

    def test_prototype_of_identical_examples_equals_that_vector(self):
        store = PrototypeStore()
        v = unit_vector(5)
        for _ in range(4):
            store.add_example("mug", v.copy())
        proto = store.class_prototype("mug")
        np.testing.assert_allclose(proto, v, atol=1e-5)

    def test_missing_class_raises_keyerror(self):
        store = PrototypeStore()
        with pytest.raises(KeyError):
            store.class_prototype("nonexistent")

    def test_all_prototypes_covers_every_class(self):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        store.add_example("bottle", unit_vector(2))
        protos = store.all_prototypes()
        assert set(protos.keys()) == {"mug", "bottle"}


# --------------------------------------------------------------------------
# PrototypeStore — best_match / open-set behavior
# --------------------------------------------------------------------------

class TestBestMatch:
    def _two_class_store(self):
        store = PrototypeStore()
        base_mug = unit_vector(100, dim=16)
        base_bottle = unit_vector(200, dim=16)
        for i in range(5):
            store.add_example("mug", jittered(base_mug, 0.05, seed=i))
        for i in range(5):
            store.add_example("bottle", jittered(base_bottle, 0.05, seed=100 + i))
        return store, base_mug, base_bottle

    def test_recognizes_close_match_mean_mode(self):
        store, base_mug, _ = self._two_class_store()
        query = jittered(base_mug, 0.05, seed=999)
        result = store.best_match(query, threshold=0.5, mode="mean")
        assert result.is_known
        assert result.label == "mug"
        assert result.similarity > 0.5

    def test_recognizes_close_match_max_mode(self):
        store, base_mug, _ = self._two_class_store()
        query = jittered(base_mug, 0.05, seed=999)
        result = store.best_match(query, threshold=0.5, mode="max")
        assert result.is_known
        assert result.label == "mug"
        assert result.matched_example_index is not None

    def test_unrelated_query_falls_back_to_unknown(self):
        store, _, _ = self._two_class_store()
        # A near-orthogonal random vector in high dimensions is very unlikely
        # to be similar to either enrolled class.
        far_query = unit_vector(999999, dim=16)
        result = store.best_match(far_query, threshold=0.9, mode="mean")
        assert not result.is_known
        assert result.label is None

    def test_empty_store_returns_unknown_not_crash(self):
        store = PrototypeStore()
        result = store.best_match(unit_vector(1), threshold=0.5)
        assert isinstance(result, MatchResult)
        assert result.label is None
        assert not result.is_known

    def test_threshold_boundary_is_inclusive(self):
        store = PrototypeStore()
        v = unit_vector(1, dim=8)
        store.add_example("mug", v)
        result = store.best_match(v, threshold=1.0, mode="mean")  # exact match, sim == 1.0
        assert result.is_known

    def test_similarity_reported_even_when_unknown(self):
        # Even a rejected match should tell the caller *how close* it was,
        # so the UI can show "63% sure it might be X, but below threshold".
        store, base_mug, _ = self._two_class_store()
        query = jittered(base_mug, 0.05, seed=999)
        result = store.best_match(query, threshold=0.999, mode="mean")
        assert not result.is_known
        assert result.similarity > 0.0  # still a real number, not thrown away

    def test_correctly_distinguishes_two_classes(self):
        store, base_mug, base_bottle = self._two_class_store()
        mug_query = jittered(base_mug, 0.05, seed=1001)
        bottle_query = jittered(base_bottle, 0.05, seed=1002)
        assert store.best_match(mug_query, threshold=0.5).label == "mug"
        assert store.best_match(bottle_query, threshold=0.5).label == "bottle"

    def test_invalid_mode_raises(self):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        with pytest.raises(ValueError):
            store.best_match(unit_vector(2), mode="bogus")


# --------------------------------------------------------------------------
# PrototypeStore — all_similarities (feeds the similarity-meter HUD)
# --------------------------------------------------------------------------

class TestAllSimilarities:
    def _two_class_store(self):
        store = PrototypeStore()
        base_mug = unit_vector(100, dim=16)
        base_bottle = unit_vector(200, dim=16)
        for i in range(5):
            store.add_example("mug", jittered(base_mug, 0.05, seed=i))
        for i in range(5):
            store.add_example("bottle", jittered(base_bottle, 0.05, seed=100 + i))
        return store, base_mug, base_bottle

    def test_empty_store_returns_empty_dict(self):
        store = PrototypeStore()
        assert store.all_similarities(unit_vector(1)) == {}

    def test_returns_one_entry_per_class(self):
        store, base_mug, base_bottle = self._two_class_store()
        result = store.all_similarities(unit_vector(999, dim=16))
        assert set(result.keys()) == {"mug", "bottle"}

    def test_values_are_plain_floats(self):
        store, _, _ = self._two_class_store()
        result = store.all_similarities(unit_vector(999, dim=16))
        for value in result.values():
            assert isinstance(value, float)

    def test_matching_class_scores_higher_than_the_other(self):
        store, base_mug, base_bottle = self._two_class_store()
        query = jittered(base_mug, 0.05, seed=999)
        result = store.all_similarities(query, mode="mean")
        assert result["mug"] > result["bottle"]

    def test_mean_mode_agrees_with_best_match_winner(self):
        store, base_mug, _ = self._two_class_store()
        query = jittered(base_mug, 0.05, seed=999)
        all_sims = store.all_similarities(query, mode="mean")
        best = store.best_match(query, threshold=0.5, mode="mean")
        winner = max(all_sims, key=all_sims.get)
        assert winner == best.label
        assert all_sims[winner] == pytest.approx(best.similarity, abs=1e-6)

    def test_max_mode_agrees_with_best_match_winner(self):
        store, base_mug, _ = self._two_class_store()
        query = jittered(base_mug, 0.05, seed=999)
        all_sims = store.all_similarities(query, mode="max")
        best = store.best_match(query, threshold=0.5, mode="max")
        winner = max(all_sims, key=all_sims.get)
        assert winner == best.label
        assert all_sims[winner] == pytest.approx(best.similarity, abs=1e-6)

    def test_max_mode_score_is_at_least_mean_mode_score(self):
        # max mode picks the single best example per class, which can only
        # be >= that class's own centroid similarity.
        store, base_mug, _ = self._two_class_store()
        query = jittered(base_mug, 0.05, seed=999)
        mean_sims = store.all_similarities(query, mode="mean")
        max_sims = store.all_similarities(query, mode="max")
        for label in mean_sims:
            assert max_sims[label] >= mean_sims[label] - 1e-6

    def test_single_class_store(self):
        store = PrototypeStore()
        v = unit_vector(1, dim=8)
        store.add_example("mug", v)
        result = store.all_similarities(v)
        assert list(result.keys()) == ["mug"]
        assert result["mug"] == pytest.approx(1.0, abs=1e-6)

    def test_invalid_mode_raises(self):
        store, _, _ = self._two_class_store()
        with pytest.raises(ValueError):
            store.all_similarities(unit_vector(1, dim=16), mode="bogus")


# --------------------------------------------------------------------------
# PrototypeStore — save/load persistence
# --------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        out = tmp_path / "prototypes.json"
        store.save(out)
        assert out.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        out = tmp_path / "nested" / "dir" / "prototypes.json"
        store.save(out)
        assert out.exists()

    def test_round_trip_preserves_data(self, tmp_path):
        store = PrototypeStore()
        v1, v2_ = unit_vector(1, dim=8), unit_vector(2, dim=8)
        store.add_example("mug", v1)
        store.add_example("mug", v2_)
        store.add_example("bottle", unit_vector(3, dim=8))
        out = tmp_path / "prototypes.json"
        store.save(out)

        loaded = PrototypeStore.load(out)
        assert loaded.embed_dim == 8
        assert set(loaded.labels()) == {"mug", "bottle"}
        assert loaded.example_count("mug") == 2
        assert loaded.example_count("bottle") == 1
        np.testing.assert_allclose(loaded.class_prototype("mug"), store.class_prototype("mug"), atol=1e-6)

    def test_round_trip_preserves_matching_behavior(self, tmp_path):
        store, base_mug, base_bottle = self._make_two_class_store()
        out = tmp_path / "prototypes.json"
        store.save(out)
        loaded = PrototypeStore.load(out)

        query = jittered(base_mug, 0.05, seed=999)
        original_result = store.best_match(query, threshold=0.5)
        loaded_result = loaded.best_match(query, threshold=0.5)
        assert original_result.label == loaded_result.label
        assert original_result.similarity == pytest.approx(loaded_result.similarity, abs=1e-5)

    def _make_two_class_store(self):
        store = PrototypeStore()
        base_mug = unit_vector(100, dim=16)
        base_bottle = unit_vector(200, dim=16)
        for i in range(5):
            store.add_example("mug", jittered(base_mug, 0.05, seed=i))
        for i in range(5):
            store.add_example("bottle", jittered(base_bottle, 0.05, seed=100 + i))
        return store, base_mug, base_bottle

    def test_load_or_empty_returns_empty_store_when_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        store = PrototypeStore.load_or_empty(missing, embed_dim=384)
        assert store.is_empty()
        assert store.embed_dim == 384

    def test_load_or_empty_loads_when_present(self, tmp_path):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        out = tmp_path / "prototypes.json"
        store.save(out)

        loaded = PrototypeStore.load_or_empty(out)
        assert loaded.example_count("mug") == 1

    def test_saved_file_is_valid_readable_json(self, tmp_path):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        out = tmp_path / "prototypes.json"
        store.save(out)
        payload = json.loads(out.read_text())
        assert "embed_dim" in payload
        assert "classes" in payload
        assert "mug" in payload["classes"]

    def test_save_is_atomic_no_leftover_tmp_file(self, tmp_path):
        store = PrototypeStore()
        store.add_example("mug", unit_vector(1))
        out = tmp_path / "prototypes.json"
        store.save(out)
        assert not (tmp_path / "prototypes.json.tmp").exists()
