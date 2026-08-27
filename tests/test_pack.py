"""
Unit tests for pack.py — export/import a shareable recognizer pack.

All pure logic (JSON I/O against tmp_path, no camera/backbone/hardware
needed), so this is tested directly and thoroughly, same rigor as
prototypes.py's own tests.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.pack import (
    export_pack,
    import_pack,
    load_pack_metadata,
    ImportSummary,
    PackFormatError,
    PackIncompatibleError,
    PACK_FORMAT_VERSION,
)
from protovision.prototypes import PrototypeStore

from conftest import unit_embedding as _unit


def store_with_classes(dim=16):
    store = PrototypeStore()
    store.add_examples("mug", [_unit(1, dim), _unit(2, dim)])
    store.add_examples("bottle", [_unit(3, dim), _unit(4, dim), _unit(5, dim)])
    return store


# --------------------------------------------------------------------------
# export_pack
# --------------------------------------------------------------------------

class TestExportPack:
    def test_empty_store_raises(self, tmp_path):
        with pytest.raises(ValueError):
            export_pack(PrototypeStore(), tmp_path / "pack.json")

    def test_creates_a_file(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "nested" / "dir" / "pack.json"
        export_pack(store, out)
        assert out.exists()

    def test_returns_exported_labels(self, tmp_path):
        store = store_with_classes()
        labels = export_pack(store, tmp_path / "pack.json")
        assert set(labels) == {"mug", "bottle"}

    def test_written_file_has_required_fields(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out)
        payload = json.loads(out.read_text())
        assert payload["pack_format_version"] == PACK_FORMAT_VERSION
        assert payload["embed_dim"] == 16
        assert "created_at" in payload
        assert "model_name" in payload
        assert set(payload["classes"].keys()) == {"mug", "bottle"}

    def test_default_exports_every_class(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out)
        payload = json.loads(out.read_text())
        assert len(payload["classes"]["mug"]) == 2
        assert len(payload["classes"]["bottle"]) == 3

    def test_exporting_specific_labels_only(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        labels = export_pack(store, out, labels=["mug"])
        assert labels == ["mug"]
        payload = json.loads(out.read_text())
        assert set(payload["classes"].keys()) == {"mug"}

    def test_exporting_unknown_label_raises_keyerror(self, tmp_path):
        store = store_with_classes()
        with pytest.raises(KeyError):
            export_pack(store, tmp_path / "pack.json", labels=["ghost"])

    def test_custom_model_name_is_recorded(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out, model_name="dinov3_vitb16")
        payload = json.loads(out.read_text())
        assert payload["model_name"] == "dinov3_vitb16"

    def test_extra_metadata_is_recorded(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out, extra_metadata={"author": "Abrar", "note": "office objects"})
        payload = json.loads(out.read_text())
        assert payload["metadata"] == {"author": "Abrar", "note": "office objects"}

    def test_no_metadata_defaults_to_empty_dict(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out)
        payload = json.loads(out.read_text())
        assert payload["metadata"] == {}

    def test_is_atomic_no_leftover_tmp_file(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out)
        assert not (tmp_path / "pack.json.tmp").exists()

    def test_exported_embeddings_round_trip_exactly(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out)
        payload = json.loads(out.read_text())
        original = store.examples_for_class("mug")[0]
        exported = np.array(payload["classes"]["mug"][0], dtype=np.float32)
        np.testing.assert_allclose(exported, original, atol=1e-6)


# --------------------------------------------------------------------------
# load_pack_metadata
# --------------------------------------------------------------------------

class TestLoadPackMetadata:
    def test_matches_what_was_exported(self, tmp_path):
        store = store_with_classes()
        out = tmp_path / "pack.json"
        export_pack(store, out, model_name="dinov3_vits16", extra_metadata={"author": "Abrar"})

        meta = load_pack_metadata(out)
        assert meta["embed_dim"] == 16
        assert meta["model_name"] == "dinov3_vits16"
        assert set(meta["labels"]) == {"mug", "bottle"}
        assert meta["metadata"] == {"author": "Abrar"}
        assert meta["pack_format_version"] == PACK_FORMAT_VERSION

    def test_missing_file_raises_packformaterror(self, tmp_path):
        with pytest.raises(PackFormatError):
            load_pack_metadata(tmp_path / "does_not_exist.json")

    def test_invalid_json_raises_packformaterror(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(PackFormatError):
            load_pack_metadata(bad)

    def test_json_array_instead_of_object_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("[1, 2, 3]")
        with pytest.raises(PackFormatError):
            load_pack_metadata(bad)

    def test_missing_required_fields_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"pack_format_version": 1}))  # no embed_dim/classes
        with pytest.raises(PackFormatError):
            load_pack_metadata(bad)

    def test_unsupported_format_version_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({
            "pack_format_version": 999,
            "embed_dim": 16,
            "classes": {},
        }))
        with pytest.raises(PackFormatError):
            load_pack_metadata(bad)

    def test_a_plain_prototypes_json_is_not_a_valid_pack(self, tmp_path):
        """prototypes.json and a pack are deliberately different formats —
        a bare prototypes.json is missing pack_format_version, so it
        should be rejected with a clear error, not silently misread."""
        store = store_with_classes()
        prototypes_path = tmp_path / "prototypes.json"
        store.save(prototypes_path)
        with pytest.raises(PackFormatError):
            load_pack_metadata(prototypes_path)


# --------------------------------------------------------------------------
# import_pack
# --------------------------------------------------------------------------

class TestImportPack:
    def test_adds_new_classes_to_empty_store(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path)

        target = PrototypeStore()
        summary = import_pack(target, pack_path)

        assert set(summary.added) == {"mug", "bottle"}
        assert target.example_count("mug") == 2
        assert target.example_count("bottle") == 3

    def test_adds_new_classes_alongside_existing_ones(self, tmp_path):
        source = store_with_classes()  # mug, bottle
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path, labels=["bottle"])

        target = PrototypeStore()
        target.add_examples("keys", [_unit(10, 16)])
        summary = import_pack(target, pack_path)

        assert summary.added == ["bottle"]
        assert set(target.labels()) == {"keys", "bottle"}

    def test_returns_import_summary_instance(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path)
        summary = import_pack(PrototypeStore(), pack_path)
        assert isinstance(summary, ImportSummary)

    def test_skip_is_the_default_conflict_policy(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path, labels=["mug"])

        target = PrototypeStore()
        original_mug = _unit(999, 16)
        target.add_example("mug", original_mug)

        summary = import_pack(target, pack_path)  # no on_conflict specified

        assert summary.skipped == ["mug"]
        assert target.example_count("mug") == 1  # untouched
        np.testing.assert_array_equal(target.examples_for_class("mug")[0], original_mug)

    def test_merge_conflict_appends_examples(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path, labels=["mug"])  # 2 examples

        target = PrototypeStore()
        target.add_example("mug", _unit(999, 16))  # 1 existing example

        summary = import_pack(target, pack_path, on_conflict="merge")

        assert summary.merged == ["mug"]
        assert target.example_count("mug") == 3  # 1 existing + 2 imported

    def test_overwrite_conflict_replaces_examples(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path, labels=["mug"])  # 2 examples

        target = PrototypeStore()
        target.add_example("mug", _unit(999, 16))
        target.add_example("mug", _unit(998, 16))
        target.add_example("mug", _unit(997, 16))  # 3 existing examples

        summary = import_pack(target, pack_path, on_conflict="overwrite")

        assert summary.overwritten == ["mug"]
        assert target.example_count("mug") == 2  # fully replaced by the pack's 2

    def test_invalid_on_conflict_raises(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path)
        with pytest.raises(ValueError):
            import_pack(PrototypeStore(), pack_path, on_conflict="bogus")

    def test_dimension_mismatch_raises_incompatible_error(self, tmp_path):
        source = store_with_classes(dim=16)
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path)

        target = PrototypeStore()
        target.add_example("keys", _unit(1, dim=384))  # different dimension

        with pytest.raises(PackIncompatibleError):
            import_pack(target, pack_path)

    def test_dimension_check_skipped_for_empty_target_store(self, tmp_path):
        """An empty store has no embed_dim yet to conflict with — any
        pack's dimension should be accepted and adopted."""
        source = store_with_classes(dim=16)
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path)

        target = PrototypeStore()  # embed_dim is None
        import_pack(target, pack_path)  # should not raise
        assert target.embed_dim == 16

    def test_model_name_mismatch_warns_but_does_not_raise(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path, model_name="dinov3_vitb16")

        target = PrototypeStore()
        summary = import_pack(target, pack_path, expected_model_name="dinov3_vits16")

        assert summary.changed_any_class  # import still happened
        assert any("dinov3_vitb16" in w for w in summary.warnings)

    def test_matching_model_name_produces_no_warning(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path, model_name="dinov3_vits16")

        target = PrototypeStore()
        summary = import_pack(target, pack_path, expected_model_name="dinov3_vits16")

        assert summary.warnings == []

    def test_none_expected_model_name_disables_the_check(self, tmp_path):
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path, model_name="anything-at-all")

        target = PrototypeStore()
        summary = import_pack(target, pack_path, expected_model_name=None)

        assert summary.warnings == []

    def test_does_not_save_to_disk(self, tmp_path):
        """import_pack only mutates the in-memory store — saving is the
        caller's responsibility, same division as the rest of
        PrototypeStore's API."""
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path)

        store_path = tmp_path / "prototypes.json"
        target = PrototypeStore()
        import_pack(target, pack_path)

        assert not store_path.exists()

    def test_invalid_pack_file_raises_packformaterror(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all")
        with pytest.raises(PackFormatError):
            import_pack(PrototypeStore(), bad)

    def test_round_trip_preserves_matching_behavior(self, tmp_path):
        """Export then import into a fresh store should behave identically
        to the original for best_match() — the whole point of a pack."""
        source = store_with_classes()
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path)

        target = PrototypeStore()
        import_pack(target, pack_path)

        query = _unit(1, 16)  # matches "mug"'s first example closely
        original_result = source.best_match(query, threshold=0.3)
        imported_result = target.best_match(query, threshold=0.3)
        assert original_result.label == imported_result.label
        assert original_result.similarity == pytest.approx(imported_result.similarity, abs=1e-6)

    def test_added_merged_overwritten_skipped_are_mutually_exclusive_per_class(self, tmp_path):
        """Sanity check on the summary bookkeeping itself: a single class
        should never appear in more than one of the four outcome lists."""
        source = store_with_classes()  # mug, bottle
        pack_path = tmp_path / "pack.json"
        export_pack(source, pack_path)

        target = PrototypeStore()
        target.add_example("mug", _unit(999, 16))  # pre-existing -> will conflict

        summary = import_pack(target, pack_path, on_conflict="merge")

        all_lists = [summary.added, summary.merged, summary.overwritten, summary.skipped]
        seen = set()
        for lst in all_lists:
            for label in lst:
                assert label not in seen, f"'{label}' appeared in more than one outcome list"
                seen.add(label)
        assert seen == {"mug", "bottle"}
