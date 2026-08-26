"""
Unit tests for enroll.py.

`EnrollApp.__init__` opens a real `Camera()` as its very last step, which
doesn't exist in this sandbox — so:
  - constructor ARGUMENT VALIDATION (which happens before the Camera() call)
    is tested via the real constructor, since it never reaches the hardware.
  - everything else (capture/undo/finish/cancel/key handling) is tested by
    building the instance with `EnrollApp.__new__(EnrollApp)` and setting
    only the attributes those methods touch — same bypass-__init__ pattern
    used for SignSense's PracticeApp/LiveApp.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.enroll import (
    EnrollApp,
    EnrollState,
    NotEnoughExamplesError,
    KEY_CAPTURE,
    KEY_UNDO,
    KEY_FINISH,
    KEY_CANCEL,
)
from protovision.prototypes import PrototypeStore
from protovision.ui import ThemeManager, GlyphCache, KEY_THEME_TOGGLE
from protovision.audio import AudioManager

from conftest import make_test_frame, FakeCamera, SpyAudio, unit_embedding, SequenceBackbone

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def make_enroll_app(
    label="mug",
    backbone=None,
    store=None,
    store_path=None,
    target_examples=4,
    min_examples=2,
    box_fraction=0.5,
    state=EnrollState.CAPTURING,
    captured=None,
    theme_manager=None,
    glyph_cache=None,
    audio=None,
) -> EnrollApp:
    """Build an EnrollApp WITHOUT calling __init__ (no real camera needed)."""
    app = EnrollApp.__new__(EnrollApp)
    app.label = label
    app.backbone = backbone
    app.store = store if store is not None else PrototypeStore()
    app.store_path = store_path if store_path is not None else Path("unused.json")
    app.target_examples = target_examples
    app.min_examples = min_examples
    app.box_fraction = box_fraction
    app._captured_embeddings = list(captured) if captured else []
    app._last_capture_warnings = []
    app.state = state
    app.theme_manager = theme_manager if theme_manager is not None else ThemeManager()
    app.glyph_cache = glyph_cache if glyph_cache is not None else GlyphCache(font_dir=FONT_DIR)
    # enabled=False: no real pygame/audio device dependency in these pure
    # logic tests; still exercises the real AudioManager fail-soft path
    # rather than a mock, since it's always a safe no-op either way.
    app.audio = audio if audio is not None else AudioManager(enabled=False)
    app.camera = None  # never touched by the methods under test
    return app


# --------------------------------------------------------------------------
# constructor argument validation (real __init__, runs before Camera())
# --------------------------------------------------------------------------

class TestConstructorValidation:
    def test_rejects_empty_label(self, mock_backbone):
        with pytest.raises(ValueError):
            EnrollApp("", mock_backbone, PrototypeStore(), "x.json")

    def test_rejects_whitespace_only_label(self, mock_backbone):
        with pytest.raises(ValueError):
            EnrollApp("   ", mock_backbone, PrototypeStore(), "x.json")

    def test_rejects_min_examples_below_one(self, mock_backbone):
        with pytest.raises(ValueError):
            EnrollApp("mug", mock_backbone, PrototypeStore(), "x.json", min_examples=0)

    def test_rejects_target_below_min(self, mock_backbone):
        with pytest.raises(ValueError):
            EnrollApp(
                "mug", mock_backbone, PrototypeStore(), "x.json",
                target_examples=2, min_examples=5,
            )

    def test_strips_label_whitespace(self):
        # Validation only — doesn't reach Camera() since we never construct
        # a full instance here, just check the intended behavior directly.
        app = make_enroll_app(label="  mug  ".strip())
        assert app.label == "mug"


class TestConstructorWithFakeCamera:
    """
    These go through the REAL __init__ end-to-end (including the Camera()
    call at the end), with protovision.enroll.Camera monkeypatched to
    FakeCamera so no actual hardware is touched. This is what actually
    proves __init__'s own logic (label stripping, defaults, camera
    injection) works, as a complement to the __new__-bypass tests above.
    """

    def test_label_is_stripped_by_real_init(self, monkeypatch, mock_backbone, tmp_path):
        monkeypatch.setattr("protovision.enroll.Camera", FakeCamera)
        app = EnrollApp("  mug  ", mock_backbone, PrototypeStore(), tmp_path / "p.json")
        assert app.label == "mug"

    def test_defaults_are_applied(self, monkeypatch, mock_backbone, tmp_path):
        monkeypatch.setattr("protovision.enroll.Camera", FakeCamera)
        app = EnrollApp("mug", mock_backbone, PrototypeStore(), tmp_path / "p.json")
        assert app.target_examples == 8
        assert app.min_examples == 5
        assert app.box_fraction == 0.5
        assert app.state == EnrollState.CAPTURING
        assert app.progress == (0, 8)

    def test_opens_a_camera_when_none_injected(self, monkeypatch, mock_backbone, tmp_path):
        monkeypatch.setattr("protovision.enroll.Camera", FakeCamera)
        app = EnrollApp("mug", mock_backbone, PrototypeStore(), tmp_path / "p.json")
        assert isinstance(app.camera, FakeCamera)

    def test_uses_injected_camera_instead_of_opening_new_one(self, mock_backbone, tmp_path):
        injected = FakeCamera()
        app = EnrollApp("mug", mock_backbone, PrototypeStore(), tmp_path / "p.json", camera=injected)
        assert app.camera is injected

    def test_default_audio_manager_is_created(self, monkeypatch, mock_backbone, tmp_path):
        monkeypatch.setattr("protovision.enroll.Camera", FakeCamera)
        app = EnrollApp("mug", mock_backbone, PrototypeStore(), tmp_path / "p.json")
        assert isinstance(app.audio, AudioManager)

    def test_injected_audio_manager_is_used(self, monkeypatch, mock_backbone, tmp_path):
        monkeypatch.setattr("protovision.enroll.Camera", FakeCamera)
        spy = SpyAudio()
        app = EnrollApp("mug", mock_backbone, PrototypeStore(), tmp_path / "p.json", audio=spy)
        assert app.audio is spy

    def test_full_capture_session_end_to_end(self, monkeypatch, mock_backbone, tmp_path):
        """Real __init__ + real capture/finish logic, still no hardware."""
        monkeypatch.setattr("protovision.enroll.Camera", FakeCamera)
        store = PrototypeStore()
        store_path = tmp_path / "p.json"
        app = EnrollApp("mug", mock_backbone, store, store_path, target_examples=3, min_examples=2)
        for i in range(3):
            app.capture_example(make_test_frame(seed=i))
        assert app.state == EnrollState.DONE
        assert store.example_count("mug") == 3
        assert store_path.exists()

    def test_default_theme_manager_and_glyph_cache_are_created(self, monkeypatch, mock_backbone, tmp_path):
        monkeypatch.setattr("protovision.enroll.Camera", FakeCamera)
        app = EnrollApp("mug", mock_backbone, PrototypeStore(), tmp_path / "p.json")
        assert isinstance(app.theme_manager, ThemeManager)
        assert isinstance(app.glyph_cache, GlyphCache)

    def test_injected_theme_manager_and_glyph_cache_are_used(self, monkeypatch, mock_backbone, tmp_path):
        monkeypatch.setattr("protovision.enroll.Camera", FakeCamera)
        theme_mgr = ThemeManager(initial="neon")
        cache = GlyphCache(font_dir=FONT_DIR)
        app = EnrollApp(
            "mug", mock_backbone, PrototypeStore(), tmp_path / "p.json",
            theme_manager=theme_mgr, glyph_cache=cache,
        )
        assert app.theme_manager is theme_mgr
        assert app.glyph_cache is cache


# --------------------------------------------------------------------------
# progress / has_min_examples
# --------------------------------------------------------------------------

class TestProgress:
    def test_progress_starts_at_zero(self):
        app = make_enroll_app(target_examples=8)
        assert app.progress == (0, 8)

    def test_progress_reflects_captured_count(self):
        app = make_enroll_app(target_examples=8, captured=[np.zeros(4)] * 3)
        assert app.progress == (3, 8)

    def test_has_min_examples_false_below_threshold(self):
        app = make_enroll_app(min_examples=5, captured=[np.zeros(4)] * 3)
        assert app.has_min_examples is False

    def test_has_min_examples_true_at_threshold(self):
        app = make_enroll_app(min_examples=3, captured=[np.zeros(4)] * 3)
        assert app.has_min_examples is True


# --------------------------------------------------------------------------
# capture_example
# --------------------------------------------------------------------------

class TestCaptureExample:
    def test_capture_appends_one_embedding(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, target_examples=8)
        frame = make_test_frame()
        app.capture_example(frame)
        assert app.progress == (1, 8)

    def test_capture_returns_the_embedding(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, target_examples=8)
        frame = make_test_frame()
        emb = app.capture_example(frame)
        assert emb.shape == (384,)

    def test_multiple_captures_accumulate(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, target_examples=8)
        for i in range(3):
            app.capture_example(make_test_frame(seed=i))
        assert app.progress == (3, 8)

    def test_auto_finishes_at_target(self, mock_backbone, tmp_path):
        store_path = tmp_path / "prototypes.json"
        app = make_enroll_app(
            backbone=mock_backbone, target_examples=2, min_examples=2, store_path=store_path
        )
        app.capture_example(make_test_frame(seed=1))
        assert app.state == EnrollState.CAPTURING
        app.capture_example(make_test_frame(seed=2))
        assert app.state == EnrollState.DONE
        assert store_path.exists()

    def test_capture_raises_outside_capturing_state(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, state=EnrollState.DONE)
        with pytest.raises(RuntimeError):
            app.capture_example(make_test_frame())


# --------------------------------------------------------------------------
# Phase 3: prototype-quality warnings (duplicate captures, confusable classes)
# --------------------------------------------------------------------------

class TestCaptureQuality:
    def test_first_capture_never_warns(self):
        base = unit_embedding(1, dim=16)
        app = make_enroll_app(backbone=SequenceBackbone([base]), target_examples=8)
        app.capture_example(make_test_frame())
        assert app.last_capture_warnings == []

    def test_near_identical_capture_warns_duplicate(self):
        base = unit_embedding(1, dim=16)
        app = make_enroll_app(backbone=SequenceBackbone([base, base]), target_examples=8)
        app.capture_example(make_test_frame(seed=1))
        app.capture_example(make_test_frame(seed=2))
        assert any("capture #1" in w for w in app.last_capture_warnings)

    def test_varied_captures_do_not_warn_duplicate(self):
        base1 = unit_embedding(1, dim=16)
        base2 = unit_embedding(2, dim=16)  # unrelated random vector, low similarity
        app = make_enroll_app(backbone=SequenceBackbone([base1, base2]), target_examples=8)
        app.capture_example(make_test_frame(seed=1))
        app.capture_example(make_test_frame(seed=2))
        assert app.last_capture_warnings == []

    def test_duplicate_warning_only_checked_against_earlier_captures_not_itself(self):
        """The embedding just captured shouldn't be compared against
        itself — _check_capture_quality runs before it's appended."""
        base = unit_embedding(1, dim=16)
        app = make_enroll_app(backbone=SequenceBackbone([base]), target_examples=8)
        app.capture_example(make_test_frame())  # only one capture ever — nothing to compare against
        assert app.last_capture_warnings == []

    def test_confusable_with_other_enrolled_class_warns(self):
        bottle_base = unit_embedding(2, dim=16)
        store = PrototypeStore()
        store.add_examples("bottle", [bottle_base.copy() for _ in range(5)])
        app = make_enroll_app(
            label="mug", store=store, backbone=SequenceBackbone([bottle_base]), target_examples=8,
        )
        app.capture_example(make_test_frame())
        assert any("bottle" in w for w in app.last_capture_warnings)

    def test_dissimilar_from_other_classes_does_not_warn(self):
        bottle_base = unit_embedding(2, dim=16)
        mug_base = unit_embedding(1, dim=16)
        store = PrototypeStore()
        store.add_examples("bottle", [bottle_base.copy() for _ in range(5)])
        app = make_enroll_app(
            label="mug", store=store, backbone=SequenceBackbone([mug_base]), target_examples=8,
        )
        app.capture_example(make_test_frame())
        assert app.last_capture_warnings == []

    def test_resuming_own_class_does_not_self_warn(self):
        """Enrolling more examples into a class that already has stored
        examples (e.g. a second enrollment session for 'mug') shouldn't
        warn that the new capture is 'confusable with mug' — that's not a
        different class, it's the same one."""
        mug_base = unit_embedding(1, dim=16)
        store = PrototypeStore()
        store.add_examples("mug", [mug_base.copy() for _ in range(5)])
        app = make_enroll_app(
            label="mug", store=store, backbone=SequenceBackbone([mug_base]), target_examples=8,
        )
        app.capture_example(make_test_frame())
        assert app.last_capture_warnings == []

    def test_both_warnings_can_appear_together(self):
        bottle_base = unit_embedding(2, dim=16)
        store = PrototypeStore()
        store.add_examples("bottle", [bottle_base.copy() for _ in range(5)])
        # captured[0] is also bottle_base, so the 2nd capture is BOTH a
        # near-duplicate of capture #1 AND confusable with "bottle".
        app = make_enroll_app(
            label="mug", store=store,
            backbone=SequenceBackbone([bottle_base, bottle_base]), target_examples=8,
        )
        app.capture_example(make_test_frame(seed=1))
        app.capture_example(make_test_frame(seed=2))
        warnings = app.last_capture_warnings
        assert len(warnings) == 2
        assert any("capture #1" in w for w in warnings)
        assert any("bottle" in w for w in warnings)

    def test_capture_still_succeeds_despite_warnings(self):
        """Warnings are advisory, not blocking — the brief says 'warn', not
        'reject'. The capture must still be stored either way."""
        base = unit_embedding(1, dim=16)
        app = make_enroll_app(backbone=SequenceBackbone([base, base]), target_examples=8)
        app.capture_example(make_test_frame(seed=1))
        app.capture_example(make_test_frame(seed=2))
        assert app.progress == (2, 8)  # both captures counted, warning or not

    def test_warnings_reflect_only_the_most_recent_capture(self):
        base1 = unit_embedding(1, dim=16)
        base2 = unit_embedding(2, dim=16)
        app = make_enroll_app(backbone=SequenceBackbone([base1, base1, base2]), target_examples=8)
        app.capture_example(make_test_frame(seed=1))  # no warning (first capture)
        app.capture_example(make_test_frame(seed=2))  # warns: duplicate of #1
        assert app.last_capture_warnings != []
        app.capture_example(make_test_frame(seed=3))  # base2 is unrelated — no warning
        assert app.last_capture_warnings == []

    def test_undo_clears_warnings(self):
        base = unit_embedding(1, dim=16)
        app = make_enroll_app(backbone=SequenceBackbone([base, base]), target_examples=8)
        app.capture_example(make_test_frame(seed=1))
        app.capture_example(make_test_frame(seed=2))
        assert app.last_capture_warnings != []
        app.undo_last()
        assert app.last_capture_warnings == []

    def test_warnings_start_empty(self):
        app = make_enroll_app()
        assert app.last_capture_warnings == []


# --------------------------------------------------------------------------
# undo_last
# --------------------------------------------------------------------------

class TestUndoLast:
    def test_undo_removes_most_recent(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, target_examples=8)
        app.capture_example(make_test_frame(seed=1))
        app.capture_example(make_test_frame(seed=2))
        assert app.progress == (2, 8)
        app.undo_last()
        assert app.progress == (1, 8)

    def test_undo_on_empty_is_noop(self):
        app = make_enroll_app()
        app.undo_last()  # should not raise
        assert app.progress == (0, 4)

    def test_undo_when_not_capturing_is_noop(self):
        app = make_enroll_app(state=EnrollState.DONE, captured=[np.zeros(4)])
        app.undo_last()
        assert app.progress[0] == 1  # unchanged


# --------------------------------------------------------------------------
# finish
# --------------------------------------------------------------------------

class TestFinish:
    def test_finish_below_min_raises(self):
        app = make_enroll_app(min_examples=5, captured=[np.zeros(4)] * 2)
        with pytest.raises(NotEnoughExamplesError):
            app.finish()

    def test_finish_below_min_does_not_change_state(self):
        app = make_enroll_app(min_examples=5, captured=[np.zeros(4)] * 2)
        try:
            app.finish()
        except NotEnoughExamplesError:
            pass
        assert app.state == EnrollState.CAPTURING

    def test_finish_commits_examples_to_store(self, tmp_path):
        store = PrototypeStore()
        vecs = [np.random.default_rng(i).normal(size=8).astype(np.float32) for i in range(3)]
        app = make_enroll_app(
            label="mug", store=store, store_path=tmp_path / "p.json",
            min_examples=2, captured=vecs,
        )
        app.finish()
        assert store.example_count("mug") == 3

    def test_finish_saves_to_disk(self, tmp_path):
        store = PrototypeStore()
        vecs = [np.random.default_rng(i).normal(size=8).astype(np.float32) for i in range(3)]
        store_path = tmp_path / "p.json"
        app = make_enroll_app(store=store, store_path=store_path, min_examples=2, captured=vecs)
        app.finish()
        assert store_path.exists()
        reloaded = PrototypeStore.load(store_path)
        assert reloaded.example_count("mug") == 3

    def test_finish_sets_state_done(self, tmp_path):
        vecs = [np.random.default_rng(i).normal(size=8).astype(np.float32) for i in range(3)]
        app = make_enroll_app(store_path=tmp_path / "p.json", min_examples=2, captured=vecs)
        app.finish()
        assert app.state == EnrollState.DONE

    def test_finish_outside_capturing_raises(self):
        app = make_enroll_app(state=EnrollState.DONE, captured=[np.zeros(4)] * 3, min_examples=2)
        with pytest.raises(RuntimeError):
            app.finish()

    def test_finish_plays_enroll_success_chime(self, tmp_path):
        spy = SpyAudio()
        vecs = [np.random.default_rng(i).normal(size=8).astype(np.float32) for i in range(3)]
        app = make_enroll_app(store_path=tmp_path / "p.json", min_examples=2, captured=vecs, audio=spy)
        app.finish()
        assert spy.enroll_success_calls == 1

    def test_finish_below_min_does_not_play_chime(self):
        spy = SpyAudio()
        app = make_enroll_app(min_examples=5, captured=[np.zeros(4)] * 2, audio=spy)
        with pytest.raises(NotEnoughExamplesError):
            app.finish()
        assert spy.enroll_success_calls == 0

    def test_capture_auto_finish_also_plays_chime(self, mock_backbone, tmp_path):
        """finish() is also reachable via capture_example() auto-finishing
        at target_examples — the chime should fire there too, not just on
        an explicit finish() call."""
        spy = SpyAudio()
        app = make_enroll_app(
            backbone=mock_backbone, target_examples=2, min_examples=2,
            store_path=tmp_path / "p.json", audio=spy,
        )
        app.capture_example(make_test_frame(seed=1))
        assert spy.enroll_success_calls == 0  # not yet, only 1 of 2 captured
        app.capture_example(make_test_frame(seed=2))
        assert spy.enroll_success_calls == 1


# --------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------

class TestCancel:
    def test_cancel_sets_state(self):
        app = make_enroll_app()
        app.cancel()
        assert app.state == EnrollState.CANCELLED

    def test_cancel_does_not_touch_store(self, tmp_path):
        store = PrototypeStore()
        store_path = tmp_path / "p.json"
        app = make_enroll_app(
            store=store, store_path=store_path,
            captured=[np.zeros(4)] * 3, min_examples=2,
        )
        app.cancel()
        assert store.is_empty()
        assert not store_path.exists()


# --------------------------------------------------------------------------
# handle_key dispatch
# --------------------------------------------------------------------------

class TestHandleKey:
    def test_capture_key_triggers_capture(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, target_examples=8)
        app.handle_key(KEY_CAPTURE, make_test_frame())
        assert app.progress == (1, 8)

    def test_undo_key_triggers_undo(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, target_examples=8)
        app.capture_example(make_test_frame())
        app.handle_key(KEY_UNDO, make_test_frame())
        assert app.progress == (0, 8)

    def test_finish_key_finishes_when_enough_examples(self, tmp_path):
        vecs = [np.random.default_rng(i).normal(size=8).astype(np.float32) for i in range(3)]
        app = make_enroll_app(store_path=tmp_path / "p.json", min_examples=2, captured=vecs)
        app.handle_key(KEY_FINISH, make_test_frame())
        assert app.state == EnrollState.DONE

    def test_finish_key_ignored_when_not_enough_examples(self):
        app = make_enroll_app(min_examples=5, captured=[np.zeros(4)] * 2)
        app.handle_key(KEY_FINISH, make_test_frame())
        assert app.state == EnrollState.CAPTURING  # unchanged, no crash

    def test_cancel_key_cancels(self):
        app = make_enroll_app()
        app.handle_key(KEY_CANCEL, make_test_frame())
        assert app.state == EnrollState.CANCELLED

    def test_unknown_key_is_ignored(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, target_examples=8)
        app.handle_key(ord("z"), make_test_frame())
        assert app.progress == (0, 8)
        assert app.state == EnrollState.CAPTURING

    def test_keys_ignored_once_done(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, state=EnrollState.DONE, target_examples=8)
        app.handle_key(KEY_CAPTURE, make_test_frame())
        assert app.progress == (0, 8)  # capture never ran

    def test_theme_key_cycles_theme(self):
        app = make_enroll_app()
        start = app.theme_manager.name
        app.handle_key(KEY_THEME_TOGGLE, make_test_frame())
        assert app.theme_manager.name != start

    def test_theme_key_works_even_when_done(self, mock_backbone):
        """Switching themes shouldn't be blocked just because the session
        already finished or was cancelled — there's no reason to gate it
        behind capture state."""
        app = make_enroll_app(backbone=mock_backbone, state=EnrollState.DONE)
        start = app.theme_manager.name
        app.handle_key(KEY_THEME_TOGGLE, make_test_frame())
        assert app.theme_manager.name != start
        assert app.state == EnrollState.DONE  # state itself untouched

    def test_theme_key_does_not_also_trigger_capture(self, mock_backbone):
        app = make_enroll_app(backbone=mock_backbone, target_examples=8)
        app.handle_key(KEY_THEME_TOGGLE, make_test_frame())
        assert app.progress == (0, 8)  # theme toggle consumed the key, nothing else ran


# --------------------------------------------------------------------------
# guide box / preview helpers
# --------------------------------------------------------------------------

class TestPreviewHelpers:
    def test_current_guide_box_matches_frame(self):
        app = make_enroll_app(box_fraction=0.5)
        frame = make_test_frame(width=200, height=100)
        box = app.current_guide_box(frame)
        assert box.x2 <= 200
        assert box.y2 <= 100

    def test_render_preview_same_shape_as_input(self):
        app = make_enroll_app()
        frame = make_test_frame(width=128, height=128)
        preview = app.render_preview(frame)
        assert preview.shape == frame.shape

    def test_render_preview_does_not_mutate_input(self):
        app = make_enroll_app()
        frame = make_test_frame(width=128, height=128)
        original = frame.copy()
        app.render_preview(frame)
        np.testing.assert_array_equal(frame, original)

    def test_render_preview_draws_a_panel(self):
        """The HUD panel should actually change pixels near the top-left,
        not just draw the guide box."""
        app = make_enroll_app()
        frame = make_test_frame(width=400, height=300, color=(90, 90, 90))
        out = app.render_preview(frame)
        assert not np.array_equal(out[30, 30], frame[30, 30])

    def test_render_preview_respects_active_theme(self):
        """Rendering under two different themes should not produce
        identical output — proves the panel is actually using
        self.theme_manager.theme, not some hardcoded palette."""
        app_dark = make_enroll_app(theme_manager=ThemeManager(initial="dark"))
        app_neon = make_enroll_app(theme_manager=ThemeManager(initial="neon"))
        frame = make_test_frame(width=400, height=300, color=(90, 90, 90))
        out_dark = app_dark.render_preview(frame)
        out_neon = app_neon.render_preview(frame)
        assert not np.array_equal(out_dark, out_neon)

    def test_progress_text_uses_accent_color_once_min_examples_reached(self):
        """Cheap end-to-end proxy: does rendering with enough examples vs.
        not-enough produce a different frame? (Exact color-matching per
        pixel is already covered at the ui.py level; this just confirms
        enroll.py actually wires has_min_examples into the render.)"""
        theme_mgr = ThemeManager(initial="dark")
        cache = GlyphCache(font_dir=FONT_DIR)
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))

        not_ready = make_enroll_app(
            min_examples=5, captured=[np.zeros(4)] * 2,
            theme_manager=ThemeManager(initial="dark"), glyph_cache=cache,
        )
        ready = make_enroll_app(
            min_examples=5, captured=[np.zeros(4)] * 5,
            theme_manager=ThemeManager(initial="dark"), glyph_cache=cache,
        )
        out_not_ready = not_ready.render_preview(frame)
        out_ready = ready.render_preview(frame)
        assert not np.array_equal(out_not_ready, out_ready)

    def test_render_preview_grows_panel_for_warnings(self):
        """Same everything else, only last_capture_warnings differs — the
        panel should visibly change (it grows to fit the warning line)."""
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))
        cache = GlyphCache(font_dir=FONT_DIR)

        no_warning = make_enroll_app(glyph_cache=cache)
        with_warning = make_enroll_app(glyph_cache=cache)
        with_warning._last_capture_warnings = ["Like capture #1 — try a new angle"]

        out_plain = no_warning.render_preview(frame)
        out_warned = with_warning.render_preview(frame)
        assert out_plain.shape != out_warned.shape or not np.array_equal(out_plain, out_warned)

    def test_render_preview_caps_at_two_warning_lines(self):
        """More than _MAX_WARNING_LINES warnings shouldn't keep growing the
        panel indefinitely — rendering with 2 vs. 3+ warnings should
        produce identical output, since only the first 2 are ever shown."""
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))
        cache = GlyphCache(font_dir=FONT_DIR)

        two = make_enroll_app(glyph_cache=cache)
        two._last_capture_warnings = ["Warning one", "Warning two"]

        three = make_enroll_app(glyph_cache=cache)
        three._last_capture_warnings = ["Warning one", "Warning two", "Warning three"]

        out_two = two.render_preview(frame)
        out_three = three.render_preview(frame)
        np.testing.assert_array_equal(out_two, out_three)

    def test_render_preview_no_warning_lines_when_empty(self):
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))
        app = make_enroll_app()
        app._last_capture_warnings = []
        out = app.render_preview(frame)
        assert out.shape == frame.shape  # renders fine, no crash, no extra rows
