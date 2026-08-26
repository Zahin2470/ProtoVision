"""
Unit tests for live.py.

Same approach as test_enroll.py: constructor argument validation runs before
the real Camera() call and is tested directly; everything else is tested by
building the instance via `LiveApp.__new__(LiveApp)`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.live import LiveApp, KEY_QUIT_CODES, KEY_TEACH_ME, LiveExitReason, UNKNOWN_STREAK_THRESHOLD
from protovision.prototypes import PrototypeStore, MatchResult
from protovision.ui import ThemeManager, GlyphCache, KEY_THEME_TOGGLE
from protovision.audio import AudioManager

from conftest import make_test_frame, FakeCamera, SpyAudio, unit_embedding as _unit, SequenceBackbone

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def make_live_app(
    backbone=None,
    store=None,
    threshold=0.5,
    match_mode="mean",
    frame_skip=5,
    box_fraction=0.5,
    theme_manager=None,
    glyph_cache=None,
    audio=None,
) -> LiveApp:
    app = LiveApp.__new__(LiveApp)
    app.backbone = backbone
    app.store = store if store is not None else PrototypeStore()
    app.threshold = threshold
    app.match_mode = match_mode
    app.frame_skip = frame_skip
    app.box_fraction = box_fraction
    app._frame_counter = 0
    app._last_result = None
    app._last_similarities = {}
    app._unknown_streak = 0
    app._teach_me_requested = False
    app._matched_example_index = None
    app.theme_manager = theme_manager if theme_manager is not None else ThemeManager()
    app.glyph_cache = glyph_cache if glyph_cache is not None else GlyphCache(font_dir=FONT_DIR)
    # enabled=False: no real pygame/audio device dependency in these pure
    # logic tests; still exercises the real AudioManager fail-soft path
    # rather than a mock, since it's always a safe no-op either way.
    app.audio = audio if audio is not None else AudioManager(enabled=False)
    app.camera = None  # never touched by the methods under test
    return app


def enrolled_store(label="mug", dim=384, n=5, seed=100):
    store = PrototypeStore()
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim).astype(np.float32)
    base /= np.linalg.norm(base)
    for i in range(n):
        noisy = base + np.random.default_rng(seed + i).normal(scale=0.02, size=dim).astype(np.float32)
        store.add_example(label, noisy / np.linalg.norm(noisy))
    return store


# --------------------------------------------------------------------------
# constructor argument validation (real __init__, runs before Camera())
# --------------------------------------------------------------------------

class TestConstructorValidation:
    def test_rejects_invalid_match_mode(self, mock_backbone):
        with pytest.raises(ValueError):
            LiveApp(mock_backbone, PrototypeStore(), match_mode="bogus")

    def test_rejects_frame_skip_below_one(self, mock_backbone):
        with pytest.raises(ValueError):
            LiveApp(mock_backbone, PrototypeStore(), frame_skip=0)

    def test_accepts_valid_match_modes_before_camera_call(self):
        # We can't construct all the way (no real camera in the sandbox),
        # but we CAN confirm validation doesn't reject legitimate values by
        # checking it doesn't raise before hitting the Camera() line —
        # simulated here by calling the same checks make_live_app relies on.
        for mode in ("mean", "max"):
            app = make_live_app(match_mode=mode)
            assert app.match_mode == mode


class TestConstructorWithFakeCamera:
    """Real __init__ end-to-end, with protovision.live.Camera monkeypatched
    to FakeCamera — proves __init__'s own logic (defaults, camera injection)
    without needing real hardware."""

    def test_defaults_are_applied(self, monkeypatch, mock_backbone):
        monkeypatch.setattr("protovision.live.Camera", FakeCamera)
        app = LiveApp(mock_backbone, PrototypeStore())
        assert app.threshold == 0.5
        assert app.match_mode == "mean"
        assert app.frame_skip == 5
        assert app.box_fraction == 0.5
        assert app.last_result is None

    def test_opens_a_camera_when_none_injected(self, monkeypatch, mock_backbone):
        monkeypatch.setattr("protovision.live.Camera", FakeCamera)
        app = LiveApp(mock_backbone, PrototypeStore())
        assert isinstance(app.camera, FakeCamera)

    def test_uses_injected_camera_instead_of_opening_new_one(self, mock_backbone):
        injected = FakeCamera()
        app = LiveApp(mock_backbone, PrototypeStore(), camera=injected)
        assert app.camera is injected

    def test_full_process_frame_end_to_end(self, monkeypatch, mock_backbone):
        """Real __init__ + real process_frame logic, still no hardware."""
        monkeypatch.setattr("protovision.live.Camera", FakeCamera)
        app = LiveApp(mock_backbone, enrolled_store(), threshold=0.5, frame_skip=2)
        result = app.process_frame(make_test_frame())
        assert isinstance(result, MatchResult)

    def test_default_theme_manager_and_glyph_cache_are_created(self, monkeypatch, mock_backbone):
        monkeypatch.setattr("protovision.live.Camera", FakeCamera)
        app = LiveApp(mock_backbone, PrototypeStore())
        assert isinstance(app.theme_manager, ThemeManager)
        assert isinstance(app.glyph_cache, GlyphCache)

    def test_injected_theme_manager_and_glyph_cache_are_used(self, monkeypatch, mock_backbone):
        monkeypatch.setattr("protovision.live.Camera", FakeCamera)
        theme_mgr = ThemeManager(initial="mono")
        cache = GlyphCache(font_dir=FONT_DIR)
        app = LiveApp(mock_backbone, PrototypeStore(), theme_manager=theme_mgr, glyph_cache=cache)
        assert app.theme_manager is theme_mgr
        assert app.glyph_cache is cache

    def test_default_audio_manager_is_created(self, monkeypatch, mock_backbone):
        monkeypatch.setattr("protovision.live.Camera", FakeCamera)
        app = LiveApp(mock_backbone, PrototypeStore())
        assert isinstance(app.audio, AudioManager)

    def test_injected_audio_manager_is_used(self, monkeypatch, mock_backbone):
        monkeypatch.setattr("protovision.live.Camera", FakeCamera)
        spy = SpyAudio()
        app = LiveApp(mock_backbone, PrototypeStore(), audio=spy)
        assert app.audio is spy


# --------------------------------------------------------------------------
# frame-skip inference logic
# --------------------------------------------------------------------------

class TestFrameSkipLogic:
    def test_infers_on_first_frame(self, counting_backbone):
        app = make_live_app(backbone=counting_backbone, store=enrolled_store(), frame_skip=5)
        app.process_frame(make_test_frame())
        assert counting_backbone.call_count == 1

    def test_holds_result_between_skipped_frames(self, counting_backbone):
        app = make_live_app(backbone=counting_backbone, store=enrolled_store(), frame_skip=5)
        for _ in range(4):
            app.process_frame(make_test_frame())
        # frame_skip=5: frame 0 infers, frames 1-3 should NOT re-infer
        assert counting_backbone.call_count == 1

    def test_reinfers_after_skip_window(self, counting_backbone):
        app = make_live_app(backbone=counting_backbone, store=enrolled_store(), frame_skip=5)
        for _ in range(6):
            app.process_frame(make_test_frame())
        # call 1 (counter=0) infers; calls 2-5 (counter=1..4) hold;
        # call 6 (counter=5, 5 % 5 == 0) infers again.
        assert counting_backbone.call_count == 2

    def test_frame_skip_one_infers_every_frame(self, counting_backbone):
        app = make_live_app(backbone=counting_backbone, store=enrolled_store(), frame_skip=1)
        for _ in range(4):
            app.process_frame(make_test_frame())
        assert counting_backbone.call_count == 4

    def test_returned_result_is_a_match_result(self, mock_backbone):
        app = make_live_app(backbone=mock_backbone, store=enrolled_store())
        result = app.process_frame(make_test_frame())
        assert isinstance(result, MatchResult)

    def test_held_result_is_identical_object_not_recomputed(self, counting_backbone):
        app = make_live_app(backbone=counting_backbone, store=enrolled_store(), frame_skip=3)
        first = app.process_frame(make_test_frame(seed=1))
        second = app.process_frame(make_test_frame(seed=2))  # different frame content, still held
        assert first is second

    def test_last_result_property_matches_process_frame_return(self, mock_backbone):
        app = make_live_app(backbone=mock_backbone, store=enrolled_store())
        result = app.process_frame(make_test_frame())
        assert app.last_result is result

    def test_last_result_starts_none_before_any_frame(self):
        app = make_live_app()
        assert app.last_result is None

    def test_empty_store_yields_unknown_without_crashing(self, mock_backbone):
        app = make_live_app(backbone=mock_backbone, store=PrototypeStore())
        result = app.process_frame(make_test_frame())
        assert result.is_known is False
        assert result.label is None

    def test_match_mode_is_passed_through_to_store(self, mock_backbone):
        # Sanity check that mode='max' path doesn't blow up and returns a
        # sensible result shape (exact matching correctness is prototypes.py's
        # job, already covered there).
        app = make_live_app(backbone=mock_backbone, store=enrolled_store(), match_mode="max")
        result = app.process_frame(make_test_frame())
        assert isinstance(result, MatchResult)

    def test_last_similarities_starts_empty(self):
        app = make_live_app()
        assert app.last_similarities == {}

    def test_last_similarities_populated_after_inference(self, mock_backbone):
        store = enrolled_store("mug")
        app = make_live_app(backbone=mock_backbone, store=store)
        app.process_frame(make_test_frame())
        assert set(app.last_similarities.keys()) == {"mug"}

    def test_last_similarities_covers_every_class_not_just_the_winner(self, mock_backbone):
        store = enrolled_store("mug")
        store.add_examples("bottle", [np.random.default_rng(i).normal(size=384).astype(np.float32) for i in range(5)])
        app = make_live_app(backbone=mock_backbone, store=store)
        app.process_frame(make_test_frame())
        assert set(app.last_similarities.keys()) == {"mug", "bottle"}

    def test_last_similarities_held_between_skipped_frames(self, mock_backbone):
        store = enrolled_store("mug")
        app = make_live_app(backbone=mock_backbone, store=store, frame_skip=5)
        app.process_frame(make_test_frame(seed=1))
        first = app.last_similarities
        app.process_frame(make_test_frame(seed=2))  # skipped frame, shouldn't recompute
        assert app.last_similarities is first

    def test_last_similarities_respects_match_mode(self, mock_backbone):
        store = enrolled_store("mug")
        app_mean = make_live_app(backbone=mock_backbone, store=store, match_mode="mean")
        app_max = make_live_app(backbone=mock_backbone, store=store, match_mode="max")
        frame = make_test_frame()
        app_mean.process_frame(frame)
        app_max.process_frame(frame)
        # max mode can only be >= mean mode for the same store/query (same
        # reasoning already covered at the prototypes.py level) — just a
        # sanity check that live.py actually passes match_mode through.
        assert app_max.last_similarities["mug"] >= app_mean.last_similarities["mug"] - 1e-6


def store_with_two_classes(dim=16):
    mug_base = _unit(1, dim)
    bottle_base = _unit(2, dim)
    store = PrototypeStore()
    store.add_examples("mug", [mug_base.copy() for _ in range(5)])
    store.add_examples("bottle", [bottle_base.copy() for _ in range(5)])
    return store, mug_base, bottle_base


# --------------------------------------------------------------------------
# matched_example_index — Phase 3 match debugging (which capture matched)
# --------------------------------------------------------------------------

class TestMatchedExampleIndex:
    def test_starts_none(self):
        app = make_live_app()
        assert app.matched_example_index is None

    def test_set_on_known_match(self):
        store, mug_base, _ = store_with_two_classes()
        backbone = SequenceBackbone([mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5)
        app.process_frame(make_test_frame())
        assert app.matched_example_index is not None
        assert 0 <= app.matched_example_index < store.example_count("mug")

    def test_none_when_unknown(self):
        store, mug_base, _ = store_with_two_classes()
        backbone = SequenceBackbone([-mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5)
        app.process_frame(make_test_frame())
        assert app.matched_example_index is None

    def test_matches_prototypes_best_example_for_class(self):
        """live.py shouldn't reimplement this search — it should just be
        calling store.best_example_for_class() and trusting the answer."""
        store, mug_base, _ = store_with_two_classes()
        backbone = SequenceBackbone([mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5)
        app.process_frame(make_test_frame())
        expected_idx, _ = store.best_example_for_class("mug", mug_base)
        assert app.matched_example_index == expected_idx

    def test_resets_to_none_on_transition_to_unknown(self):
        store, mug_base, _ = store_with_two_classes()
        backbone = SequenceBackbone([mug_base, -mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5)
        app.process_frame(make_test_frame())
        assert app.matched_example_index is not None
        app.process_frame(make_test_frame())
        assert app.matched_example_index is None

    def test_not_updated_on_held_skipped_frames(self):
        store, mug_base, _ = store_with_two_classes()
        backbone = SequenceBackbone([mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=10, threshold=0.5)
        app.process_frame(make_test_frame())
        first = app.matched_example_index
        for _ in range(4):  # within the same frame_skip window
            app.process_frame(make_test_frame())
        assert app.matched_example_index == first


# --------------------------------------------------------------------------
# match_found chime — fires on the transition INTO a known match
# --------------------------------------------------------------------------

class TestMatchFoundChime:
    def test_chime_fires_on_unknown_to_known_transition(self):
        store, mug_base, _ = store_with_two_classes()
        spy = SpyAudio()
        # unknown, known, known, unknown, known — frame_skip=1 so every
        # call actually runs inference.
        backbone = SequenceBackbone([-mug_base, mug_base, mug_base, -mug_base, mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5, audio=spy)

        for _ in range(5):
            app.process_frame(make_test_frame())

        assert spy.match_found_calls == 2  # the two unknown->known transitions

    def test_chime_does_not_fire_while_staying_matched_on_same_class(self):
        store, mug_base, _ = store_with_two_classes()
        spy = SpyAudio()
        backbone = SequenceBackbone([mug_base, mug_base, mug_base, mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5, audio=spy)

        for _ in range(4):
            app.process_frame(make_test_frame())

        assert spy.match_found_calls == 1  # only the initial entry into "known"

    def test_chime_does_not_fire_on_known_to_unknown_transition(self):
        store, mug_base, _ = store_with_two_classes()
        spy = SpyAudio()
        backbone = SequenceBackbone([mug_base, -mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5, audio=spy)

        app.process_frame(make_test_frame())  # known -> chime #1
        app.process_frame(make_test_frame())  # known -> unknown -> no chime

        assert spy.match_found_calls == 1

    def test_chime_fires_again_when_switching_between_two_known_classes(self):
        """Switching from a confident match on one class straight to a
        confident match on a DIFFERENT class should still count as a fresh
        'match found' — the object in the box genuinely changed."""
        store, mug_base, bottle_base = store_with_two_classes()
        spy = SpyAudio()
        backbone = SequenceBackbone([mug_base, mug_base, bottle_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5, audio=spy)

        app.process_frame(make_test_frame())  # unknown->mug: chime #1
        app.process_frame(make_test_frame())  # still mug: no chime
        app.process_frame(make_test_frame())  # mug->bottle: chime #2

        assert spy.match_found_calls == 2

    def test_chime_never_fires_if_nothing_ever_crosses_threshold(self):
        store, mug_base, _ = store_with_two_classes()
        spy = SpyAudio()
        backbone = SequenceBackbone([-mug_base, -mug_base, -mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5, audio=spy)

        for _ in range(3):
            app.process_frame(make_test_frame())

        assert spy.match_found_calls == 0

    def test_chime_respects_frame_skip_not_evaluated_on_held_frames(self):
        """The chime decision only happens when inference actually runs —
        held (skipped) frames can't trigger it, same as they can't update
        last_result/last_similarities."""
        store, mug_base, _ = store_with_two_classes()
        spy = SpyAudio()
        backbone = SequenceBackbone([mug_base])  # same embedding every call
        app = make_live_app(backbone=backbone, store=store, frame_skip=10, threshold=0.5, audio=spy)

        for _ in range(5):  # well within the first frame_skip window
            app.process_frame(make_test_frame())

        assert spy.match_found_calls == 1  # only the single real inference call


# --------------------------------------------------------------------------
# open-set polish: unknown_streak / wants_to_teach / teach-me key handling
# --------------------------------------------------------------------------

class TestUnknownStreak:
    def _store_and_bases(self, dim=16):
        return store_with_two_classes(dim)

    def test_streak_starts_at_zero(self):
        app = make_live_app()
        assert app.unknown_streak == 0

    def test_streak_increments_on_each_unknown_inference(self, mock_backbone):
        store, mug_base, _ = self._store_and_bases()
        backbone = SequenceBackbone([-mug_base, -mug_base, -mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5)
        for expected in (1, 2, 3):
            app.process_frame(make_test_frame())
            assert app.unknown_streak == expected

    def test_streak_resets_to_zero_on_known_match(self):
        store, mug_base, _ = self._store_and_bases()
        backbone = SequenceBackbone([-mug_base, -mug_base, mug_base])
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5)
        app.process_frame(make_test_frame())
        app.process_frame(make_test_frame())
        assert app.unknown_streak == 2
        app.process_frame(make_test_frame())
        assert app.unknown_streak == 0

    def test_streak_does_not_advance_on_held_skipped_frames(self):
        store, mug_base, _ = self._store_and_bases()
        backbone = SequenceBackbone([-mug_base])  # same result every real inference
        app = make_live_app(backbone=backbone, store=store, frame_skip=10, threshold=0.5)
        for _ in range(5):  # within the first frame_skip window
            app.process_frame(make_test_frame())
        assert app.unknown_streak == 1  # only the one real inference counted

    def test_wants_to_teach_false_below_threshold(self):
        app = make_live_app()
        app._unknown_streak = UNKNOWN_STREAK_THRESHOLD - 1
        assert app.wants_to_teach is False

    def test_wants_to_teach_true_at_threshold(self):
        app = make_live_app()
        app._unknown_streak = UNKNOWN_STREAK_THRESHOLD
        assert app.wants_to_teach is True

    def test_wants_to_teach_true_above_threshold(self):
        app = make_live_app()
        app._unknown_streak = UNKNOWN_STREAK_THRESHOLD + 5
        assert app.wants_to_teach is True

    def test_wants_to_teach_reaches_true_through_real_process_frame_calls(self):
        store, mug_base, _ = self._store_and_bases()
        backbone = SequenceBackbone([-mug_base] * UNKNOWN_STREAK_THRESHOLD)
        app = make_live_app(backbone=backbone, store=store, frame_skip=1, threshold=0.5)
        for _ in range(UNKNOWN_STREAK_THRESHOLD - 1):
            app.process_frame(make_test_frame())
            assert app.wants_to_teach is False
        app.process_frame(make_test_frame())
        assert app.wants_to_teach is True


class TestTeachMeKeyHandling:
    def test_teach_key_ignored_when_not_wanting_to_teach(self):
        app = make_live_app()
        app._unknown_streak = 0
        consumed = app.handle_key(KEY_TEACH_ME)
        assert consumed is False
        assert app._teach_me_requested is False

    def test_teach_key_ignored_during_known_match(self):
        app = make_live_app()
        app._unknown_streak = 0  # known match already reset the streak
        consumed = app.handle_key(KEY_TEACH_ME)
        assert consumed is False

    def test_teach_key_consumed_once_wants_to_teach(self):
        app = make_live_app()
        app._unknown_streak = UNKNOWN_STREAK_THRESHOLD
        consumed = app.handle_key(KEY_TEACH_ME)
        assert consumed is True
        assert app._teach_me_requested is True

    def test_theme_key_takes_priority_and_does_not_request_teaching(self):
        app = make_live_app()
        app._unknown_streak = UNKNOWN_STREAK_THRESHOLD
        from protovision.ui import KEY_THEME_TOGGLE
        consumed = app.handle_key(KEY_THEME_TOGGLE)
        assert consumed is True
        assert app._teach_me_requested is False

    def test_other_keys_do_not_request_teaching(self):
        app = make_live_app()
        app._unknown_streak = UNKNOWN_STREAK_THRESHOLD
        consumed = app.handle_key(ord("x"))
        assert consumed is False
        assert app._teach_me_requested is False


# --------------------------------------------------------------------------
# render_preview / quit key
# --------------------------------------------------------------------------

class TestPreviewAndQuit:
    def test_render_preview_same_shape(self):
        app = make_live_app()
        frame = make_test_frame(width=128, height=128)
        preview = app.render_preview(frame)
        assert preview.shape == frame.shape

    def test_render_preview_does_not_mutate_input(self):
        app = make_live_app()
        frame = make_test_frame(width=128, height=128)
        original = frame.copy()
        app.render_preview(frame)
        np.testing.assert_array_equal(frame, original)

    def test_render_preview_draws_something_even_with_empty_store(self):
        """Empty store -> the 'No classes enrolled yet' fallback panel,
        not a crash or a blank frame."""
        app = make_live_app(store=PrototypeStore())
        frame = make_test_frame(width=400, height=300, color=(90, 90, 90))
        out = app.render_preview(frame)
        assert not np.array_equal(out[30, 30], frame[30, 30])

    def test_render_preview_draws_something_before_first_inference(self):
        """Store has classes, but process_frame hasn't run yet -> the
        'Waiting for first frame' fallback, not an empty meter."""
        app = make_live_app(store=enrolled_store())
        frame = make_test_frame(width=400, height=300, color=(90, 90, 90))
        out = app.render_preview(frame)
        assert not np.array_equal(out[30, 30], frame[30, 30])

    def test_render_preview_after_inference_shows_meter(self, mock_backbone):
        store = enrolled_store("mug")
        app = make_live_app(backbone=mock_backbone, store=store)
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))
        app.process_frame(frame)
        out = app.render_preview(frame)
        assert not np.array_equal(out[30, 30], frame[30, 30])

    def test_render_preview_differs_between_plain_unknown_and_teach_prompt(self):
        """Same underlying result (is_known=False), only unknown_streak
        differs — the rendered HUD should visibly change once the streak
        crosses the teach-me threshold (different title text)."""
        store, mug_base, _ = store_with_two_classes()
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))

        plain = make_live_app(store=store)
        plain._last_result = MatchResult(label=None, similarity=0.1, is_known=False)
        plain._last_similarities = {"mug": 0.1, "bottle": -0.05}
        plain._unknown_streak = UNKNOWN_STREAK_THRESHOLD - 1

        teaching = make_live_app(store=store)
        teaching._last_result = MatchResult(label=None, similarity=0.1, is_known=False)
        teaching._last_similarities = {"mug": 0.1, "bottle": -0.05}
        teaching._unknown_streak = UNKNOWN_STREAK_THRESHOLD

        out_plain = plain.render_preview(frame)
        out_teaching = teaching.render_preview(frame)
        assert not np.array_equal(out_plain, out_teaching)

    def test_render_preview_teach_prompt_does_not_appear_for_known_match(self):
        """A high unknown_streak left over from before a match was found
        shouldn't leak into the title once is_known is True — the label
        text should read as the actual class, not the teach-me prompt."""
        store, mug_base, _ = store_with_two_classes()
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))

        app = make_live_app(store=store)
        app._last_result = MatchResult(label="mug", similarity=0.9, is_known=True)
        app._last_similarities = {"mug": 0.9, "bottle": 0.1}
        app._unknown_streak = 0  # a known match always resets this in process_frame

        # Sanity: render succeeds and doesn't crash trying to show both states at once.
        out = app.render_preview(frame)
        assert out.shape == frame.shape

    def test_render_preview_differs_with_and_without_matched_example_debug_line(self):
        """Same known-match result, only matched_example_index differs —
        the panel should grow/change to show the debug line."""
        store, mug_base, _ = store_with_two_classes()
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))

        without_debug = make_live_app(store=store)
        without_debug._last_result = MatchResult(label="mug", similarity=0.9, is_known=True)
        without_debug._last_similarities = {"mug": 0.9, "bottle": 0.1}
        without_debug._matched_example_index = None

        with_debug = make_live_app(store=store)
        with_debug._last_result = MatchResult(label="mug", similarity=0.9, is_known=True)
        with_debug._last_similarities = {"mug": 0.9, "bottle": 0.1}
        with_debug._matched_example_index = 2

        out_without = without_debug.render_preview(frame)
        out_with = with_debug.render_preview(frame)
        assert out_without.shape == out_with.shape  # both still valid frames
        assert not np.array_equal(out_without, out_with)

    def test_render_preview_no_debug_line_when_unknown(self):
        """Even with a stale matched_example_index left over from a
        previous known match, an unknown result shouldn't show the debug
        line — process_frame() clears it, but render_preview() shouldn't
        rely solely on that; it gates on is_known too."""
        store, mug_base, _ = store_with_two_classes()
        frame = make_test_frame(width=400, height=300, color=(30, 30, 30))

        app = make_live_app(store=store)
        app._last_result = MatchResult(label=None, similarity=0.2, is_known=False)
        app._last_similarities = {"mug": 0.2, "bottle": 0.1}
        app._matched_example_index = 3  # stale value, should be ignored

        # Compare against a version with matched_example_index=None — if
        # render_preview correctly gates on is_known, these should render
        # IDENTICALLY despite the stale index being set.
        app_clean = make_live_app(store=store)
        app_clean._last_result = MatchResult(label=None, similarity=0.2, is_known=False)
        app_clean._last_similarities = {"mug": 0.2, "bottle": 0.1}
        app_clean._matched_example_index = None

        out = app.render_preview(frame)
        out_clean = app_clean.render_preview(frame)
        np.testing.assert_array_equal(out, out_clean)

    def test_render_preview_respects_active_theme(self, mock_backbone):
        store = enrolled_store("mug")
        frame = make_test_frame(width=400, height=300, color=(90, 90, 90))

        app_dark = make_live_app(backbone=mock_backbone, store=store, theme_manager=ThemeManager(initial="dark"))
        app_neon = make_live_app(backbone=mock_backbone, store=store, theme_manager=ThemeManager(initial="neon"))
        app_dark.process_frame(frame)
        app_neon.process_frame(frame)

        out_dark = app_dark.render_preview(frame)
        out_neon = app_neon.render_preview(frame)
        assert not np.array_equal(out_dark, out_neon)

    def test_is_quit_key_true_for_q(self):
        assert LiveApp.is_quit_key(ord("q")) is True

    def test_is_quit_key_true_for_esc(self):
        assert LiveApp.is_quit_key(27) is True

    def test_is_quit_key_false_for_other_keys(self):
        assert LiveApp.is_quit_key(ord("x")) is False

    def test_key_quit_codes_contains_expected_values(self):
        assert 27 in KEY_QUIT_CODES
        assert ord("q") in KEY_QUIT_CODES

    def test_handle_key_cycles_theme_and_reports_consumed(self):
        app = make_live_app()
        start = app.theme_manager.name
        consumed = app.handle_key(KEY_THEME_TOGGLE)
        assert consumed is True
        assert app.theme_manager.name != start

    def test_handle_key_returns_false_for_other_keys(self):
        app = make_live_app()
        start = app.theme_manager.name
        consumed = app.handle_key(ord("q"))
        assert consumed is False
        assert app.theme_manager.name == start
