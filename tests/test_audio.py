"""
Unit tests for audio.py.

Two layers of testing here, deliberately:
  1. Fail-soft LOGIC (caching, name dispatch, error handling, volume
     clamping) is tested against a fake pygame double via monkeypatch —
     same pattern as FakeCamera for enroll.py/live.py — so it's fast,
     deterministic, and doesn't depend on any audio subsystem actually
     working in this sandbox.
  2. A small integration check at the bottom loads the REAL bundled .wav
     files through REAL pygame, using SDL's "dummy" audio driver (no
     physical audio device needed — just proves the generated files are
     valid and actually loadable, the same confidence-building step used
     for the real Poppins fonts in test_ui.py).
"""

import os
import sys
from pathlib import Path

import pytest

# SDL_AUDIODRIVER=dummy is set globally in conftest.py (before any test
# file's imports run), so real pygame mixer calls in this file and in
# main.py's cmd_enroll/cmd_live tests never try to probe for real hardware.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.audio import AudioManager, SFX_FILES, AMBIENT_FILENAME

REAL_AUDIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio"


# --------------------------------------------------------------------------
# fake pygame double
# --------------------------------------------------------------------------

class _FakeSound:
    def __init__(self, path, raise_on_play=False):
        self.path = path
        self.play_count = 0
        self._raise_on_play = raise_on_play

    def play(self):
        if self._raise_on_play:
            raise RuntimeError("simulated playback failure")
        self.play_count += 1


class _FakeMusic:
    def __init__(self):
        self.loaded_path = None
        self.volume = None
        self.playing = False
        self.loops = None
        self.stop_count = 0
        self.raise_on_load = False
        self.raise_on_stop = False

    def load(self, path):
        if self.raise_on_load:
            raise RuntimeError("simulated load failure")
        self.loaded_path = path

    def set_volume(self, v):
        self.volume = v

    def play(self, loops=0):
        self.playing = True
        self.loops = loops

    def stop(self):
        if self.raise_on_stop:
            raise RuntimeError("simulated stop failure")
        self.stop_count += 1
        self.playing = False


class _FakeMixer:
    def __init__(self, fail_init=False, fail_sound_for=None, raise_on_sound_play_for=None):
        self.fail_init = fail_init
        self.initialized = False
        self.fail_sound_for = fail_sound_for or set()
        self.raise_on_sound_play_for = raise_on_sound_play_for or set()
        self.sound_load_calls = []
        self.music = _FakeMusic()

    def init(self):
        if self.fail_init:
            raise RuntimeError("simulated: no audio device")
        self.initialized = True

    def Sound(self, path):
        self.sound_load_calls.append(path)
        name = Path(path).name
        if name in self.fail_sound_for:
            raise RuntimeError(f"simulated: cannot load {name}")
        return _FakeSound(path, raise_on_play=(name in self.raise_on_sound_play_for))


class _FakePygame:
    def __init__(self, **mixer_kwargs):
        self.mixer = _FakeMixer(**mixer_kwargs)


# --------------------------------------------------------------------------
# construction / fail-soft behavior
# --------------------------------------------------------------------------

class TestConstruction:
    def test_disabled_skips_init_entirely(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        mgr = AudioManager(sfx_dir=tmp_path, enabled=False)
        assert mgr.available is False
        assert fake.mixer.initialized is False

    def test_pygame_not_installed(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", None)
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.available is False
        assert "not available" in mgr.last_error

    def test_mixer_init_failure_is_caught(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", _FakePygame(fail_init=True))
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.available is False
        assert "failed" in mgr.last_error

    def test_successful_init(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", _FakePygame())
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.available is True
        assert mgr.last_error is None

    def test_construction_never_raises_even_when_everything_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", _FakePygame(fail_init=True))
        try:
            AudioManager(sfx_dir=tmp_path)
        except Exception as exc:  # pragma: no cover - the whole point is this doesn't happen
            pytest.fail(f"AudioManager() raised unexpectedly: {exc}")


# --------------------------------------------------------------------------
# SFX playback
# --------------------------------------------------------------------------

class TestPlay:
    def _touch(self, tmp_path, filename):
        (tmp_path / filename).write_bytes(b"not a real wav, but Sound() is faked anyway")

    def test_play_false_when_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", None)
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.play("enroll_success") is False

    def test_play_false_when_disabled(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch(tmp_path, SFX_FILES["enroll_success"])
        mgr = AudioManager(sfx_dir=tmp_path, enabled=False)
        assert mgr.play("enroll_success") is False

    def test_play_unknown_name_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", _FakePygame())
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.play("not_a_real_sound") is False

    def test_play_missing_file_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", _FakePygame())
        mgr = AudioManager(sfx_dir=tmp_path)  # tmp_path is empty, no .wav files
        assert mgr.play("enroll_success") is False

    def test_play_success_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", _FakePygame())
        self._touch(tmp_path, SFX_FILES["enroll_success"])
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.play("enroll_success") is True

    def test_play_actually_calls_sound_play(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch(tmp_path, SFX_FILES["match_found"])
        mgr = AudioManager(sfx_dir=tmp_path)
        mgr.play("match_found")
        sound = mgr._sounds["match_found"]
        assert sound.play_count == 1

    def test_play_caches_sound_after_first_load(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch(tmp_path, SFX_FILES["enroll_success"])
        mgr = AudioManager(sfx_dir=tmp_path)
        mgr.play("enroll_success")
        mgr.play("enroll_success")
        mgr.play("enroll_success")
        assert len(fake.mixer.sound_load_calls) == 1  # loaded once, played 3 times

    def test_sound_load_exception_is_caught(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "protovision.audio.pygame",
            _FakePygame(fail_sound_for={SFX_FILES["enroll_success"]}),
        )
        self._touch(tmp_path, SFX_FILES["enroll_success"])
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.play("enroll_success") is False
        assert "Failed to load" in mgr.last_error

    def test_sound_play_exception_is_caught(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "protovision.audio.pygame",
            _FakePygame(raise_on_sound_play_for={SFX_FILES["enroll_success"]}),
        )
        self._touch(tmp_path, SFX_FILES["enroll_success"])
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.play("enroll_success") is False
        assert "Playback failed" in mgr.last_error

    def test_play_enroll_success_convenience_method(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch(tmp_path, SFX_FILES["enroll_success"])
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.play_enroll_success() is True
        assert mgr._sounds["enroll_success"].play_count == 1

    def test_play_match_found_convenience_method(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch(tmp_path, SFX_FILES["match_found"])
        mgr = AudioManager(sfx_dir=tmp_path)
        assert mgr.play_match_found() is True
        assert mgr._sounds["match_found"].play_count == 1

    def test_enroll_success_and_match_found_are_independent_sounds(self, monkeypatch, tmp_path):
        """The two SFX must be genuinely distinct files, not the same
        sound played twice — that's the whole point of having two."""
        assert SFX_FILES["enroll_success"] != SFX_FILES["match_found"]


# --------------------------------------------------------------------------
# ambient loop
# --------------------------------------------------------------------------

class TestAmbient:
    def _touch_music(self, tmp_path):
        (tmp_path / AMBIENT_FILENAME).write_bytes(b"not a real wav, but music.load() is faked anyway")

    def test_start_ambient_false_when_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", None)
        mgr = AudioManager(music_dir=tmp_path)
        assert mgr.start_ambient() is False

    def test_start_ambient_false_when_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", _FakePygame())
        mgr = AudioManager(music_dir=tmp_path)
        assert mgr.start_ambient() is False

    def test_start_ambient_success(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch_music(tmp_path)
        mgr = AudioManager(music_dir=tmp_path)
        assert mgr.start_ambient(volume=0.5) is True
        assert fake.mixer.music.playing is True
        assert fake.mixer.music.loops == -1  # loop forever
        assert fake.mixer.music.volume == 0.5

    def test_start_ambient_clamps_volume_above_one(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch_music(tmp_path)
        mgr = AudioManager(music_dir=tmp_path)
        mgr.start_ambient(volume=5.0)
        assert fake.mixer.music.volume == 1.0

    def test_start_ambient_clamps_volume_below_zero(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch_music(tmp_path)
        mgr = AudioManager(music_dir=tmp_path)
        mgr.start_ambient(volume=-2.0)
        assert fake.mixer.music.volume == 0.0

    def test_start_ambient_load_exception_is_caught(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        fake.mixer.music.raise_on_load = True
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch_music(tmp_path)
        mgr = AudioManager(music_dir=tmp_path)
        assert mgr.start_ambient() is False
        assert "Failed to start ambient" in mgr.last_error

    def test_stop_ambient_calls_music_stop(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch_music(tmp_path)
        mgr = AudioManager(music_dir=tmp_path)
        mgr.start_ambient()
        mgr.stop_ambient()
        assert fake.mixer.music.stop_count == 1
        assert fake.mixer.music.playing is False

    def test_stop_ambient_when_unavailable_is_a_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr("protovision.audio.pygame", None)
        mgr = AudioManager(music_dir=tmp_path)
        mgr.stop_ambient()  # should not raise

    def test_stop_ambient_exception_is_caught(self, monkeypatch, tmp_path):
        fake = _FakePygame()
        monkeypatch.setattr("protovision.audio.pygame", fake)
        self._touch_music(tmp_path)
        mgr = AudioManager(music_dir=tmp_path)
        mgr.start_ambient()
        fake.mixer.music.raise_on_stop = True
        mgr.stop_ambient()  # should not raise, error captured instead
        assert "Failed to stop ambient" in mgr.last_error


# --------------------------------------------------------------------------
# real pygame + real bundled files, dummy SDL driver (no hardware needed)
# --------------------------------------------------------------------------

class TestRealAudioIntegration:
    """Loads the ACTUAL bundled .wav files through REAL pygame — not a
    fake — using SDL's dummy audio driver. Proves the generated audio
    assets are genuinely valid, playable files, the same kind of
    confidence check test_ui.py does against the real Poppins fonts."""

    def test_real_pygame_mixer_initializes(self):
        mgr = AudioManager(sfx_dir=REAL_AUDIO_DIR / "sfx", music_dir=REAL_AUDIO_DIR / "music")
        assert mgr.available is True, f"pygame mixer failed to init: {mgr.last_error}"

    def test_real_enroll_success_sfx_loads_and_plays(self):
        mgr = AudioManager(sfx_dir=REAL_AUDIO_DIR / "sfx", music_dir=REAL_AUDIO_DIR / "music")
        assert mgr.play_enroll_success() is True, mgr.last_error

    def test_real_match_found_sfx_loads_and_plays(self):
        mgr = AudioManager(sfx_dir=REAL_AUDIO_DIR / "sfx", music_dir=REAL_AUDIO_DIR / "music")
        assert mgr.play_match_found() is True, mgr.last_error

    def test_real_ambient_pad_loads_and_plays(self):
        mgr = AudioManager(sfx_dir=REAL_AUDIO_DIR / "sfx", music_dir=REAL_AUDIO_DIR / "music")
        assert mgr.start_ambient() is True, mgr.last_error
        mgr.stop_ambient()

    def test_real_sfx_files_actually_exist_on_disk(self):
        for filename in SFX_FILES.values():
            assert (REAL_AUDIO_DIR / "sfx" / filename).exists()
        assert (REAL_AUDIO_DIR / "music" / AMBIENT_FILENAME).exists()
