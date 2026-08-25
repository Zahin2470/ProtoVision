"""
audio.py — ambient audio + SFX, fail-soft.

Reuses the fail-soft pygame pattern from VisionPuzzle Studio/SignSenseLive:
audio is a nice-to-have, never a hard dependency. If pygame isn't installed,
no audio device exists (a real possibility in some environments, this
sandbox's own test runs included), or a specific sound file is missing,
ProtoVision keeps running silently rather than crashing. This is
deliberately the one part of the visual/audio design system allowed to just
not work sometimes — nothing else in the app depends on audio succeeding.

Two SFX are wired into enroll.py/live.py:
  - enroll_success: plays once when an enrollment session actually finishes
    (finish() succeeds — i.e. min_examples was met and the prototype saved).
  - match_found: plays when a live prediction CROSSES the confidence
    threshold — transitioning from unknown (or a different class) into a
    known match — not on every single frame that happens to already be a
    known match. That distinction lives in live.py's process_frame(), not
    here; this module just knows how to play a named sound.

The bundled .wav files (assets/audio/sfx/, assets/audio/music/) are
procedurally generated sine-wave tones, not sourced audio — see
assets/audio/NOTICE.md. Deliberately simple placeholders, not final polish.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

# Suppress pygame's "Hello from the pygame community" banner, printed at
# import time regardless of whether mixer.init() ever succeeds — noise a
# CLI tool shouldn't produce on every run. Must be set before `import
# pygame`; harmless if pygame ends up unavailable anyway.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
    _PYGAME_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on whether pygame is installed
    pygame = None  # type: ignore[assignment]
    _PYGAME_IMPORT_ERROR = exc

DEFAULT_SFX_DIR = "assets/audio/sfx"
DEFAULT_MUSIC_DIR = "assets/audio/music"

# Logical sound name -> filename. Kept as an explicit mapping (rather than
# assuming name == f"{name}.wav") so callers never need to know the actual
# filenames, same spirit as ui.py's FONT_FILES.
SFX_FILES: Dict[str, str] = {
    "enroll_success": "enroll_success.wav",
    "match_found": "match_found.wav",
}

AMBIENT_FILENAME = "ambient_pad.wav"


class AudioManager:
    """
    Fail-soft audio: named SFX playback plus an optional looping ambient
    pad. Every public method is safe to call regardless of whether audio
    actually works on this machine — construction never raises, and
    playback failures are swallowed rather than propagated, matching the
    "camera loop" philosophy elsewhere in this project of keeping
    hardware-dependent failure contained to a single well-understood spot.
    """

    def __init__(
        self,
        sfx_dir: "str | Path" = DEFAULT_SFX_DIR,
        music_dir: "str | Path" = DEFAULT_MUSIC_DIR,
        enabled: bool = True,
    ):
        self.sfx_dir = Path(sfx_dir)
        self.music_dir = Path(music_dir)
        self.enabled = enabled

        self._sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._available = False
        self._last_error: Optional[str] = None

        if not enabled:
            return
        if pygame is None:
            self._last_error = f"pygame not available: {_PYGAME_IMPORT_ERROR}"
            return
        try:
            pygame.mixer.init()
            self._available = True
        except Exception as exc:  # e.g. no audio device on this machine
            self._last_error = f"pygame.mixer.init() failed: {exc}"
            self._available = False

    @property
    def available(self) -> bool:
        """True only if audio actually initialized successfully. Callers
        don't need to check this before calling play()/start_ambient() —
        those are always safe — but it's here for anything that wants to
        show a small "audio unavailable" indicator rather than silently
        wondering why nothing plays."""
        return self._available

    @property
    def last_error(self) -> Optional[str]:
        """Human-readable reason audio isn't available, if it isn't.
        None when everything's fine (or nothing's been attempted yet)."""
        return self._last_error

    # -- SFX -----------------------------------------------------

    def _get_sound(self, name: str) -> Optional["pygame.mixer.Sound"]:
        if not self._available:
            return None
        if name in self._sounds:
            return self._sounds[name]

        filename = SFX_FILES.get(name)
        if filename is None:
            self._last_error = f"Unknown SFX name {name!r}, available: {list(SFX_FILES)}"
            return None

        path = self.sfx_dir / filename
        if not path.exists():
            self._last_error = f"SFX file not found: '{path}'"
            return None

        try:
            sound = pygame.mixer.Sound(str(path))
        except Exception as exc:
            self._last_error = f"Failed to load '{path}': {exc}"
            return None

        self._sounds[name] = sound
        return sound

    def play(self, name: str) -> bool:
        """Play a named SFX (see SFX_FILES for valid names). Returns True
        if it actually played, False if it silently did nothing — either
        way, this never raises."""
        if not self.enabled:
            return False
        sound = self._get_sound(name)
        if sound is None:
            return False
        try:
            sound.play()
            return True
        except Exception as exc:
            self._last_error = f"Playback failed for {name!r}: {exc}"
            return False

    def play_enroll_success(self) -> bool:
        return self.play("enroll_success")

    def play_match_found(self) -> bool:
        return self.play("match_found")

    # -- ambient loop -----------------------------------------------------

    def start_ambient(self, volume: float = 0.4) -> bool:
        """Start (or restart) the looping ambient pad. Returns True if
        playback actually started."""
        if not self.enabled or not self._available:
            return False
        path = self.music_dir / AMBIENT_FILENAME
        if not path.exists():
            self._last_error = f"Ambient audio file not found: '{path}'"
            return False
        try:
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
            pygame.mixer.music.play(loops=-1)
            return True
        except Exception as exc:
            self._last_error = f"Failed to start ambient audio: {exc}"
            return False

    def stop_ambient(self) -> None:
        if not self._available:
            return
        try:
            pygame.mixer.music.stop()
        except Exception as exc:
            self._last_error = f"Failed to stop ambient audio: {exc}"
