"""
Shared fixtures for tests that need "a backbone" but shouldn't need real
DINOv3 weights — same mock-model approach as test_backbone.py, reused here
so enroll/live tests aren't coupled to real hardware/weights either.
"""

import os
import sys
from pathlib import Path

# Must be set before pygame's mixer is ever initialized anywhere in this
# test session (any test file that constructs a real, non-muted
# AudioManager — including main.py's cmd_enroll/cmd_live tests — will
# trigger this). Tells SDL to use a no-op audio backend instead of probing
# for real hardware, which doesn't exist in this sandbox and could
# otherwise hang or error unpredictably depending on the environment. Set
# here (not in test_audio.py) so it's guaranteed to apply regardless of
# test collection order.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.backbone import DinoV3Backbone, EMBED_DIM


class _MockDinoModel(nn.Module):
    """Same forward_features(x) -> {"x_norm_clstoken": ...} contract as real
    DINOv3, backed by a tiny seeded conv net — deterministic, content-sensitive,
    good enough to test pipeline wiring without real weights."""

    def __init__(self, embed_dim: int = EMBED_DIM, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(3, 16, kernel_size=7, stride=4)
        self.fc = nn.Linear(16, embed_dim)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g))

    def forward_features(self, x: torch.Tensor):
        feats = self.conv(x)
        pooled = feats.mean(dim=[2, 3])
        return {"x_norm_clstoken": self.fc(pooled)}


class CountingBackbone:
    """Wraps a DinoV3Backbone and counts how many times `.embed()` is
    actually called — used to verify frame-skip logic in LiveApp without
    caring about the embedding values themselves."""

    def __init__(self, inner: DinoV3Backbone):
        self._inner = inner
        self.call_count = 0

    def embed(self, image, input_is_bgr: bool = True) -> np.ndarray:
        self.call_count += 1
        return self._inner.embed(image, input_is_bgr=input_is_bgr)


@pytest.fixture
def mock_backbone() -> DinoV3Backbone:
    return DinoV3Backbone(model=_MockDinoModel(seed=42))


@pytest.fixture
def counting_backbone(mock_backbone) -> CountingBackbone:
    return CountingBackbone(mock_backbone)


class FakeCamera:
    """
    Stand-in for protovision.capture.Camera that never touches real hardware.
    Used (via monkeypatch) to test the FULL real __init__ of EnrollApp/LiveApp
    — including logic like label-stripping and default values — without a
    webcam, as a complement to the __new__-bypass pattern used for the rest
    of those classes' methods.
    """

    def __init__(self, *args, **kwargs):
        self.released = False

    def read(self):
        return None

    def release(self):
        self.released = True


def make_test_frame(width: int = 128, height: int = 128, color=(100, 120, 140), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :] = color
    noise = rng.integers(-15, 15, size=base.shape)
    return np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)


def unit_embedding(seed: int, dim: int = 16) -> np.ndarray:
    """A deterministic, seeded unit-length vector — a stand-in embedding
    for tests that need to control exact cosine-similarity relationships
    (e.g. "this must be a known match", "this must be unknown") rather than
    relying on a mock backbone's real-but-arbitrary output crossing a
    threshold by chance."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


class SequenceBackbone:
    """Returns pre-programmed embeddings in order, one per .embed() call —
    lets a test control exactly what embedding (and therefore what
    similarity) each process_frame()/capture_example() call sees, without
    needing pixel-level control over a real/mock backbone's output.
    Repeats the last embedding if called more times than the sequence has
    entries."""

    def __init__(self, embeddings):
        self._embeddings = list(embeddings)
        self._index = 0

    def embed(self, image, input_is_bgr: bool = True) -> np.ndarray:
        idx = min(self._index, len(self._embeddings) - 1)
        self._index += 1
        return self._embeddings[idx]


class SpyAudio:
    """Stand-in for AudioManager that records calls without touching pygame
    at all — for verifying WHEN enroll.py/live.py decide to play a sound,
    independent of whether audio playback itself works (that's audio.py's
    own test file's job)."""

    def __init__(self):
        self.enroll_success_calls = 0
        self.match_found_calls = 0

    def play_enroll_success(self) -> bool:
        self.enroll_success_calls += 1
        return True

    def play_match_found(self) -> bool:
        self.match_found_calls += 1
        return True
