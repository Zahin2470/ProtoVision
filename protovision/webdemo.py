"""
webdemo.py — pure logic behind streamlit_app.py, kept separate and fully
unit tested the same way everything else "adjacent to something untestable"
in this project is (enroll.py/live.py's `run()` loops, main.py's CLI
dispatch functions): the Streamlit script itself (`st.title()`,
`st.file_uploader()`, ...) only runs meaningfully inside a real Streamlit
session and isn't something pytest can drive directly, so every piece of
actual LOGIC the demo needs lives here instead, where it can be.

Two responsibilities:
  1. `get_backbone()` — tries the real, gated DINOv3 backbone first (exactly
     the same `load_default_backbone()` the CLI uses — no special-casing),
     and falls back to `PlaceholderBackbone` if it isn't configured, so the
     demo's INTERFACE can still be tried immediately without weights.
  2. `image_to_array()` / `embed_uploaded_image()` — turn whatever a
     browser file upload hands over (a PIL Image, in practice) into the
     RGB uint8 array the rest of this project's embedding pipeline expects.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .backbone import DinoV3Backbone, DinoV3NotAvailableError, load_default_backbone


class PlaceholderBackbone:
    """
    Stand-in for the real DINOv3 backbone when it isn't configured in this
    environment (see docs/DINOV3_SETUP.md) — NOT a trained model. A small,
    fixed-seed convolutional feature extractor: deterministic (same image
    in, same embedding out) and genuinely content-sensitive (different
    images produce different embeddings), but without DINOv3's learned
    semantic understanding. This exists purely so `streamlit_app.py`'s
    enroll → recognize workflow can be tried immediately by anyone, with no
    gated download required — recognition quality here will NOT match the
    real thing, and the demo says so on screen whenever this is active.

    Same mock-model pattern already used in tests/conftest.py, promoted
    here because the web demo needs a working fallback for real visitors,
    not just for pytest.
    """

    def __init__(self, seed: int = 42, embed_dim: int = 64, image_size: int = 128):
        import torch
        import torch.nn as nn

        self.embed_dim = embed_dim
        self.image_size = image_size
        self._torch = torch

        generator = torch.Generator().manual_seed(seed)
        self._model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, embed_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        with torch.no_grad():
            for param in self._model.parameters():
                param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
        self._model.eval()

    def embed(self, image: np.ndarray, input_is_bgr: bool = True) -> np.ndarray:
        """Same `.embed()` interface as DinoV3Backbone (see backbone.py),
        so callers (webdemo.py's own helpers, streamlit_app.py) don't need
        to know or care which backbone is actually active."""
        from PIL import Image as PILImage

        rgb = image if not input_is_bgr else image[:, :, ::-1]
        resized = np.asarray(
            PILImage.fromarray(rgb).convert("RGB").resize((self.image_size, self.image_size))
        )
        tensor = (
            self._torch.from_numpy(resized.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        with self._torch.no_grad():
            features = self._model(tensor).flatten(1).squeeze(0).numpy()

        norm = np.linalg.norm(features)
        return (features / norm if norm > 0 else features).astype(np.float32)


def get_backbone(**backbone_kwargs) -> Tuple[object, bool]:
    """
    Try the real DINOv3 backbone first (`load_default_backbone()` —
    identical call to what `main.py`'s CLI uses, so a machine that already
    has real weights set up gets real recognition in the demo too, with no
    separate configuration). Falls back to `PlaceholderBackbone` if it's
    not available.

    Returns (backbone, is_real). `is_real` is what the UI uses to decide
    whether to show the "this is DINOv3" or "this is a placeholder, see
    docs/DINOV3_SETUP.md" banner — the demo should never claim to be doing
    real DINOv3 recognition when it isn't.
    """
    try:
        return load_default_backbone(**backbone_kwargs), True
    except DinoV3NotAvailableError:
        return PlaceholderBackbone(), False


def image_to_array(pil_image) -> np.ndarray:
    """
    Convert a PIL Image (whatever mode a browser upload happens to decode
    to — RGB, RGBA, palette, grayscale, ...) into an RGB uint8 numpy array,
    the format this project's embedding pipeline expects everywhere else.
    """
    return np.asarray(pil_image.convert("RGB"))


def embed_uploaded_image(backbone, pil_image) -> np.ndarray:
    """Convert an uploaded image and embed it in one step — `pil_image` is
    RGB once converted, so `input_is_bgr=False` (this never goes through
    OpenCV/webcam capture, which is the only reason anything else in this
    project defaults to BGR)."""
    rgb = image_to_array(pil_image)
    return backbone.embed(rgb, input_is_bgr=False)
