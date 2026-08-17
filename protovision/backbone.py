"""
backbone.py — Frozen DINOv3 ViT-S/16 backbone loading + embedding extraction.

Design intent
-------------
Loading the model (slow, once) is kept completely separate from extracting an
embedding from a frame (fast, every call). `DinoV3Backbone` is a thin wrapper
around an already-constructed torch model, so it can be unit-tested with a
fake model that has the same interface — we don't need real DINOv3 weights
to verify the preprocessing math, the L2-normalization, or the output
contract (shape, dtype).

Real DINOv3 weights are gated by Meta (request-access form -> emailed
checkpoint URLs) and are NOT reachable from this sandbox's network allowlist,
so end-to-end verification with the *real* backbone has to happen on your
machine. See `docs/DINOV3_SETUP.md` for the exact steps.

Model facts (confirmed against the official facebookresearch/dinov3 repo,
hubconf.py / dinov3/hub/backbones.py, Aug 2026):
  - Entry point name : "dinov3_vits16"
  - Embedding dim     : 384 (CLS token, key "x_norm_clstoken" from forward_features)
  - Patch size        : 16 (input H/W should be multiples of 16; default 224x224)
  - Pretraining data  : LVD-1689M (web images) -> standard ImageNet normalization
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        (NOTE: the SAT-493M satellite checkpoints use different
                        stats — irrelevant here, we're using the web-pretrained one)
  - Expected weights file name (LVD1689M, vits16): matches the hash baked into
    the hub entrypoint itself (hash="08c60483"), i.e.
    dinov3_vits16_pretrain_lvd1689m-08c60483.pth
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_NAME = "dinov3_vits16"
EMBED_DIM = 384
INPUT_SIZE = 224  # multiple of patch_size(16), matches the model's pretraining resolution

# LVD-1689M (web-image) normalization stats — NOT the satellite ones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Where to look by default; overridable via env vars so nothing is hardcoded
# to one machine.
DEFAULT_REPO_DIR = os.environ.get("PROTOVISION_DINOV3_REPO", "./dinov3_repo")
DEFAULT_WEIGHTS_PATH = os.environ.get(
    "PROTOVISION_DINOV3_WEIGHTS",
    "./data/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
)


class DinoV3NotAvailableError(RuntimeError):
    """Raised when the real DINOv3 repo/weights aren't present locally."""


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def build_transform(resize_size: int = INPUT_SIZE) -> v2.Compose:
    """
    Build the exact preprocessing pipeline DINOv3 (LVD-1689M, web-pretrained
    checkpoints) expects: resize -> float [0,1] -> ImageNet normalize.

    resize_size must be a multiple of 16 (the ViT-S/16 patch size) — we assert
    this rather than silently rounding, since a silent mismatch would corrupt
    every embedding without throwing an obvious error.
    """
    if resize_size % 16 != 0:
        raise ValueError(f"resize_size must be a multiple of 16 (patch size), got {resize_size}")
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _to_rgb_uint8(image: Union[np.ndarray, "PIL.Image.Image"], input_is_bgr: bool) -> np.ndarray:
    """
    Normalize any accepted input (OpenCV BGR ndarray, RGB ndarray, or PIL
    Image) down to an RGB uint8 HxWx3 ndarray, which torchvision's v2.ToImage
    can consume directly.
    """
    # Local import so PIL isn't a hard dependency for callers that only ever
    # pass ndarrays (e.g. everything coming out of OpenCV).
    try:
        from PIL import Image as PILImage
    except ImportError:  # pragma: no cover - PIL ships with torchvision anyway
        PILImage = None

    if PILImage is not None and isinstance(image, PILImage.Image):
        return np.array(image.convert("RGB"))

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if image.ndim == 2:  # grayscale -> RGB
        image = np.stack([image] * 3, axis=-1)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 image, got shape {image.shape}")

    if image.dtype != np.uint8:
        # Guide-box crops should already be uint8 straight off the camera;
        # this is a safety net, not the expected path.
        image = image.astype(np.uint8)

    if input_is_bgr:
        image = image[:, :, ::-1]  # BGR -> RGB, no copy-heavy cv2 dependency needed here

    return np.ascontiguousarray(image)


# --------------------------------------------------------------------------
# Backbone wrapper
# --------------------------------------------------------------------------

@dataclass
class DinoV3Backbone:
    """
    Thin, testable wrapper around a loaded DINOv3 (or DINOv3-shaped) model.

    `model` just needs a `.forward_features(tensor)` method returning a dict
    containing `"x_norm_clstoken"` of shape (B, EMBED_DIM) — that's the real
    DINOv3 ViT interface, and it's also what the mock model in tests
    implements, which is what makes this class testable without real weights.
    """

    model: nn.Module
    device: str = "cpu"
    transform: v2.Compose = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.transform is None:
            self.transform = build_transform()
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def embed(self, image: Union[np.ndarray, "PIL.Image.Image"], input_is_bgr: bool = True) -> np.ndarray:
        """
        Extract an L2-normalized embedding vector from a single image.

        Parameters
        ----------
        image : np.ndarray (HxWx3, BGR by default — i.e. straight from
            OpenCV/cv2) or a PIL.Image.
        input_is_bgr : set False if you're passing an already-RGB ndarray.

        Returns
        -------
        np.ndarray, shape (EMBED_DIM,), dtype float32, unit L2 norm.
        """
        rgb = _to_rgb_uint8(image, input_is_bgr=input_is_bgr)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        features = self.model.forward_features(tensor)
        cls_token = features["x_norm_clstoken"]  # (1, EMBED_DIM)

        embedding = cls_token.squeeze(0).float().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def embed_batch(self, images, input_is_bgr: bool = True) -> np.ndarray:
        """Convenience loop wrapper (not batched on the GPU/CPU — fine for
        enrollment where you process 5-10 images once, not a live loop)."""
        return np.stack([self.embed(img, input_is_bgr=input_is_bgr) for img in images])


# --------------------------------------------------------------------------
# Real-model loading (requires the cloned repo + gated weights on disk)
# --------------------------------------------------------------------------

def load_default_backbone(
    repo_dir: Optional[str] = None,
    weights_path: Optional[str] = None,
    device: str = "cpu",
) -> DinoV3Backbone:
    """
    Load the real, frozen DINOv3 ViT-S/16 backbone via torch.hub against a
    local clone of facebookresearch/dinov3, with a locally-downloaded (gated)
    checkpoint. See docs/DINOV3_SETUP.md for how to get both.

    This is intentionally the ONLY function in this module that touches disk
    for the real model — everything else (DinoV3Backbone, build_transform)
    works with any injected model, which is how we test the rest of the
    pipeline without needing the actual weights.
    """
    repo_dir = repo_dir or DEFAULT_REPO_DIR
    weights_path = weights_path or DEFAULT_WEIGHTS_PATH

    repo_path = Path(repo_dir)
    if not repo_path.exists():
        raise DinoV3NotAvailableError(
            f"DINOv3 repo not found at '{repo_dir}'. Clone it first:\n"
            f"  git clone https://github.com/facebookresearch/dinov3 {repo_dir}\n"
            f"(See docs/DINOV3_SETUP.md.)"
        )

    weights_path_obj = Path(weights_path)
    is_url = str(weights_path).startswith("http")
    if not is_url and not weights_path_obj.exists():
        raise DinoV3NotAvailableError(
            f"DINOv3 weights not found at '{weights_path}'.\n"
            "Weights are gated by Meta — request access, then either point this at\n"
            "the emailed URL directly, or `wget` it locally first. "
            "See docs/DINOV3_SETUP.md."
        )

    model = torch.hub.load(
        repo_dir,
        MODEL_NAME,
        source="local",
        weights=str(weights_path),
    )
    return DinoV3Backbone(model=model, device=device)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two (already or not-yet normalized) vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))
