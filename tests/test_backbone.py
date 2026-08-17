"""
Unit tests for backbone.py.

IMPORTANT — what these tests DO and DON'T prove:
  - They DO verify the preprocessing pipeline (resize/normalize), the
    BGR->RGB handling, the L2-normalization, the output shape/dtype
    contract, and that the wiring from image -> transform -> model ->
    embedding is correct.
  - They do NOT verify real DINOv3's semantic quality (i.e. that it
    actually produces embeddings where the same real-world object is
    closer than a different one — that requires the actual gated
    weights, which aren't available in this sandbox).

To stand in for the real model, `MockDinoBackbone` below implements the same
interface real DINOv3 exposes (`forward_features(x) -> {"x_norm_clstoken": ...}`)
but is a small, fixed, seeded random conv net — deterministic and fast, and
critically, DIFFERENT/consistent images map to different/consistent outputs,
which is enough to test the pipeline logic end-to-end.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.backbone import (
    DinoV3Backbone,
    build_transform,
    cosine_similarity,
    EMBED_DIM,
    INPUT_SIZE,
    _to_rgb_uint8,
)


class MockDinoBackbone(nn.Module):
    """
    Same call contract as the real DINOv3 ViT: forward_features(x) returns a
    dict with "x_norm_clstoken" of shape (B, EMBED_DIM). Internally it's just
    a tiny seeded conv -> global average pool -> linear, which is enough to
    be a deterministic, content-sensitive function of the input image.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(3, 16, kernel_size=7, stride=4)
        self.fc = nn.Linear(16, embed_dim)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g))

    def forward_features(self, x: torch.Tensor):
        feats = self.conv(x)
        pooled = feats.mean(dim=[2, 3])  # global average pool
        cls = self.fc(pooled)
        return {"x_norm_clstoken": cls}


def make_test_image(color: tuple, size: int = 64, noise_seed: int = 0) -> np.ndarray:
    """Solid-color-ish synthetic BGR image with a little seeded texture, so
    it's not a degenerate flat input the conv could trivially collapse."""
    rng = np.random.default_rng(noise_seed)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[:, :] = color
    noise = rng.integers(-15, 15, size=base.shape)
    img = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def mock_backbone():
    return DinoV3Backbone(model=MockDinoBackbone(seed=42))


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

class TestBuildTransform:
    def test_default_size_matches_patch_multiple(self):
        assert INPUT_SIZE % 16 == 0

    def test_rejects_non_multiple_of_16(self):
        with pytest.raises(ValueError):
            build_transform(resize_size=100)

    def test_accepts_multiple_of_16(self):
        transform = build_transform(resize_size=32)
        assert transform is not None

    def test_output_tensor_shape_and_range(self):
        transform = build_transform(resize_size=64)
        img = make_test_image((10, 20, 30), size=64)
        tensor = transform(img)
        assert tensor.shape == (3, 64, 64)
        # normalized (not raw 0-255), should have both positive and negative values typically
        assert tensor.dtype == torch.float32


class TestToRgbUint8:
    def test_bgr_to_rgb_swaps_channels(self):
        img_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = 10   # B
        img_bgr[:, :, 1] = 20   # G
        img_bgr[:, :, 2] = 30   # R
        rgb = _to_rgb_uint8(img_bgr, input_is_bgr=True)
        assert rgb[0, 0, 0] == 30  # R
        assert rgb[0, 0, 1] == 20  # G
        assert rgb[0, 0, 2] == 10  # B

    def test_already_rgb_not_swapped(self):
        img_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        img_rgb[:, :, 0] = 99
        rgb = _to_rgb_uint8(img_rgb, input_is_bgr=False)
        assert rgb[0, 0, 0] == 99

    def test_grayscale_expanded_to_three_channels(self):
        gray = np.full((4, 4), 128, dtype=np.uint8)
        rgb = _to_rgb_uint8(gray, input_is_bgr=False)
        assert rgb.shape == (4, 4, 3)

    def test_rejects_wrong_shape(self):
        bad = np.zeros((4, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            _to_rgb_uint8(bad, input_is_bgr=True)

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            _to_rgb_uint8("not an image", input_is_bgr=True)

    def test_pil_image_input(self):
        from PIL import Image
        pil_img = Image.new("RGB", (8, 8), color=(1, 2, 3))
        rgb = _to_rgb_uint8(pil_img, input_is_bgr=True)  # flag ignored for PIL, already RGB
        assert rgb.shape == (8, 8, 3)
        assert tuple(rgb[0, 0]) == (1, 2, 3)


# --------------------------------------------------------------------------
# DinoV3Backbone.embed — contract tests against the mock model
# --------------------------------------------------------------------------

class TestEmbedContract:
    def test_output_shape(self, mock_backbone):
        img = make_test_image((100, 50, 200))
        emb = mock_backbone.embed(img)
        assert emb.shape == (EMBED_DIM,)

    def test_output_dtype(self, mock_backbone):
        img = make_test_image((100, 50, 200))
        emb = mock_backbone.embed(img)
        assert emb.dtype == np.float32

    def test_output_is_l2_normalized(self, mock_backbone):
        img = make_test_image((100, 50, 200))
        emb = mock_backbone.embed(img)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_deterministic_for_same_image(self, mock_backbone):
        img = make_test_image((100, 50, 200), noise_seed=7)
        emb1 = mock_backbone.embed(img.copy())
        emb2 = mock_backbone.embed(img.copy())
        np.testing.assert_allclose(emb1, emb2, atol=1e-6)

    def test_two_crops_of_same_color_are_more_similar_than_different_colors(self, mock_backbone):
        """
        This mirrors the acceptance test described in the brief: two crops of
        "the same object" (here, stand-ins: two independently-noised crops of
        the same base color) should be MORE similar than two crops of
        genuinely different colors — proving the embed() pipeline preserves
        and doesn't scramble image content.
        Note: this validates the *pipeline*, not real DINOv3's semantic
        quality (see module docstring).
        """
        red_a = make_test_image((0, 0, 220), noise_seed=1)
        red_b = make_test_image((0, 0, 220), noise_seed=2)
        blue = make_test_image((220, 0, 0), noise_seed=3)

        emb_red_a = mock_backbone.embed(red_a)
        emb_red_b = mock_backbone.embed(red_b)
        emb_blue = mock_backbone.embed(blue)

        same_object_sim = cosine_similarity(emb_red_a, emb_red_b)
        different_object_sim = cosine_similarity(emb_red_a, emb_blue)

        assert same_object_sim > different_object_sim

    def test_accepts_rgb_ndarray_when_flagged(self, mock_backbone):
        img_rgb = make_test_image((5, 6, 7))
        emb = mock_backbone.embed(img_rgb, input_is_bgr=False)
        assert emb.shape == (EMBED_DIM,)

    def test_accepts_pil_image(self, mock_backbone):
        from PIL import Image
        pil_img = Image.fromarray(make_test_image((1, 2, 3)))
        emb = mock_backbone.embed(pil_img)
        assert emb.shape == (EMBED_DIM,)

    def test_embed_batch_stacks_correctly(self, mock_backbone):
        imgs = [make_test_image((i * 10, i * 5, i * 2), noise_seed=i) for i in range(4)]
        embs = mock_backbone.embed_batch(imgs)
        assert embs.shape == (4, EMBED_DIM)

    def test_model_is_set_to_eval_mode(self, mock_backbone):
        assert mock_backbone.model.training is False

    def test_no_gradients_leak_from_embed(self, mock_backbone):
        img = make_test_image((1, 2, 3))
        mock_backbone.embed(img)
        for p in mock_backbone.model.parameters():
            assert p.grad is None


class TestCosineSimilarityHelper:
    def test_identical_embeddings(self):
        v = np.random.default_rng(0).normal(size=384).astype(np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)
