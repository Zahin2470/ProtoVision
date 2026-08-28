"""
Unit tests for webdemo.py. Runs the REAL PlaceholderBackbone (a small
seeded conv net, not a mock of it) — cheap and fast to actually execute,
same "test the real thing when it's feasible" approach used for Poppins
and pygame elsewhere in this project. `get_backbone()`'s fallback behavior
is tested via monkeypatching `load_default_backbone`, same pattern as
main.py's `_load_backbone_or_exit` tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.backbone import DinoV3NotAvailableError
from protovision.webdemo import (
    PlaceholderBackbone,
    get_backbone,
    image_to_array,
    embed_uploaded_image,
)


def make_pil_image(mode="RGB", color=(200, 0, 0), size=(64, 64)):
    return Image.new(mode, size, color)


# --------------------------------------------------------------------------
# PlaceholderBackbone
# --------------------------------------------------------------------------

class TestPlaceholderBackbone:
    def test_output_shape_matches_embed_dim(self):
        bb = PlaceholderBackbone(embed_dim=64)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert bb.embed(img, input_is_bgr=False).shape == (64,)

    def test_output_is_l2_normalized(self):
        bb = PlaceholderBackbone()
        img = np.random.default_rng(0).integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
        emb = bb.embed(img, input_is_bgr=False)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_deterministic_for_same_image(self):
        bb = PlaceholderBackbone()
        img = np.random.default_rng(1).integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
        e1 = bb.embed(img.copy(), input_is_bgr=False)
        e2 = bb.embed(img.copy(), input_is_bgr=False)
        np.testing.assert_allclose(e1, e2, atol=1e-6)

    def test_two_instances_with_same_seed_produce_same_embedding(self):
        """A fresh PlaceholderBackbone() each Streamlit session should
        behave identically — the seed is what guarantees that, not shared
        in-memory state."""
        img = np.random.default_rng(2).integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
        e1 = PlaceholderBackbone(seed=42).embed(img.copy(), input_is_bgr=False)
        e2 = PlaceholderBackbone(seed=42).embed(img.copy(), input_is_bgr=False)
        np.testing.assert_allclose(e1, e2, atol=1e-6)

    def test_different_seeds_produce_different_embeddings(self):
        img = np.random.default_rng(2).integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
        e1 = PlaceholderBackbone(seed=1).embed(img.copy(), input_is_bgr=False)
        e2 = PlaceholderBackbone(seed=2).embed(img.copy(), input_is_bgr=False)
        assert not np.allclose(e1, e2)

    def test_similar_colored_images_are_more_similar_than_different_ones(self):
        """The actual acceptance check this whole project has used since
        Phase 1's real backbone test: same-ish content should embed more
        similarly than clearly different content — even for an untrained
        placeholder, since it's still a real (if random) function of pixel
        values, not noise."""
        bb = PlaceholderBackbone()

        def colored_image(color, seed):
            rng = np.random.default_rng(seed)
            base = np.zeros((100, 100, 3), dtype=np.uint8)
            base[:, :] = color
            noise = rng.integers(-10, 10, size=base.shape)
            return np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)

        red_a = colored_image((0, 0, 200), seed=1)
        red_b = colored_image((0, 0, 200), seed=2)
        blue = colored_image((200, 0, 0), seed=3)

        e_red_a = bb.embed(red_a, input_is_bgr=False)
        e_red_b = bb.embed(red_b, input_is_bgr=False)
        e_blue = bb.embed(blue, input_is_bgr=False)

        same_sim = float(np.dot(e_red_a, e_red_b))
        diff_sim = float(np.dot(e_red_a, e_blue))
        assert same_sim > diff_sim

    def test_accepts_bgr_and_rgb_flags(self):
        bb = PlaceholderBackbone()
        img = np.random.default_rng(0).integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
        # Should not raise either way — the flag just controls a channel flip.
        bb.embed(img, input_is_bgr=True)
        bb.embed(img, input_is_bgr=False)

    def test_handles_non_square_images(self):
        bb = PlaceholderBackbone()
        img = np.random.default_rng(0).integers(0, 255, size=(60, 140, 3), dtype=np.uint8)
        emb = bb.embed(img, input_is_bgr=False)
        assert emb.shape == (64,)


# --------------------------------------------------------------------------
# get_backbone
# --------------------------------------------------------------------------

class TestGetBackbone:
    def test_returns_real_backbone_when_available(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr("protovision.webdemo.load_default_backbone", lambda **kwargs: sentinel)
        backbone, is_real = get_backbone()
        assert backbone is sentinel
        assert is_real is True

    def test_falls_back_to_placeholder_when_unavailable(self, monkeypatch):
        def raiser(**kwargs):
            raise DinoV3NotAvailableError("no weights configured")

        monkeypatch.setattr("protovision.webdemo.load_default_backbone", raiser)
        backbone, is_real = get_backbone()
        assert isinstance(backbone, PlaceholderBackbone)
        assert is_real is False

    def test_fallback_backbone_is_actually_usable(self, monkeypatch):
        """Not just 'returns something' — the fallback should be able to
        embed a real image without raising, since the demo calls this
        immediately after get_backbone() falls back."""
        def raiser(**kwargs):
            raise DinoV3NotAvailableError("no weights configured")

        monkeypatch.setattr("protovision.webdemo.load_default_backbone", raiser)
        backbone, _ = get_backbone()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        emb = backbone.embed(img, input_is_bgr=False)
        assert emb.shape[0] > 0


# --------------------------------------------------------------------------
# image_to_array / embed_uploaded_image
# --------------------------------------------------------------------------

class TestImageToArray:
    def test_rgb_image_converts_cleanly(self):
        img = make_pil_image(mode="RGB", color=(10, 20, 30))
        arr = image_to_array(img)
        assert arr.shape == (64, 64, 3)
        assert tuple(arr[0, 0]) == (10, 20, 30)

    def test_rgba_image_is_converted_to_rgb(self):
        img = Image.new("RGBA", (32, 32), (10, 20, 30, 128))
        arr = image_to_array(img)
        assert arr.shape == (32, 32, 3)

    def test_grayscale_image_is_converted_to_three_channels(self):
        img = Image.new("L", (32, 32), 128)
        arr = image_to_array(img)
        assert arr.shape == (32, 32, 3)

    def test_palette_image_is_converted(self):
        img = Image.new("P", (32, 32))
        arr = image_to_array(img)
        assert arr.shape == (32, 32, 3)

    def test_output_is_uint8(self):
        img = make_pil_image()
        arr = image_to_array(img)
        assert arr.dtype == np.uint8


class TestEmbedUploadedImage:
    def test_produces_an_embedding(self):
        backbone = PlaceholderBackbone()
        img = make_pil_image(color=(50, 100, 150))
        emb = embed_uploaded_image(backbone, img)
        assert emb.shape == (64,)

    def test_uses_rgb_not_bgr(self):
        """A pure-red PIL image should embed the same whether passed
        through embed_uploaded_image or manually converted + embedded
        with input_is_bgr=False — proving no accidental channel swap."""
        backbone = PlaceholderBackbone()
        img = make_pil_image(color=(200, 10, 10))

        via_helper = embed_uploaded_image(backbone, img)
        manual = backbone.embed(image_to_array(img), input_is_bgr=False)

        np.testing.assert_allclose(via_helper, manual, atol=1e-6)

    def test_different_uploaded_images_give_different_embeddings(self):
        backbone = PlaceholderBackbone()
        red = make_pil_image(color=(200, 0, 0))
        blue = make_pil_image(color=(0, 0, 200))
        assert not np.allclose(
            embed_uploaded_image(backbone, red),
            embed_uploaded_image(backbone, blue),
        )
