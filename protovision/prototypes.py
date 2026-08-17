"""
prototypes.py — store per-class example embeddings on disk, and match a
live embedding against them (open-set: "unknown" below a similarity threshold).

Storage strategy
-----------------
We keep *every* enrolled example embedding per class (not just a running
mean), because:
  - it lets us do proper k-NN-style matching ("max" mode) as well as the
    simpler centroid ("mean") mode, so we can compare both later,
  - it's what Phase 3's "show which stored example matched closest" needs,
  - re-computing the mean from scratch is trivial and cheap at this scale
    (tens of examples per class, not thousands).

File format is plain JSON (human-inspectable, diff-friendly, no numpy
pickle security footguns): {"embed_dim": 384, "classes": {label: [[...], ...]}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Safe against zero vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


@dataclass
class MatchResult:
    label: Optional[str]          # None means "unknown / no confident match"
    similarity: float             # best similarity found, even if below threshold
    matched_example_index: Optional[int] = None  # which stored example, if mode="max"
    is_known: bool = False        # True iff similarity >= threshold and label is not None


class PrototypeStore:
    """In-memory store of {label: [embedding, embedding, ...]}, with disk I/O."""

    def __init__(self, embed_dim: Optional[int] = None):
        self.embed_dim = embed_dim
        self._classes: Dict[str, List[np.ndarray]] = {}

    # -- building -----------------------------------------------------

    def add_example(self, label: str, embedding: np.ndarray) -> None:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if self.embed_dim is None:
            self.embed_dim = embedding.shape[0]
        elif embedding.shape[0] != self.embed_dim:
            raise ValueError(
                f"Embedding has dim {embedding.shape[0]}, but this store is locked to "
                f"dim {self.embed_dim} (mixing embeddings from different backbones "
                "will silently corrupt matching, so this is a hard error, not a warning)."
            )
        self._classes.setdefault(label, []).append(embedding)

    def add_examples(self, label: str, embeddings) -> None:
        for e in embeddings:
            self.add_example(label, e)

    def remove_class(self, label: str) -> None:
        self._classes.pop(label, None)

    def clear(self) -> None:
        self._classes.clear()

    # -- inspecting -----------------------------------------------------

    def labels(self) -> List[str]:
        return list(self._classes.keys())

    def example_count(self, label: str) -> int:
        return len(self._classes.get(label, []))

    def is_empty(self) -> bool:
        return len(self._classes) == 0

    def class_prototype(self, label: str) -> np.ndarray:
        """Mean (centroid) embedding for a class, re-normalized to unit length."""
        examples = self._classes.get(label)
        if not examples:
            raise KeyError(f"No examples stored for class '{label}'")
        mean = np.mean(np.stack(examples), axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    def all_prototypes(self) -> Dict[str, np.ndarray]:
        return {label: self.class_prototype(label) for label in self._classes}

    # -- matching -----------------------------------------------------

    def best_match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
        mode: str = "mean",
    ) -> MatchResult:
        """
        Compare `embedding` against every stored class and return the best match.

        mode="mean": compare against each class's centroid (fast, smooths out
            noisy individual examples, but can blur distinct example clusters).
        mode="max": compare against every stored example individually and take
            the single best (classic k-NN with k=1; more sensitive to outliers
            but tells you exactly which example matched, useful for debugging
            bad prototypes per the Phase 3 idea in the brief).

        If the store is empty, or the best similarity is below `threshold`,
        returns a MatchResult with label=None and is_known=False — the
        open-set "unknown / new object?" case — while still reporting the
        best similarity found so the caller can show *how close* it was.
        """
        if mode not in ("mean", "max"):
            raise ValueError(f"mode must be 'mean' or 'max', got {mode!r}")

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if self.is_empty():
            return MatchResult(label=None, similarity=float("-inf"), is_known=False)

        best_label: Optional[str] = None
        best_sim = float("-inf")
        best_index: Optional[int] = None

        if mode == "mean":
            for label, proto in self.all_prototypes().items():
                sim = cosine_similarity(embedding, proto)
                if sim > best_sim:
                    best_sim, best_label = sim, label
        else:  # mode == "max"
            for label, examples in self._classes.items():
                for idx, ex in enumerate(examples):
                    sim = cosine_similarity(embedding, ex)
                    if sim > best_sim:
                        best_sim, best_label, best_index = sim, label, idx

        is_known = best_sim >= threshold
        return MatchResult(
            label=best_label if is_known else None,
            similarity=best_sim,
            matched_example_index=best_index if is_known else best_index,
            is_known=is_known,
        )

    # -- persistence -----------------------------------------------------

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embed_dim": self.embed_dim,
            "classes": {
                label: [ex.tolist() for ex in examples]
                for label, examples in self._classes.items()
            },
        }
        # Write to a temp file then replace, so a crash mid-write can't leave
        # a truncated/corrupt prototypes.json behind.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: PathLike) -> "PrototypeStore":
        path = Path(path)
        payload = json.loads(path.read_text())
        store = cls(embed_dim=payload.get("embed_dim"))
        for label, examples in payload.get("classes", {}).items():
            for ex in examples:
                store.add_example(label, np.array(ex, dtype=np.float32))
        return store

    @classmethod
    def load_or_empty(cls, path: PathLike, embed_dim: Optional[int] = None) -> "PrototypeStore":
        """Convenience for CLI entry points: don't blow up if the file doesn't exist yet."""
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls(embed_dim=embed_dim)
