"""
pack.py — export/import a shareable "recognizer pack": Phase 3's third
enrichment idea.

A recognizer pack bundles a PrototypeStore's classes + raw example
embeddings into ONE self-describing file, so someone's enrolled classes can
be handed to another person (or another machine) and merged into their own
store — not just replace it wholesale.

Format: plain JSON, same philosophy as prototypes.json itself (human
inspectable, diff-friendly, no pickle security footguns). What a pack adds
on top of a bare prototypes.json is metadata a shared file actually needs:
which backbone/embedding dimension produced these vectors (so an
incompatible pack is caught with a clear error before it silently corrupts
matching, not after), a pack format version (so a future format change can
be detected instead of misread), and when it was created.

Deliberately NOT bundled: the actual crops/photos. ProtoVision only ever
stores embeddings, never images (see backbone.py/prototypes.py) — a pack
shares what the system learned, not the pictures that taught it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np

from .backbone import MODEL_NAME as DEFAULT_MODEL_NAME
from .prototypes import PrototypeStore

PACK_FORMAT_VERSION = 1

PathLike = Union[str, Path]
ConflictPolicy = Literal["skip", "merge", "overwrite"]


class PackFormatError(ValueError):
    """Raised when a file doesn't parse as JSON, is missing required
    fields, or declares a pack format version this build doesn't support."""


class PackIncompatibleError(ValueError):
    """
    Raised when a pack's embedding dimension doesn't match the target
    store's. This is a hard error, not a warning — embeddings from a
    different backbone (or even a different ViT variant of the same
    family) simply aren't comparable via cosine similarity, so importing
    them wouldn't just be "probably fine", it would silently corrupt every
    future match against the affected classes.
    """


@dataclass
class ImportSummary:
    """What actually happened when importing a pack — for a caller (CLI or
    otherwise) to report back to the person, rather than importing
    silently and leaving them to guess what changed."""

    added: List[str] = field(default_factory=list)         # brand-new classes
    merged: List[str] = field(default_factory=list)         # existing classes, examples appended
    overwritten: List[str] = field(default_factory=list)    # existing classes, fully replaced
    skipped: List[str] = field(default_factory=list)        # existing classes, left untouched
    warnings: List[str] = field(default_factory=list)       # e.g. model_name mismatch

    @property
    def changed_any_class(self) -> bool:
        return bool(self.added or self.merged or self.overwritten)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export_pack(
    store: PrototypeStore,
    path: PathLike,
    model_name: str = DEFAULT_MODEL_NAME,
    labels: Optional[Sequence[str]] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Write `store`'s classes to `path` as a self-describing recognizer pack.

    `labels`, if given, exports only those classes (e.g. sharing just
    "mug" and "bottle" out of a store with a dozen enrolled classes)
    rather than everything. Defaults to every enrolled class.

    Returns the list of labels actually exported. Raises ValueError on an
    empty store (nothing to pack) and KeyError if `labels` names a class
    the store doesn't have.
    """
    if store.is_empty():
        raise ValueError("Cannot export an empty store — nothing to pack.")

    selected_labels = list(labels) if labels is not None else store.labels()
    missing = [label for label in selected_labels if label not in store.labels()]
    if missing:
        raise KeyError(f"Store has no examples for: {sorted(missing)}")

    classes_payload = {
        label: [example.tolist() for example in store.examples_for_class(label)]
        for label in selected_labels
    }

    payload = {
        "pack_format_version": PACK_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "embed_dim": store.embed_dim,
        "classes": classes_payload,
        "metadata": dict(extra_metadata) if extra_metadata else {},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then replace, same atomicity reasoning as
    # PrototypeStore.save() — a crash mid-write shouldn't leave a
    # truncated, unreadable pack behind.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)

    return selected_labels


# --------------------------------------------------------------------------
# reading (shared by import and metadata preview)
# --------------------------------------------------------------------------

def _read_pack_payload(path: PathLike) -> dict:
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError as exc:
        raise PackFormatError(f"Could not read '{path}': {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PackFormatError(f"'{path}' is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise PackFormatError(f"'{path}' is not a recognizer pack (expected a JSON object).")

    required_fields = {"pack_format_version", "embed_dim", "classes"}
    missing_fields = required_fields - payload.keys()
    if missing_fields:
        raise PackFormatError(
            f"'{path}' is missing required field(s): {sorted(missing_fields)} — "
            "not a valid recognizer pack."
        )

    version = payload["pack_format_version"]
    if version != PACK_FORMAT_VERSION:
        raise PackFormatError(
            f"'{path}' is pack format version {version}, but this build only "
            f"supports version {PACK_FORMAT_VERSION}."
        )

    return payload


def load_pack_metadata(path: PathLike) -> dict:
    """
    Read a pack's header WITHOUT importing anything — lets a caller (e.g.
    the CLI's `import --info`) preview what's inside before committing to
    a merge. Returns a plain dict: pack_format_version, created_at,
    model_name, embed_dim, labels (list), metadata (dict).
    """
    payload = _read_pack_payload(path)
    return {
        "pack_format_version": payload["pack_format_version"],
        "created_at": payload.get("created_at"),
        "model_name": payload.get("model_name"),
        "embed_dim": payload["embed_dim"],
        "labels": list(payload.get("classes", {}).keys()),
        "metadata": payload.get("metadata", {}),
    }


# --------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------

def import_pack(
    store: PrototypeStore,
    path: PathLike,
    on_conflict: ConflictPolicy = "skip",
    expected_model_name: Optional[str] = DEFAULT_MODEL_NAME,
) -> ImportSummary:
    """
    Import a recognizer pack's classes into `store` IN PLACE — merged
    alongside whatever `store` already has, not a wholesale replacement.
    Does not save `store` to disk; that's the caller's job (same division
    of responsibility as PrototypeStore itself: this mutates the in-memory
    store, saving is separate).

    Compatibility: an embedding-dimension mismatch is a hard
    PackIncompatibleError (see that class's docstring for why). A
    model_name mismatch is only a soft warning recorded on the returned
    ImportSummary — dimension is what actually determines whether cosine
    similarity is meaningful, and someone may have legitimately renamed or
    retrained under a different label for the same architecture.

    `on_conflict` controls what happens for each class in the pack that
    ALREADY has examples in `store`:
      - "skip" (default): leave the existing class untouched
      - "merge": append the pack's examples onto the existing class
      - "overwrite": replace the existing class's examples entirely
    Classes not already present in `store` are always added, regardless
    of `on_conflict` — there's no conflict to resolve for a brand-new class.
    """
    if on_conflict not in ("skip", "merge", "overwrite"):
        raise ValueError(f"on_conflict must be 'skip', 'merge', or 'overwrite', got {on_conflict!r}")

    payload = _read_pack_payload(path)
    pack_dim = payload["embed_dim"]

    if store.embed_dim is not None and pack_dim != store.embed_dim:
        raise PackIncompatibleError(
            f"Pack embedding dimension ({pack_dim}) doesn't match this store's "
            f"({store.embed_dim}) — embeddings from different backbones aren't "
            "comparable, so importing would silently corrupt matching."
        )

    summary = ImportSummary()

    pack_model_name = payload.get("model_name")
    if expected_model_name is not None and pack_model_name and pack_model_name != expected_model_name:
        summary.warnings.append(
            f"Pack was created with model '{pack_model_name}', this session expects "
            f"'{expected_model_name}' — same embedding size, but double-check it's "
            "genuinely the same backbone before trusting matches against it."
        )

    existing_labels = set(store.labels())

    for label, raw_examples in payload["classes"].items():
        examples = [np.asarray(example, dtype=np.float32) for example in raw_examples]

        if label not in existing_labels:
            store.add_examples(label, examples)
            summary.added.append(label)
            continue

        if on_conflict == "skip":
            summary.skipped.append(label)
        elif on_conflict == "merge":
            store.add_examples(label, examples)
            summary.merged.append(label)
        else:  # "overwrite"
            store.remove_class(label)
            store.add_examples(label, examples)
            summary.overwritten.append(label)

    return summary
