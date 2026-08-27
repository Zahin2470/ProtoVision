#!/usr/bin/env python3
"""
main.py — ProtoVision CLI.

Usage:
    python main.py enroll --label mug
    python main.py enroll --label mug --target-examples 10 --min-examples 6
    python main.py live
    python main.py live --threshold 0.6 --match-mode max --frame-skip 3
    python main.py live --mute
    python main.py list
    python main.py export my_pack.json
    python main.py export my_pack.json --label mug --label bottle
    python main.py import my_pack.json --on-conflict merge
    python main.py import my_pack.json --info

'live' also supports on-the-fly teaching: once an object has stayed
unrecognized long enough, the HUD offers "New object? Press N" — pressing N
drops straight into an enrollment session for it (reusing the already-loaded
backbone, not reloading DINOv3), then automatically resumes live recognition.

This file is intentionally thin: argument parsing + wiring only. All real
logic lives in enroll.py/live.py/prototypes.py/backbone.py/audio.py, each
already unit tested on its own. What's tested HERE (see tests/test_main.py)
is the argument parsing and command dispatch — using a fake backbone/app so
no real camera or DINOv3 weights are needed — not the real camera runs
themselves, which need actual hardware.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from protovision.backbone import DinoV3NotAvailableError, load_default_backbone
from protovision.enroll import EnrollApp, EnrollState
from protovision.live import LiveApp, LiveExitReason
from protovision.prototypes import PrototypeStore
from protovision.pack import export_pack, import_pack, load_pack_metadata, PackFormatError, PackIncompatibleError
from protovision.ui import ThemeManager
from protovision.audio import AudioManager

DEFAULT_STORE_PATH = "data/prototypes.json"


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # Shared flags, attached to every subcommand via `parents=`, so they can
    # be given after the subcommand name (`enroll --label mug --device cpu`)
    # rather than only before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--store", default=DEFAULT_STORE_PATH,
        help=f"Path to the prototypes JSON file (default: {DEFAULT_STORE_PATH}).",
    )
    common.add_argument(
        "--repo", default=None,
        help="Path to a local DINOv3 repo clone (default: $PROTOVISION_DINOV3_REPO or ./dinov3_repo).",
    )
    common.add_argument(
        "--weights", default=None,
        help="Path or URL to DINOv3 weights (default: $PROTOVISION_DINOV3_WEIGHTS or ./data/weights/...).",
    )
    common.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Torch device to run the backbone on (default: cpu). See docs/DINOV3_SETUP.md before trying mps.",
    )
    common.add_argument(
        "--mute", action="store_true",
        help="Disable all audio (SFX and ambient). Audio is already fail-soft — this is for "
        "explicitly turning it off, not for working around a broken audio device.",
    )

    parser = argparse.ArgumentParser(
        prog="protovision",
        description="Self-supervised few-shot object recognition with a frozen DINOv3 backbone.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll_p = subparsers.add_parser(
        "enroll", parents=[common], help="Capture example images for a new (or existing) object class."
    )
    enroll_p.add_argument("--label", required=True, help="Class name to enroll, e.g. 'mug'.")
    enroll_p.add_argument(
        "--target-examples", type=int, default=8,
        help="How many example crops to capture before auto-finishing (default: 8).",
    )
    enroll_p.add_argument(
        "--min-examples", type=int, default=5,
        help="Minimum examples required before you're allowed to finish early with Enter (default: 5).",
    )
    enroll_p.add_argument(
        "--box-fraction", type=float, default=0.5,
        help="Guide box size as a fraction of the shorter frame dimension (default: 0.5).",
    )

    live_p = subparsers.add_parser(
        "live", parents=[common], help="Recognize objects live against stored prototypes."
    )
    live_p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Minimum cosine similarity to count as a known match, below this shows 'unknown' (default: 0.5).",
    )
    live_p.add_argument(
        "--match-mode", choices=["mean", "max"], default="mean",
        help="'mean' compares against each class's centroid; 'max' compares against every stored "
        "example individually (k-NN style). Default: mean.",
    )
    live_p.add_argument(
        "--frame-skip", type=int, default=5,
        help="Run the backbone every Nth frame and hold the prediction in between (default: 5). "
        "Use 1 to infer every frame.",
    )
    live_p.add_argument(
        "--box-fraction", type=float, default=0.5,
        help="Guide box size as a fraction of the shorter frame dimension (default: 0.5).",
    )

    subparsers.add_parser(
        "list", parents=[common],
        help="List enrolled classes and example counts. Reads the store only — no camera or backbone needed.",
    )

    # export/import only ever touch the store file — no camera, backbone,
    # or audio involved, so they get a minimal parent (just --store)
    # instead of `common`'s full set of camera/backbone/audio flags.
    store_only = argparse.ArgumentParser(add_help=False)
    store_only.add_argument(
        "--store", default=DEFAULT_STORE_PATH,
        help=f"Path to the prototypes JSON file (default: {DEFAULT_STORE_PATH}).",
    )

    export_p = subparsers.add_parser(
        "export", parents=[store_only],
        help="Export enrolled classes as a shareable recognizer pack file.",
    )
    export_p.add_argument("output", help="Path to write the recognizer pack (e.g. my_objects_pack.json).")
    export_p.add_argument(
        "--label", action="append", dest="labels", default=None,
        help="Class to include (repeatable, e.g. --label mug --label bottle). Default: every enrolled class.",
    )

    import_p = subparsers.add_parser(
        "import", parents=[store_only],
        help="Import a recognizer pack's classes into the current store.",
    )
    import_p.add_argument("pack_path", help="Path to the recognizer pack file to import.")
    import_p.add_argument(
        "--on-conflict", choices=["skip", "merge", "overwrite"], default="skip",
        help="What to do when a class in the pack already exists locally (default: skip).",
    )
    import_p.add_argument(
        "--info", action="store_true",
        help="Preview the pack's contents (classes, model, embed_dim) without importing anything.",
    )

    return parser


# --------------------------------------------------------------------------
# shared helper
# --------------------------------------------------------------------------

def _load_backbone_or_exit(args: argparse.Namespace):
    """Load the real DINOv3 backbone, or print a clear error and exit(1)
    rather than a raw traceback if the repo/weights aren't set up yet."""
    try:
        return load_default_backbone(repo_dir=args.repo, weights_path=args.weights, device=args.device)
    except DinoV3NotAvailableError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    """No camera, no backbone — just reads the store. Fully unit-testable."""
    store = PrototypeStore.load_or_empty(args.store)
    if store.is_empty():
        print(f"No classes enrolled yet in '{args.store}'.")
        return 0
    print(f"Enrolled classes in '{args.store}':")
    for label in sorted(store.labels()):
        print(f"  {label}: {store.example_count(label)} example(s)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """No camera, no backbone — reads the store and writes a pack file.
    Fully unit-testable, same as cmd_list."""
    store = PrototypeStore.load_or_empty(args.store)
    if store.is_empty():
        print(f"Nothing to export — '{args.store}' has no enrolled classes.", file=sys.stderr)
        return 1
    try:
        exported = export_pack(store, args.output, labels=args.labels)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Exported {len(exported)} class(es) to '{args.output}': {', '.join(sorted(exported))}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """No camera, no backbone — reads a pack file and updates the store.
    Fully unit-testable, same as cmd_list/cmd_export."""
    if args.info:
        try:
            meta = load_pack_metadata(args.pack_path)
        except PackFormatError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"Recognizer pack: '{args.pack_path}'")
        print(f"  format version : {meta['pack_format_version']}")
        print(f"  created        : {meta['created_at']}")
        print(f"  model          : {meta['model_name']}")
        print(f"  embed_dim      : {meta['embed_dim']}")
        print(f"  classes ({len(meta['labels'])}): {', '.join(sorted(meta['labels']))}")
        if meta["metadata"]:
            print(f"  metadata       : {meta['metadata']}")
        return 0

    store = PrototypeStore.load_or_empty(args.store)
    try:
        summary = import_pack(store, args.pack_path, on_conflict=args.on_conflict)
    except (PackFormatError, PackIncompatibleError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for warning in summary.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if summary.changed_any_class:
        store.save(args.store)

    if summary.added:
        print(f"Added: {', '.join(sorted(summary.added))}")
    if summary.merged:
        print(f"Merged (examples appended): {', '.join(sorted(summary.merged))}")
    if summary.overwritten:
        print(f"Overwritten: {', '.join(sorted(summary.overwritten))}")
    if summary.skipped:
        print(
            "Skipped (already exists — use --on-conflict merge/overwrite to change this): "
            f"{', '.join(sorted(summary.skipped))}"
        )
    if not summary.changed_any_class:
        print("Nothing changed.")
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    backbone = _load_backbone_or_exit(args)
    store = PrototypeStore.load_or_empty(args.store)
    audio = AudioManager(enabled=not args.mute)
    app = EnrollApp(
        label=args.label,
        backbone=backbone,
        store=store,
        store_path=args.store,
        target_examples=args.target_examples,
        min_examples=args.min_examples,
        box_fraction=args.box_fraction,
        audio=audio,
    )
    print(f"Enrolling '{app.label}' — SPACE=capture  u=undo  Enter=finish early  Esc=cancel  T=theme")
    final_state = app.run()
    if final_state == EnrollState.DONE:
        captured, _ = app.progress
        print(f"Saved {captured} example(s) for '{app.label}' to '{args.store}'.")
        return 0
    print("Enrollment cancelled — nothing saved.")
    return 1


def cmd_live(args: argparse.Namespace) -> int:
    """
    Runs live recognition, and — if the user presses 'N' once the open-set
    "New object? Press N" prompt appears (see live.py's unknown_streak /
    wants_to_teach) — hands off into an enrollment session for whatever's
    in the box right now, then automatically resumes live recognition
    afterward. Backbone, store, audio, and theme are all shared/reused
    across that handoff rather than reloaded: DINOv3 loading is the
    expensive part, and there's no reason to pay for it twice just because
    the user taught the system a new object mid-session.
    """
    backbone = _load_backbone_or_exit(args)
    store = PrototypeStore.load_or_empty(args.store)
    audio = AudioManager(enabled=not args.mute)
    theme_manager = ThemeManager()  # persists across the live<->enroll handoff below

    while True:
        if store.is_empty():
            print(
                f"warning: no classes enrolled yet in '{args.store}' — everything will show as "
                "'unknown'. Run 'python main.py enroll --label <name>' first, or press N once the "
                "open-set prompt appears during live recognition.",
                file=sys.stderr,
            )
        app = LiveApp(
            backbone=backbone,
            store=store,
            threshold=args.threshold,
            match_mode=args.match_mode,
            frame_skip=args.frame_skip,
            box_fraction=args.box_fraction,
            audio=audio,
            theme_manager=theme_manager,
        )
        print(
            "Live recognition running — 'q'/Esc to quit, 'T' to cycle themes, "
            "'N' to teach a new object once prompted."
        )
        audio.start_ambient()
        try:
            reason = app.run()
        finally:
            audio.stop_ambient()

        if reason != LiveExitReason.TEACH_ME_REQUESTED:
            return 0

        label = input("New object detected — what should I call it? (blank to cancel): ").strip()
        if not label:
            print("Cancelled — resuming live recognition.")
            continue

        enroll_app = EnrollApp(
            label=label,
            backbone=backbone,
            store=store,
            store_path=args.store,
            audio=audio,
            theme_manager=theme_manager,
        )
        print(f"Enrolling '{label}' — SPACE=capture  u=undo  Enter=finish early  Esc=cancel  T=theme")
        final_state = enroll_app.run()
        if final_state == EnrollState.DONE:
            captured, _ = enroll_app.progress
            print(f"Saved {captured} example(s) for '{label}'. Resuming live recognition...")
        else:
            print(f"Enrollment of '{label}' cancelled. Resuming live recognition...")
        # loop back around into live recognition


COMMANDS = {
    "enroll": cmd_enroll,
    "live": cmd_live,
    "list": cmd_list,
    "export": cmd_export,
    "import": cmd_import,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
