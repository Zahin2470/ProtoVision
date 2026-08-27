"""
benchmark.py — measure DINOv3 embedding latency on THIS machine, and
suggest a `--frame-skip` value for live.py accordingly. Phase 3's fourth
and final enrichment idea from the brief: "a short benchmark script...
so I know what frame rate to expect."

Deliberately separate from backbone.py: loading the model is one concern
(see backbone.py's own module docstring on why loading is split from
per-frame extraction), timing repeated calls to an already-loaded model is
a different one. `run_benchmark()` takes an already-loaded backbone (or
anything duck-typed with the same `.embed()` method) rather than loading
one itself — same "inject the dependency" pattern used throughout this
project for testability. That, plus an injectable `clock` callable instead
of calling `time.perf_counter()` directly, is what makes this fully unit
tested without needing real DINOv3 weights OR any actual wall-clock
sleeping: a fake backbone and a scripted fake clock stand in for real
inference timing, deterministically.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing statistics from run_benchmark(). `latencies_ms` holds one
    entry per timed run (warmup runs are excluded); every other stat is
    derived from it."""

    num_runs: int
    warmup_runs: int
    image_size: int
    device_label: str
    latencies_ms: List[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.latencies_ms) != self.num_runs:
            raise ValueError(
                f"latencies_ms has {len(self.latencies_ms)} entries, "
                f"expected num_runs={self.num_runs}"
            )

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def stdev_ms(self) -> float:
        # A single run has no spread to speak of — 0.0, not a
        # statistics.StatisticsError.
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms)

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms)

    def percentile_ms(self, pct: float) -> float:
        """pct in [0, 100]. Linear interpolation between closest ranks
        (numpy's default) — a rough operational number for "how bad can a
        slow frame get", not a rigorous statistical estimator."""
        if not (0 <= pct <= 100):
            raise ValueError(f"pct must be in [0, 100], got {pct}")
        return float(np.percentile(self.latencies_ms, pct))

    @property
    def inferences_per_second(self) -> float:
        """How many embed() calls this machine can sustain per second,
        back to back, based on the mean latency."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else float("inf")

    def suggested_frame_skip(self, display_fps: float) -> int:
        """
        The smallest frame_skip such that inference can keep up with a
        camera delivering `display_fps` frames/second — i.e. there's
        enough wall-clock time between inference calls (frame_skip frames
        worth, at display_fps) for one embed() call (mean_ms) to actually
        finish before the next one is due. Always at least 1 — live.py's
        own LiveApp already requires frame_skip >= 1.
        """
        if display_fps <= 0:
            raise ValueError(f"display_fps must be positive, got {display_fps}")
        frame_interval_ms = 1000.0 / display_fps
        return max(1, math.ceil(self.mean_ms / frame_interval_ms))


def run_benchmark(
    backbone,
    num_runs: int = 50,
    warmup_runs: int = 5,
    image_size: int = 224,
    device_label: str = "cpu",
    seed: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """
    Time `num_runs` calls to `backbone.embed()` on one synthetic RGB image,
    after `warmup_runs` untimed calls first (lets any one-time costs — lazy
    CUDA/MPS initialization, first-call JIT/caching, page faults on first
    touch of the weight tensors — happen before measuring, so they don't
    skew the numbers toward "how slow is the FIRST call" instead of "how
    slow is a typical call").

    `backbone` only needs an `.embed(image, input_is_bgr=...)` method —
    the same duck-typed interface used everywhere else in this project
    (DinoV3Backbone, the mock model in tests, etc.).

    The synthetic image is random noise, not a real photo — DINOv3's
    compute graph runs the same regardless of pixel content, so timing
    shouldn't meaningfully depend on what's actually pictured, only on
    resolution (`image_size`), which this DOES control.

    `clock` defaults to `time.perf_counter` for real use; tests inject a
    scripted fake instead of real `time.sleep()`-based timing, so this
    function's tests are fast and deterministic rather than slow and
    dependent on actual system scheduling jitter.
    """
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)

    for _ in range(warmup_runs):
        backbone.embed(image)

    latencies_ms: List[float] = []
    for _ in range(num_runs):
        start = clock()
        backbone.embed(image)
        end = clock()
        latencies_ms.append((end - start) * 1000.0)

    return BenchmarkResult(
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        image_size=image_size,
        device_label=device_label,
        latencies_ms=latencies_ms,
    )


def format_benchmark_report(result: BenchmarkResult) -> str:
    """
    Human-readable report for the CLI: timing stats plus concrete
    `--frame-skip` suggestions at a few common camera frame rates, so the
    output is immediately actionable rather than a pile of numbers someone
    has to do their own arithmetic on.
    """
    lines = [
        f"DINOv3 embedding latency — {result.num_runs} run(s) on {result.device_label}, "
        f"{result.image_size}x{result.image_size} image "
        f"({result.warmup_runs} warmup run(s) excluded):",
        f"  mean   : {result.mean_ms:.1f} ms",
        f"  median : {result.median_ms:.1f} ms",
        f"  stdev  : {result.stdev_ms:.1f} ms",
        f"  min    : {result.min_ms:.1f} ms",
        f"  max    : {result.max_ms:.1f} ms",
        f"  p95    : {result.percentile_ms(95):.1f} ms",
        f"  p99    : {result.percentile_ms(99):.1f} ms",
        "",
        f"Sustained rate: ~{result.inferences_per_second:.1f} inference(s)/sec if run every frame.",
        "",
        "Suggested --frame-skip (so inference keeps up with the camera):",
    ]
    for fps in (30, 24, 15):
        lines.append(f"  {fps} fps camera -> --frame-skip {result.suggested_frame_skip(fps)}")
    return "\n".join(lines)
