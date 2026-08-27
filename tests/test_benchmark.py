"""
Unit tests for benchmark.py.

Timing is made fully deterministic via an injectable `clock` callable
(rather than real time.sleep()-based measurement) — a scripted fake clock
returns exact, known elapsed times per call, so these tests are fast and
never flaky due to real system scheduling jitter, while still exercising
the real timing/statistics code path end to end.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.benchmark import BenchmarkResult, run_benchmark, format_benchmark_report


class NoopBackbone:
    """A backbone whose embed() does no real work — timing comes entirely
    from the injected fake clock, not from anything this actually does."""

    def embed(self, image, input_is_bgr: bool = True) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


def make_fake_clock(elapsed_ms_per_call):
    """
    Returns a zero-argument callable that, called in (start, end) pairs,
    yields elapsed times matching `elapsed_ms_per_call` (in milliseconds),
    one value consumed per pair. run_benchmark() calls the clock exactly
    twice per timed run (start, end) and never during warmup runs.
    """
    values = []
    for elapsed_ms in elapsed_ms_per_call:
        values.append(0.0)
        values.append(elapsed_ms / 1000.0)  # run_benchmark converts back to ms itself
    it = iter(values)
    return lambda: next(it)


# --------------------------------------------------------------------------
# run_benchmark — validation
# --------------------------------------------------------------------------

class TestRunBenchmarkValidation:
    def test_rejects_zero_runs(self):
        with pytest.raises(ValueError):
            run_benchmark(NoopBackbone(), num_runs=0)

    def test_rejects_negative_runs(self):
        with pytest.raises(ValueError):
            run_benchmark(NoopBackbone(), num_runs=-1)

    def test_rejects_negative_warmup(self):
        with pytest.raises(ValueError):
            run_benchmark(NoopBackbone(), warmup_runs=-1)

    def test_accepts_zero_warmup(self):
        clock = make_fake_clock([5.0])
        result = run_benchmark(NoopBackbone(), num_runs=1, warmup_runs=0, clock=clock)
        assert result.warmup_runs == 0

    def test_rejects_non_positive_image_size(self):
        with pytest.raises(ValueError):
            run_benchmark(NoopBackbone(), image_size=0)
        with pytest.raises(ValueError):
            run_benchmark(NoopBackbone(), image_size=-10)


# --------------------------------------------------------------------------
# run_benchmark — timing behavior (deterministic via fake clock)
# --------------------------------------------------------------------------

class TestRunBenchmarkTiming:
    def test_records_exactly_num_runs_latencies(self):
        clock = make_fake_clock([1.0, 2.0, 3.0])
        result = run_benchmark(NoopBackbone(), num_runs=3, warmup_runs=0, clock=clock)
        assert len(result.latencies_ms) == 3

    def test_latencies_match_the_injected_clock_exactly(self):
        clock = make_fake_clock([5.0, 10.0, 15.0])
        result = run_benchmark(NoopBackbone(), num_runs=3, warmup_runs=0, clock=clock)
        assert result.latencies_ms == pytest.approx([5.0, 10.0, 15.0])

    def test_warmup_runs_are_not_timed_or_recorded(self):
        # Only 2 clock-pairs provided — if warmup consumed the clock too,
        # this would raise StopIteration.
        clock = make_fake_clock([1.0, 2.0])
        result = run_benchmark(NoopBackbone(), num_runs=2, warmup_runs=3, clock=clock)
        assert len(result.latencies_ms) == 2

    def test_warmup_runs_call_backbone_embed(self):
        class CountingBackbone:
            def __init__(self):
                self.calls = 0

            def embed(self, image, input_is_bgr=True):
                self.calls += 1
                return np.zeros(8, dtype=np.float32)

        backbone = CountingBackbone()
        clock = make_fake_clock([1.0, 1.0])
        run_benchmark(backbone, num_runs=2, warmup_runs=3, clock=clock)
        assert backbone.calls == 5  # 3 warmup + 2 timed

    def test_default_clock_is_real_time_and_produces_positive_latencies(self):
        """Sanity check with the REAL default clock (no fake) — just
        confirms actual wall-clock timing works end to end and produces
        sane (positive, finite) numbers, without asserting exact values."""
        result = run_benchmark(NoopBackbone(), num_runs=3, warmup_runs=1)
        assert len(result.latencies_ms) == 3
        assert all(latency >= 0 for latency in result.latencies_ms)

    def test_records_image_size_and_device_label(self):
        clock = make_fake_clock([1.0])
        result = run_benchmark(NoopBackbone(), num_runs=1, clock=clock, image_size=128, device_label="mps")
        assert result.image_size == 128
        assert result.device_label == "mps"

    def test_same_seed_produces_same_image_each_run(self):
        """Not directly observable from BenchmarkResult, but confirms the
        function doesn't crash across repeated calls with the same seed
        and that results are structurally consistent."""
        clock1 = make_fake_clock([1.0, 1.0])
        clock2 = make_fake_clock([1.0, 1.0])
        r1 = run_benchmark(NoopBackbone(), num_runs=2, clock=clock1, seed=42)
        r2 = run_benchmark(NoopBackbone(), num_runs=2, clock=clock2, seed=42)
        assert r1.latencies_ms == r2.latencies_ms


# --------------------------------------------------------------------------
# BenchmarkResult — derived statistics
# --------------------------------------------------------------------------

class TestBenchmarkResultStats:
    def _result(self, latencies):
        return BenchmarkResult(
            num_runs=len(latencies), warmup_runs=2, image_size=224,
            device_label="cpu", latencies_ms=latencies,
        )

    def test_constructor_validates_latency_count_matches_num_runs(self):
        with pytest.raises(ValueError):
            BenchmarkResult(num_runs=3, warmup_runs=0, image_size=224, device_label="cpu", latencies_ms=[1.0, 2.0])

    def test_mean(self):
        result = self._result([10.0, 20.0, 30.0])
        assert result.mean_ms == pytest.approx(20.0)

    def test_median(self):
        result = self._result([10.0, 20.0, 30.0])
        assert result.median_ms == pytest.approx(20.0)

    def test_min_max(self):
        result = self._result([15.0, 5.0, 25.0])
        assert result.min_ms == pytest.approx(5.0)
        assert result.max_ms == pytest.approx(25.0)

    def test_stdev_of_single_run_is_zero(self):
        result = self._result([42.0])
        assert result.stdev_ms == 0.0

    def test_stdev_of_identical_runs_is_zero(self):
        result = self._result([10.0, 10.0, 10.0])
        assert result.stdev_ms == pytest.approx(0.0, abs=1e-9)

    def test_stdev_is_positive_for_varied_runs(self):
        result = self._result([10.0, 20.0, 30.0])
        assert result.stdev_ms > 0

    def test_percentile_bounds(self):
        result = self._result(list(range(1, 101)))  # 1..100
        assert result.percentile_ms(0) == pytest.approx(1.0)
        assert result.percentile_ms(100) == pytest.approx(100.0)
        assert result.percentile_ms(50) == pytest.approx(50.5, abs=1.0)

    def test_percentile_rejects_out_of_range(self):
        result = self._result([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            result.percentile_ms(-1)
        with pytest.raises(ValueError):
            result.percentile_ms(101)

    def test_inferences_per_second_matches_mean(self):
        result = self._result([100.0, 100.0])  # mean = 100ms -> 10/sec
        assert result.inferences_per_second == pytest.approx(10.0)

    def test_faster_mean_gives_higher_inferences_per_second(self):
        fast = self._result([10.0, 10.0])
        slow = self._result([100.0, 100.0])
        assert fast.inferences_per_second > slow.inferences_per_second


# --------------------------------------------------------------------------
# BenchmarkResult.suggested_frame_skip
# --------------------------------------------------------------------------

class TestSuggestedFrameSkip:
    def test_fast_inference_suggests_frame_skip_one(self):
        # mean latency 5ms, well under a 30fps frame interval (~33ms)
        result = BenchmarkResult(num_runs=1, warmup_runs=0, image_size=224, device_label="cpu", latencies_ms=[5.0])
        assert result.suggested_frame_skip(display_fps=30) == 1

    def test_slow_inference_suggests_higher_frame_skip(self):
        # mean latency 100ms vs. a 30fps (~33ms) frame interval -> need to
        # skip several frames per inference to keep up
        result = BenchmarkResult(num_runs=1, warmup_runs=0, image_size=224, device_label="cpu", latencies_ms=[100.0])
        skip = result.suggested_frame_skip(display_fps=30)
        assert skip >= 3

    def test_never_returns_less_than_one(self):
        result = BenchmarkResult(num_runs=1, warmup_runs=0, image_size=224, device_label="cpu", latencies_ms=[0.001])
        assert result.suggested_frame_skip(display_fps=30) >= 1

    def test_slower_display_fps_needs_higher_or_equal_frame_skip(self):
        """A slower camera has MORE time between frames, so the same
        inference latency needs a smaller (or equal) frame_skip to keep up
        — i.e. suggested_frame_skip should be monotonically non-decreasing
        in display_fps."""
        result = BenchmarkResult(num_runs=1, warmup_runs=0, image_size=224, device_label="cpu", latencies_ms=[50.0])
        skip_15fps = result.suggested_frame_skip(display_fps=15)
        skip_30fps = result.suggested_frame_skip(display_fps=30)
        assert skip_30fps >= skip_15fps

    def test_rejects_non_positive_display_fps(self):
        result = BenchmarkResult(num_runs=1, warmup_runs=0, image_size=224, device_label="cpu", latencies_ms=[5.0])
        with pytest.raises(ValueError):
            result.suggested_frame_skip(display_fps=0)
        with pytest.raises(ValueError):
            result.suggested_frame_skip(display_fps=-5)

    def test_exact_boundary_case(self):
        # mean latency exactly equals one frame interval at 20fps (50ms) ->
        # frame_skip of 1 is exactly enough, not 2.
        result = BenchmarkResult(num_runs=1, warmup_runs=0, image_size=224, device_label="cpu", latencies_ms=[50.0])
        assert result.suggested_frame_skip(display_fps=20) == 1


# --------------------------------------------------------------------------
# format_benchmark_report
# --------------------------------------------------------------------------

class TestFormatBenchmarkReport:
    def _result(self):
        return BenchmarkResult(
            num_runs=10, warmup_runs=3, image_size=224, device_label="cpu",
            latencies_ms=[50.0, 52.0, 48.0, 51.0, 49.0, 53.0, 47.0, 50.0, 51.0, 49.0],
        )

    def test_report_is_a_nonempty_string(self):
        report = format_benchmark_report(self._result())
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_mentions_run_count_and_device(self):
        report = format_benchmark_report(self._result())
        assert "10 run" in report
        assert "cpu" in report

    def test_report_mentions_image_size(self):
        report = format_benchmark_report(self._result())
        assert "224x224" in report

    def test_report_includes_mean_and_percentiles(self):
        report = format_benchmark_report(self._result())
        assert "mean" in report
        assert "p95" in report
        assert "p99" in report

    def test_report_includes_frame_skip_suggestions(self):
        report = format_benchmark_report(self._result())
        assert "--frame-skip" in report
        assert "30 fps" in report
        assert "15 fps" in report

    def test_report_does_not_crash_on_single_run(self):
        result = BenchmarkResult(num_runs=1, warmup_runs=0, image_size=224, device_label="cpu", latencies_ms=[42.0])
        report = format_benchmark_report(result)
        assert "42.0" in report
