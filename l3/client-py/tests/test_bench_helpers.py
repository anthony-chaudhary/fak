"""Tier 0 — Benchmark helper function tests.

Tests _percentile, _LCG, and ZipfGenerator from bench.py.
Pure Python, runs anywhere.
"""

import importlib
import sys
import os

import pytest


# ---------------------------------------------------------------------------
# Import bench.py as a module (it's a script, not a package)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench():
    bench_path = os.path.join(
        os.path.dirname(__file__), os.pardir, "bench.py"
    )
    bench_path = os.path.normpath(bench_path)
    spec = importlib.util.spec_from_file_location("bench", bench_path)
    mod = importlib.util.module_from_spec(spec)
    # Prevent main() from running if there's any top-level execution guard issue
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# _percentile
# ======================================================================

class TestPercentile:
    def test_single_element(self, bench):
        assert bench._percentile([5.0], 50) == 5.0

    def test_p0_is_min(self, bench):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert bench._percentile(data, 0) == 1.0

    def test_p100_is_max(self, bench):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert bench._percentile(data, 100) == 5.0

    def test_p50_median_odd(self, bench):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert bench._percentile(data, 50) == 3.0

    def test_p50_median_even(self, bench):
        data = [1.0, 2.0, 3.0, 4.0]
        result = bench._percentile(data, 50)
        assert abs(result - 2.5) < 1e-9

    def test_interpolation(self, bench):
        data = [10.0, 20.0]
        p25 = bench._percentile(data, 25)
        assert abs(p25 - 12.5) < 1e-9

    def test_empty_returns_zero(self, bench):
        assert bench._percentile([], 50) == 0.0

    def test_p99_9_large_dataset(self, bench):
        data = list(range(10_000))
        data_float = [float(x) for x in data]
        p999 = bench._percentile(data_float, 99.9)
        # Should be close to 9990
        assert 9989.0 <= p999 <= 9999.0


# ======================================================================
# _LCG
# ======================================================================

class TestLCG:
    def test_deterministic_same_seed(self, bench):
        rng1 = bench._LCG(12345)
        rng2 = bench._LCG(12345)
        seq1 = [rng1.next_int() for _ in range(100)]
        seq2 = [rng2.next_int() for _ in range(100)]
        assert seq1 == seq2

    def test_different_seeds_diverge(self, bench):
        rng1 = bench._LCG(1)
        rng2 = bench._LCG(2)
        seq1 = [rng1.next_int() for _ in range(10)]
        seq2 = [rng2.next_int() for _ in range(10)]
        assert seq1 != seq2

    def test_next_float_range(self, bench):
        rng = bench._LCG(42)
        for _ in range(1000):
            f = rng.next_float()
            assert 0.0 <= f < 1.0

    def test_next_int_returns_positive(self, bench):
        rng = bench._LCG(0)
        for _ in range(100):
            val = rng.next_int()
            assert val >= 0


# ======================================================================
# ZipfGenerator
# ======================================================================

class TestZipfGenerator:
    def test_samples_in_range(self, bench):
        n = 100
        zipf = bench.ZipfGenerator(n)
        rng = bench._LCG(42)
        for _ in range(1000):
            s = zipf.sample(rng)
            assert 0 <= s < n

    def test_skew_toward_low_indices(self, bench):
        n = 1000
        zipf = bench.ZipfGenerator(n, s=1.0)
        rng = bench._LCG(42)
        counts = [0] * n
        num_samples = 50_000
        for _ in range(num_samples):
            s = zipf.sample(rng)
            counts[s] += 1
        top10 = sum(counts[:10])
        bottom10 = sum(counts[-10:])
        # Zipfian: top-10 keys should get far more hits than bottom-10
        assert top10 > bottom10 * 5, f"top10={top10}, bottom10={bottom10}"

    def test_n_equals_1(self, bench):
        zipf = bench.ZipfGenerator(1)
        rng = bench._LCG(99)
        for _ in range(100):
            assert zipf.sample(rng) == 0
