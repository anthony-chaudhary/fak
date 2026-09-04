"""Tier 0 — bench_worker unit tests.

Tests WorkerConfig serialization, WorkerResult aggregation, and saturation
detection logic. Pure Python, runs anywhere (no server required).
"""

import importlib
import os
import pickle
import sys

import pytest


# ---------------------------------------------------------------------------
# Import bench_worker as a module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bw():
    bw_path = os.path.join(
        os.path.dirname(__file__), os.pardir, "bench_worker.py"
    )
    bw_path = os.path.normpath(bw_path)
    spec = importlib.util.spec_from_file_location("bench_worker", bw_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_worker"] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# WorkerConfig serialization
# ======================================================================

class TestWorkerConfigSerialization:
    def test_pickle_roundtrip(self, bw):
        cfg = bw.WorkerConfig(
            worker_id=7,
            transport="rdma",
            addr="10.0.0.1",
            port=18001,
            value_size=5_242_880,
            keys=["key_0", "key_1", "key_2"],
            read_ratio=0.8,
            distribution="zipfian",
            zipf_n=3,
            endpoint={"ip": "10.0.0.1", "port": 18001, "device": "mlx5_0"},
            workload="sglang",
            batch_size=16,
            io_workers=16,
            key_multiplier=2,
        )
        data = pickle.dumps(cfg)
        cfg2 = pickle.loads(data)
        assert cfg2.worker_id == 7
        assert cfg2.transport == "rdma"
        assert cfg2.keys == ["key_0", "key_1", "key_2"]
        assert cfg2.endpoint["device"] == "mlx5_0"
        assert cfg2.batch_size == 16
        assert cfg2.key_multiplier == 2

    def test_default_values(self, bw):
        cfg = bw.WorkerConfig()
        assert cfg.worker_id == 0
        assert cfg.transport == "tcp"
        assert cfg.keys == []
        assert cfg.endpoint is None
        assert cfg.batch_size == 16
        assert cfg.io_workers == 16

    def test_all_basic_types(self, bw):
        """WorkerConfig should only contain basic types for pickling."""
        cfg = bw.WorkerConfig(keys=["a", "b"], endpoint={"ip": "1.2.3.4", "port": 18000})
        data = pickle.dumps(cfg)
        assert len(data) > 0  # serialized successfully


# ======================================================================
# WorkerResult aggregation
# ======================================================================

class TestWorkerResultAggregation:
    def test_aggregate_empty(self, bw):
        assert bw.WorkerResult.aggregate([]) == {}

    def test_aggregate_single(self, bw):
        r = bw.WorkerResult(
            worker_id=0, ops=1000, gets=800, sets=200, hits=600,
            bytes_read=4_000_000, bytes_write=1_000_000,
            elapsed=10.0, p50_ms=0.5, p99_ms=2.0,
        )
        agg = bw.WorkerResult.aggregate([r])
        assert agg["total_ops"] == 1000
        assert agg["total_gets"] == 800
        assert agg["total_sets"] == 200
        assert agg["total_hits"] == 600
        assert abs(agg["ops_per_sec"] - 100.0) < 0.01
        assert abs(agg["p50_ms"] - 0.5) < 0.01

    def test_aggregate_multiple(self, bw):
        r1 = bw.WorkerResult(
            worker_id=0, ops=1000, gets=800, sets=200, hits=600,
            bytes_read=4_000_000_000, bytes_write=1_000_000_000,
            elapsed=10.0, p50_ms=0.5, p99_ms=2.0,
        )
        r2 = bw.WorkerResult(
            worker_id=1, ops=1200, gets=900, sets=300, hits=700,
            bytes_read=4_500_000_000, bytes_write=1_500_000_000,
            elapsed=10.0, p50_ms=0.6, p99_ms=3.0,
        )
        agg = bw.WorkerResult.aggregate([r1, r2])
        assert agg["total_ops"] == 2200
        assert agg["total_gets"] == 1700
        assert abs(agg["ops_per_sec"] - 220.0) < 0.01
        # p99 should be max of workers
        assert abs(agg["p99_ms"] - 3.0) < 0.01
        # Bandwidth: (4+4.5+1+1.5)GB / 10s = 1.1 GB/s
        assert abs(agg["bandwidth_gbps"] - 1.1) < 0.01

    def test_aggregate_hit_rate_exists_dominant(self, bw):
        """When exists_ops > gets, use exists hit rate."""
        r = bw.WorkerResult(
            worker_id=0, ops=1000,
            gets=100, hits=50,
            exists_ops=900, exists_hits=800,
            elapsed=10.0,
        )
        agg = bw.WorkerResult.aggregate([r])
        expected_hr = 800 / 900 * 100
        assert abs(agg["hit_rate"] - expected_hr) < 0.01

    def test_aggregate_hit_rate_get_dominant(self, bw):
        """When gets > exists_ops, use GET hit rate."""
        r = bw.WorkerResult(
            worker_id=0, ops=1000,
            gets=800, hits=600,
            exists_ops=100, exists_hits=80,
            elapsed=10.0,
        )
        agg = bw.WorkerResult.aggregate([r])
        expected_hr = 600 / 800 * 100
        assert abs(agg["hit_rate"] - expected_hr) < 0.01

    def test_aggregate_no_ops_hit_rate(self, bw):
        """No gets and no exists_ops -> hit_rate = -1."""
        r = bw.WorkerResult(worker_id=0, ops=100, sets=100, elapsed=10.0)
        agg = bw.WorkerResult.aggregate([r])
        assert agg["hit_rate"] == -1.0


# ======================================================================
# Latency helpers
# ======================================================================

class TestLatencyHelpers:
    def test_percentile_empty(self, bw):
        assert bw._percentile([], 50) == 0.0

    def test_percentile_single(self, bw):
        assert bw._percentile([1.0], 99) == 1.0

    def test_percentile_p50(self, bw):
        data = [float(i) for i in range(100)]
        p50 = bw._percentile(data, 50)
        assert abs(p50 - 49.5) < 0.01

    def test_compute_latency_stats_empty(self, bw):
        avg, p50, p99, p999 = bw._compute_latency_stats([])
        assert avg == 0.0
        assert p50 == 0.0

    def test_compute_latency_stats_values(self, bw):
        lats = [0.001 * i for i in range(1, 101)]  # 1ms to 100ms
        avg, p50, p99, p999 = bw._compute_latency_stats(lats)
        # avg = 50.5ms, p50 ~ 50ms, p99 ~ 99ms
        assert 40 < avg < 60
        assert 40 < p50 < 60
        assert 90 < p99 < 110


# ======================================================================
# Saturation detection
# ======================================================================

class TestSaturationDetection:
    def test_detect_clear_plateau(self, bw):
        """Plateau at 8 clients when improvement drops below threshold."""
        rows = [
            (1,   4200,  8.4, 0.31, 100.0),
            (2,   8200, 16.4, 0.35, 95.2),
            (4,  14000, 28.0, 0.52, 70.7),
            (8,  17800, 35.6, 1.20, 27.1),
            (16, 18000, 36.0, 2.40, 1.1),
        ]
        # The saturation point should be row before the one with < threshold improvement
        sat = bw.detect_saturation(rows)
        assert sat is not None
        assert sat == 8  # row before 16 which had 1.1% improvement

    def test_no_saturation_reached(self, bw):
        """All rows show significant improvement."""
        rows = [
            (1,  4200,  8.4, 0.31, 100.0),
            (2,  8200, 16.4, 0.35, 95.2),
            (4, 14000, 28.0, 0.52, 70.7),
        ]
        sat = bw.detect_saturation(rows)
        # Should return last client count
        assert sat == 4

    def test_single_row(self, bw):
        rows = [(1, 4200, 8.4, 0.31, 100.0)]
        sat = bw.detect_saturation(rows)
        assert sat == 1


# ======================================================================
# RNG / Zipf (duplicated in bench_worker — verify consistency)
# ======================================================================

class TestWorkerRNG:
    def test_lcg_deterministic(self, bw):
        rng1 = bw._LCG(12345)
        rng2 = bw._LCG(12345)
        assert [rng1.next_int() for _ in range(50)] == [rng2.next_int() for _ in range(50)]

    def test_zipf_in_range(self, bw):
        zipf = bw._ZipfGenerator(100, s=1.0)
        rng = bw._LCG(42)
        for _ in range(1000):
            s = zipf.sample(rng)
            assert 0 <= s < 100
