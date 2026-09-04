"""Tier 1 — C++ get_stats/reset_stats schema validation.

Requires compiled _l3_rdma.so but no running server.
"""

import pytest

_l3_rdma = pytest.importorskip("l3_client._l3_rdma")
RDMATransport = _l3_rdma.RDMATransport


EXPECTED_KEYS = {
    "roundtrip_count",
    "rdma_read_count",
    "avg_roundtrip_us",
    "avg_rdma_read_us",
}


class TestGetStats:
    def test_returns_dict_with_expected_keys(self):
        t = RDMATransport()
        stats = t.get_stats()
        assert isinstance(stats, dict)
        assert set(stats.keys()) == EXPECTED_KEYS
        t.close()

    def test_counts_are_integers(self):
        t = RDMATransport()
        stats = t.get_stats()
        assert isinstance(stats["roundtrip_count"], int)
        assert isinstance(stats["rdma_read_count"], int)
        t.close()

    def test_averages_are_floats(self):
        t = RDMATransport()
        stats = t.get_stats()
        assert isinstance(stats["avg_roundtrip_us"], float)
        assert isinstance(stats["avg_rdma_read_us"], float)
        t.close()

    def test_fresh_transport_has_all_zeros(self):
        t = RDMATransport()
        stats = t.get_stats()
        assert stats["roundtrip_count"] == 0
        assert stats["rdma_read_count"] == 0
        assert stats["avg_roundtrip_us"] == 0.0
        assert stats["avg_rdma_read_us"] == 0.0
        t.close()


class TestResetStats:
    def test_reset_clears_to_zero(self):
        t = RDMATransport()
        t.reset_stats()
        stats = t.get_stats()
        assert stats["roundtrip_count"] == 0
        assert stats["rdma_read_count"] == 0
        assert stats["avg_roundtrip_us"] == 0.0
        assert stats["avg_rdma_read_us"] == 0.0
        t.close()


class TestIndependentStats:
    def test_multiple_transports_independent(self):
        """Each transport instance has its own stats counters."""
        t1 = RDMATransport()
        t2 = RDMATransport()

        s1 = t1.get_stats()
        s2 = t2.get_stats()
        assert s1["roundtrip_count"] == 0
        assert s2["roundtrip_count"] == 0

        # Even after resetting one, the other is unaffected
        t1.reset_stats()
        s2_after = t2.get_stats()
        assert s2_after["roundtrip_count"] == 0

        t1.close()
        t2.close()
