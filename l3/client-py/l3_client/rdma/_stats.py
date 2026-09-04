"""Transfer sub-phase timing accumulation and averaging."""

from __future__ import annotations


def average_get_timings(timings: list[dict]) -> dict:
    """Swap-and-average GET sub-phase timings. Returns stat dict entries.

    Handles both standard mget_rdma timings (t_ctrl_ms, t_meta_ms, etc.)
    and raw-mode timings (t_total_ms) by using .get() with defaults.
    """
    if timings:
        gc = len(timings)
        return {
            "get_avg_ctrl_ms": sum(t.get("t_ctrl_ms", 0.0) for t in timings) / gc,
            "get_avg_meta_ms": sum(t.get("t_meta_ms", 0.0) for t in timings) / gc,
            "get_avg_read_ms": sum(t.get("t_read_ms", 0.0) for t in timings) / gc,
            "get_avg_ack_ms": sum(t.get("t_ack_ms", 0.0) for t in timings) / gc,
            "get_batch_count": gc,
            "get_avg_keys": sum(t.get("n_keys", 0) for t in timings) / gc,
            "get_avg_bytes": sum(t.get("total_bytes", 0) for t in timings) / gc,
        }
    return {
        "get_avg_ctrl_ms": 0.0, "get_avg_meta_ms": 0.0,
        "get_avg_read_ms": 0.0, "get_avg_ack_ms": 0.0,
        "get_batch_count": 0, "get_avg_keys": 0.0, "get_avg_bytes": 0.0,
    }


def average_set_timings(timings: list[dict]) -> dict:
    """Swap-and-average SET sub-phase timings. Returns stat dict entries."""
    if timings:
        sc = len(timings)
        return {
            "set_avg_serialize_ms": sum(t["t_serialize_ms"] for t in timings) / sc,
            "set_avg_send_ms": sum(t["t_send_ms"] for t in timings) / sc,
            "set_batch_count": sc,
            "set_avg_keys": sum(t["n_keys"] for t in timings) / sc,
            "set_avg_bytes": sum(t["total_bytes"] for t in timings) / sc,
            "set_avg_sub_batches": sum(t.get("n_sub_batches", 1) for t in timings) / sc,
        }
    return {
        "set_avg_serialize_ms": 0.0, "set_avg_send_ms": 0.0,
        "set_batch_count": 0, "set_avg_keys": 0.0, "set_avg_bytes": 0.0,
        "set_avg_sub_batches": 0.0,
    }


def compute_nic_balance(per_nic_ops: list[int]) -> float:
    """NIC balance percentage. 100 = perfectly balanced, low = skewed."""
    if not per_nic_ops:
        return 100.0
    mx = max(per_nic_ops)
    mn = min(per_nic_ops)
    return round((mn / mx * 100) if mx > 0 else 100.0, 2)
