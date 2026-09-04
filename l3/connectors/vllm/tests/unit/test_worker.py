"""Tests for l3_vllm.worker.CamaConnectorWorker using a mock PriskvClient."""

import torch

from l3_vllm.config import CamaConnectorConfig
from l3_vllm.key_scheme import KeyScheme, KeySchemeConfig
from l3_vllm.metadata import CamaKVConnectorMetadata, LoadSpec, SaveSpec
from l3_vllm.worker import CamaConnectorWorker


def make_worker():
    cfg = CamaConnectorConfig.from_extra_config({})
    ks = KeyScheme(KeySchemeConfig(is_mla=False))
    return CamaConnectorWorker(cfg, ks)


def test_register_kv_caches(patch_cama_client):
    w = make_worker()
    # 4 blocks of 1024 bytes each
    t1 = torch.zeros((4, 16, 16), dtype=torch.uint8)
    t2 = torch.zeros((4, 16, 16), dtype=torch.uint8)
    w.register_kv_caches({"layer.0": t1, "layer.1": t2})
    assert len(patch_cama_client.reg_calls) == 2
    assert w._kv_cache_generation == 1


def test_save_and_load_roundtrip(patch_cama_client):
    w = make_worker()
    t = torch.zeros((4, 16, 16), dtype=torch.uint8)
    w.register_kv_caches({"layer.0": t})

    # SAVE path
    meta_save = CamaKVConnectorMetadata(
        loads=[],
        saves=[SaveSpec(request_id="r1", block_hashes=["aa"], block_ids=[2])],
    )
    w.bind_metadata(meta_save)
    # Save needs save_kv_layer called and then wait_for_save.
    w.save_kv_layer("layer.0", t, attn_metadata=None)
    w.wait_for_save()
    # mset must have run
    assert any("aa" in "_".join(call) for call in patch_cama_client.mset_calls)

    # LOAD path
    meta_load = CamaKVConnectorMetadata(
        loads=[LoadSpec(request_id="r2", block_hashes=["aa"], block_ids=[2], num_external_tokens=16)],
        saves=[],
    )
    w.bind_metadata(meta_load)
    w.start_load_kv(forward_context=None)
    # mget should have been called with at least one key
    assert len(patch_cama_client.mget_calls) >= 1
    # All layer events should be set
    for ev in w._load_events.values():
        assert ev.is_set()


def test_load_with_unconnected_signals_events(patch_cama_client):
    """No metadata -> still safely signal all events."""
    w = make_worker()
    t = torch.zeros((4, 16, 16), dtype=torch.uint8)
    w.register_kv_caches({"layer.0": t})
    w.bind_metadata(None)
    w.start_load_kv(forward_context=None)
    # No mget_rdma should have happened
    pre_count = len(patch_cama_client.mget_calls)
    assert pre_count == 0
    # but events signaled
    for ev in w._load_events.values():
        assert ev.is_set()


def test_circuit_breaker_short_circuits_save(patch_cama_client):
    w = make_worker()
    t = torch.zeros((4, 16, 16), dtype=torch.uint8)
    w.register_kv_caches({"layer.0": t})
    # Force CB open
    for _ in range(w.config.cb_failure_threshold):
        w._cb.on_failure()
    assert w._cb.state.value == "open"
    meta_save = CamaKVConnectorMetadata(
        loads=[],
        saves=[SaveSpec(request_id="r1", block_hashes=["aa"], block_ids=[0])],
    )
    w.bind_metadata(meta_save)
    pre = len(patch_cama_client.mset_calls)
    w.save_kv_layer("layer.0", t, attn_metadata=None)
    w.wait_for_save()
    assert len(patch_cama_client.mset_calls) == pre


def test_shutdown_dereg_and_close(patch_cama_client):
    w = make_worker()
    t = torch.zeros((4, 16, 16), dtype=torch.uint8)
    w.register_kv_caches({"layer.0": t})
    w.shutdown()
    assert patch_cama_client.closed is True
    assert len(patch_cama_client.dereg_calls) >= 1
