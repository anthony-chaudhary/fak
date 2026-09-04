"""Smoke test: connector is discoverable / constructible through vLLM's APIs.

Does NOT spin up the full inference server — that requires a GPU or a long
CPU run. Instead:
1. Force-load our `register()` (entry_points are auto-discovered, but on
   pip install -e they only fire if the env has the deps wired).
2. Build a minimal VllmConfig with our connector selected.
3. Construct CamaConnectorV1 in SCHEDULER and WORKER roles.
4. Verify the round-trip key/metadata path works against a fresh
   CamaKVConnectorMetadata.

If this passes, `vllm serve --kv-transfer-config ...` will discover our
connector at startup.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from contextlib import closing

import pytest


CAMA_SERVER_BIN = os.environ.get(
    "CAMA_SERVER_BIN", "/root/work/cama-complete/cama-server/cama-server"
)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.3)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.2)
    return False


def test_factory_registration_discovers_cama_connector():
    """vLLM's KVConnectorFactory should be able to resolve 'CamaConnector'."""
    from l3_vllm.connector import register
    register()
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    cls = KVConnectorFactory.get_connector_class_by_name("CamaConnector")
    assert cls.__name__ == "CamaConnectorV1"


def test_construct_via_kv_transfer_config():
    """Build a KVTransferConfig and construct both connector roles."""
    if not os.path.exists(CAMA_SERVER_BIN):
        pytest.skip("cama-server binary not available")

    port = _free_port()
    log = open(f"/tmp/cama-server-smoke-{port}.log", "w")
    server = subprocess.Popen(
        [
            CAMA_SERVER_BIN, "server",
            "-listen", f"0.0.0.0:{port}",
            "-metrics", "", "-pprof", "",
            "-no-preflight", "-huge-pages=false", "-mlockall=false",
            "-auto-disable-swap=false", "-auto-panic-reboot=false",
            "-auto-alloc-huge-pages=false", "-max-memory-gb", "1",
        ],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        assert _wait_port(port, 30), "cama-server did not start"
        time.sleep(2.0)

        # Build a VllmConfig dataclass tree by hand. We don't need a real
        # model — just enough for the connector __init__ to read fields.
        from types import SimpleNamespace
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        # Compose a minimal "VllmConfig" sufficient for our constructor.
        kv_transfer = SimpleNamespace(
            kv_connector="CamaConnector",
            kv_connector_module_path="l3_vllm.connector",
            kv_role="kv_both",
            engine_id="test",
            kv_connector_extra_config={
                "remote_addr": "127.0.0.1",
                "remote_port": port,
                "use_rdma": False,
            },
        )
        cache_config = SimpleNamespace(block_size=16)
        parallel_config = SimpleNamespace(
            tensor_parallel_rank=0,
            pipeline_parallel_rank=0,
            pipeline_parallel_size=1,
        )
        model_config = SimpleNamespace(hf_config=SimpleNamespace(), use_mla=False)
        vllm_config = SimpleNamespace(
            kv_transfer_config=kv_transfer,
            cache_config=cache_config,
            parallel_config=parallel_config,
            model_config=model_config,
        )

        from l3_vllm.connector import CamaConnectorV1
        sched = CamaConnectorV1(vllm_config, KVConnectorRole.SCHEDULER)
        wrk = CamaConnectorV1(vllm_config, KVConnectorRole.WORKER)

        assert sched._scheduler is not None
        assert wrk._worker is not None

        # Exercise the scheduler flow end-to-end.
        req = SimpleNamespace(
            request_id="r1",
            block_hashes=["aa", "bb"],
        )
        matched, is_async = sched.get_num_new_matched_tokens(req, 0)
        assert matched == 0 and is_async is False  # nothing in cache yet

        # Report a SAVE, build meta, hand to worker, run it, then a LOAD.
        sched.request_finished(req, block_ids=[0, 1])
        meta = sched.build_connector_meta(SimpleNamespace())
        assert len(meta.saves) == 1

        import torch
        t = torch.arange(0, 4 * 16 * 16, dtype=torch.uint8).reshape(4, 16, 16)
        wrk.register_kv_caches({"layer.0": t.clone()})
        wrk._worker.bind_metadata(meta)
        wrk.save_kv_layer("layer.0", t, attn_metadata=None)
        wrk.wait_for_save()

        # Tell the scheduler the block is now present.
        sched._scheduler.report_blocks_present(["aa", "bb"])
        matched2, is_async2 = sched.get_num_new_matched_tokens(req, 0)
        assert matched2 == 32  # 2 blocks * 16 tokens
        assert is_async2 is True

        wrk.shutdown()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
