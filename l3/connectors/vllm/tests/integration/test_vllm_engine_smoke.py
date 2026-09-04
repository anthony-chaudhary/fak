"""Smoke test driving vLLM's engine assembly with our connector wired in.

We don't need to run a forward pass — vLLM constructs Scheduler + Worker
connectors at engine init, so just initializing the connectors through the
KVConnectorFactory proves the wire is connected end-to-end.

The test fails if any of:
- vLLM cannot resolve `CamaConnector` by name.
- Our scheduler / worker classes don't satisfy KVConnectorBase_V1 abstract methods.
- The required_kvcache_layout classmethod isn't callable.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from contextlib import closing
from types import SimpleNamespace

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


def test_concrete_subclass_satisfies_abc():
    """No TypeError from leftover @abstractmethod."""
    from l3_vllm.connector import CamaConnectorV1
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    assert issubclass(CamaConnectorV1, KVConnectorBase_V1)
    # Verify no abstract methods remain
    abstracts = getattr(CamaConnectorV1, "__abstractmethods__", set())
    assert not abstracts, f"unimplemented abstract methods: {abstracts}"


def test_required_layout_classmethod():
    """vLLM calls cls.get_required_kvcache_layout(vllm_config) at startup."""
    from l3_vllm.connector import CamaConnectorV1
    vllm_config = SimpleNamespace()
    result = CamaConnectorV1.get_required_kvcache_layout(vllm_config)
    assert result is None  # we accept any layout


def test_requires_piecewise_default():
    from l3_vllm.connector import CamaConnectorV1
    assert CamaConnectorV1.requires_piecewise_for_cudagraph({}) is False


def test_full_engine_init_path_with_real_server():
    """Walk the actual vLLM factory creation code with a real cama-server."""
    if not os.path.exists(CAMA_SERVER_BIN):
        pytest.skip("cama-server not available")
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    port = _free_port()
    log = open(f"/tmp/cama-server-engine-{port}.log", "w")
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
        assert _wait_port(port, 30)
        time.sleep(2.0)

        kc = KVTransferConfig(
            kv_connector="CamaConnector",
            kv_connector_module_path="l3_vllm.connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "remote_addr": "127.0.0.1",
                "remote_port": port,
                "use_rdma": False,
            },
        )

        # Validate via vLLM's own resolver — this is the same path vllm serve uses.
        cls = KVConnectorFactory.get_connector_class(kc)
        assert cls.__name__ == "CamaConnectorV1"

        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        cache_config = SimpleNamespace(block_size=16)
        parallel_config = SimpleNamespace(
            tensor_parallel_rank=0, pipeline_parallel_rank=0, pipeline_parallel_size=1
        )
        model_config = SimpleNamespace(hf_config=SimpleNamespace(), use_mla=False)
        vllm_config = SimpleNamespace(
            kv_transfer_config=kc,
            cache_config=cache_config,
            parallel_config=parallel_config,
            model_config=model_config,
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        )

        # Instantiate using vLLM's own factory path.
        sched = cls(vllm_config, KVConnectorRole.SCHEDULER)
        wrk = cls(vllm_config, KVConnectorRole.WORKER)

        import torch
        t = torch.zeros((4, 16, 16), dtype=torch.uint8)
        wrk.register_kv_caches({"layer.0": t})
        wrk.shutdown()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
