"""Unit tests for cama_storage.py — no live PrisKV server required.

All PrisKV interactions are mocked via unittest.mock.  Tests are organized by
the component they cover:

  TestMultiNICDiscovery  (8 tests)  — Multi-NIC endpoint discovery & reconnect
  TestWarmup             (5 tests)  — 3-phase warmup validation
  TestConfigLoading      (5 tests)  — Triple-source configuration priority
  TestKeyNaming          (4 tests)  — MHA/MLA suffix construction
  TestBatchPostprocess   (3 tests)  — Per-sub-key → per-page boolean conversion
"""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np


# ---------------------------------------------------------------------------
# Minimal stubs for SGLang types that cama_storage.py imports at module level.
# These must be in place *before* we import cama_storage.
# ---------------------------------------------------------------------------

# Stub envs module
_ENV_DEFAULTS = {
    "SGLANG_CAMA_REMOTE_ADDR": "127.0.0.1",
    "SGLANG_CAMA_REMOTE_PORT": 18001,
    "SGLANG_CAMA_PASSWORD": "",
    "SGLANG_CAMA_USE_MPUT_MGET": True,
    "SGLANG_CAMA_CHECK_SERVER": False,
    "SGLANG_CAMA_OP_TIMEOUT_S": 10.0,
    "SGLANG_CAMA_CONFIG_PATH": None,
    "SGLANG_CAMA_IO_WORKERS": 16,
}


class _StubEnvVar:
    """Minimal stand-in for sglang.srt.environ.EnvStr / EnvInt / EnvBool."""

    def __init__(self, name, default):
        self.name = name
        self.default = default
        self._override = None

    def get(self):
        if self._override is not None:
            return self._override
        val = os.environ.get(self.name)
        if val is not None:
            if isinstance(self.default, bool):
                return val.lower() in ("1", "true", "yes")
            if isinstance(self.default, int):
                return int(val)
            if isinstance(self.default, float):
                return float(val)
            return val
        return self.default

    def is_set(self):
        return self._override is not None or self.name in os.environ


class _StubEnvs:
    pass


_stub_envs = _StubEnvs()
for _k, _v in _ENV_DEFAULTS.items():
    setattr(_stub_envs, _k, _StubEnvVar(_k, _v))

# Stub HiCache types
@dataclass
class _StubStorageConfig:
    is_mla_model: bool = False
    tp_rank: int = 0
    pp_rank: int = 0
    pp_size: int = 1
    extra_config: dict = None


class _StubHiCacheStorage:
    def __init__(self, *a, **kw):
        pass


class _StubHiCacheStorageConfig:
    pass


class _StubHiCacheStorageExtraInfo:
    pass


class _StubHostKVCache:
    pass


class _StubStorageMetrics:
    def __init__(self):
        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []
        self.get_errors = 0
        self.get_successes = 0
        self.set_errors = 0
        self.set_successes = 0
        self.exists_errors = 0
        self.exists_timeouts = 0


# Stub profiling decorators (no-ops)
def _nvtx_noop(msg, domain):
    def decorator(func):
        return func
    return decorator


from contextlib import contextmanager

@contextmanager
def _tag_noop(tags):
    yield


# ---------------------------------------------------------------------------
# Patch sys.modules so cama_storage.py can import without real SGLang / priskv
# ---------------------------------------------------------------------------
import sys

# priskv / cama_client stubs
_mock_kv_mod = MagicMock()
_mock_kv_mod.__version__ = "0.0.0-test"
_MockPriskvClient = MagicMock
_mock_kv_mod.SGL = MagicMock

# RC constants (CAMA convention)
class _MockRC:
    EXISTS_FOUND = 1
    EXISTS_MISSING = 0
    SET_OK = 0
    GET_OK = 0
    GET_MISS = -1
    DELETE_OK = 0


_mock_rc_mod = MagicMock()
_mock_rc_mod.EXISTS_FOUND = 1
_mock_rc_mod.EXISTS_MISSING = 0
_mock_rc_mod.SET_OK = 0
_mock_rc_mod.GET_OK = 0
_mock_rc_mod.GET_MISS = -1
_mock_rc_mod.DELETE_OK = 0

sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("sglang", MagicMock())
sys.modules.setdefault("sglang.srt", MagicMock())
sys.modules.setdefault("sglang.srt.environ", MagicMock())
sys.modules["sglang.srt.environ"].envs = _stub_envs
sys.modules.setdefault("sglang.srt.mem_cache", MagicMock())
sys.modules.setdefault("sglang.srt.mem_cache.hicache_storage", MagicMock())
sys.modules["sglang.srt.mem_cache.hicache_storage"].HiCacheStorage = _StubHiCacheStorage
sys.modules["sglang.srt.mem_cache.hicache_storage"].HiCacheStorageConfig = _StubHiCacheStorageConfig
sys.modules["sglang.srt.mem_cache.hicache_storage"].HiCacheStorageExtraInfo = _StubHiCacheStorageExtraInfo
sys.modules.setdefault("sglang.srt.mem_cache.memory_pool_host", MagicMock())
sys.modules["sglang.srt.mem_cache.memory_pool_host"].HostKVCache = _StubHostKVCache
sys.modules.setdefault("sglang.srt.mem_cache.storage", MagicMock())
sys.modules.setdefault("sglang.srt.mem_cache.storage.cama", MagicMock())
sys.modules.setdefault("sglang.srt.mem_cache.storage.cama.profiling", MagicMock())
sys.modules["sglang.srt.mem_cache.storage.cama.profiling"].nvtx_range = _nvtx_noop
sys.modules["sglang.srt.mem_cache.storage.cama.profiling"].tag_wrapper = _tag_noop
sys.modules.setdefault("sglang.srt.metrics", MagicMock())
sys.modules.setdefault("sglang.srt.metrics.collector", MagicMock())
sys.modules["sglang.srt.metrics.collector"].StorageMetrics = _StubStorageMetrics

# Provide cama_client so the import guard succeeds
sys.modules.setdefault("cama_client", _mock_kv_mod)
sys.modules.setdefault("l3_client", _mock_kv_mod)
sys.modules.setdefault("cama_client._version", MagicMock())
sys.modules.setdefault("l3_client._version", sys.modules["cama_client._version"])
_mock_cama_version = sys.modules["cama_client._version"]
_mock_cama_version.check_sglang_compatibility = MagicMock()

# errors sub-module (CamaServerOverloadError, CamaNotReadyError)
_mock_errors_mod = MagicMock()
_mock_errors_mod.CamaServerOverloadError = type("CamaServerOverloadError", (RuntimeError,), {})
_mock_errors_mod.CamaNotReadyError = type("CamaNotReadyError", (RuntimeError,), {})
sys.modules.setdefault("cama_client.errors", _mock_errors_mod)
sys.modules.setdefault("l3_client.errors", _mock_errors_mod)

# rc sub-module
sys.modules.setdefault("cama_client.rc", _mock_rc_mod)
sys.modules.setdefault("l3_client.rc", _mock_rc_mod)
_mock_kv_mod.rc = _mock_rc_mod
_mock_kv_mod.PriskvClient = MagicMock
_mock_kv_mod.PriskvClient.__name__ = "PriskvClientRDMA"

# reconnect sub-module (ReconnectConfig dataclass)
_mock_reconnect_mod = MagicMock()
_mock_reconnect_mod.ReconnectConfig = MagicMock
sys.modules.setdefault("cama_client.reconnect", _mock_reconnect_mod)
sys.modules.setdefault("l3_client.reconnect", _mock_reconnect_mod)

# NOW import cama_storage (after all stubs are in place)
# We import the module so we can patch its globals.
import importlib
import cama_module.cama_storage as cs_mod

# Ensure RC constants match
_RC = cs_mod._RC


# ===================================================================
# Helper: build a CamaStorage with fully mocked PriskvClient
# ===================================================================

def _make_storage(
    *,
    remote_addr="10.0.0.1",
    remote_port=18001,
    password="",
    rdma_endpoints=None,
    is_mla=False,
    tp_rank=0,
    pp_rank=0,
    pp_size=1,
    extra_config=None,
    op_timeout_s=10.0,
    check_server=False,
):
    """Instantiate CamaStorage with a mock PriskvClient.

    ``rdma_endpoints`` controls what ``conn.rdma_endpoints()`` returns:
      - list of dicts  → returned as-is
      - None           → method not present (hasattr guard → [])
      - "raise"        → raises RuntimeError
    """
    mock_conn = MagicMock()
    mock_conn.setstr.return_value = 0
    mock_conn.getstr.return_value = "ok"
    mock_conn.set.return_value = 0
    mock_conn.get.return_value = 0
    mock_conn.exists.side_effect = lambda k: (
        _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
    )
    mock_conn.delete.return_value = 0
    mock_conn.mset.side_effect = lambda keys, sgls, **kw: [0] * len(keys)
    mock_conn.mset_striped = MagicMock(
        side_effect=lambda keys, sgls, **kw: [0] * len(keys))
    mock_conn.mexists.side_effect = lambda keys: [
        _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
        for k in keys
    ]
    mock_conn.mdel.return_value = 0
    mock_conn.reg_memory.return_value = 12345  # non-zero handle
    mock_conn.dereg_memory.return_value = 0
    mock_conn.set_timeout = MagicMock()
    mock_conn._server_info = None

    if rdma_endpoints == "raise":
        mock_conn.rdma_endpoints.side_effect = RuntimeError("discovery boom")
    elif rdma_endpoints is not None:
        mock_conn.rdma_endpoints.return_value = rdma_endpoints
    else:
        # Remove the attribute so hasattr returns False
        del mock_conn.rdma_endpoints

    if extra_config is None:
        extra_config = {
            "remote_addr": remote_addr,
            "remote_port": remote_port,
            "password": password,
            "op_timeout_s": op_timeout_s,
            "check_server": check_server,
        }

    storage_config = _StubStorageConfig(
        is_mla_model=is_mla,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        pp_size=pp_size,
        extra_config=extra_config,
    )

    # Patch _make_connection — the single entry point for all connection creation.
    # This covers both pool_size=1 (_PriskvClient) and pool_size>1 (create_pool).
    _mock_kv_mod.create_pool = MagicMock(return_value=mock_conn)

    with patch.object(cs_mod, "_make_connection", return_value=mock_conn) as mock_cls, \
         patch("numpy.array_equal", return_value=True):
        storage = cs_mod.CamaStorage(storage_config, None)

    return storage, mock_conn, mock_cls


# ===================================================================
# Test Classes
# ===================================================================


class TestMultiNICDiscovery(unittest.TestCase):
    """Tests for the Multi-NIC discovery block in __init__ (lines 241-277)."""

    def test_multi_nic_striping_reconnect(self):
        """4 endpoints with striping → reconnects to endpoints[0] with all endpoints passed."""
        eps = [
            {"ip": "10.0.0.10", "port": 6380, "device": "mlx5_0"},
            {"ip": "10.0.0.11", "port": 6381, "device": "mlx5_1"},
            {"ip": "10.0.0.12", "port": 6382, "device": "mlx5_2"},
            {"ip": "10.0.0.13", "port": 6383, "device": "mlx5_3"},
        ]
        storage, mock_conn, mock_cls = _make_storage(
            rdma_endpoints=eps, tp_rank=1,
        )
        # Initial connect + striping reconnect
        self.assertEqual(mock_cls.call_count, 2)
        # Striping: second call targets endpoints[0] with all endpoints
        _, args, kwargs = mock_cls.mock_calls[1]
        self.assertEqual(args[0], "10.0.0.10")  # addr = first endpoint
        self.assertEqual(args[1], 6380)          # port = first endpoint
        self.assertIn("endpoints", kwargs)
        self.assertEqual(len(kwargs["endpoints"]), 4)
        # Config updated to first endpoint
        self.assertEqual(storage.config.remote_addr, "10.0.0.10")
        self.assertEqual(storage.config.remote_port, 6380)

    def test_multi_nic_round_robin_no_striping(self):
        """2 endpoints, nic_striping=False, ranks 0-3 → round-robin assignment."""
        eps = [
            {"ip": "10.0.0.10", "port": 6380, "device": "mlx5_0"},
            {"ip": "10.0.0.11", "port": 6381, "device": "mlx5_1"},
        ]
        expected = {
            0: ("10.0.0.10", 6380),
            1: ("10.0.0.11", 6381),
            2: ("10.0.0.10", 6380),
            3: ("10.0.0.11", 6381),
        }
        for rank, (exp_ip, exp_port) in expected.items():
            storage, _, _ = _make_storage(
                rdma_endpoints=eps, tp_rank=rank,
                extra_config={
                    "remote_addr": "10.0.0.1", "remote_port": 18001,
                    "password": "", "nic_striping": False,
                },
            )
            self.assertEqual(storage.config.remote_addr, exp_ip,
                             f"rank {rank} got wrong IP")
            self.assertEqual(storage.config.remote_port, exp_port,
                             f"rank {rank} got wrong port")

    def test_multi_nic_striping_no_reconnect_same_pool(self):
        """Striping: initial pool already has multi-endpoint → no reconnect."""
        eps = [
            {"ip": "10.0.0.1", "port": 18001, "device": "mlx5_0"},
            {"ip": "10.0.0.11", "port": 6381, "device": "mlx5_1"},
        ]
        # Create a mock that already has multi-endpoint pool
        mock_conn = MagicMock()
        mock_conn.setstr.return_value = 0
        mock_conn.getstr.return_value = "ok"
        mock_conn.set.return_value = 0
        mock_conn.get.return_value = 0
        mock_conn.exists.side_effect = lambda k: (
            _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
        )
        mock_conn.delete.return_value = 0
        mock_conn.mset.side_effect = lambda keys, sgls, **kw: [0] * len(keys)
        mock_conn.mset_striped = MagicMock(
            side_effect=lambda keys, sgls, **kw: [0] * len(keys))
        mock_conn.mexists.side_effect = lambda keys: [
            _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
            for k in keys
        ]
        mock_conn.mdel.return_value = 0
        mock_conn.reg_memory.return_value = 12345
        mock_conn.dereg_memory.return_value = 0
        mock_conn.set_timeout = MagicMock()
        mock_conn._server_info = None
        mock_conn._endpoints = [("10.0.0.1", 18001), ("10.0.0.11", 6381)]
        mock_conn.rdma_endpoints.return_value = eps

        extra = {"remote_addr": "10.0.0.1", "remote_port": 18001, "password": ""}
        sc = _StubStorageConfig(extra_config=extra)
        _mock_kv_mod.create_pool = MagicMock(return_value=mock_conn)

        with patch.object(cs_mod, "_make_connection", return_value=mock_conn) as mock_cls, \
             patch("numpy.array_equal", return_value=True):
            storage = cs_mod.CamaStorage(sc, None)

        # Only the initial connection — no NIC reconnect needed
        self.assertEqual(mock_cls.call_count, 1)
        mock_conn.close.assert_not_called()

    def test_multi_nic_timeout_reapplied(self):
        """After reconnect, set_timeout() called on new connection."""
        eps = [
            {"ip": "10.0.0.10", "port": 6380, "device": "mlx5_0"},
            {"ip": "10.0.0.11", "port": 6381, "device": "mlx5_1"},
        ]
        storage, _, mock_cls = _make_storage(
            rdma_endpoints=eps, tp_rank=1, op_timeout_s=5.0,
        )
        # The reconnected client (second call's return value) should have set_timeout called
        reconnected_conn = mock_cls.return_value
        reconnected_conn.set_timeout.assert_called_with(5.0)

    def test_single_endpoint_no_reconnect(self):
        """1 endpoint → no reconnect, info log emitted."""
        eps = [{"ip": "10.0.0.10", "port": 6380, "device": "mlx5_0"}]
        storage, mock_conn, mock_cls = _make_storage(rdma_endpoints=eps)
        # Only initial connection
        self.assertEqual(mock_cls.call_count, 1)
        mock_conn.close.assert_not_called()

    def test_no_endpoints_tcp_fallback(self):
        """Empty list → no reconnect, debug log."""
        storage, mock_conn, mock_cls = _make_storage(rdma_endpoints=[])
        self.assertEqual(mock_cls.call_count, 1)
        mock_conn.close.assert_not_called()

    def test_no_rdma_endpoints_method(self):
        """Client lacks rdma_endpoints attr → hasattr guard returns [], no error."""
        storage, mock_conn, mock_cls = _make_storage(rdma_endpoints=None)
        self.assertEqual(mock_cls.call_count, 1)
        mock_conn.close.assert_not_called()

    def test_discovery_exception_keeps_original(self):
        """rdma_endpoints() raises → warning logged, original conn preserved."""
        storage, mock_conn, _ = _make_storage(rdma_endpoints="raise")
        # Original connection preserved — close() NOT called
        mock_conn.close.assert_not_called()
        # Storage should still work (warmup passed)
        self.assertIsNotNone(storage.conn)


class TestWarmup(unittest.TestCase):
    """Tests for the _warmup() method (3-phase validation)."""

    def test_warmup_string_roundtrip_success(self):
        """setstr returns 0, getstr returns 'ok' → no assertion error."""
        # _make_storage runs warmup internally; if it doesn't raise, it passed.
        storage, _, _ = _make_storage()
        self.assertIsNotNone(storage)

    def test_warmup_setstr_failure(self):
        """setstr returns non-zero → AssertionError."""
        mock_conn = MagicMock()
        mock_conn.setstr.return_value = -1  # failure
        mock_conn._server_info = None
        mock_conn.set_timeout = MagicMock()
        del mock_conn.rdma_endpoints

        extra = {"remote_addr": "127.0.0.1", "remote_port": 18001, "password": ""}
        sc = _StubStorageConfig(extra_config=extra)

        with patch.object(cs_mod, "_make_connection", return_value=mock_conn):
            with self.assertRaises(AssertionError):
                cs_mod.CamaStorage(sc, None)

    def test_warmup_rdma_reg_failure(self):
        """reg_memory returns 0 → AssertionError."""
        mock_conn = MagicMock()
        mock_conn.setstr.return_value = 0
        mock_conn.getstr.return_value = "ok"
        mock_conn.delete.return_value = 0
        mock_conn.reg_memory.return_value = 0  # RDMA reg failure
        mock_conn._server_info = None
        mock_conn.set_timeout = MagicMock()
        del mock_conn.rdma_endpoints

        extra = {"remote_addr": "127.0.0.1", "remote_port": 18001, "password": ""}
        sc = _StubStorageConfig(extra_config=extra)

        with patch.object(cs_mod, "_make_connection", return_value=mock_conn):
            with self.assertRaises(AssertionError):
                cs_mod.CamaStorage(sc, None)

    def test_warmup_data_mismatch(self):
        """get succeeds but recv buffer differs from send → AssertionError."""
        mock_conn = MagicMock()
        mock_conn.setstr.return_value = 0
        mock_conn.getstr.return_value = "ok"
        mock_conn.delete.return_value = 0
        mock_conn.reg_memory.return_value = 99
        mock_conn.set.return_value = 0
        # get returns 0 (success) but does NOT modify recv_buf → data mismatch
        mock_conn.get.return_value = 0
        mock_conn.exists.side_effect = lambda k: (
            _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
        )
        mock_conn.dereg_memory.return_value = 0
        mock_conn._server_info = None
        mock_conn.set_timeout = MagicMock()
        del mock_conn.rdma_endpoints

        extra = {"remote_addr": "127.0.0.1", "remote_port": 18001, "password": ""}
        sc = _StubStorageConfig(extra_config=extra)

        with patch.object(cs_mod, "_make_connection", return_value=mock_conn):
            # The warmup creates send_buf with np.arange pattern and recv_buf as zeros.
            # Since our mock get() doesn't modify recv_buf, they won't match.
            # However, in the mock environment numpy operations run for real,
            # so send_buf != recv_buf → AssertionError.
            with self.assertRaises(AssertionError):
                cs_mod.CamaStorage(sc, None)

    def test_warmup_batch_mset_returns_list(self):
        """Warmup iterates mset result — must be list[int], not scalar."""
        mock_conn = MagicMock()
        mock_conn.setstr.return_value = 0
        mock_conn.getstr.return_value = "ok"
        mock_conn.delete.return_value = 0
        mock_conn.reg_memory.return_value = 99
        mock_conn.set.return_value = 0
        mock_conn.get.return_value = 0
        mock_conn.exists.side_effect = lambda k: (
            _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
        )
        mock_conn.dereg_memory.return_value = 0
        mock_conn._server_info = None
        mock_conn.set_timeout = MagicMock()
        del mock_conn.rdma_endpoints
        mock_conn.mexists.side_effect = lambda keys: [
            _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
            for k in keys
        ]
        mock_conn.mdel.return_value = 0
        mock_conn.mset_striped = MagicMock(
            side_effect=lambda keys, sgls, **kw: [0] * len(keys))

        # BUG: returning scalar 0 instead of list → TypeError in warmup
        mock_conn.mset.return_value = 0

        extra = {"remote_addr": "127.0.0.1", "remote_port": 18001, "password": ""}
        sc = _StubStorageConfig(extra_config=extra)

        with patch.object(cs_mod, "_make_connection", return_value=mock_conn), \
             patch("numpy.array_equal", return_value=True):
            with self.assertRaises(TypeError):
                cs_mod.CamaStorage(sc, None)

    def test_warmup_exists_missing_key_wrong_code(self):
        """exists(nonexistent) returns EXISTS_FOUND → AssertionError (pybind11 bug detection)."""
        mock_conn = MagicMock()
        mock_conn.setstr.return_value = 0
        mock_conn.getstr.return_value = "ok"
        mock_conn.delete.return_value = 0
        mock_conn.reg_memory.return_value = 99
        mock_conn.set.return_value = 0
        mock_conn.get.return_value = 0
        mock_conn.dereg_memory.return_value = 0
        mock_conn._server_info = None
        mock_conn.set_timeout = MagicMock()
        del mock_conn.rdma_endpoints

        # exists() always returns EXISTS_FOUND — even for non-existent keys
        mock_conn.exists.return_value = _RC.EXISTS_FOUND

        extra = {"remote_addr": "127.0.0.1", "remote_port": 18001, "password": ""}
        sc = _StubStorageConfig(extra_config=extra)

        with patch.object(cs_mod, "_make_connection", return_value=mock_conn):
            # Patch numpy so send == recv to get past the data comparison,
            # then the exists() check for nonexistent key should fail.
            with patch("numpy.array_equal", return_value=True):
                with self.assertRaises(AssertionError):
                    cs_mod.CamaStorage(sc, None)


class TestConfigLoading(unittest.TestCase):
    """Tests for CamaConfig triple-source loading."""

    def test_config_from_extra_config(self):
        """extra_config with remote_addr → CamaConfig fields match."""
        cfg = cs_mod.CamaConfig.load_from_extra_config({
            "remote_addr": "192.168.1.100",
            "remote_port": 7000,
            "password": "secret",
        })
        self.assertEqual(cfg.remote_addr, "192.168.1.100")
        self.assertEqual(cfg.remote_port, 7000)
        self.assertEqual(cfg.password, "secret")

    def test_config_from_env(self):
        """Env vars set → CamaConfig.load_from_env() reads them."""
        with patch.dict(os.environ, {
            "SGLANG_CAMA_REMOTE_ADDR": "10.1.1.1",
            "SGLANG_CAMA_REMOTE_PORT": "8888",
            "SGLANG_CAMA_PASSWORD": "pw",
        }):
            cfg = cs_mod.CamaConfig.load_from_env()
        self.assertEqual(cfg.remote_addr, "10.1.1.1")
        self.assertEqual(cfg.remote_port, 8888)
        self.assertEqual(cfg.password, "pw")

    def test_config_from_file(self):
        """JSON file → CamaConfig.from_file() parses correctly."""
        config_data = {
            "remote_addr": "10.2.2.2",
            "remote_port": 9999,
            "password": "file_pw",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            tmp_path = f.name

        try:
            # Make the env var point to our temp file
            _stub_envs.SGLANG_CAMA_CONFIG_PATH._override = tmp_path
            cfg = cs_mod.CamaConfig.from_file()
            self.assertEqual(cfg.remote_addr, "10.2.2.2")
            self.assertEqual(cfg.remote_port, 9999)
            self.assertEqual(cfg.password, "file_pw")
        finally:
            _stub_envs.SGLANG_CAMA_CONFIG_PATH._override = None
            os.unlink(tmp_path)

    def test_config_priority_extra_over_env(self):
        """Both extra_config and env set → extra_config wins."""
        with patch.dict(os.environ, {
            "SGLANG_CAMA_REMOTE_ADDR": "env_addr",
        }):
            storage, _, _ = _make_storage(remote_addr="extra_addr")
        self.assertEqual(storage.config.remote_addr, "extra_addr")

    def test_config_missing_remote_addr(self):
        """extra_config without remote_addr → ValueError."""
        with self.assertRaises(ValueError):
            cs_mod.CamaConfig.load_from_extra_config({"remote_port": 18001})

    def test_config_resilient_to_missing_env_attrs(self):
        """load_from_env() works even if envs is missing CAMA attributes (stale patch)."""
        # Temporarily remove attributes to simulate an older environ.py patch
        saved_attrs = {}
        for attr in ["SGLANG_CAMA_OP_TIMEOUT_S", "SGLANG_CAMA_IO_WORKERS"]:
            if hasattr(_stub_envs, attr):
                saved_attrs[attr] = getattr(_stub_envs, attr)
                delattr(_stub_envs, attr)
        try:
            cfg = cs_mod.CamaConfig.load_from_env()
            self.assertEqual(cfg.op_timeout_s, 10.0)  # hardcoded default
            self.assertEqual(cfg.io_workers, 8)        # hardcoded default
        finally:
            for attr, val in saved_attrs.items():
                setattr(_stub_envs, attr, val)

    def test_config_from_extra_config_resilient_to_missing_env_attrs(self):
        """load_from_extra_config() works even if envs is missing CAMA attributes."""
        saved_attrs = {}
        for attr in ["SGLANG_CAMA_OP_TIMEOUT_S", "SGLANG_CAMA_IO_WORKERS",
                      "SGLANG_CAMA_REMOTE_ADDR"]:
            if hasattr(_stub_envs, attr):
                saved_attrs[attr] = getattr(_stub_envs, attr)
                delattr(_stub_envs, attr)
        try:
            cfg = cs_mod.CamaConfig.load_from_extra_config({
                "remote_addr": "10.0.0.1",
            })
            self.assertEqual(cfg.remote_addr, "10.0.0.1")
            self.assertEqual(cfg.op_timeout_s, 10.0)
            self.assertEqual(cfg.io_workers, 8)
        finally:
            for attr, val in saved_attrs.items():
                setattr(_stub_envs, attr, val)


class TestKeyNaming(unittest.TestCase):
    """Tests for MHA/MLA suffix construction and key building."""

    def test_mha_suffix_no_pp(self):
        """pp_size=1, tp_rank=2 → suffix '2', keys '{hash}_2_k', '{hash}_2_v'."""
        storage, _, _ = _make_storage(tp_rank=2, pp_size=1)
        self.assertEqual(storage.mha_suffix, "2")
        self.assertFalse(storage.enable_pp)

    def test_mha_suffix_with_pp(self):
        """pp_size=2, tp_rank=1, pp_rank=0 → suffix '1_0'."""
        storage, _, _ = _make_storage(tp_rank=1, pp_rank=0, pp_size=2)
        self.assertEqual(storage.mha_suffix, "1_0")
        self.assertTrue(storage.enable_pp)

    def test_mla_suffix_no_pp(self):
        """MLA, pp_size=1 → suffix '', key '{hash}__k'."""
        storage, _, _ = _make_storage(is_mla=True, pp_size=1)
        self.assertEqual(storage.mla_suffix, "")

    def test_extra_backend_tag(self):
        """tag='prod' → keys prefixed 'prod_{hash}_...'."""
        extra = {
            "remote_addr": "10.0.0.1",
            "remote_port": 18001,
            "password": "",
            "extra_backend_tag": "prod",
        }
        storage, _, _ = _make_storage(extra_config=extra)
        self.assertEqual(storage.extra_backend_tag, "prod")
        tagged = storage._apply_tag(["abc123"])
        self.assertEqual(tagged, ["prod_abc123"])


class TestBatchPostprocess(unittest.TestCase):
    """Tests for _batch_postprocess (per-sub-key → per-page booleans)."""

    def test_mha_postprocess(self):
        """[0,0,0,1] → [True, False] (K+V pairs)."""
        storage, _, _ = _make_storage(is_mla=False)
        result = storage._batch_postprocess([0, 0, 0, 1], is_set_operate=False)
        self.assertEqual(result, [True, False])

    def test_mla_postprocess(self):
        """[0,1,0] → [True, False, True] (1:1)."""
        storage, _, _ = _make_storage(is_mla=True)
        result = storage._batch_postprocess([0, 1, 0], is_set_operate=False)
        self.assertEqual(result, [True, False, True])

    def test_set_vs_get_ok(self):
        """Uses _RC.SET_OK vs _RC.GET_OK correctly."""
        storage, _, _ = _make_storage(is_mla=True)
        # Both SET_OK and GET_OK are 0 in CAMA convention
        set_result = storage._batch_postprocess([0, 1], is_set_operate=True)
        get_result = storage._batch_postprocess([0, 1], is_set_operate=False)
        self.assertEqual(set_result, [True, False])
        self.assertEqual(get_result, [True, False])


class TestSGLangMetrics(unittest.TestCase):
    """Tests for SGLang metrics forwarding via update_sglang_metrics / get_stats."""

    def test_update_sglang_metrics_stores_data(self):
        """update_sglang_metrics() stores the dict for later use."""
        storage, _, _ = _make_storage()
        metrics = {
            "cache_hit_rate": 0.75,
            "token_usage": 0.6,
            "num_running_reqs": 10,
            "gen_throughput": 42.5,
            "evictable_ratio": 0.35,
        }
        storage.update_sglang_metrics(metrics)
        self.assertEqual(storage._sglang_metrics, metrics)

    def test_get_stats_includes_sglang_metrics_in_report(self):
        """get_stats() merges _sglang_metrics into report_stats() payload."""
        storage, mock_conn, _ = _make_storage()
        metrics = {
            "cache_hit_rate": 0.85,
            "token_usage": 0.5,
            "num_running_reqs": 5,
        }
        storage.update_sglang_metrics(metrics)
        storage.get_stats()

        # report_stats should have been called with a dict containing our metrics
        mock_conn.report_stats.assert_called_once()
        reported = mock_conn.report_stats.call_args[0][0]
        self.assertAlmostEqual(reported["cache_hit_rate"], 0.85)
        self.assertAlmostEqual(reported["token_usage"], 0.5)
        self.assertEqual(reported["num_running_reqs"], 5)

    def test_get_stats_works_without_sglang_metrics(self):
        """get_stats() doesn't break when no SGLang metrics have been set."""
        storage, mock_conn, _ = _make_storage()
        # Don't call update_sglang_metrics — _sglang_metrics is empty dict
        result = storage.get_stats()
        self.assertIsNotNone(result)
        # report_stats should still be called (with error counters at minimum)
        mock_conn.report_stats.assert_called_once()
        reported = mock_conn.report_stats.call_args[0][0]
        # SGLang keys should not be present
        self.assertNotIn("cache_hit_rate", reported)

    def test_update_sglang_metrics_merges_not_replaces(self):
        """Two callers pushing different keys → both sets of keys coexist."""
        storage, _, _ = _make_storage()
        # Simulate backup thread pushing its keys
        storage.update_sglang_metrics({
            "backup_ops_completed": 100,
            "backup_ops_failed": 2,
        })
        # Simulate prefetch thread pushing its keys
        storage.update_sglang_metrics({
            "prefetch_received": 50,
            "prefetch_io_completed": 48,
        })
        # Both sets of keys must be present
        self.assertEqual(storage._sglang_metrics["backup_ops_completed"], 100)
        self.assertEqual(storage._sglang_metrics["backup_ops_failed"], 2)
        self.assertEqual(storage._sglang_metrics["prefetch_received"], 50)
        self.assertEqual(storage._sglang_metrics["prefetch_io_completed"], 48)

    def test_update_sglang_metrics_none_is_noop(self):
        """Passing None does not clear previously stored metrics."""
        storage, _, _ = _make_storage()
        storage.update_sglang_metrics({"cache_hit_rate": 0.9})
        storage.update_sglang_metrics(None)
        self.assertAlmostEqual(storage._sglang_metrics["cache_hit_rate"], 0.9)

    def test_get_stats_includes_prefetch_bandwidth(self):
        """get_stats() derives prefetch_bandwidth_gbps from prefetch_bandwidth samples."""
        storage, mock_conn, _ = _make_storage()
        # Simulate bandwidth samples collected during an interval
        storage.prefetch_bandwidth.extend([1.5, 2.5, 3.0])
        storage.update_sglang_metrics({"cache_hit_rate": 0.5})
        storage.get_stats()

        reported = mock_conn.report_stats.call_args[0][0]
        # Average of [1.5, 2.5, 3.0] = 2.333...
        self.assertIn("prefetch_bandwidth_gbps", reported)
        self.assertAlmostEqual(reported["prefetch_bandwidth_gbps"], 7.0 / 3, places=3)


class TestCodecIntegration(unittest.TestCase):
    """Integration tests for codec paths in CamaStorage."""

    def test_codec_disabled_by_default(self):
        """_codec is None when config.codec is empty string."""
        storage, _, _ = _make_storage()
        self.assertIsNone(storage._codec)

    def test_codec_enabled_via_extra_config(self):
        """_codec is set when config.codec='int8'."""
        extra = {
            "remote_addr": "10.0.0.1",
            "remote_port": 18001,
            "password": "",
            "codec": "int8",
        }
        storage, _, _ = _make_storage(extra_config=extra)
        self.assertIsNotNone(storage._codec)
        self.assertEqual(storage._codec.name, "int8")
        self.assertTrue(storage._codec.is_lossy)

    def test_codec_chain_via_extra_config(self):
        """Chained codec 'int8+shuffle_zstd' is created."""
        extra = {
            "remote_addr": "10.0.0.1",
            "remote_port": 18001,
            "password": "",
            "codec": "int8+shuffle_zstd",
        }
        storage, _, _ = _make_storage(extra_config=extra)
        self.assertIsNotNone(storage._codec)
        self.assertEqual(storage._codec.name, "int8+shuffle_zstd")
        self.assertTrue(storage._codec.is_lossy)

    def test_codec_stats_in_get_stats(self):
        """get_stats() includes codec info when codec is enabled."""
        extra = {
            "remote_addr": "10.0.0.1",
            "remote_port": 18001,
            "password": "",
            "codec": "int8",
        }
        storage, mock_conn, _ = _make_storage(extra_config=extra)
        storage.get_stats()

        mock_conn.report_stats.assert_called_once()
        reported = mock_conn.report_stats.call_args[0][0]
        self.assertEqual(reported["codec"], "int8")
        self.assertTrue(reported["codec_lossy"])

    def test_codec_stats_absent_when_disabled(self):
        """get_stats() does not include codec keys when codec is disabled."""
        storage, mock_conn, _ = _make_storage()
        storage.get_stats()

        mock_conn.report_stats.assert_called_once()
        reported = mock_conn.report_stats.call_args[0][0]
        self.assertNotIn("codec", reported)
        self.assertNotIn("codec_lossy", reported)

    def test_config_codec_from_env(self):
        """SGLANG_CAMA_CODEC env var sets codec field via _env_get."""
        # Add the env var stub so _env_get can find it
        _stub_envs.SGLANG_CAMA_CODEC = _StubEnvVar("SGLANG_CAMA_CODEC", "")
        try:
            with patch.dict(os.environ, {"SGLANG_CAMA_CODEC": "int8"}):
                cfg = cs_mod.CamaConfig.load_from_env()
            self.assertEqual(cfg.codec, "int8")
        finally:
            delattr(_stub_envs, "SGLANG_CAMA_CODEC")

    def test_config_codec_default_empty(self):
        """Default codec is empty string (disabled)."""
        cfg = cs_mod.CamaConfig.load_from_env()
        self.assertEqual(cfg.codec, "")
        self.assertEqual(cfg.codec_zstd_level, 3)

    def test_codec_zstd_level_does_not_mutate_global(self):
        """BUG 2: Creating storage with codec_zstd_level=19 must NOT mutate the global singleton."""
        from cama_module.codec import _REGISTRY, ShuffleZstdCodec

        global_zstd = _REGISTRY["shuffle_zstd"]
        original_level = global_zstd._level

        extra = {
            "remote_addr": "10.0.0.1",
            "remote_port": 18001,
            "password": "",
            "codec": "shuffle_zstd",
            "codec_zstd_level": 19,
        }
        storage, _, _ = _make_storage(extra_config=extra)

        # Global singleton must NOT have been mutated
        self.assertEqual(global_zstd._level, original_level,
                         "Global ShuffleZstdCodec singleton was mutated!")
        # But the storage's codec should use level 19
        self.assertIsInstance(storage._codec, ShuffleZstdCodec)
        self.assertEqual(storage._codec._level, 19)
        self.assertIsNot(storage._codec, global_zstd)


class TestInitFailureCleanup(unittest.TestCase):
    """Tests for resource cleanup when __init__ fails partway through."""

    def _mock_conn(self):
        """Return a mock connection pre-configured for warmup success."""
        mock = MagicMock()
        mock.setstr.return_value = 0
        mock.getstr.return_value = "ok"
        mock.set.return_value = 0
        mock.get.return_value = 0
        mock.exists.side_effect = lambda k: (
            _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
        )
        mock.delete.return_value = 0
        mock.mset.side_effect = lambda keys, sgls, **kw: [0] * len(keys)
        mock.mset_striped = MagicMock(
            side_effect=lambda keys, sgls, **kw: [0] * len(keys))
        mock.mexists.side_effect = lambda keys: [
            _RC.EXISTS_FOUND if "nonexistent" not in k else _RC.EXISTS_MISSING
            for k in keys
        ]
        mock.mdel.return_value = 0
        mock.reg_memory.return_value = 12345
        mock.dereg_memory.return_value = 0
        mock.set_timeout = MagicMock()
        mock._server_info = None
        del mock.rdma_endpoints  # no RDMA
        return mock

    def test_warmup_failure_closes_connection(self):
        """If warmup (phase 5) fails, connection must be closed."""
        mock_conn = self._mock_conn()
        # Make warmup fail — getstr returns wrong value
        mock_conn.getstr.return_value = "WRONG"

        extra = {
            "remote_addr": "10.0.0.1",
            "remote_port": 18001,
            "password": "",
        }
        storage_config = _StubStorageConfig(extra_config=extra)
        _mock_kv_mod.create_pool = MagicMock(return_value=mock_conn)

        with self.assertRaises(Exception):
            with patch.object(cs_mod, "_PriskvClient", return_value=mock_conn), \
                 patch("numpy.array_equal", return_value=True):
                cs_mod.CamaStorage(storage_config, None)

        mock_conn.close.assert_called_once()

    def test_nic_double_failure_raises(self):
        """If multi-NIC and fallback both fail, init raises (not AttributeError on None)."""
        mock_conn = self._mock_conn()
        # Give it endpoints so it enters the multi-NIC path
        mock_conn.rdma_endpoints = MagicMock(return_value=[
            {"ip": "10.0.0.10", "port": 6380, "device": "mlx5_0"},
            {"ip": "10.0.0.11", "port": 6381, "device": "mlx5_1"},
        ])

        extra = {
            "remote_addr": "10.0.0.1",
            "remote_port": 18001,
            "password": "",
            "nic_striping": False,  # legacy single-NIC path
        }
        storage_config = _StubStorageConfig(tp_rank=1, extra_config=extra)
        _mock_kv_mod.create_pool = MagicMock(return_value=mock_conn)

        call_count = [0]
        def _failing_make_connection(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_conn  # initial connection succeeds
            raise ConnectionRefusedError("server down")

        with self.assertRaises(RuntimeError) as ctx:
            with patch.object(cs_mod, "_PriskvClient", return_value=mock_conn), \
                 patch.object(cs_mod, "_make_connection", side_effect=_failing_make_connection), \
                 patch("numpy.array_equal", return_value=True):
                cs_mod.CamaStorage(storage_config, None)

        self.assertIn("no usable connection", str(ctx.exception))

    def test_config_failure_no_cleanup_crash(self):
        """ValueError from config does not crash cleanup (no conn yet)."""
        # Config that triggers load_from_extra_config (has remote_addr) but
        # then we force a ValueError by patching resolve.
        storage_config = _StubStorageConfig(extra_config={
            "remote_addr": "10.0.0.1", "remote_port": 18001,
        })

        with self.assertRaises(ValueError):
            with patch.object(cs_mod.CamaConfig, "resolve",
                              side_effect=ValueError("bad config")), \
                 patch.object(cs_mod, "_PriskvClient", return_value=MagicMock()):
                cs_mod.CamaStorage(storage_config, None)
        # No AttributeError from _cleanup_partial_init when conn doesn't exist


if __name__ == "__main__":
    unittest.main()
