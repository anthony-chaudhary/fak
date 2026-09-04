"""CamaConnectorWorker — worker-side I/O for l3-vllm-connector.

Lifecycle (per forward):
1. start_load_kv(forward_context) — fires one mget_rdma covering every
   (request, block) pair from KVConnectorMetadata.loads. Sets per-layer events.
2. For each layer:
     wait_for_layer_load(name)  -> events[name].wait(...)
     <model runs forward for that layer>
     save_kv_layer(name, kv, attn_meta) -> append to pending save list
3. wait_for_save() — drains pending saves into one mset (or chunked).

RDMA buffer registration happens once in register_kv_caches; we hold tensor
refs strongly per the cama-client buf= GC-safety contract.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from l3_vllm.circuit_breaker import CircuitBreaker
from l3_vllm.config import CamaConnectorConfig
from l3_vllm.key_scheme import KeyScheme
from l3_vllm.metadata import CamaKVConnectorMetadata

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata

logger = logging.getLogger(__name__)


@dataclass
class _LayerRegistration:
    """One layer's CAMA memory-region registration."""

    name: str
    tensor: torch.Tensor       # held strongly — buf= GC-safety contract
    base_ptr: int
    reg_handle: int
    block_bytes: int
    num_blocks: int


class L3ConnectorWorker:
    def __init__(self, config: CamaConnectorConfig, key_scheme: KeyScheme):
        self.config = config
        self.key_scheme = key_scheme

        self._client: Any = None              # PriskvClient
        self._SGL: type | None = None         # cama_client.SGL constructor
        self._cb = CircuitBreaker(
            failure_threshold=config.cb_failure_threshold,
            overload_rate_per_sec=config.cb_overload_rate_per_sec,
            probe_interval_s=config.cb_probe_interval_s,
            close_after_successes=config.cb_close_after_successes,
        )

        self._layers: dict[str, _LayerRegistration] = {}
        self._kv_cache_generation = 0
        self._layer_order: list[str] = []

        # Per-forward state
        self._load_events: dict[str, threading.Event] = {}
        self._pending_saves: list[tuple[str, list[Any], list[int]]] = []  # (layer, hashes, block_ids)
        self._finished_loads: set[str] = set()
        self._finished_saves: set[str] = set()
        self._load_error_blocks: set[int] = set()
        self._metadata: CamaKVConnectorMetadata | None = None
        self._lock = threading.Lock()

        # Reported-present hashes for the scheduler (drained via
        # build_connector_worker_meta).
        self._reports_present: list[Any] = []
        self._reports_missing: list[Any] = []

    # ----- Connection management -----------------------------------------

    def _ensure_client(self) -> bool:
        """Lazy-connect on first use. Returns False if connection failed."""
        if self._client is not None:
            return True
        try:
            try:
                import l3_client as client_mod
            except ImportError:
                import cama_client as client_mod
            self._SGL = client_mod.SGL
            self._client = client_mod.PriskvClient(
                self.config.remote_addr,
                self.config.remote_port,
                password=self.config.password,
            )
            logger.info(
                "cama-vllm worker connected to %s:%d (transport=%s)",
                self.config.remote_addr,
                self.config.remote_port,
                type(self._client).__name__,
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("cama-vllm: connection failed (%s) — going OPEN", e)
            self._cb.on_failure()
            return False

    # ----- vLLM hooks ----------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register every layer's KV tensor as an RDMA memory region."""
        if not self._ensure_client():
            return
        try:
            self._kv_cache_generation += 1
            new_layers: dict[str, _LayerRegistration] = {}
            self._layer_order = list(kv_caches.keys())
            for name, tensor in kv_caches.items():
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                base_ptr = tensor.data_ptr()
                nbytes = tensor.element_size() * tensor.numel()
                # block_bytes: first dim of layer tensor is usually num_blocks.
                num_blocks = tensor.shape[0] if tensor.dim() >= 1 else 1
                block_bytes = nbytes // num_blocks if num_blocks > 0 else nbytes
                # buf= is the RDMA GC-safety contract; TCP fallback client
                # doesn't accept the kwarg. Try with first, fall back.
                try:
                    handle = self._client.reg_memory(base_ptr, nbytes, buf=tensor)
                except TypeError:
                    handle = self._client.reg_memory(base_ptr, nbytes)
                new_layers[name] = _LayerRegistration(
                    name=name,
                    tensor=tensor,           # strong ref — required
                    base_ptr=base_ptr,
                    reg_handle=handle,
                    block_bytes=block_bytes,
                    num_blocks=num_blocks,
                )
            # Deregister previous-generation handles.
            for old in self._layers.values():
                try:
                    self._client.dereg_memory(old.reg_handle)
                except Exception:  # noqa: BLE001
                    pass
            self._layers = new_layers
            self._cb.on_success()
            logger.info(
                "cama-vllm worker registered %d layers (gen=%d)",
                len(new_layers),
                self._kv_cache_generation,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("register_kv_caches failed: %s", e)
            self._cb.on_failure()

    def bind_metadata(self, metadata: CamaKVConnectorMetadata) -> None:
        self._metadata = metadata

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Fire one mget_rdma for all (request, block) pairs from metadata."""
        meta = self._metadata
        with self._lock:
            self._load_events = {name: threading.Event() for name in self._layer_order}
            self._finished_loads.clear()
            self._load_error_blocks.clear()

        if not self._cb.allow() or not self._ensure_client() or meta is None or not meta.loads:
            self._signal_all_loaded()
            return

        # Build the master list of (key, sgl) for every (load_spec, block, layer).
        all_keys: list[str] = []
        all_sgls: list[Any] = []
        for spec in meta.loads:
            for bh, bid in zip(spec.block_hashes, spec.block_ids):
                keys = self.key_scheme.keys_for_block(bh)
                for layer_name in self._layer_order:
                    reg = self._layers.get(layer_name)
                    if reg is None:
                        continue
                    if bid >= reg.num_blocks:
                        continue
                    block_ptr = reg.base_ptr + bid * reg.block_bytes
                    for k in keys:
                        all_keys.append(self._with_layer(k, layer_name))
                        all_sgls.append(self._SGL(
                            ptr=block_ptr,
                            size=reg.block_bytes // max(1, self.key_scheme.keys_per_block),
                            reg_handle=reg.reg_handle,
                        ))

        if not all_keys:
            self._signal_all_loaded()
            return

        try:
            results = self._client.mget_rdma(all_keys, all_sgls)
            ok_count = sum(1 for r in results if r == 0)
            logger.debug(
                "cama-vllm mget_rdma: %d/%d hits", ok_count, len(results)
            )
            # Track per-request finished — for v1 we treat all loads as
            # synchronous-complete once mget returns.
            for spec in meta.loads:
                self._finished_loads.add(spec.request_id)
                # Report status back to scheduler.
                self._reports_present.extend(spec.block_hashes)
            self._cb.on_success()
        except Exception as e:  # noqa: BLE001
            logger.warning("mget_rdma failed: %s", e)
            self._cb.on_failure()
            # Drop all loads — vLLM will recompute.
        finally:
            self._signal_all_loaded()

    def wait_for_layer_load(self, layer_name: str) -> None:
        ev = self._load_events.get(layer_name)
        if ev is None:
            return
        ev.wait(timeout=self.config.request_timeout_s)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        meta = self._metadata
        if meta is None or not meta.saves:
            return
        if not self._cb.allow():
            return
        # Enqueue — wait_for_save flushes in one batch.
        for spec in meta.saves:
            self._pending_saves.append((layer_name, spec.block_hashes, spec.block_ids))

    def wait_for_save(self) -> None:
        if not self._pending_saves:
            return
        if not self._cb.allow() or not self._ensure_client():
            self._pending_saves.clear()
            return
        try:
            all_keys: list[str] = []
            all_sgls: list[Any] = []
            seen_hashes: set[str] = set()
            for layer_name, hashes, block_ids in self._pending_saves:
                reg = self._layers.get(layer_name)
                if reg is None:
                    continue
                for bh, bid in zip(hashes, block_ids):
                    if bid >= reg.num_blocks:
                        continue
                    keys = self.key_scheme.keys_for_block(bh)
                    block_ptr = reg.base_ptr + bid * reg.block_bytes
                    sub_size = reg.block_bytes // max(1, self.key_scheme.keys_per_block)
                    for k in keys:
                        decorated = self._with_layer(k, layer_name)
                        if decorated in seen_hashes:
                            continue
                        seen_hashes.add(decorated)
                        all_keys.append(decorated)
                        all_sgls.append(self._SGL(
                            ptr=block_ptr,
                            size=sub_size,
                            reg_handle=reg.reg_handle,
                        ))

            if not all_keys:
                return

            # Chunk to honor send_buf_size.
            chunk = max(1, self.config.save_chunk_size)
            for i in range(0, len(all_keys), chunk):
                ks = all_keys[i:i + chunk]
                ss = all_sgls[i:i + chunk]
                results = self._client.mset(ks, ss)
                # Track for scheduler reports.
            for layer_name, hashes, _ in self._pending_saves:
                self._reports_present.extend(hashes)
            for spec in (self._metadata.saves if self._metadata else []):
                self._finished_saves.add(spec.request_id)
            self._cb.on_success()
        except Exception as e:  # noqa: BLE001
            logger.warning("wait_for_save mset failed: %s", e)
            self._cb.on_failure()
        finally:
            self._pending_saves.clear()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        with self._lock:
            sends = self._finished_saves & finished_req_ids
            recvs = self._finished_loads & finished_req_ids
            self._finished_saves -= sends
            self._finished_loads -= recvs
        return (sends or None, recvs or None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._lock:
            errs = set(self._load_error_blocks)
            self._load_error_blocks.clear()
        return errs

    def shutdown(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        for reg in self._layers.values():
            try:
                client.dereg_memory(reg.reg_handle)
            except Exception:  # noqa: BLE001
                pass
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        self._layers.clear()

    # ----- helpers --------------------------------------------------------

    def _signal_all_loaded(self) -> None:
        for ev in self._load_events.values():
            ev.set()

    def _with_layer(self, base_key: str, layer_name: str) -> str:
        """Decorate a block key with the per-layer suffix.

        cama_storage.py uses the layer slot index via `get_page_buffer_meta`;
        here we just include the layer name in the key so two layers don't
        collide. Safe for round-trip if both sides agree on the convention.
        """
        return f"{base_key}__L{layer_name}"


# Backward compatibility alias
CamaConnectorWorker = L3ConnectorWorker
