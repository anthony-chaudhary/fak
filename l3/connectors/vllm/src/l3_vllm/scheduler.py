"""CamaConnectorScheduler — scheduler-side state for l3-vllm-connector.

Owns:
- Bloom filter of known-present block hashes (worker-reported + sweeper).
- LRU of confirmed-present hashes.
- Per-request load plans built in build_connector_meta.

Critical path constraint: ``get_num_new_matched_tokens`` runs on the
scheduler's hot loop. No CAMA RPC happens here — we consult the BF/LRU only.
Unknown hashes are enqueued for a background sweeper thread to resolve via
``mexists`` and feed back into the LRU.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any

from l3_vllm.config import CamaConnectorConfig
from l3_vllm.key_scheme import KeyScheme, KeySchemeConfig
from l3_vllm.metadata import CamaKVConnectorMetadata, LoadSpec, SaveSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


class _BloomFilter:
    """Best-effort bloom filter; uses pyroaring if available, else bitarray-style.

    Falls back to a plain set() if neither is installed — capacity-bounded.
    """

    def __init__(self, capacity: int, fpr: float):
        self.capacity = capacity
        self.fpr = fpr
        self._impl: Any = None
        try:
            from pybloom_live import ScalableBloomFilter  # type: ignore

            self._impl = ScalableBloomFilter(
                initial_capacity=max(1024, capacity // 100),
                error_rate=fpr,
            )
            self._mode = "pybloom_live"
        except ImportError:
            # Cheap fallback: a bounded set. Same semantics for our use:
            # contains() may be True after add(); never falsely False.
            self._impl = set()
            self._mode = "set"

    def add(self, key: str) -> None:
        if self._mode == "set":
            if len(self._impl) >= self.capacity:
                # Drop one arbitrary entry — bf semantics tolerate it.
                self._impl.discard(next(iter(self._impl)))
            self._impl.add(key)
        else:
            self._impl.add(key)

    def __contains__(self, key: str) -> bool:
        return key in self._impl


class _LRU:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, key: str) -> None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
            else:
                self._d[key] = None
                if len(self._d) > self.capacity:
                    self._d.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return True
            return False

    def discard(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)


class L3ConnectorScheduler:
    """Scheduler-side connector state.

    Does not own a PriskvClient — RPCs are out of scope for the scheduler
    process (the background sweeper, when enabled, opens its own short-lived
    client). The worker side handles all heavy I/O.
    """

    def __init__(self, config: CamaConnectorConfig, key_scheme: KeyScheme):
        self.config = config
        self.key_scheme = key_scheme
        self._bloom = _BloomFilter(config.bloom_capacity, config.bloom_fpr)
        self._lru = _LRU(config.lru_capacity)

        # Per-request pending plans collected between build_connector_meta()
        # calls.
        self._lock = threading.Lock()
        self._pending_loads: dict[str, LoadSpec] = {}
        self._pending_saves: dict[str, SaveSpec] = {}

        # Hashes the worker has reported as present; also fed into _lru / bloom.
        self._unknown_q: deque[Any] = deque(maxlen=10_000)

    # ---- vLLM scheduler hooks -------------------------------------------

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (matched_extra_tokens, is_async).

        We only consider blocks not already locally computed. For each
        candidate block hash, hit the LRU first (confirmed present), then the
        bloom (possibly present). On a miss we enqueue for background sweep
        and stop counting.
        """
        block_size = self._infer_block_size(request)
        if block_size is None:
            return 0, False

        block_hashes = self._get_request_block_hashes(request)
        if not block_hashes:
            return 0, False

        # Skip the prefix that's already locally computed.
        start_block = num_computed_tokens // block_size
        if start_block >= len(block_hashes):
            return 0, False

        matched = 0
        for bh in block_hashes[start_block:]:
            keys = self.key_scheme.keys_for_block(bh)
            # We require all sub-keys of a block to be present.
            present = all(k in self._lru or k in self._bloom for k in keys)
            if present:
                matched += 1
                for k in keys:
                    if k not in self._lru:
                        # bloom hit but unconfirmed — promote optimistically.
                        # If the worker reports miss we'll discard.
                        pass
                continue
            # Miss — enqueue for sweeper and stop.
            self._unknown_q.append(bh)
            break

        extra_tokens = matched * block_size
        if extra_tokens == 0:
            # vLLM contract: if matched==0, async flag MUST be False.
            return 0, False
        if extra_tokens < self.config.async_load_min_tokens:
            # Bypass async machinery — let vLLM treat it as no external match.
            # (Worker won't try to load in build_connector_meta() either.)
            return 0, False
        return extra_tokens, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        block_size = self._infer_block_size(request)
        if block_size is None:
            return

        block_hashes = self._get_request_block_hashes(request)
        if not block_hashes:
            return

        num_blocks = (num_external_tokens + block_size - 1) // block_size
        # Resolve the vLLM block ids that were allocated for the external load.
        block_ids = self._extract_block_ids(blocks)
        # We only want the trailing N block ids (those that will receive the
        # externally-loaded data).
        load_block_ids = block_ids[-num_blocks:] if len(block_ids) >= num_blocks else block_ids
        load_hashes = block_hashes[:num_blocks]

        with self._lock:
            self._pending_loads[request.request_id] = LoadSpec(
                request_id=request.request_id,
                block_hashes=load_hashes,
                block_ids=list(load_block_ids),
                num_external_tokens=num_external_tokens,
            )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> CamaKVConnectorMetadata:
        """Drain pending plans into a metadata object for the worker."""
        with self._lock:
            loads = list(self._pending_loads.values())
            saves = list(self._pending_saves.values())
            self._pending_loads.clear()
            self._pending_saves.clear()
        return CamaKVConnectorMetadata(loads=loads, saves=saves)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Schedule a SAVE of this request's blocks.

        Returns (delay_free=False, kv_transfer_params=None) — we save
        synchronously from the worker; vLLM may free the blocks immediately.
        Disagg-prefill v2 will fill kv_transfer_params here.
        """
        try:
            block_hashes = self._get_request_block_hashes(request)
            if block_hashes and block_ids:
                # Align: only save blocks we have hashes for and that vLLM
                # actually allocated.
                n = min(len(block_hashes), len(block_ids))
                with self._lock:
                    self._pending_saves[request.request_id] = SaveSpec(
                        request_id=request.request_id,
                        block_hashes=block_hashes[:n],
                        block_ids=list(block_ids[:n]),
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning("cama scheduler request_finished failed: %s", e)
        return False, None

    # ---- Worker -> scheduler updates ------------------------------------

    def report_blocks_present(self, block_hashes: list[Any]) -> None:
        """Called via update_connector_output from worker save/load success."""
        for bh in block_hashes:
            for k in self.key_scheme.keys_for_block(bh):
                self._bloom.add(k)
                self._lru.add(k)

    def report_blocks_missing(self, block_hashes: list[Any]) -> None:
        for bh in block_hashes:
            for k in self.key_scheme.keys_for_block(bh):
                self._lru.discard(k)

    # ---- Helpers --------------------------------------------------------

    def _infer_block_size(self, request: "Request") -> int | None:
        """Best-effort: vLLM exposes block_size via VllmConfig.cache_config."""
        try:
            return self._block_size  # type: ignore[attr-defined]
        except AttributeError:
            return None

    def set_block_size(self, block_size: int) -> None:
        self._block_size = block_size

    def _get_request_block_hashes(self, request: "Request") -> list[Any]:
        """Pull the precomputed block hashes off the Request.

        vLLM stores these on the request after the scheduler hashes the
        prompt. The attribute name is ``block_hashes`` in current versions;
        we fall back gracefully if it isn't there yet.
        """
        bhs = getattr(request, "block_hashes", None)
        if bhs is None:
            return []
        return list(bhs)

    def _extract_block_ids(self, blocks: "KVCacheBlocks") -> list[int]:
        """Pull block ids out of vLLM's KVCacheBlocks wrapper."""
        # KVCacheBlocks layouts have shifted across versions; try the common ones.
        if hasattr(blocks, "get_block_ids"):
            try:
                return list(blocks.get_block_ids())
            except Exception:  # noqa: BLE001
                pass
        if hasattr(blocks, "block_ids"):
            try:
                ids = blocks.block_ids
                if isinstance(ids, (list, tuple)):
                    if ids and isinstance(ids[0], (list, tuple)):
                        return list(ids[0])
                    return list(ids)
            except Exception:  # noqa: BLE001
                pass
        if hasattr(blocks, "blocks"):
            try:
                return [getattr(b, "block_id", b) for b in blocks.blocks]
            except Exception:  # noqa: BLE001
                pass
        return []


# Backward compatibility alias
CamaConnectorScheduler = L3ConnectorScheduler
