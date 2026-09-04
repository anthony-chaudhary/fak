from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.
    """

    def __init__(
        self, stop_event, buffer_count: int = 3, max_buffer_size: int = 1024
    ) -> None:
        self.stop_event = stop_event
        self.buffers = Queue(maxsize=buffer_count)
        # todo: adjust the buffer size based on throughput profile of the system
        self.max_buffer_size = max_buffer_size

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error("[CacheOperationBuffer] put() failed: %s", e, exc_info=True)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error("[CacheOperationBuffer] get() failed: %s", e, exc_info=True)

    def clear(self):
        self.buffers.queue.clear()


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys
        self._source_ops = None

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: "StorageOperation"):
        return self.id < other.id


class BackupCoalescer:
    """Drain multiple 1-page StorageOperations and merge into one large batch.

    Single-consumer: called only from backup_thread_func.
    """

    def __init__(self, queue, stop_event, max_pages=2048, deadline_ms=20.0,
                 max_pages_ref=None, deadline_ref=None):
        self._queue = queue
        self._stop_event = stop_event
        self._max_pages = max_pages
        self._max_pages_ref = max_pages_ref  # callable returning live batch size
        self._deadline_ms = deadline_ms
        self._deadline_ref = deadline_ref  # callable returning effective deadline ms
        # Stats (read by health log, reset every 60s)
        self._coalesced_ops = 0
        self._coalesced_batches = 0

    def drain(self):
        """Drain ops from the queue, merging into one large StorageOperation.

        Phase 1: Blocking get for first op (matches original behavior).
        Phase 2: Non-blocking drain until max_pages reached or deadline expires.
        Returns None if queue is empty after timeout.
        """
        try:
            first = self._queue.get(block=True, timeout=1)
        except Empty:
            return None
        if first is None:
            return None

        ops = [first]
        pages = len(first.hash_value)
        effective_deadline = self._deadline_ref() if self._deadline_ref else self._deadline_ms
        deadline = time.monotonic() + (effective_deadline / 1000.0)

        max_pages = self._max_pages_ref() if self._max_pages_ref else self._max_pages
        while pages < max_pages and not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                op = self._queue.get(block=True, timeout=remaining)
            except Empty:
                break
            if op is None:
                continue
            ops.append(op)
            pages += len(op.hash_value)

        self._coalesced_ops += len(ops)
        self._coalesced_batches += 1

        if len(ops) == 1:
            ops[0]._source_ops = ops
            return ops[0]
        return self._merge_ops(ops)

    @staticmethod
    def _merge_ops(ops):
        """Merge multiple StorageOperations into one.

        NOTE: prefix_keys is set to None for multi-op merges because coalesced
        ops may originate from different radix tree branches with different prefixes.
        NOTE: total pages may slightly exceed max_pages because the budget
        check happens *before* the blocking get, not after.  This is benign
        (one extra op at most) and avoids complicating the hot loop.
        """
        host_indices = torch.cat([op.host_indices for op in ops])
        token_ids = []
        hash_value = []
        for op in ops:
            token_ids.extend(op.token_ids)
            hash_value.extend(op.hash_value)
        merged = StorageOperation(
            host_indices=host_indices,
            token_ids=token_ids,
            last_hash=ops[0].last_hash,
            hash_value=hash_value,
            prefix_keys=None,
        )
        merged._source_ops = ops
        return merged

    @property
    def avg_ops_per_batch(self):
        return self._coalesced_ops / self._coalesced_batches if self._coalesced_batches else 0.0

    def reset_stats(self):
        self._coalesced_ops = 0
        self._coalesced_batches = 0


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.pp_rank = pp_rank
        self.pp_size = pp_size

        if storage_backend is not None:
            self.storage_backend_type = storage_backend
            from sglang.srt.mem_cache.hicache_storage import get_hash_str

            self.get_hash_str = get_hash_str
            self.storage_config = self._generate_storage_config(
                model_name, storage_backend_extra_config
            )
            # for MLA models, only one rank needs to backup the KV cache
            self.backup_skip = (
                self.storage_config.is_mla_model
                # todo: load balancing
                and self.storage_config.tp_rank != 0
            )

            # Use storage backend factory for dynamic backend creation
            from sglang.srt.mem_cache.storage import StorageBackendFactory

            try:
                self.storage_backend = StorageBackendFactory.create_backend(
                    storage_backend, self.storage_config, self.mem_pool_host
                )
            except ValueError as e:
                raise ValueError(f"Failed to create storage backend: {e}") from e

            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            self.prefetch_capacity_limit = int(
                0.8 * (self.mem_pool_host.size - self.mem_pool_device.size)
            )
            # granularity of batch storage IO operations, in number of pages
            self.storage_batch_size = max(1, (storage_backend_extra_config or {}).get("storage_batch_size", 2048))
            # number of concurrent prefetch IO workers
            self.prefetch_io_workers = max(1, (storage_backend_extra_config or {}).get("prefetch_io_workers", 2))
            # number of concurrent backup IO workers (2 = matches prefetch_io_workers, one connection each)
            self.backup_io_workers = max(1, (storage_backend_extra_config or {}).get("backup_io_workers", 2))
            # per-sub-batch jitter to prevent thundering herd across TP ranks
            _raw_jitter = (storage_backend_extra_config or {}).get("backup_jitter_ms", 10)
            self.backup_jitter_ms = max(0, min(500, _raw_jitter))
            # adaptive batch sizing: auto-tune storage_batch_size based on observed latency
            self.batch_size_auto = bool((storage_backend_extra_config or {}).get("batch_size_auto", True))
            self.batch_size_latency_target_ms = max(1.0, float(
                (storage_backend_extra_config or {}).get("batch_size_latency_target_ms", 200)
            ))
            self._batch_size_max = max(1, (storage_backend_extra_config or {}).get(
                "batch_size_max", max(self.storage_batch_size, 4096)
            ))  # auto-tune ceiling; defaults above configured batch size
            self._batch_size_min = 32  # never go below this
            # Backup queue coalescing: merge multiple 1-page ops into one batch
            self._coalesce_backup_ops = bool(
                (storage_backend_extra_config or {}).get("coalesce_backup_ops", True)
            )
            self._coalesce_deadline_ms = max(0.0, min(500.0, float(
                (storage_backend_extra_config or {}).get("coalesce_deadline_ms", 20.0)
            )))
            # Prefetch queue coalescing: merge multiple small prefetch ops
            self._coalesce_prefetch_ops = bool(
                (storage_backend_extra_config or {}).get("coalesce_prefetch_ops", True)
            )
            self._prefetch_coalesce_deadline_ms = max(0.0, min(100.0, float(
                (storage_backend_extra_config or {}).get("prefetch_coalesce_deadline_ms", 5.0)
            )))
            self._prefetch_coalesce_max_pages = max(1, int(
                (storage_backend_extra_config or {}).get("prefetch_coalesce_max_pages", self.storage_batch_size)
            ))

            # Warmup controller: prevents cold-start death spiral
            from sglang.srt.mem_cache.storage.cama.warmup import WarmupConfig, WarmupController
            _ec = storage_backend_extra_config or {}
            _warmup_cfg = WarmupConfig(
                enabled=bool(_ec.get("warmup_enabled", True)),
                cold_batch_size=int(_ec.get("warmup_cold_batch_size", self._batch_size_max)),
                cold_jitter_ms=float(_ec.get("warmup_cold_jitter_ms", 0.0)),
                cold_deadline_ms=float(_ec.get("warmup_cold_deadline_ms", 2.0)),
                min_batch_size=int(_ec.get("warmup_min_batch_size", 256)),
                server_poll_interval_s=float(_ec.get("warmup_server_poll_interval_s", 2.0)),
                server_poll_timeout_s=float(_ec.get("warmup_server_poll_timeout_s", 300.0)),
                tp_size=getattr(self, "tp_size", 8),
                # Backward-compat: pass old keys so _apply_compat_mapping can map them
                aggressive_batch_size=int(_ec.get("warmup_aggressive_batch_size", 0)),
                aggressive_jitter_ms=float(_ec.get("warmup_aggressive_jitter_ms", -1.0)),
                aggressive_deadline_ms=float(_ec.get("warmup_aggressive_deadline_ms", -1.0)),
            )
            self._warmup = WarmupController(
                _warmup_cfg,
                steady_batch_size=self.storage_batch_size,
                steady_jitter_ms=self.backup_jitter_ms,
                steady_deadline_ms=self._coalesce_deadline_ms,
                local_rank=self.storage_config.tp_rank,
            )

            # Wire warmup cold check so CamaStorage can skip dedup during COLD
            if hasattr(self.storage_backend, "set_warmup_cold_check"):
                self.storage_backend.set_warmup_cold_check(self._warmup.is_cache_cold)
            elif hasattr(self.storage_backend, "set_warmup_phase_ref"):
                # Legacy fallback
                self.storage_backend.set_warmup_phase_ref(self._warmup.is_cache_cold)

            # Wire warmup reset so reconnect resets warmup state
            if hasattr(self.storage_backend, "set_warmup_reset_ref"):
                self.storage_backend.set_warmup_reset_ref(self._warmup.reset)

            # Wire conn_factory so poller can check server readiness
            if hasattr(self.storage_backend, "conn"):
                self._warmup.set_conn_factory(lambda: self.storage_backend.conn)

            # host alloc drop counter (only written from main scheduler thread)
            self._host_alloc_drops = 0
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # --- Load-back device headroom ---
            _headroom_pct = max(0.05, min(0.50, float(
                (storage_backend_extra_config or {}).get("load_back_headroom_pct", 0.20)
            )))
            self._device_headroom_tokens = int(_headroom_pct * self.mem_pool_device_allocator.size)
            self._load_back_capacity = self.mem_pool_device_allocator.size - self._device_headroom_tokens
            self.load_tokens_pending = 0
            self._load_back_skipped = 0

            # create a new communication group for synchronizing storage operations across TP workers
            self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
            if self.tp_world_size > 1:
                from sglang.srt.distributed.parallel_state import (
                    create_custom_parallel_group,
                )

                group_ranks = torch.distributed.get_process_group_ranks(tp_group)
                self.prefetch_tp_group = create_custom_parallel_group(
                    group_ranks=group_ranks, backend="gloo"
                )

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (self.storage_backend_type in ["hf3fs", "mooncake", "eic", "cama"]) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_queue = Queue()
            self.backup_queue = Queue()
            self._coalescer = BackupCoalescer(
                queue=self.backup_queue,
                stop_event=self.stop_event,
                max_pages=self.storage_batch_size,
                deadline_ms=self._coalesce_deadline_ms,
                max_pages_ref=lambda: self._warmup.effective_batch_size(),
                deadline_ref=lambda: self._warmup.effective_deadline_ms(),
            ) if self._coalesce_backup_ops else None

            self.prefetch_revoke_queue = Queue()
            self.ack_backup_queue = Queue()
            self.host_mem_release_queue = Queue()

            self.prefetch_thread.start()
            self.backup_thread.start()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        is_mla_backend = isinstance(self.mem_pool_device, MLATokenToKVPool)

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            is_mla_model=is_mla_backend,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            extra_config=storage_backend_extra_config,
        )

    def push_scheduler_metrics(self, metrics: dict) -> None:
        """Forward scheduler-level metrics to storage backend for Prometheus."""
        if hasattr(self, 'storage_backend') and self.storage_backend is not None:
            try:
                self.storage_backend.update_sglang_metrics(metrics)
            except Exception:
                pass

    def reset(self):
        logger.info("[reset] starting: clearing queues and restarting threads")
        self.stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()
            if self._coalescer is not None:
                self._coalescer.reset_stats()

        self.stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()
        logger.info("[reset] complete")

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            self._host_alloc_drops += 1
            # Throttle: log first 5 drops at WARNING, then every 50th.
            # Host alloc drops are normal under memory pressure.
            drops = self._host_alloc_drops
            if drops <= 5 or drops % 50 == 0:
                logger.warning(
                    "[write] host alloc failed: requested=%d indices, node_id=%d (total_drops=%d)",
                    len(device_indices), node_id, drops,
                )
            else:
                logger.debug(
                    "[write] host alloc failed: requested=%d indices, node_id=%d (total_drops=%d)",
                    len(device_indices), node_id, drops,
                )
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
        self.write_queue.clear()
        logger.debug(
            "[write] start_writing: indices=%d, node_ids=%s",
            len(host_indices), op.node_ids,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        need_size = len(host_indices)

        # Backpressure: refuse if load queue would exceed capacity
        if hasattr(self, "load_tokens_pending"):
            if self.load_tokens_pending + need_size > self._load_back_capacity:
                self._load_back_skipped += 1
                if self._load_back_skipped % 100 == 1:
                    logger.warning(
                        "[load] skipped: pending=%d + need=%d > capacity=%d (skipped=%d)",
                        self.load_tokens_pending, need_size,
                        self._load_back_capacity, self._load_back_skipped,
                    )
                return None

            # Refuse if device headroom for decode would be breached
            avail = self.mem_pool_device_allocator.available_size()
            if avail - need_size < self._device_headroom_tokens:
                self._load_back_skipped += 1
                if self._load_back_skipped % 100 == 1:
                    logger.warning(
                        "[load] skipped: avail=%d - need=%d < headroom=%d (skipped=%d)",
                        avail, need_size,
                        self._device_headroom_tokens, self._load_back_skipped,
                    )
                return None

        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            logger.warning(
                "[load] device alloc failed: requested=%d indices, node_id=%d",
                len(host_indices), node_id,
            )
            return None

        if hasattr(self, "load_tokens_pending"):
            self.load_tokens_pending += need_size

        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def move_indices(self, op: CacheOperation):
        host_indices, device_indices = op.host_indices, op.device_indices
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        # Decrement pending counter for all items about to be loaded
        if hasattr(self, "load_tokens_pending"):
            drained = sum(len(op.host_indices) for op in self.load_queue)
            self.load_tokens_pending = max(0, self.load_tokens_pending - drained)

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(op)
        self.load_queue.clear()
        logger.info(
            "[prefetch][start_loading] indices=%d, layers=%d, producer_id=%d",
            len(host_indices), self.layer_num, producer_id,
        )
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )
                producer_event.complete(i)
            logger.debug(
                "[prefetch][start_loading] all %d layers transferred, producer_id=%d",
                self.layer_num, producer_id,
            )
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        logger.debug(
            "[prefetch][entry] request=%s enqueued: tokens=%d, pages=%d, op_id=%d",
            request_id, len(new_input_tokens),
            len(new_input_tokens) // self.page_size, operation.id,
        )
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        logger.debug(
            "[prefetch][terminate] request=%s completed_tokens=%d, dispatched_pages=%d, elapsed=%.3fs",
            operation.request_id, operation.completed_tokens,
            len(operation.hash_value),
            time.monotonic() - operation.start_time,
        )
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        failed_at = -1
        for i in range(len(hash_values)):
            if not results[i]:
                failed_at = i
                break
            inc += self.page_size
        if failed_at >= 0:
            logger.warning(
                "[prefetch][page_get] request=%s FAIL: succeeded=%d/%d pages",
                operation.request_id, failed_at, len(hash_values),
            )
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            logger.warning(
                "[prefetch][page_get] request=%s FAIL: batch_get returned None for %d pages",
                operation.request_id, len(hash_values),
            )
            return
        succeeded = 0
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    "[prefetch][page_get] request=%s FAIL: %d/%d pages succeeded (first_failure_idx=%d)",
                    operation.request_id, succeeded, len(hash_values), i,
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller
            succeeded += 1

    def _page_transfer(self, operation):
        # Transfer batch by batch — use warmup-aware batch size (mirrors _page_backup)
        prefix_keys = operation.prefix_keys
        eff_batch = self._warmup.effective_batch_size() if hasattr(self, '_warmup') else self.storage_batch_size
        for i in range(0, len(operation.hash_value), eff_batch):
            if self.stop_event.is_set():
                operation.mark_terminate()
                break
            batch_hashes = operation.hash_value[i : i + eff_batch]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            expected_tokens = prev_completed_tokens + len(batch_hashes) * self.page_size
            if operation.completed_tokens != expected_tokens:
                logger.warning(
                    "[prefetch][transfer] request=%s batch FAIL: expected=%d, actual=%d tokens (%d pages). Terminating.",
                    operation.request_id,
                    expected_tokens, operation.completed_tokens, len(batch_hashes),
                )
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    @staticmethod
    def _merge_prefetch_ops(ops):
        """Merge multiple PrefetchOperations into one for batched IO.

        Similar to BackupCoalescer._merge_ops but for PrefetchOperation.
        prefix_keys is set to None for multi-op merges (same rationale as backup).
        """
        host_indices = torch.cat([op.host_indices for op in ops])
        token_ids = []
        hash_value = []
        for op in ops:
            token_ids.extend(op.token_ids)
            hash_value.extend(op.hash_value)
        merged = PrefetchOperation(
            request_id=f"{ops[0].request_id}+{len(ops) - 1}",
            host_indices=host_indices,
            token_ids=token_ids,
            prefix_keys=None,
        )
        merged.hash_value = hash_value
        merged._source_ops = ops
        return merged

    _PrefetchIOResult = NamedTuple("_PrefetchIOResult", [
        ("success", bool), ("tokens", int), ("pages", int), ("elapsed_s", float),
    ])

    def _prefetch_io_task(self, operation) -> "_PrefetchIOResult":
        """Execute a single prefetch IO operation (runs in executor thread).

        Returns a _PrefetchIOResult so the reaping loop can accumulate
        tokens/pages/latency for the enriched health log.
        """
        io_start = time.monotonic()
        try:
            self._page_transfer(operation)
            io_elapsed = time.monotonic() - io_start
            tokens = operation.completed_tokens
            pages = len(operation.hash_value)
            if operation.is_terminated():
                logger.warning(
                    "[prefetch][io] request=%s FAIL: partial transfer, "
                    "completed=%d/%d tokens, elapsed=%.3fs",
                    operation.request_id, tokens,
                    pages * self.page_size, io_elapsed,
                )
                return self._PrefetchIOResult(False, tokens, pages, io_elapsed)
            else:
                logger.debug(
                    "[prefetch][io] request=%s SUCCESS: transferred=%d tokens (%d pages), elapsed=%.3fs",
                    operation.request_id, tokens, pages, io_elapsed,
                )
                return self._PrefetchIOResult(True, tokens, pages, io_elapsed)
        except Exception as e:
            io_elapsed = time.monotonic() - io_start
            logger.warning(
                "[prefetch][io] request=%s FAIL: exception after %.3fs, completed=%d tokens: %s",
                operation.request_id, io_elapsed, operation.completed_tokens, e,
                exc_info=True,
            )
            operation.mark_terminate()
            return self._PrefetchIOResult(False, operation.completed_tokens, 0, io_elapsed)
        finally:
            self.append_host_mem_release(
                operation.host_indices[operation.completed_tokens:]
            )

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            now = time.monotonic()
            if now - getattr(self, '_last_rl_log', 0) >= 10:
                logger.info(
                    "Prefetch rate-limited: occupied=%d >= limit=%d",
                    self.prefetch_tokens_occupied, self.prefetch_capacity_limit,
                )
                self._last_rl_log = now
            return True
        # Stop prefetching if device memory is under pressure
        if hasattr(self, "_device_headroom_tokens"):
            avail = self.mem_pool_device_allocator.available_size()
            if avail < self._device_headroom_tokens * 2:
                now = time.monotonic()
                if now - getattr(self, '_last_dev_rl_log', 0) >= 10:
                    logger.info(
                        "Prefetch rate-limited (device pressure): avail=%d < 2*headroom=%d",
                        avail, self._device_headroom_tokens * 2,
                    )
                    self._last_dev_rl_log = now
                return True
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []

        for start in range(
            0, len(tokens_to_fetch), self.page_size * self.storage_batch_size
        ):
            end = min(
                start + self.page_size * self.storage_batch_size, len(tokens_to_fetch)
            )
            batch_tokens = tokens_to_fetch[start:end]
            batch_hashes = []
            for i in range(0, len(batch_tokens), self.page_size):
                last_hash = self.get_hash_str(
                    batch_tokens[i : i + self.page_size], last_hash
                )
                batch_hashes.append(last_hash)
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            logger.debug(
                "[prefetch][hit_query] batch: hit=%d/%d pages, offset=%d",
                hit_page_num, len(batch_hashes), start,
            )
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        executor = ThreadPoolExecutor(
            max_workers=self.prefetch_io_workers,
            thread_name_prefix="prefetch_io",
        )
        futures = []
        total_received = 0
        total_dispatched = 0
        total_revoked = 0
        total_coalesced = 0
        io_completed = 0
        io_failed = 0
        total_tokens = 0
        total_pages = 0
        total_io_s = 0.0
        consecutive_failures = 0
        last_health_log = time.monotonic()

        # Prefetch coalescing state
        _coalesce = self._coalesce_prefetch_ops
        _coalesce_deadline = self._prefetch_coalesce_deadline_ms
        _coalesce_max_pages = self._prefetch_coalesce_max_pages
        _pending_ops = []
        _pending_pages = 0
        _pending_start = None

        def _flush_pending():
            nonlocal _pending_ops, _pending_pages, _pending_start, total_coalesced
            if not _pending_ops:
                return
            try:
                if len(_pending_ops) == 1:
                    futures.append(executor.submit(self._prefetch_io_task, _pending_ops[0]))
                else:
                    merged = self._merge_prefetch_ops(_pending_ops)
                    total_coalesced += len(_pending_ops) - 1
                    logger.debug(
                        "[prefetch][coalesce] merged %d ops (%d pages) into one batch",
                        len(_pending_ops), _pending_pages,
                    )
                    futures.append(executor.submit(self._prefetch_io_task, merged))
            except RuntimeError:
                # Executor shut down (interpreter exiting) — normal during teardown
                logger.debug("[prefetch_thread] executor shut down during flush")
            _pending_ops = []
            _pending_pages = 0
            _pending_start = None

        logger.info(
            "[prefetch_thread] Started with ThreadPoolExecutor (workers=%d)",
            self.prefetch_io_workers,
        )

        try:
            while (not self.stop_event.is_set()) or not self.prefetch_queue.empty():
                try:
                    # Reap completed futures
                    still_pending = []
                    for f in futures:
                        if f.done():
                            try:
                                result = f.result()
                                if result.success:
                                    io_completed += 1
                                else:
                                    io_failed += 1
                                total_tokens += result.tokens
                                total_pages += result.pages
                                total_io_s += result.elapsed_s
                            except Exception:
                                io_failed += 1
                        else:
                            still_pending.append(f)
                    futures = still_pending

                    # Periodic health log
                    now = time.monotonic()
                    if now - last_health_log >= 60:
                        completed_ios = io_completed + io_failed
                        avg_io_ms = (total_io_s / completed_ios * 1000) if completed_ios > 0 else 0.0
                        tok_label = f"{total_tokens / 1000:.0f}K" if total_tokens >= 1000 else str(total_tokens)
                        _lb_skip = getattr(self, '_load_back_skipped', 0)
                        _lb_pend = getattr(self, 'load_tokens_pending', 0)
                        logger.info(
                            "[prefetch_thread] health: received=%d, dispatched=%d, revoked=%d, coalesced=%d, "
                            "io_completed=%d, io_failed=%d, io_in_flight=%d, "
                            "total_tokens=%s, total_pages=%d, avg_io=%.1fms, "
                            "query_failures=%d, queue_size=%d, "
                            "load_back_skipped=%d, load_tokens_pending=%d",
                            total_received, total_dispatched, total_revoked, total_coalesced,
                            io_completed, io_failed, len(futures),
                            tok_label, total_pages, avg_io_ms,
                            consecutive_failures, self.prefetch_queue.qsize(),
                            _lb_skip, _lb_pend,
                        )
                        last_health_log = now

                        # Push prefetch metrics to storage backend for Prometheus
                        try:
                            self.storage_backend.update_sglang_metrics({
                                "prefetch_received": total_received,
                                "prefetch_dispatched": total_dispatched,
                                "prefetch_revoked": total_revoked,
                                "prefetch_coalesced": total_coalesced,
                                "prefetch_io_completed": io_completed,
                                "prefetch_io_failed": io_failed,
                                "prefetch_tokens": total_tokens,
                                "prefetch_pages": total_pages,
                                "prefetch_avg_io_ms": avg_io_ms,
                                "prefetch_queue_depth": self.prefetch_queue.qsize(),
                                "prefetch_in_flight": len(futures),
                                "load_back_skipped": _lb_skip,
                                "load_tokens_pending": _lb_pend,
                            })
                        except Exception:
                            pass

                        # Push evictable_ratio from host KV pool
                        if hasattr(self, 'token_to_kv_pool_host') and self.token_to_kv_pool_host is not None:
                            try:
                                pool = self.token_to_kv_pool_host
                                avail = pool.available_size() if callable(getattr(pool, 'available_size', None)) else 0
                                total = getattr(pool, 'size', 0)
                                if total > 0:
                                    self.storage_backend.update_sglang_metrics({"evictable_ratio": avail / total})
                            except Exception:
                                pass

                    operation = self.prefetch_queue.get(block=True, timeout=1)
                    if operation is None:
                        continue

                    total_received += 1
                    logger.debug(
                        "[prefetch][pickup] request=%s received: tokens=%d, queue_wait=%.3fs, total_received=%d",
                        operation.request_id, len(operation.token_ids),
                        time.monotonic() - operation.start_time, total_received,
                    )

                    try:
                        query_start = time.monotonic()
                        hash_value, storage_hit_count = self._storage_hit_query(operation)
                        logger.debug(
                            "[prefetch][query] request=%s SUCCESS: hit_pages=%d, hit_tokens=%d, elapsed=%.3fs",
                            operation.request_id, len(hash_value), storage_hit_count,
                            time.monotonic() - query_start,
                        )
                    except Exception as e:
                        consecutive_failures += 1
                        logger.warning(
                            "[prefetch][query] request=%s FAIL: elapsed=%.3fs, consecutive=%d, error=%s",
                            operation.request_id, time.monotonic() - query_start,
                            consecutive_failures, e, exc_info=(consecutive_failures <= 3),
                        )
                        if consecutive_failures == 10:
                            logger.error("[prefetch][query] 10 consecutive failures -- storage may be down")
                        self.prefetch_revoke_queue.put(operation.request_id)
                        self.append_host_mem_release(operation.host_indices)
                        total_revoked += 1
                        continue

                    consecutive_failures = 0

                    if self.tp_world_size > 1:
                        storage_hit_count_tensor = torch.tensor(
                            storage_hit_count, dtype=torch.int
                        )
                        torch.distributed.all_reduce(
                            storage_hit_count_tensor,
                            op=torch.distributed.ReduceOp.MIN,
                            group=self.prefetch_tp_group,
                        )
                        storage_hit_count = storage_hit_count_tensor.item()

                    if storage_hit_count < self.prefetch_threshold:
                        # not to prefetch if not enough benefits
                        self.prefetch_revoke_queue.put(operation.request_id)
                        self.append_host_mem_release(operation.host_indices)
                        total_revoked += 1
                        logger.debug(
                            "[prefetch][threshold] request=%s REVOKED: hit_tokens=%d < threshold=%d, elapsed=%.3fs",
                            operation.request_id, storage_hit_count, self.prefetch_threshold,
                            time.monotonic() - operation.start_time,
                        )
                    else:
                        operation.hash_value = hash_value[
                            : (storage_hit_count // self.page_size)
                        ]
                        # free the pre-allocated memory for pages that are not hit
                        self.append_host_mem_release(
                            operation.host_indices[storage_hit_count:]
                        )
                        operation.host_indices = operation.host_indices[:storage_hit_count]
                        total_dispatched += 1
                        logger.debug(
                            "[prefetch][threshold] request=%s DISPATCHED: pages=%d, tokens=%d, elapsed=%.3fs",
                            operation.request_id, len(operation.hash_value), storage_hit_count,
                            time.monotonic() - operation.start_time,
                        )
                        if _coalesce:
                            _pending_ops.append(operation)
                            _pending_pages += len(operation.hash_value)
                            if _pending_start is None:
                                _pending_start = time.monotonic()
                            # Flush if page budget or deadline exceeded
                            if _pending_pages >= _coalesce_max_pages or \
                               (time.monotonic() - _pending_start) * 1000 >= _coalesce_deadline:
                                _flush_pending()
                        else:
                            try:
                                futures.append(executor.submit(self._prefetch_io_task, operation))
                            except RuntimeError:
                                logger.debug("[prefetch_thread] executor shut down, exiting")
                                break

                except Empty:
                    # Queue empty — flush any pending coalesced ops
                    if _coalesce:
                        _flush_pending()
                    continue
                except Exception as e:
                    logger.error("[prefetch_thread] unexpected error: %s", e, exc_info=True)

                # Check deadline for pending ops (may not have hit page budget)
                if _coalesce and _pending_start is not None and \
                   (time.monotonic() - _pending_start) * 1000 >= _coalesce_deadline:
                    _flush_pending()
        finally:
            # Flush remaining pending ops before shutdown
            if _coalesce:
                _flush_pending()
            executor.shutdown(wait=True, cancel_futures=True)

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        qsize = self.backup_queue.qsize()
        if qsize > 0 and qsize % 100 == 0:
            logger.warning("[write_storage] backup_queue depth=%d", qsize)
        logger.debug(
            "[backup_thread] write_storage enqueued: op_id=%d, tokens=%d, pages=%d",
            operation.id, len(token_ids),
            len(hash_value) if hash_value is not None else 0,
        )
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _jitter_sleep(self) -> float:
        """Sleep random [0, effective_jitter] ms. Returns actual ms, 0.0 if disabled."""
        effective_jitter = self._warmup.effective_jitter_ms()
        if effective_jitter <= 0 or self.stop_event.is_set():
            return 0.0
        delay_ms = random.uniform(0, effective_jitter)
        # stop_event.wait() instead of time.sleep() — wakes on shutdown
        self.stop_event.wait(timeout=delay_ms / 1000.0)
        return delay_ms

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = list(operation.prefix_keys) if operation.prefix_keys else None
        total_jitter_ms = 0.0
        batch_count = 0
        last_batch_end = None
        gap_sum_ms = 0.0
        gap_count = 0

        eff_batch = self._warmup.effective_batch_size()
        for i in range(0, len(operation.hash_value), eff_batch):
            # Jitter between sub-batches (not before the first)
            if batch_count > 0:
                total_jitter_ms += self._jitter_sleep()
                if self.stop_event.is_set():
                    break

            batch_hashes = operation.hash_value[i : i + eff_batch]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            should_skip_dedup = self.write_policy == "write_back"
            extra_info = HiCacheStorageExtraInfo(
                prefix_keys=prefix_keys,
                extra_info={"skip_dedup": should_skip_dedup},
            )
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)

            now = time.monotonic()
            if last_batch_end is not None:
                gap_sum_ms += (now - last_batch_end) * 1000
                gap_count += 1
            last_batch_end = now

            if not success:
                total_tokens = len(operation.hash_value) * self.page_size if operation.hash_value else 0
                logger.warning(
                    "[backup_thread] page_backup failed: %d pages at offset=%d, op_id=%d, "
                    "completed=%d/%d tokens",
                    len(batch_hashes), i, operation.id,
                    operation.completed_tokens, total_tokens,
                )
                break

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)
            batch_count += 1
            logger.debug(
                "[backup_thread] page_backup batch: pages=%d, completed_tokens=%d, op_id=%d",
                len(batch_hashes), operation.completed_tokens, operation.id,
            )

        # Store stats on operation for _backup_io_task to harvest
        operation._jitter_ms = total_jitter_ms
        operation._avg_gap_ms = (gap_sum_ms / gap_count) if gap_count > 0 else 0.0

    _BackupResult = NamedTuple("_BackupResult", [
        ("success", bool), ("elapsed_ms", float), ("pages", int),
        ("jitter_ms", float), ("avg_gap_ms", float),
    ])

    def _backup_io_task(self, operation) -> "_BackupResult":
        """Execute one backup operation (runs in executor thread)."""
        t0 = time.monotonic()
        pages = 0
        success = False
        try:
            if not self.backup_skip:
                self._page_backup(operation)
            success = True
        except Exception as e:
            logger.warning("[backup_io] op_id=%d failed: %s", operation.id, e)
        finally:
            # Always credit partial progress to warmup — even failed ops
            # may have written some batches before the exception.
            pages = operation.completed_tokens // self.page_size
            if pages > 0:
                self._warmup.record_pages(pages)
            for src_op in getattr(operation, '_source_ops', None) or [operation]:
                self.ack_backup_queue.put(src_op)
        jitter_ms = getattr(operation, '_jitter_ms', 0.0)
        avg_gap_ms = getattr(operation, '_avg_gap_ms', 0.0)
        return self._BackupResult(success, (time.monotonic() - t0) * 1000, pages, jitter_ms, avg_gap_ms)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.

        Uses a ThreadPoolExecutor with ``backup_io_workers`` threads (default 2).
        The executor pattern mirrors ``prefetch_thread_func`` for consistency.
        """
        executor = ThreadPoolExecutor(
            max_workers=self.backup_io_workers, thread_name_prefix="backup_io"
        )
        futures = []
        ops_completed = 0
        ops_failed = 0
        backup_latencies_ms: list = []
        backup_jitter_ms_total: float = 0.0
        backup_gap_ms_samples: list = []
        last_health_log = time.monotonic()

        _banner = (
            "\n"
            + "=" * 60 + "\n"
            + "  CAMA Backup Thread Settings\n"
            + "=" * 60 + "\n"
            + f"  backup_io_workers      : {self.backup_io_workers}\n"
            + f"  storage_batch_size     : {self.storage_batch_size}\n"
            + f"  coalesce_backup_ops    : {'ON' if self._coalesce_backup_ops else 'OFF'}\n"
            + f"  coalesce_deadline_ms   : {self._coalesce_deadline_ms:.1f}\n"
            + f"  batch_size_auto        : {'ON' if self.batch_size_auto else 'OFF'}\n"
            + f"  batch_size_max         : {self._batch_size_max}\n"
            + f"  batch_size_target_ms   : {self.batch_size_latency_target_ms:.0f}\n"
            + f"  backup_jitter_ms       : {self.backup_jitter_ms}\n"
            + f"  prefetch_io_workers    : {self.prefetch_io_workers}\n"
            + f"  coalesce_prefetch_ops  : {'ON' if self._coalesce_prefetch_ops else 'OFF'}\n"
            + f"  prefetch_coalesce_ms   : {self._prefetch_coalesce_deadline_ms:.1f}\n"
            + f"  warmup_enabled         : {'ON' if self._warmup._config.enabled else 'OFF'}\n"
            + f"  warmup_mode            : data-driven (COLD -> STEADY)\n"
            + f"  warmup_poll_timeout_s  : {self._warmup._config.server_poll_timeout_s}\n"
            + f"  warmup_min_batch_size  : {self._warmup._config.min_batch_size}\n"
            + "=" * 60 + "\n"
        )
        logger.info(_banner)
        try:
            while not self.stop_event.is_set():
                try:
                    # Reap completed futures
                    still_pending = []
                    for f in futures:
                        if f.done():
                            try:
                                result = f.result()
                                if result.success:
                                    ops_completed += 1
                                else:
                                    ops_failed += 1
                                backup_latencies_ms.append(result.elapsed_ms)
                                backup_jitter_ms_total += result.jitter_ms
                                if result.avg_gap_ms > 0:
                                    backup_gap_ms_samples.append(result.avg_gap_ms)
                            except Exception:
                                ops_failed += 1
                        else:
                            still_pending.append(f)
                    futures = still_pending

                    # Health log every 60s + push metrics
                    now = time.monotonic()
                    if now - last_health_log >= 60:
                        avg_lat = (
                            (sum(backup_latencies_ms) / len(backup_latencies_ms))
                            if backup_latencies_ms
                            else 0.0
                        )
                        avg_gap = (
                            (sum(backup_gap_ms_samples) / len(backup_gap_ms_samples))
                            if backup_gap_ms_samples
                            else 0.0
                        )
                        coalesce_avg = self._coalescer.avg_ops_per_batch if self._coalescer else 1.0
                        warmup_info = self._warmup.phase_info()
                        _bt_lvl = logging.INFO if getattr(self, "tp_rank", 0) == 0 else logging.DEBUG
                        logger.log(
                            _bt_lvl,
                            "[backup_thread] health: completed=%d, failed=%d, in_flight=%d, "
                            "queue_depth=%d, avg_latency=%.1fms, jitter_total=%.1fms, "
                            "avg_gap=%.1fms, jitter_cfg=%dms, coalesce_avg_ops=%.1f, "
                            "write_policy=%s, warmup_phase=%s, eff_batch=%d, "
                            "eff_jitter=%.0fms, eff_deadline=%.0fms",
                            ops_completed, ops_failed, len(futures),
                            self.backup_queue.qsize(), avg_lat,
                            backup_jitter_ms_total, avg_gap, self.backup_jitter_ms,
                            coalesce_avg, self.write_policy,
                            warmup_info["phase"], warmup_info["effective_batch_size"],
                            warmup_info["effective_jitter_ms"], warmup_info["effective_deadline_ms"],
                        )
                        try:
                            self.storage_backend.update_sglang_metrics({
                                "host_alloc_drops": self._host_alloc_drops,
                                "backup_queue_depth": self.backup_queue.qsize(),
                                "backup_ops_completed": ops_completed,
                                "backup_ops_failed": ops_failed,
                                "backup_in_flight": len(futures),
                                "backup_avg_latency_ms": avg_lat,
                                "backup_jitter_total_ms": backup_jitter_ms_total,
                                "backup_avg_gap_ms": avg_gap,
                                "backup_jitter_cfg_ms": self.backup_jitter_ms,
                                "backup_batch_size": self.storage_batch_size,
                                "backup_coalesce_avg_ops": coalesce_avg,
                                "warmup_phase": warmup_info["phase"],
                                "warmup_effective_batch_size": warmup_info["effective_batch_size"],
                            })
                        except Exception:
                            pass

                        # Adaptive batch sizing: adjust storage_batch_size based on latency
                        if self.batch_size_auto and len(backup_latencies_ms) >= 3:
                            target = self.batch_size_latency_target_ms
                            min_floor = max(self._batch_size_min, self._warmup._config.min_batch_size)
                            old_bs = self.storage_batch_size
                            if avg_lat > target and old_bs > min_floor:
                                self.storage_batch_size = max(min_floor, old_bs // 2)
                            elif avg_lat < target * 0.5 and old_bs < self._batch_size_max:
                                self.storage_batch_size = min(self._batch_size_max, old_bs * 2)
                            if self.storage_batch_size != old_bs:
                                self._warmup.update_steady_batch_size(self.storage_batch_size)
                                logger.log(
                                    _bt_lvl,
                                    "[backup_thread] auto_batch_size: %d -> %d "
                                    "(avg_lat=%.1fms, target=%.1fms)",
                                    old_bs, self.storage_batch_size, avg_lat, target,
                                )

                        backup_latencies_ms.clear()
                        backup_jitter_ms_total = 0.0
                        backup_gap_ms_samples.clear()
                        if self._coalescer:
                            self._coalescer.reset_stats()
                        last_health_log = now

                    if self._coalescer is not None:
                        operation = self._coalescer.drain()
                    else:
                        try:
                            operation = self.backup_queue.get(block=True, timeout=1)
                        except Empty:
                            operation = None
                    if operation is None:
                        continue
                    try:
                        futures.append(executor.submit(self._backup_io_task, operation))
                    except RuntimeError:
                        # Executor shut down (interpreter exiting) — normal during teardown
                        logger.debug("[backup_thread] executor shut down, exiting")
                        break

                except Empty:
                    continue
                except Exception as e:
                    logger.error("[backup_thread] unexpected: %s", e, exc_info=True)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            _sd_lvl = logging.INFO if getattr(self, "tp_rank", 0) == 0 else logging.DEBUG
            logger.log(
                _sd_lvl,
                "[backup_thread] shutdown: completed=%d, failed=%d",
                ops_completed, ops_failed,
            )

    def detach_storage_backend(self):
        """Cleanly shut down the storage backend (called during server teardown)."""
        if not self.enable_storage:
            return
        self.stop_event.set()
        logger.info("[detach] joining prefetch and backup threads")
        try:
            self.prefetch_thread.join(timeout=5)
            self.backup_thread.join(timeout=5)
        except Exception as e:
            logger.warning("Error joining storage threads: %s", e)
        try:
            self.storage_backend.close()
        except Exception as e:
            logger.warning("Error closing storage backend: %s", e)
        self.enable_storage = False
        logger.info("Storage backend detached.")
