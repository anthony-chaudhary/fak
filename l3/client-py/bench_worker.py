"""Multiprocessing, saturation, and SGLang-realistic workers for CAMA bench.

Separated from bench.py for picklability (multiprocessing requires top-level
functions and serializable config objects).
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Process, Queue, Event as MPEvent, Barrier as MPBarrier, Value
from typing import Any


# ---------------------------------------------------------------------------
# Picklable config / result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class WorkerConfig:
    """All fields are basic types — safe for multiprocessing serialization."""
    worker_id: int = 0
    transport: str = "tcp"
    addr: str = "127.0.0.1"
    port: int = 18000
    value_size: int = 5_242_880
    keys: list[str] = field(default_factory=list)
    read_ratio: float = 0.8
    exists_ratio: float = 0.0
    delete_ratio: float = 0.0
    ttl_ms: int = 0
    distribution: str = "zipfian"
    zipf_n: int = 100
    zipf_s: float = 1.0
    debug: bool = False
    endpoint: dict[str, Any] | None = None
    # SGLang-specific
    workload: str = "mixed"
    batch_size: int = 16
    io_workers: int = 16
    key_multiplier: int = 2
    compute_delay_ms: float = 0.0
    prefix_hit_rate: float = 0.6


@dataclass
class WorkerResult:
    """Per-worker summary — computed in-process to avoid serializing raw latencies."""
    worker_id: int = 0
    ops: int = 0
    gets: int = 0
    sets: int = 0
    hits: int = 0
    exists_ops: int = 0
    exists_hits: int = 0
    deletes: int = 0
    errors: int = 0
    bytes_read: int = 0
    bytes_write: int = 0
    elapsed: float = 0.0
    # Pre-computed latency percentiles (ms)
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    p999_ms: float = 0.0
    avg_ms: float = 0.0
    # Per-op latency percentiles
    get_p50_ms: float = 0.0
    get_p99_ms: float = 0.0
    set_p50_ms: float = 0.0
    set_p99_ms: float = 0.0
    exists_p50_ms: float = 0.0
    exists_p99_ms: float = 0.0
    # NIC info
    nic_device: str = ""
    nic_ip: str = ""
    nic_port: int = 0
    # C++ stats
    cpp_avg_roundtrip_us: float = 0.0
    cpp_avg_rdma_read_us: float = 0.0
    # SGLang-specific
    requests: int = 0
    prefix_hits_total: int = 0
    prefix_checks_total: int = 0
    phase_exists_ops: int = 0
    phase_get_ops: int = 0
    phase_set_ops: int = 0
    phase_exists_lats: list[float] = field(default_factory=list)
    phase_get_lats: list[float] = field(default_factory=list)
    phase_set_lats: list[float] = field(default_factory=list)

    @staticmethod
    def aggregate(results: list[WorkerResult]) -> dict[str, Any]:
        """Combine multiple WorkerResults into a summary dict."""
        if not results:
            return {}
        total_ops = sum(r.ops for r in results)
        total_gets = sum(r.gets for r in results)
        total_sets = sum(r.sets for r in results)
        total_hits = sum(r.hits for r in results)
        total_exists_ops = sum(r.exists_ops for r in results)
        total_exists_hits = sum(r.exists_hits for r in results)
        total_deletes = sum(r.deletes for r in results)
        total_errors = sum(r.errors for r in results)
        total_bytes_read = sum(r.bytes_read for r in results)
        total_bytes_write = sum(r.bytes_write for r in results)
        max_elapsed = max(r.elapsed for r in results) if results else 1.0
        ops_per_sec = total_ops / max_elapsed if max_elapsed > 0 else 0.0
        total_bw = (total_bytes_read + total_bytes_write) / max_elapsed / 1e9 if max_elapsed > 0 else 0.0

        if total_exists_ops > total_gets and total_exists_ops > 0:
            hit_rate = total_exists_hits / total_exists_ops * 100
        elif total_gets > 0:
            hit_rate = total_hits / total_gets * 100
        else:
            hit_rate = -1.0

        # Weighted-average latency percentiles
        weighted_p50 = sum(r.p50_ms * r.ops for r in results) / total_ops if total_ops > 0 else 0.0
        weighted_p99 = max(r.p99_ms for r in results) if results else 0.0

        return {
            "total_ops": total_ops,
            "total_gets": total_gets,
            "total_sets": total_sets,
            "total_hits": total_hits,
            "total_exists_ops": total_exists_ops,
            "total_exists_hits": total_exists_hits,
            "total_deletes": total_deletes,
            "total_errors": total_errors,
            "total_bytes_read": total_bytes_read,
            "total_bytes_write": total_bytes_write,
            "elapsed": max_elapsed,
            "ops_per_sec": ops_per_sec,
            "bandwidth_gbps": total_bw,
            "hit_rate": hit_rate,
            "p50_ms": weighted_p50,
            "p99_ms": weighted_p99,
        }


# ---------------------------------------------------------------------------
# Shared helpers (mirroring bench.py internals)
# ---------------------------------------------------------------------------
class _LCG:
    __slots__ = ("_state",)

    def __init__(self, seed: int):
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def next_int(self) -> int:
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self._state

    def next_float(self) -> float:
        return (self.next_int() >> 11) / (1 << 53)


class _ZipfGenerator:
    def __init__(self, n: int, s: float = 1.0):
        self.n = n
        self.s = s
        self.h_n = self._harmonic(float(n), s)
        self.h_1 = self._harmonic(1.0, s)

    @staticmethod
    def _harmonic(x: float, s: float) -> float:
        if s == 1.0:
            return math.log(x)
        return (math.pow(x, 1.0 - s) - 1.0) / (1.0 - s)

    def _h_inv(self, h: float) -> float:
        if self.s == 1.0:
            return math.exp(h)
        return math.pow(h * (1.0 - self.s) + 1.0, 1.0 / (1.0 - self.s))

    def sample(self, rng: _LCG) -> int:
        u = rng.next_float()
        h = self.h_1 + u * (self.h_n - self.h_1)
        x = self._h_inv(h)
        k = int(x + 0.5)
        k = max(1, min(k, self.n))
        return k - 1


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def _compute_latency_stats(latencies: list[float]) -> tuple[float, float, float, float]:
    """Return (avg_ms, p50_ms, p99_ms, p999_ms) from raw latency list (in seconds)."""
    if not latencies:
        return 0.0, 0.0, 0.0, 0.0
    latencies.sort()
    avg = sum(latencies) / len(latencies) * 1000
    p50 = _percentile(latencies, 50) * 1000
    p99 = _percentile(latencies, 99) * 1000
    p999 = _percentile(latencies, 99.9) * 1000
    return avg, p50, p99, p999


def _make_client(transport: str, addr: str, port: int, debug: bool = False):
    if transport == "rdma":
        from l3_client.rdma_client import RDMAClient
        return RDMAClient(addr, port=port, debug=debug)
    else:
        from l3_client.client import CamaClient
        return CamaClient(addr, port=port)


def _make_sgl(transport: str, client, ptr: int, size: int, buf=None):
    from l3_client.sgl import SGL
    if transport == "rdma":
        handle = client.reg_memory(ptr, size, buf=buf)
        return SGL(ptr=ptr, size=size, reg_handle=handle), handle
    return SGL(ptr=ptr, size=size), None


# ---------------------------------------------------------------------------
# mp_worker — process-safe mixed/exists/ttl-churn/delete-heavy worker
# ---------------------------------------------------------------------------
def mp_worker(
    cfg: WorkerConfig,
    result_queue: Queue,
    stop_event,       # multiprocessing.Event
    barrier,          # multiprocessing.Barrier
    live_ops,         # multiprocessing.Value('L')
) -> None:
    """Process-safe worker entry point. Mirrors bench.py _worker() logic."""
    addr = cfg.addr
    port = cfg.port
    nic_device = ""
    nic_ip = ""
    nic_port = 0
    if cfg.endpoint is not None:
        addr = cfg.endpoint["ip"]
        port = cfg.endpoint["port"]
        nic_device = cfg.endpoint.get("device", "unknown")
        nic_ip = addr
        nic_port = port

    client = _make_client(cfg.transport, addr, port, cfg.debug)

    buf = ctypes.create_string_buffer(os.urandom(cfg.value_size), cfg.value_size)
    ptr = ctypes.addressof(buf)
    sgl, handle = _make_sgl(cfg.transport, client, ptr, cfg.value_size, buf=buf)

    zipf = _ZipfGenerator(cfg.zipf_n, cfg.zipf_s) if cfg.distribution == "zipfian" else None
    rng = _LCG(cfg.worker_id * 2654435761 + 1)

    keys = cfg.keys
    n_keys = len(keys)
    ops = 0
    gets = 0
    sets = 0
    hits = 0
    exists_ops = 0
    exists_hits = 0
    deletes = 0
    errors = 0
    bytes_read = 0
    bytes_write = 0
    get_latencies: list[float] = []
    set_latencies: list[float] = []
    exists_latencies: list[float] = []
    del_latencies: list[float] = []

    delete_thresh = cfg.delete_ratio
    exists_thresh = cfg.delete_ratio + cfg.exists_ratio
    remaining = 1.0 - exists_thresh
    read_thresh = exists_thresh + remaining * cfg.read_ratio

    try:
        barrier.wait()
    except Exception:
        result_queue.put(WorkerResult(worker_id=cfg.worker_id, errors=1))
        return

    worker_start = time.monotonic()

    try:
        while not stop_event.is_set():
            if zipf is not None:
                key_idx = zipf.sample(rng)
            else:
                key_idx = rng.next_int() % n_keys
            key = keys[key_idx]

            r = rng.next_float()

            try:
                t_op = time.monotonic()
                if r < delete_thresh:
                    client.delete(key)
                    del_latencies.append(time.monotonic() - t_op)
                    deletes += 1
                elif r < exists_thresh:
                    rc = client.exists(key)
                    exists_latencies.append(time.monotonic() - t_op)
                    exists_ops += 1
                    if rc == 1:
                        exists_hits += 1
                elif r < read_thresh:
                    rc = client.get(key, sgl)
                    get_latencies.append(time.monotonic() - t_op)
                    gets += 1
                    if rc == 0:
                        hits += 1
                        bytes_read += cfg.value_size
                else:
                    rc = client.set(key, sgl, ttl_ms=cfg.ttl_ms)
                    set_latencies.append(time.monotonic() - t_op)
                    sets += 1
                    if rc == 0:
                        bytes_write += cfg.value_size
            except Exception:
                errors += 1

            ops += 1
            if ops % 64 == 0:
                with live_ops.get_lock():
                    live_ops.value += 64
    finally:
        worker_elapsed = time.monotonic() - worker_start

        cpp_rt = 0.0
        cpp_rd = 0.0
        if cfg.transport == "rdma" and hasattr(client, '_transport'):
            try:
                cs = client._transport.get_stats()
                if cs.get("roundtrip_count", 0) > 0:
                    cpp_rt = cs["avg_roundtrip_us"]
                if cs.get("rdma_read_count", 0) > 0:
                    cpp_rd = cs["avg_rdma_read_us"]
            except Exception:
                pass

        if handle is not None:
            try:
                client.dereg_memory(handle)
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass

    all_lats = get_latencies + set_latencies + exists_latencies + del_latencies
    avg_ms, p50_ms, p99_ms, p999_ms = _compute_latency_stats(all_lats)
    _, get_p50, get_p99, _ = _compute_latency_stats(get_latencies)
    _, set_p50, set_p99, _ = _compute_latency_stats(set_latencies)
    _, ex_p50, ex_p99, _ = _compute_latency_stats(exists_latencies)

    result_queue.put(WorkerResult(
        worker_id=cfg.worker_id,
        ops=ops, gets=gets, sets=sets, hits=hits,
        exists_ops=exists_ops, exists_hits=exists_hits,
        deletes=deletes, errors=errors,
        bytes_read=bytes_read, bytes_write=bytes_write,
        elapsed=worker_elapsed,
        p50_ms=p50_ms, p99_ms=p99_ms, p999_ms=p999_ms, avg_ms=avg_ms,
        get_p50_ms=get_p50, get_p99_ms=get_p99,
        set_p50_ms=set_p50, set_p99_ms=set_p99,
        exists_p50_ms=ex_p50, exists_p99_ms=ex_p99,
        nic_device=nic_device, nic_ip=nic_ip, nic_port=nic_port,
        cpp_avg_roundtrip_us=cpp_rt, cpp_avg_rdma_read_us=cpp_rd,
    ))


# ---------------------------------------------------------------------------
# run_multiprocess_benchmark — spawn + collect
# ---------------------------------------------------------------------------
def run_multiprocess_benchmark(
    args,
    key_list: list[str],
    endpoints,
    workload_name: str | None = None,
    duration: int | None = None,
    quiet: bool = False,
) -> list[WorkerResult]:
    """Spawn N processes, run the mixed/exists/ttl-churn/delete-heavy workload, collect results."""
    from bench import WORKLOAD_DEFAULTS, _prepopulate

    wl = workload_name or args.workload
    wl_read, wl_exists, wl_delete, wl_ttl = WORKLOAD_DEFAULTS[wl]
    dur = duration or args.duration

    # Use explicit ratio overrides if set (only for primary workload, not sweep sub-runs)
    if workload_name is None:
        if args.read_ratio is not None:
            wl_read = args.read_ratio
        if args.exists_ratio is not None:
            wl_exists = args.exists_ratio
        if args.delete_ratio is not None:
            wl_delete = args.delete_ratio
        if args.ttl_ms is not None:
            wl_ttl = args.ttl_ms

    # Pre-populate (from main process)
    if not args.no_flush:
        if not quiet:
            print(f"[bench] Flushing server before '{wl}'...", flush=True)
        fc = _make_client(args.transport, args.addr, args.port)
        fc.flush()
        fc.close()

    if not quiet:
        _prepopulate(args.transport, args.addr, args.port, key_list, args.value_size, ttl_ms=wl_ttl)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    stop_event = ctx.Event()
    barrier = ctx.Barrier(args.clients + 1, timeout=120)
    live_ops = ctx.Value("L", 0)

    processes: list[Process] = []
    for i in range(args.clients):
        ep = endpoints[i % len(endpoints)] if endpoints else None
        cfg = WorkerConfig(
            worker_id=i,
            transport=args.transport,
            addr=args.addr,
            port=args.port,
            value_size=args.value_size,
            keys=key_list,
            read_ratio=wl_read,
            exists_ratio=wl_exists,
            delete_ratio=wl_delete,
            ttl_ms=wl_ttl,
            distribution=args.distribution,
            zipf_n=len(key_list),
            zipf_s=1.0,
            debug=args.debug,
            endpoint=ep,
            workload=wl,
        )
        p = ctx.Process(target=mp_worker, args=(cfg, result_queue, stop_event, barrier, live_ops))
        p.daemon = True
        processes.append(p)

    connect_start = time.monotonic()
    for p in processes:
        p.start()

    try:
        barrier.wait()
    except Exception:
        print("[bench] ERROR: barrier broken — workers failed to connect", file=sys.stderr)
        for p in processes:
            p.terminate()
        sys.exit(1)

    connect_elapsed = time.monotonic() - connect_start
    if not quiet:
        print(f"[bench] All {args.clients} workers connected in {connect_elapsed:.1f}s", flush=True)
        print(f"[bench] Running '{wl}' for {dur}s (engine=process)...", flush=True)

    start_time = time.monotonic()

    # Progress ticker
    def _progress():
        while not stop_event.is_set():
            time.sleep(5)
            if stop_event.is_set():
                break
            elapsed = time.monotonic() - start_time
            with live_ops.get_lock():
                cur_ops = live_ops.value
            print(f"[bench] {elapsed:8.1f}s: {cur_ops:10d} ops, "
                  f"{cur_ops / elapsed:10.0f} ops/s", flush=True)

    if not quiet:
        progress_thread = threading.Thread(target=_progress, daemon=True)
        progress_thread.start()

    time.sleep(dur)
    stop_event.set()

    for p in processes:
        p.join(timeout=30)

    results: list[WorkerResult] = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    return results


# ---------------------------------------------------------------------------
# Saturation finder
# ---------------------------------------------------------------------------
def detect_saturation(rows: list[tuple], threshold: float = 5.0) -> int | None:
    """Given [(clients, ops/s, bw, p99, improvement%), ...], return the
    saturation point client count — the last row that still showed meaningful
    improvement before it dropped below threshold. Returns last row's client
    count if no saturation detected."""
    if not rows:
        return None
    # The last row in the list is either:
    # - the one where improvement < threshold (saturated), in which case
    #   the saturation point is the row before it
    # - the final row tested (no saturation), return its client count
    for i in range(1, len(rows)):
        if rows[i][4] < threshold:
            return rows[i - 1][0]
    return rows[-1][0]


def run_saturation(args, key_list: list[str], endpoints) -> None:
    """Auto-ramp client count until throughput plateaus."""
    from bench import SweepResult, _run_timed_workload, WORKLOAD_DEFAULTS
    import copy

    n = 1
    prev_ops = 0.0
    rows: list[tuple] = []

    print(f"\nSATURATION ANALYSIS ({args.preset or 'custom'}, "
          f"{args.transport.upper()}, {args.workload})")
    print(f"{'Clients':>8s}    {'Ops/s':>10s}    {'BW GB/s':>9s}    {'p99 ms':>9s}    {'Improvement':>12s}")
    print("-" * 60)

    while n <= args.saturate_max:
        args_copy = copy.copy(args)
        args_copy.clients = n
        # Use the engine for saturation runs — if process mode, use multiprocess
        if args.engine == "process":
            worker_results = run_multiprocess_benchmark(
                args_copy, key_list, endpoints,
                workload_name=args.workload,
                duration=args.saturate_duration,
                quiet=True,
            )
            agg = WorkerResult.aggregate(worker_results)
            sr = SweepResult(
                workload=args.workload,
                ops_per_sec=agg.get("ops_per_sec", 0),
                bandwidth_gbps=agg.get("bandwidth_gbps", 0),
                p50_ms=agg.get("p50_ms", 0),
                p99_ms=agg.get("p99_ms", 0),
                errors=agg.get("total_errors", 0),
                total_ops=agg.get("total_ops", 0),
                hit_rate=agg.get("hit_rate", -1),
            )
        else:
            sr = _run_timed_workload(
                args_copy, args.workload, key_list,
                _ZipfGenerator(len(key_list)) if args.distribution == "zipfian" else None,
                endpoints, args.saturate_duration, quiet=True,
            )

        improvement = ((sr.ops_per_sec - prev_ops) / prev_ops * 100) if prev_ops > 0 else 100.0
        rows.append((n, sr.ops_per_sec, sr.bandwidth_gbps, sr.p99_ms, improvement))

        marker = ""
        if improvement < args.saturate_threshold and n > 1:
            marker = "  <- saturated"
        imp_str = f"+{improvement:.1f}%" if n > 1 else "---"
        print(f"{n:>8d}    {sr.ops_per_sec:>10,.0f}    {sr.bandwidth_gbps:>9.2f}    "
              f"{sr.p99_ms:>9.2f}    {imp_str:>12s}{marker}", flush=True)

        if improvement < args.saturate_threshold and n > 1:
            break
        prev_ops = sr.ops_per_sec
        n *= 2

    print()
    if len(rows) >= 2:
        # Saturation point is the row before the one that triggered the break
        sat_row = rows[-2] if rows[-1][4] < args.saturate_threshold else rows[-1]
        print(f"Saturation point: {sat_row[0]} clients -> "
              f"{sat_row[1]:,.0f} ops/s, {sat_row[2]:.2f} GB/s")
    elif rows:
        print(f"Only tested 1 level: {rows[0][0]} clients -> {rows[0][1]:,.0f} ops/s")
    print()


# ---------------------------------------------------------------------------
# SGLang-realistic worker (single conn + ThreadPoolExecutor per rank)
# ---------------------------------------------------------------------------
def _sglang_realistic_worker(
    cfg: WorkerConfig,
    result_queue: Queue,
    stop_event,
    barrier,
) -> None:
    """Mirrors cama_storage.py: 1 connection + ThreadPoolExecutor(io_workers)
    sharing it. Runs exists->get->compute->set lifecycle per batch."""
    addr = cfg.addr
    port = cfg.port
    if cfg.endpoint is not None:
        addr = cfg.endpoint["ip"]
        port = cfg.endpoint["port"]

    client = _make_client(cfg.transport, addr, port, cfg.debug)

    # Register one large buffer = batch_size * key_multiplier * value_size
    n_sub_keys = cfg.batch_size * cfg.key_multiplier
    total_buf_size = n_sub_keys * cfg.value_size
    big_buf = ctypes.create_string_buffer(os.urandom(min(total_buf_size, 32 * 1024 * 1024)), total_buf_size)
    base_ptr = ctypes.addressof(big_buf)

    handle = None
    if cfg.transport == "rdma":
        handle = client.reg_memory(base_ptr, total_buf_size, buf=big_buf)

    from l3_client.sgl import SGL
    sgls = []
    for i in range(n_sub_keys):
        offset = i * cfg.value_size
        rh = handle if handle is not None else 1
        sgls.append(SGL(ptr=base_ptr + offset, size=cfg.value_size, reg_handle=rh))

    rng = _LCG(cfg.worker_id * 2654435761 + 1)
    keys = cfg.keys
    n_keys = len(keys)

    requests = 0
    prefix_hits_total = 0
    prefix_checks_total = 0
    phase_exists_ops = 0
    phase_get_ops = 0
    phase_set_ops = 0
    errors = 0
    bytes_read = 0
    bytes_write = 0
    exists_lats: list[float] = []
    get_lats: list[float] = []
    set_lats: list[float] = []

    # Pre-populate prefix_hit_rate fraction of keys
    n_prepop = int(n_keys * cfg.prefix_hit_rate)

    pool = ThreadPoolExecutor(max_workers=cfg.io_workers, thread_name_prefix="sglang_io")

    try:
        barrier.wait()
    except Exception:
        result_queue.put(WorkerResult(worker_id=cfg.worker_id, errors=1))
        return

    try:
        while not stop_event.is_set():
            # Pick a batch of consecutive keys (simulating prefix sequence)
            start_idx = rng.next_int() % max(1, n_keys - cfg.batch_size)
            batch_keys = keys[start_idx:start_idx + cfg.batch_size]

            # Expand to sub-keys (MHA=2x, MLA=1x)
            sub_keys = []
            for k in batch_keys:
                if cfg.key_multiplier == 2:
                    sub_keys.append(f"{k}_k")
                    sub_keys.append(f"{k}_v")
                else:
                    sub_keys.append(f"{k}_k")

            n_sub = len(sub_keys)

            # Phase 1: EXISTS check (concurrent via thread pool)
            t0 = time.monotonic()

            def _do_exists(key):
                try:
                    return client.exists(key)
                except Exception:
                    return -1

            exist_futures = {pool.submit(_do_exists, sk): i for i, sk in enumerate(sub_keys)}
            exist_results = [0] * n_sub
            for fut in as_completed(exist_futures):
                idx = exist_futures[fut]
                exist_results[idx] = fut.result()

            exists_lat = time.monotonic() - t0
            exists_lats.append(exists_lat)
            phase_exists_ops += n_sub

            # Count consecutive prefix hits (per page, not per sub-key)
            prefix_hits = 0
            for p in range(cfg.batch_size):
                base = p * cfg.key_multiplier
                all_found = all(
                    exist_results[base + j] == 1
                    for j in range(cfg.key_multiplier)
                )
                if all_found:
                    prefix_hits += 1
                else:
                    break  # first miss stops prefix matching

            prefix_hits_total += prefix_hits
            prefix_checks_total += cfg.batch_size

            # Phase 2: GET cached prefix (concurrent via thread pool)
            cached_count = prefix_hits * cfg.key_multiplier
            if cached_count > 0:
                t1 = time.monotonic()

                def _do_get(key, sgl_ref):
                    try:
                        return client.get(key, sgl_ref)
                    except Exception:
                        return -1

                get_futures = {}
                for gi in range(cached_count):
                    get_futures[pool.submit(_do_get, sub_keys[gi], sgls[gi])] = gi
                for fut in as_completed(get_futures):
                    rc = fut.result()
                    if rc == 0:
                        bytes_read += cfg.value_size

                get_lat = time.monotonic() - t1
                get_lats.append(get_lat)
                phase_get_ops += cached_count

            # Phase 3: COMPUTE DELAY
            if cfg.compute_delay_ms > 0:
                time.sleep(cfg.compute_delay_ms / 1000)

            # Phase 4: SET new suffix (keys that weren't found)
            new_start = cached_count
            new_count = n_sub - new_start
            if new_count > 0:
                t2 = time.monotonic()

                def _do_set(key, sgl_ref):
                    try:
                        return client.set(key, sgl_ref)
                    except Exception:
                        return -1

                set_futures = {}
                for si in range(new_count):
                    actual_idx = new_start + si
                    set_futures[pool.submit(_do_set, sub_keys[actual_idx], sgls[actual_idx])] = si
                for fut in as_completed(set_futures):
                    rc = fut.result()
                    if rc == 0:
                        bytes_write += cfg.value_size

                set_lat = time.monotonic() - t2
                set_lats.append(set_lat)
                phase_set_ops += new_count

            requests += 1

    except Exception:
        errors += 1
    finally:
        elapsed = time.monotonic() - (barrier._state if hasattr(barrier, '_state') else time.monotonic())
        pool.shutdown(wait=False)

        if handle is not None:
            try:
                client.dereg_memory(handle)
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass

    total_ops = phase_exists_ops + phase_get_ops + phase_set_ops
    all_lats = exists_lats + get_lats + set_lats
    avg_ms, p50_ms, p99_ms, p999_ms = _compute_latency_stats(all_lats)

    result_queue.put(WorkerResult(
        worker_id=cfg.worker_id,
        ops=total_ops,
        gets=phase_get_ops,
        sets=phase_set_ops,
        exists_ops=phase_exists_ops,
        errors=errors,
        bytes_read=bytes_read,
        bytes_write=bytes_write,
        elapsed=0.0,  # set by orchestrator
        p50_ms=p50_ms, p99_ms=p99_ms, p999_ms=p999_ms, avg_ms=avg_ms,
        requests=requests,
        prefix_hits_total=prefix_hits_total,
        prefix_checks_total=prefix_checks_total,
        phase_exists_ops=phase_exists_ops,
        phase_get_ops=phase_get_ops,
        phase_set_ops=phase_set_ops,
        phase_exists_lats=exists_lats,
        phase_get_lats=get_lats,
        phase_set_lats=set_lats,
    ))


# ---------------------------------------------------------------------------
# SGLang-stress worker (separate connections, batch ops)
# ---------------------------------------------------------------------------
def _sglang_stress_worker(
    cfg: WorkerConfig,
    result_queue: Queue,
    stop_event,
    barrier,
) -> None:
    """Stress mode: each worker has its own connection, runs batch
    exists->get->set cycles sequentially (no thread pool)."""
    addr = cfg.addr
    port = cfg.port
    if cfg.endpoint is not None:
        addr = cfg.endpoint["ip"]
        port = cfg.endpoint["port"]

    client = _make_client(cfg.transport, addr, port, cfg.debug)

    n_sub_keys = cfg.batch_size * cfg.key_multiplier
    total_buf_size = n_sub_keys * cfg.value_size
    big_buf = ctypes.create_string_buffer(os.urandom(min(total_buf_size, 32 * 1024 * 1024)), total_buf_size)
    base_ptr = ctypes.addressof(big_buf)

    handle = None
    if cfg.transport == "rdma":
        handle = client.reg_memory(base_ptr, total_buf_size, buf=big_buf)

    from l3_client.sgl import SGL
    sgls = []
    for i in range(n_sub_keys):
        offset = i * cfg.value_size
        rh = handle if handle is not None else 1
        sgls.append(SGL(ptr=base_ptr + offset, size=cfg.value_size, reg_handle=rh))

    rng = _LCG(cfg.worker_id * 2654435761 + 1)
    keys = cfg.keys
    n_keys = len(keys)

    requests = 0
    phase_exists_ops = 0
    phase_get_ops = 0
    phase_set_ops = 0
    errors = 0
    bytes_read = 0
    bytes_write = 0
    exists_lats: list[float] = []
    get_lats: list[float] = []
    set_lats: list[float] = []

    try:
        barrier.wait()
    except Exception:
        result_queue.put(WorkerResult(worker_id=cfg.worker_id, errors=1))
        return

    try:
        while not stop_event.is_set():
            start_idx = rng.next_int() % max(1, n_keys - cfg.batch_size)
            batch_keys = keys[start_idx:start_idx + cfg.batch_size]

            sub_keys = []
            for k in batch_keys:
                if cfg.key_multiplier == 2:
                    sub_keys.append(f"{k}_k")
                    sub_keys.append(f"{k}_v")
                else:
                    sub_keys.append(f"{k}_k")

            n_sub = len(sub_keys)

            # Phase 1: Sequential EXISTS
            t0 = time.monotonic()
            exist_results = []
            for sk in sub_keys:
                try:
                    exist_results.append(client.exists(sk))
                except Exception:
                    exist_results.append(-1)
                    errors += 1
            exists_lats.append(time.monotonic() - t0)
            phase_exists_ops += n_sub

            # Count prefix hits
            prefix_hits = 0
            for p in range(cfg.batch_size):
                base = p * cfg.key_multiplier
                all_found = all(exist_results[base + j] == 1 for j in range(cfg.key_multiplier))
                if all_found:
                    prefix_hits += 1
                else:
                    break

            # Phase 2: Sequential GET for cached prefix
            cached_count = prefix_hits * cfg.key_multiplier
            if cached_count > 0:
                t1 = time.monotonic()
                for gi in range(cached_count):
                    try:
                        rc = client.get(sub_keys[gi], sgls[gi])
                        if rc == 0:
                            bytes_read += cfg.value_size
                    except Exception:
                        errors += 1
                get_lats.append(time.monotonic() - t1)
                phase_get_ops += cached_count

            # Phase 3: Sequential SET for new suffix
            new_start = cached_count
            new_count = n_sub - new_start
            if new_count > 0:
                t2 = time.monotonic()
                for si in range(new_count):
                    actual_idx = new_start + si
                    try:
                        rc = client.set(sub_keys[actual_idx], sgls[actual_idx])
                        if rc == 0:
                            bytes_write += cfg.value_size
                    except Exception:
                        errors += 1
                set_lats.append(time.monotonic() - t2)
                phase_set_ops += new_count

            requests += 1

    except Exception:
        errors += 1
    finally:
        if handle is not None:
            try:
                client.dereg_memory(handle)
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass

    total_ops = phase_exists_ops + phase_get_ops + phase_set_ops
    all_lats = exists_lats + get_lats + set_lats
    avg_ms, p50_ms, p99_ms, p999_ms = _compute_latency_stats(all_lats)

    result_queue.put(WorkerResult(
        worker_id=cfg.worker_id,
        ops=total_ops,
        gets=phase_get_ops,
        sets=phase_set_ops,
        exists_ops=phase_exists_ops,
        errors=errors,
        bytes_read=bytes_read,
        bytes_write=bytes_write,
        requests=requests,
        phase_exists_ops=phase_exists_ops,
        phase_get_ops=phase_get_ops,
        phase_set_ops=phase_set_ops,
        phase_exists_lats=exists_lats,
        phase_get_lats=get_lats,
        phase_set_lats=set_lats,
        p50_ms=p50_ms, p99_ms=p99_ms, p999_ms=p999_ms, avg_ms=avg_ms,
    ))


# ---------------------------------------------------------------------------
# Batch worker (raw m-op throughput)
# ---------------------------------------------------------------------------
def _batch_worker(
    cfg: WorkerConfig,
    result_queue: Queue,
    stop_event,
    barrier,
) -> None:
    """Raw batch operation throughput: mget/mset/mexists based on read_ratio."""
    addr = cfg.addr
    port = cfg.port
    if cfg.endpoint is not None:
        addr = cfg.endpoint["ip"]
        port = cfg.endpoint["port"]

    client = _make_client(cfg.transport, addr, port, cfg.debug)

    total_buf_size = cfg.batch_size * cfg.value_size
    big_buf = ctypes.create_string_buffer(os.urandom(min(total_buf_size, 32 * 1024 * 1024)), total_buf_size)
    base_ptr = ctypes.addressof(big_buf)

    handle = None
    if cfg.transport == "rdma":
        handle = client.reg_memory(base_ptr, total_buf_size, buf=big_buf)

    from l3_client.sgl import SGL
    sgls = []
    for i in range(cfg.batch_size):
        offset = i * cfg.value_size
        rh = handle if handle is not None else 1
        sgls.append(SGL(ptr=base_ptr + offset, size=cfg.value_size, reg_handle=rh))

    rng = _LCG(cfg.worker_id * 2654435761 + 1)
    keys = cfg.keys
    n_keys = len(keys)

    ops = 0
    gets = 0
    sets = 0
    errors = 0
    bytes_read = 0
    bytes_write = 0
    batch_lats: list[float] = []

    try:
        barrier.wait()
    except Exception:
        result_queue.put(WorkerResult(worker_id=cfg.worker_id, errors=1))
        return

    try:
        while not stop_event.is_set():
            start_idx = rng.next_int() % max(1, n_keys - cfg.batch_size)
            batch_keys = keys[start_idx:start_idx + cfg.batch_size]
            r = rng.next_float()

            t0 = time.monotonic()
            if r < cfg.read_ratio:
                # Batch GET
                for i, k in enumerate(batch_keys):
                    try:
                        rc = client.get(k, sgls[i])
                        if rc == 0:
                            bytes_read += cfg.value_size
                    except Exception:
                        errors += 1
                gets += len(batch_keys)
            else:
                # Batch SET
                for i, k in enumerate(batch_keys):
                    try:
                        rc = client.set(k, sgls[i])
                        if rc == 0:
                            bytes_write += cfg.value_size
                    except Exception:
                        errors += 1
                sets += len(batch_keys)

            batch_lats.append(time.monotonic() - t0)
            ops += len(batch_keys)

    except Exception:
        errors += 1
    finally:
        if handle is not None:
            try:
                client.dereg_memory(handle)
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass

    avg_ms, p50_ms, p99_ms, p999_ms = _compute_latency_stats(batch_lats)

    result_queue.put(WorkerResult(
        worker_id=cfg.worker_id,
        ops=ops, gets=gets, sets=sets,
        errors=errors,
        bytes_read=bytes_read, bytes_write=bytes_write,
        p50_ms=p50_ms, p99_ms=p99_ms, p999_ms=p999_ms, avg_ms=avg_ms,
    ))


# ---------------------------------------------------------------------------
# SGLang orchestrators
# ---------------------------------------------------------------------------
def run_sglang_benchmark(args, key_list: list[str], endpoints, stress: bool = False) -> list[WorkerResult]:
    """Orchestrate SGLang-realistic or SGLang-stress benchmark."""
    import multiprocessing as mp

    # Pre-populate prefix_hit_rate fraction of keys
    n_prepop = int(len(key_list) * args.prefix_hit_rate)
    if n_prepop > 0:
        if not args.no_flush:
            print(f"[bench] Flushing server before sglang workload...", flush=True)
            fc = _make_client(args.transport, args.addr, args.port)
            fc.flush()
            fc.close()

        from bench import _prepopulate
        # Pre-populate first n_prepop keys (simulates prefix cache warmth)
        prepop_keys = key_list[:n_prepop]
        # For MHA/MLA sub-keys, expand
        expanded = []
        for k in prepop_keys:
            if args.key_multiplier == 2:
                expanded.append(f"{k}_k")
                expanded.append(f"{k}_v")
            else:
                expanded.append(f"{k}_k")
        _prepopulate(args.transport, args.addr, args.port, expanded, args.value_size)

    n_ranks = args.clients
    worker_fn = _sglang_stress_worker if stress else _sglang_realistic_worker

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    stop_event = ctx.Event()
    barrier = ctx.Barrier(n_ranks + 1, timeout=120)

    processes: list[Process] = []
    for i in range(n_ranks):
        ep = endpoints[i % len(endpoints)] if endpoints else None
        cfg = WorkerConfig(
            worker_id=i,
            transport=args.transport,
            addr=args.addr,
            port=args.port,
            value_size=args.value_size,
            keys=key_list,
            read_ratio=args.read_ratio,
            distribution=args.distribution,
            zipf_n=len(key_list),
            debug=args.debug,
            endpoint=ep,
            workload="sglang" if not stress else "sglang-stress",
            batch_size=args.batch_size,
            io_workers=args.io_workers,
            key_multiplier=args.key_multiplier,
            compute_delay_ms=args.compute_delay_ms,
            prefix_hit_rate=args.prefix_hit_rate,
        )
        p = ctx.Process(target=worker_fn, args=(cfg, result_queue, stop_event, barrier))
        p.daemon = True
        processes.append(p)

    connect_start = time.monotonic()
    for p in processes:
        p.start()

    try:
        barrier.wait()
    except Exception:
        print("[bench] ERROR: barrier broken — workers failed to connect", file=sys.stderr)
        for p in processes:
            p.terminate()
        sys.exit(1)

    connect_elapsed = time.monotonic() - connect_start
    mode_name = "sglang-stress" if stress else "sglang"
    print(f"[bench] All {n_ranks} ranks connected in {connect_elapsed:.1f}s", flush=True)
    print(f"[bench] Running '{mode_name}' for {args.duration}s...", flush=True)

    start_time = time.monotonic()
    time.sleep(args.duration)
    stop_event.set()

    for p in processes:
        p.join(timeout=30)

    elapsed = time.monotonic() - start_time

    results: list[WorkerResult] = []
    while not result_queue.empty():
        r = result_queue.get_nowait()
        r.elapsed = elapsed
        results.append(r)

    return results


def run_batch_benchmark(args, key_list: list[str], endpoints) -> list[WorkerResult]:
    """Orchestrate raw batch throughput benchmark."""
    import multiprocessing as mp

    if not args.no_flush:
        print("[bench] Flushing server before batch workload...", flush=True)
        fc = _make_client(args.transport, args.addr, args.port)
        fc.flush()
        fc.close()

    from bench import _prepopulate
    _prepopulate(args.transport, args.addr, args.port, key_list, args.value_size)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    stop_event = ctx.Event()
    barrier = ctx.Barrier(args.clients + 1, timeout=120)

    processes: list[Process] = []
    for i in range(args.clients):
        ep = endpoints[i % len(endpoints)] if endpoints else None
        cfg = WorkerConfig(
            worker_id=i,
            transport=args.transport,
            addr=args.addr,
            port=args.port,
            value_size=args.value_size,
            keys=key_list,
            read_ratio=args.read_ratio,
            distribution=args.distribution,
            zipf_n=len(key_list),
            debug=args.debug,
            endpoint=ep,
            workload="batch",
            batch_size=args.batch_size,
        )
        p = ctx.Process(target=_batch_worker, args=(cfg, result_queue, stop_event, barrier))
        p.daemon = True
        processes.append(p)

    connect_start = time.monotonic()
    for p in processes:
        p.start()

    try:
        barrier.wait()
    except Exception:
        print("[bench] ERROR: barrier broken — workers failed to connect", file=sys.stderr)
        for p in processes:
            p.terminate()
        sys.exit(1)

    connect_elapsed = time.monotonic() - connect_start
    print(f"[bench] All {args.clients} workers connected in {connect_elapsed:.1f}s", flush=True)
    print(f"[bench] Running 'batch' for {args.duration}s...", flush=True)

    start_time = time.monotonic()
    time.sleep(args.duration)
    stop_event.set()

    for p in processes:
        p.join(timeout=30)

    elapsed = time.monotonic() - start_time

    results: list[WorkerResult] = []
    while not result_queue.empty():
        r = result_queue.get_nowait()
        r.elapsed = elapsed
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# SGLang results printer
# ---------------------------------------------------------------------------
def print_sglang_results(args, results: list[WorkerResult], stress: bool = False) -> None:
    """Print SGLang-style results with phase breakdown."""
    if not results:
        print("[bench] No results collected")
        return

    elapsed = max(r.elapsed for r in results) if results else 1.0
    total_requests = sum(r.requests for r in results)
    total_prefix_hits = sum(r.prefix_hits_total for r in results)
    total_prefix_checks = sum(r.prefix_checks_total for r in results)
    total_exists_ops = sum(r.phase_exists_ops for r in results)
    total_get_ops = sum(r.phase_get_ops for r in results)
    total_set_ops = sum(r.phase_set_ops for r in results)
    total_ops = total_exists_ops + total_get_ops + total_set_ops
    total_errors = sum(r.errors for r in results)
    total_bytes_read = sum(r.bytes_read for r in results)
    total_bytes_write = sum(r.bytes_write for r in results)
    total_bw = (total_bytes_read + total_bytes_write) / elapsed / 1e9 if elapsed > 0 else 0.0

    avg_prefix = total_prefix_hits / total_prefix_checks * args.batch_size if total_prefix_checks > 0 else 0.0
    prefix_pct = total_prefix_hits / total_prefix_checks * 100 if total_prefix_checks > 0 else 0.0

    mode_name = "SGLang-Stress" if stress else "SGLang-Realistic"
    mha_str = "MHA" if args.key_multiplier == 2 else "MLA"
    n_sub = args.batch_size * args.key_multiplier

    print()
    print(f"=== CAMA {mode_name} Results ===")
    print(f"Ranks:           {args.clients}")
    if not stress:
        print(f"IO workers/rank: {args.io_workers}")
    print(f"Batch size:      {args.batch_size} pages ({n_sub} sub-keys, {mha_str})")
    print(f"Duration:        {elapsed:.2f}s")
    print()
    print(f"Requests:        {total_requests:,}  ({total_requests / elapsed:.1f} req/s)")
    print(f"Avg prefix hits: {avg_prefix:.1f} / {args.batch_size} ({prefix_pct:.1f}%)")
    print()

    # Phase breakdown
    all_exists_lats = []
    all_get_lats = []
    all_set_lats = []
    for r in results:
        all_exists_lats.extend(r.phase_exists_lats)
        all_get_lats.extend(r.phase_get_lats)
        all_set_lats.extend(r.phase_set_lats)

    print(f"{'Phase':<20s} {'avg':>10s} {'p99':>10s} {'ops':>12s}")
    print("-" * 55)

    for name, lats, ops in [
        ("batch_exists", all_exists_lats, total_exists_ops),
        ("batch_get", all_get_lats, total_get_ops),
        ("batch_set", all_set_lats, total_set_ops),
    ]:
        if lats:
            lats.sort()
            avg = sum(lats) / len(lats) * 1000
            p99 = _percentile(lats, 99) * 1000
            print(f"  {name:<18s} {avg:>8.2f}ms {p99:>8.2f}ms   ({ops:,} ops)")
        else:
            print(f"  {name:<18s}      ---       ---   (0 ops)")

    print()
    print(f"Total sub-key ops: {total_ops:,}")
    print(f"Throughput:        {total_ops / elapsed:,.0f} ops/s")
    print(f"Bandwidth:         {total_bw:.2f} GB/s")
    if total_errors:
        print(f"Errors:            {total_errors:,}")
    print("=====================================")


def print_batch_results(args, results: list[WorkerResult]) -> None:
    """Print batch benchmark results."""
    if not results:
        print("[bench] No results collected")
        return

    elapsed = max(r.elapsed for r in results) if results else 1.0
    total_ops = sum(r.ops for r in results)
    total_gets = sum(r.gets for r in results)
    total_sets = sum(r.sets for r in results)
    total_errors = sum(r.errors for r in results)
    total_bytes_read = sum(r.bytes_read for r in results)
    total_bytes_write = sum(r.bytes_write for r in results)
    total_bw = (total_bytes_read + total_bytes_write) / elapsed / 1e9 if elapsed > 0 else 0.0
    ops_per_sec = total_ops / elapsed if elapsed > 0 else 0.0

    # Aggregate batch latencies
    avg_ms = sum(r.avg_ms * r.ops for r in results) / total_ops if total_ops > 0 else 0.0
    p99_ms = max(r.p99_ms for r in results) if results else 0.0

    print()
    print("=== CAMA Batch Benchmark Results ===")
    print(f"Transport:       {args.transport.upper()}")
    print(f"Clients:         {args.clients}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Duration:        {elapsed:.2f}s")
    print(f"Total ops:       {total_ops:,}  (GETs: {total_gets:,}, SETs: {total_sets:,})")
    print(f"Throughput:      {ops_per_sec:,.0f} ops/s")
    print(f"Bandwidth:       {total_bw:.2f} GB/s")
    print(f"Batch latency:   avg={avg_ms:.2f}ms  p99={p99_ms:.2f}ms")
    if total_errors:
        print(f"Errors:          {total_errors:,}")
    print("=====================================")
