"""CAMA Python benchmark — measures throughput and bandwidth for CamaClient (TCP)
and RDMAClient (RDMA) against a running CAMA server.

Usage examples:
    # TCP baseline (Llama 70B, 10k-token context)
    python bench.py --preset llama70b-10k --transport tcp --clients 32 --duration 60

    # RDMA
    python bench.py --preset llama70b-10k --transport rdma --addr 10.0.0.1 --clients 16 --duration 60

    # 100k-token context, read-heavy
    python bench.py --preset llama70b-100k --read-ratio 0.95 --transport rdma --clients 32

    # Debug mode (per-op timing in client logs)
    CAMA_DEBUG=1 python bench.py --transport rdma --clients 2 --duration 5 --debug

    # EXISTS-heavy (prefetch pipeline validation)
    python bench.py --workload exists --transport rdma --clients 16 --duration 30

    # TTL churn (tests TTL sweep + memory reclamation)
    python bench.py --workload ttl-churn --transport tcp --clients 32 --duration 60

    # DELETE-heavy (tombstone compaction stress)
    python bench.py --workload delete-heavy --transport tcp --clients 32 --duration 60

    # Mixed (post-benchmark server stats are on by default)
    python bench.py --workload mixed --transport tcp --duration 30

    # Load-probe (SET all keys, then GET all keys — canonical KV cache pattern)
    python bench.py --workload load-probe --keys 50 --value-size 1048576 --transport tcp --clients 4

    # Sweep mode — run multiple workloads and compare
    python bench.py --sweep --transport tcp --clients 16
    python bench.py --sweep --preset small-1k --transport tcp
    python bench.py --sweep --sweep-workloads mixed,exists --transport rdma

    # Multiprocessing (bypass GIL ceiling for TCP / small-value RDMA)
    python bench.py --engine process --transport tcp --clients 8 --duration 30

    # Saturation finder (auto-ramp until throughput plateaus)
    python bench.py --saturate --preset large-2m --transport rdma --saturate-max 16

    # SGLang-realistic (mirrors cama_storage.py: 1 conn + ThreadPoolExecutor per rank)
    python bench.py --workload sglang --preset llama70b-10k --clients 4 --batch-size 16

    # SGLang-stress (separate connections per worker, batch ops)
    python bench.py --workload sglang-stress --preset llama70b-10k --clients 32 --batch-size 16

    # Batch throughput (raw m-op performance)
    python bench.py --workload batch --preset medium-64k --clients 16 --batch-size 64
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Preset table: (value_size_bytes, key_count)
# ---------------------------------------------------------------------------
PRESETS = {
    # LLM KV cache presets — 16-token pages (default)
    # key_count = ceil(context_tokens / page_tokens) per request
    # value_size = per-token KV bytes × page_tokens
    "llama70b-10k":  (5_242_880, 625),
    "llama70b-100k": (5_242_880, 6250),
    "llama8b-10k":   (524_288,   625),
    "llama8b-100k":  (524_288,   6250),
    # 64-token pages: 4x larger values, 4x fewer keys
    "llama70b-10k-p64":  (20_971_520, 157),
    "llama70b-100k-p64": (20_971_520, 1563),
    "llama8b-10k-p64":   (2_097_152,  157),
    "llama8b-100k-p64":  (2_097_152,  1563),
    # 256-token pages: 16x larger values, 16x fewer keys
    "llama70b-10k-p256":  (83_886_080, 40),
    "llama70b-100k-p256": (83_886_080, 391),
    "llama8b-10k-p256":   (8_388_608,  40),
    "llama8b-100k-p256":  (8_388_608,  391),
    # Generic presets (model-agnostic)
    "small-1k":      (1_024,      10_000),
    "medium-64k":    (65_536,     1_000),
    "large-1m":      (1_048_576,  500),
    "large-2m":      (2_097_152,  250),
    "large-4m":      (4_194_304,  125),
}

# ---------------------------------------------------------------------------
# Workload mode defaults: (read_ratio, exists_ratio, delete_ratio, ttl_ms)
# ---------------------------------------------------------------------------
WORKLOAD_DEFAULTS = {
    "mixed":        (0.8, 0.0, 0.0, 0),
    "exists":       (0.1, 0.8, 0.0, 0),
    "ttl-churn":    (0.4, 0.1, 0.0, 5000),
    "delete-heavy": (0.2, 0.0, 0.6, 0),
    "load-probe":   (1.0, 0.0, 0.0, 0),  # ratios unused — phases are hardcoded
    "sglang":       (0.8, 0.0, 0.0, 0),  # ratios unused — lifecycle-driven
    "sglang-stress":(0.8, 0.0, 0.0, 0),  # ratios unused — lifecycle-driven
    "batch":        (0.8, 0.0, 0.0, 0),  # read_ratio used for GET vs SET split
}


# ---------------------------------------------------------------------------
# Sweep result — per-workload metrics collected by _run_timed_workload
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    workload: str = ""
    ops_per_sec: float = 0.0
    bandwidth_gbps: float = 0.0
    hit_rate: float = -1.0       # -1 means N/A
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    errors: int = 0
    total_ops: int = 0


# ---------------------------------------------------------------------------
# Zipfian generator (harmonic approximation, s=1.0)
# Same algorithm as the Go bench implementation.
# ---------------------------------------------------------------------------
class ZipfGenerator:
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


class _LCG:
    """Fast per-thread LCG RNG — same constants as Go bench."""
    __slots__ = ("_state",)

    def __init__(self, seed: int):
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def next_int(self) -> int:
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self._state

    def next_float(self) -> float:
        return (self.next_int() >> 11) / (1 << 53)


# ---------------------------------------------------------------------------
# Per-thread worker
# ---------------------------------------------------------------------------
def _worker(
    thread_id: int,
    transport: str,
    addr: str,
    port: int,
    value_size: int,
    keys: list[str],
    read_ratio: float,
    distribution: str,
    zipf: ZipfGenerator | None,
    stop_event: threading.Event,
    barrier: threading.Barrier,
    result: dict,
    debug: bool = False,
    endpoint: dict | None = None,
    exists_ratio: float = 0.0,
    delete_ratio: float = 0.0,
    ttl_ms: int = 0,
) -> None:
    # Override addr/port if a specific NIC endpoint was assigned
    if endpoint is not None:
        addr = endpoint["ip"]
        port = endpoint["port"]
        result["nic_device"] = endpoint.get("device", "unknown")
        result["nic_ip"] = addr
        result["nic_port"] = port

    # Import inside thread so import errors are caught per-thread
    if transport == "rdma":
        from l3_client.rdma_client import RDMAClient
        from l3_client.sgl import SGL
        client = RDMAClient(addr, port=port, debug=debug)
        if debug:
            # Full timing resolution in debug mode (default is 1/64 sampling)
            client._transport.set_sample_rate(1)
    else:
        from l3_client.client import CamaClient
        from l3_client.sgl import SGL
        client = CamaClient(addr, port=port)

    # Allocate one buffer filled with random bytes
    buf = ctypes.create_string_buffer(os.urandom(value_size), value_size)
    ptr = ctypes.addressof(buf)
    sgl = None

    if transport == "rdma":
        handle = client.reg_memory(ptr, value_size, buf=buf)
        sgl = SGL(ptr=ptr, size=value_size, reg_handle=handle)
    else:
        sgl = SGL(ptr=ptr, size=value_size)

    rng = _LCG(thread_id * 2654435761 + 1)

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

    # Pre-compute op-selection thresholds
    delete_thresh = delete_ratio
    exists_thresh = delete_ratio + exists_ratio
    # Remaining probability split by read_ratio into GET vs SET
    remaining = 1.0 - exists_thresh
    read_thresh = exists_thresh + remaining * read_ratio

    # Wait for all workers to be ready before starting benchmark loop
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        result["error"] = "barrier broken"
        return

    worker_start = time.monotonic()

    try:
        while not stop_event.is_set():
            # Pick key index
            if zipf is not None:
                key_idx = zipf.sample(rng)
            else:
                key_idx = rng.next_int() % n_keys
            key = keys[key_idx]

            r = rng.next_float()

            try:
                t_op = time.monotonic()
                if r < delete_thresh:
                    rc = client.delete(key)
                    lat = time.monotonic() - t_op
                    del_latencies.append(lat)
                    deletes += 1
                elif r < exists_thresh:
                    rc = client.exists(key)
                    lat = time.monotonic() - t_op
                    exists_latencies.append(lat)
                    exists_ops += 1
                    if rc == 1:  # EXISTS_FOUND
                        exists_hits += 1
                elif r < read_thresh:
                    rc = client.get(key, sgl)
                    lat = time.monotonic() - t_op
                    get_latencies.append(lat)
                    gets += 1
                    if rc == 0:
                        hits += 1
                        bytes_read += value_size
                else:
                    rc = client.set(key, sgl, ttl_ms=ttl_ms)
                    lat = time.monotonic() - t_op
                    set_latencies.append(lat)
                    sets += 1
                    if rc == 0:
                        bytes_write += value_size
            except Exception:
                errors += 1

            ops += 1

            # Update shared result dict periodically for live progress stats
            if ops % 64 == 0:
                result["ops"] = ops
                result["bytes_read"] = bytes_read
                result["bytes_write"] = bytes_write
    finally:
        worker_elapsed = time.monotonic() - worker_start

        # Collect C++ transport stats before closing
        cpp_stats = None
        if transport == "rdma" and hasattr(client._transport, 'get_stats'):
            try:
                cpp_stats = client._transport.get_stats()
            except Exception:
                pass

        if transport == "rdma":
            try:
                client.dereg_memory(handle)
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass

    result["ops"] = ops
    result["gets"] = gets
    result["sets"] = sets
    result["hits"] = hits
    result["exists_ops"] = exists_ops
    result["exists_hits"] = exists_hits
    result["deletes"] = deletes
    result["errors"] = errors
    result["bytes_read"] = bytes_read
    result["bytes_write"] = bytes_write
    result["worker_elapsed"] = worker_elapsed
    result["get_latencies"] = get_latencies
    result["set_latencies"] = set_latencies
    result["exists_latencies"] = exists_latencies
    result["del_latencies"] = del_latencies
    if cpp_stats is not None:
        result["cpp_stats"] = cpp_stats


# ---------------------------------------------------------------------------
# Latency percentile helper
# ---------------------------------------------------------------------------
def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile from already-sorted data (0 <= p <= 100)."""
    if not sorted_data:
        return 0.0
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ---------------------------------------------------------------------------
# Pre-populate helper
# ---------------------------------------------------------------------------
def _prepopulate(transport: str, addr: str, port: int, keys: list[str], value_size: int, ttl_ms: int = 0) -> None:
    from l3_client.sgl import SGL

    if transport == "rdma":
        from l3_client.rdma_client import RDMAClient
        client = RDMAClient(addr, port=port)
    else:
        from l3_client.client import CamaClient
        client = CamaClient(addr, port=port)

    buf = ctypes.create_string_buffer(os.urandom(value_size), value_size)
    ptr = ctypes.addressof(buf)
    sgl = SGL(ptr=ptr, size=value_size)

    if transport == "rdma":
        handle = client.reg_memory(ptr, value_size, buf=buf)
        sgl = SGL(ptr=ptr, size=value_size, reg_handle=handle)

    print(f"[bench] Pre-populating {len(keys)} keys...", flush=True)
    for key in keys:
        try:
            client.set(key, sgl, ttl_ms=ttl_ms)
        except Exception as e:
            print(f"[bench] pre-populate error for {key}: {e}", file=sys.stderr)

    if transport == "rdma":
        try:
            client.dereg_memory(handle)
        except Exception:
            pass
    client.close()
    print("[bench] Pre-population complete", flush=True)


# ---------------------------------------------------------------------------
# Load-probe worker (two-phase: SET all, then GET all)
# ---------------------------------------------------------------------------
def _load_probe_worker(
    worker_id: int,
    transport: str,
    addr: str,
    port: int,
    assigned_keys: list[str],
    value_size: int,
    phase_barrier: threading.Barrier,
    result: dict,
    debug: bool = False,
    endpoint: dict | None = None,
) -> None:
    if endpoint is not None:
        addr = endpoint["ip"]
        port = endpoint["port"]
        result["nic_device"] = endpoint.get("device", "unknown")
        result["nic_ip"] = addr
        result["nic_port"] = port

    if transport == "rdma":
        from l3_client.rdma_client import RDMAClient
        from l3_client.sgl import SGL
        client = RDMAClient(addr, port=port, debug=debug)
        if debug:
            client._transport.set_sample_rate(1)
    else:
        from l3_client.client import CamaClient
        from l3_client.sgl import SGL
        client = CamaClient(addr, port=port)

    buf = ctypes.create_string_buffer(os.urandom(value_size), value_size)
    ptr = ctypes.addressof(buf)
    handle = None

    if transport == "rdma":
        handle = client.reg_memory(ptr, value_size, buf=buf)
        sgl = SGL(ptr=ptr, size=value_size, reg_handle=handle)
    else:
        sgl = SGL(ptr=ptr, size=value_size)

    set_latencies: list[float] = []
    get_latencies: list[float] = []
    bytes_write = 0
    bytes_read = 0
    hits = 0
    misses = 0
    set_errors = 0
    get_errors = 0

    # --- Phase 1: Load (SET) ---
    try:
        phase_barrier.wait()
    except threading.BrokenBarrierError:
        result["error"] = "barrier broken (load start)"
        return

    load_start = time.monotonic()
    for key in assigned_keys:
        try:
            t0 = time.monotonic()
            rc = client.set(key, sgl)
            lat = time.monotonic() - t0
            set_latencies.append(lat)
            if rc == 0:
                bytes_write += value_size
        except Exception:
            set_errors += 1
    load_elapsed = time.monotonic() - load_start

    # Signal load phase complete
    try:
        phase_barrier.wait()
    except threading.BrokenBarrierError:
        result["error"] = "barrier broken (load end)"
        return

    # --- Phase 2: Probe (GET) ---
    try:
        phase_barrier.wait()
    except threading.BrokenBarrierError:
        result["error"] = "barrier broken (probe start)"
        return

    probe_start = time.monotonic()
    for key in assigned_keys:
        try:
            t0 = time.monotonic()
            rc = client.get(key, sgl)
            lat = time.monotonic() - t0
            get_latencies.append(lat)
            if rc == 0:
                hits += 1
                bytes_read += value_size
            else:
                misses += 1
        except Exception:
            get_errors += 1
    probe_elapsed = time.monotonic() - probe_start

    # Signal probe phase complete
    try:
        phase_barrier.wait()
    except threading.BrokenBarrierError:
        pass

    # Cleanup
    if transport == "rdma" and handle is not None:
        try:
            client.dereg_memory(handle)
        except Exception:
            pass
    try:
        client.close()
    except Exception:
        pass

    result["set_count"] = len(assigned_keys)
    result["get_count"] = len(assigned_keys)
    result["set_latencies"] = set_latencies
    result["get_latencies"] = get_latencies
    result["bytes_write"] = bytes_write
    result["bytes_read"] = bytes_read
    result["hits"] = hits
    result["misses"] = misses
    result["set_errors"] = set_errors
    result["get_errors"] = get_errors
    result["load_elapsed"] = load_elapsed
    result["probe_elapsed"] = probe_elapsed


def _run_load_probe(args, key_list: list[str], endpoints, quiet: bool = False) -> SweepResult:
    """Two-phase benchmark: SET all keys, then GET all keys."""

    n_clients = args.clients
    n_keys = len(key_list)

    # Partition keys round-robin across workers
    partitions: list[list[str]] = [[] for _ in range(n_clients)]
    for i, key in enumerate(key_list):
        partitions[i % n_clients].append(key)

    # Barrier: n_clients + 1 (main thread), used 4 times:
    #   1) load start, 2) load end, 3) probe start, 4) probe end
    phase_barrier = threading.Barrier(n_clients + 1, timeout=300)
    results: list[dict] = [{} for _ in range(n_clients)]
    threads: list[threading.Thread] = []

    connect_start = time.monotonic()

    for i in range(n_clients):
        ep = endpoints[i % len(endpoints)] if endpoints else None
        t = threading.Thread(
            target=_load_probe_worker,
            args=(
                i,
                args.transport,
                args.addr,
                args.port,
                partitions[i],
                args.value_size,
                phase_barrier,
                results[i],
                args.debug,
                ep,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # --- Phase 1: Load ---
    try:
        phase_barrier.wait()  # load start
    except threading.BrokenBarrierError:
        print("[bench] ERROR: barrier broken — workers failed to connect", file=sys.stderr)
        sys.exit(1)

    connect_elapsed = time.monotonic() - connect_start
    if not quiet:
        print(f"[bench] All {n_clients} workers connected in {connect_elapsed:.1f}s", flush=True)
        print(f"[bench] Load phase: SET {n_keys} keys...", flush=True)
    load_wall_start = time.monotonic()

    try:
        phase_barrier.wait()  # load end
    except threading.BrokenBarrierError:
        print("[bench] ERROR: barrier broken during load phase", file=sys.stderr)
        sys.exit(1)

    load_wall = time.monotonic() - load_wall_start

    # --- Phase 2: Probe ---
    if not quiet:
        print(f"[bench] Probe phase: GET {n_keys} keys...", flush=True)
    probe_wall_start = time.monotonic()

    try:
        phase_barrier.wait()  # probe start
    except threading.BrokenBarrierError:
        print("[bench] ERROR: barrier broken before probe phase", file=sys.stderr)
        sys.exit(1)

    try:
        phase_barrier.wait()  # probe end
    except threading.BrokenBarrierError:
        pass

    probe_wall = time.monotonic() - probe_wall_start

    for t in threads:
        t.join()

    # --- Aggregate results ---
    total_sets = sum(r.get("set_count", 0) for r in results)
    total_gets = sum(r.get("get_count", 0) for r in results)
    total_hits = sum(r.get("hits", 0) for r in results)
    total_misses = sum(r.get("misses", 0) for r in results)
    total_bytes_write = sum(r.get("bytes_write", 0) for r in results)
    total_bytes_read = sum(r.get("bytes_read", 0) for r in results)
    total_set_errors = sum(r.get("set_errors", 0) for r in results)
    total_get_errors = sum(r.get("get_errors", 0) for r in results)

    all_set_lats: list[float] = []
    all_get_lats: list[float] = []
    for r in results:
        all_set_lats.extend(r.get("set_latencies", []))
        all_get_lats.extend(r.get("get_latencies", []))

    value_mb = args.value_size / (1024 * 1024)

    # --- Load phase report ---
    set_ops_sec = total_sets / load_wall if load_wall > 0 else 0
    set_bw = total_bytes_write / load_wall / 1e9 if load_wall > 0 else 0

    if not quiet:
        print()
        print("--- Load Phase (SET) ---")
        print(f"  keys:       {total_sets:,}")
        print(f"  elapsed:    {load_wall:.2f}s")
        print(f"  ops/sec:    {set_ops_sec:,.0f}")
        print(f"  bandwidth:  {set_bw:.2f} GB/s (write)")
        if total_set_errors:
            print(f"  errors:     {total_set_errors:,}")

    if all_set_lats:
        all_set_lats.sort()
        p50_s = _percentile(all_set_lats, 50) * 1000
        p99_s = _percentile(all_set_lats, 99) * 1000
        if not quiet:
            avg_s = sum(all_set_lats) / len(all_set_lats) * 1000
            p999_s = _percentile(all_set_lats, 99.9) * 1000
            print(f"  SET avg:    {avg_s:.2f}ms  p50: {p50_s:.2f}ms  p99: {p99_s:.2f}ms  p99.9: {p999_s:.2f}ms")

    # --- Probe phase report ---
    get_ops_sec = total_gets / probe_wall if probe_wall > 0 else 0
    get_bw = total_bytes_read / probe_wall / 1e9 if probe_wall > 0 else 0
    hit_rate = (total_hits / total_gets * 100) if total_gets > 0 else 0.0

    if not quiet:
        print()
        print("--- Probe Phase (GET) ---")
        print(f"  keys:       {total_gets:,}")
        print(f"  elapsed:    {probe_wall:.2f}s")
        print(f"  ops/sec:    {get_ops_sec:,.0f}")
        print(f"  bandwidth:  {get_bw:.2f} GB/s (read)")
        print(f"  hit_rate:   {hit_rate:.1f}%")
        if total_get_errors:
            print(f"  errors:     {total_get_errors:,}")

    p50_g = 0.0
    p99_g = 0.0
    if all_get_lats:
        all_get_lats.sort()
        p50_g = _percentile(all_get_lats, 50) * 1000
        p99_g = _percentile(all_get_lats, 99) * 1000
        if not quiet:
            avg_g = sum(all_get_lats) / len(all_get_lats) * 1000
            p999_g = _percentile(all_get_lats, 99.9) * 1000
            print(f"  GET avg:    {avg_g:.2f}ms  p50: {p50_g:.2f}ms  p99: {p99_g:.2f}ms  p99.9: {p999_g:.2f}ms")

    if not quiet:
        print()
        print("=====================================")

    # Combined ops/s for sweep: SET phase + GET phase
    total_wall = load_wall + probe_wall
    total_lp_ops = total_sets + total_gets
    combined_ops_sec = total_lp_ops / total_wall if total_wall > 0 else 0
    combined_bw = (total_bytes_read + total_bytes_write) / total_wall / 1e9 if total_wall > 0 else 0

    return SweepResult(
        workload="load-probe",
        ops_per_sec=combined_ops_sec,
        bandwidth_gbps=combined_bw,
        hit_rate=hit_rate,
        p50_ms=p50_g,
        p99_ms=p99_g,
        errors=total_set_errors + total_get_errors,
        total_ops=total_lp_ops,
    )


# ---------------------------------------------------------------------------
# Post-benchmark server stats (shared by all workloads)
# ---------------------------------------------------------------------------
def _post_stats(args) -> None:
    try:
        if args.transport == "rdma":
            from l3_client.rdma_client import RDMAClient
            _sc = RDMAClient(args.addr, port=args.port)
        else:
            from l3_client.client import CamaClient
            _sc = CamaClient(args.addr, port=args.port)
        srv = _sc.stats()
        _sc.close()

        totals = srv.get("totals", {})
        print()
        print("--- Server Stats ---")
        s_gets = totals.get("gets", 0)
        s_get_hits = totals.get("hits", 0)
        s_get_hr = (s_get_hits / s_gets * 100) if s_gets > 0 else 0.0
        print(f"  gets:                {s_gets:,}  (hit_rate: {s_get_hr:.1f}%)")
        print(f"  sets:                {totals.get('sets', 0):,}")
        s_exists = totals.get("exists", 0)
        s_exists_hits = totals.get("exists_hits", 0)
        s_exists_hr = (s_exists_hits / s_exists * 100) if s_exists > 0 else 0.0
        print(f"  exists:              {s_exists:,}  (hit_rate: {s_exists_hr:.1f}%)")
        print(f"  deletes:             {totals.get('deletes', 0):,}")
        s_evictions = totals.get("evictions", 0)
        s_sets = totals.get("sets", 0)
        s_evict_rate = (s_evictions / s_sets * 100) if s_sets > 0 else 0.0
        print(f"  evictions:           {s_evictions:,}  (eviction_rate: {s_evict_rate:.1f}%)")
        print(f"  entries:             {totals.get('entries', 0):,}")
        print(f"  active_gb:           {totals.get('active_gb', 0):.2f} GB")

        # Value-size detection
        vsd = srv.get("value_size_detection", {})
        if vsd:
            print()
            print("--- Value-Size Detection ---")
            print(f"  status:              {vsd.get('status', 'unknown')}")
            print(f"  dominant_size:       {vsd.get('dominant_size', 0)} bytes")
            print(f"  dominant_freq:       {vsd.get('dominant_freq', 0):.1f}%")
            print(f"  waste:               {vsd.get('waste', 0):.1f}%")

        # Slab utilization
        slabs = srv.get("slab_classes", [])
        if slabs:
            slabs_sorted = sorted(slabs, key=lambda s: s.get("alloc_count", 0), reverse=True)
            top = slabs_sorted[:5]
            print()
            print("--- Slab Utilization (top 5 by alloc_count) ---")
            print(f"  {'Size':>12s}  {'Used/Total':>14s}  {'Allocs':>10s}  {'SlotUtil%':>10s}")
            for sl in top:
                sz = sl.get("size", 0)
                used = sl.get("used_slots", 0)
                total = sl.get("total_slots", 0)
                allocs = sl.get("alloc_count", 0)
                slot_util = sl.get("slot_utilization", 0.0)
                print(f"  {sz:>12d}  {used:>6d}/{total:<6d}  {allocs:>10d}  {slot_util:>9.1f}%")
    except Exception as e:
        print(f"\n[bench] --post-stats failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# _run_timed_workload — runs a single workload and returns SweepResult
# ---------------------------------------------------------------------------
def _run_timed_workload(
    args,
    workload_name: str,
    key_list: list[str],
    zipf: ZipfGenerator | None,
    endpoints,
    duration: int,
    quiet: bool = False,
) -> SweepResult:
    """Run one workload for *duration* seconds and return a SweepResult.

    Applies workload ratios from WORKLOAD_DEFAULTS, flushes the server,
    pre-populates, launches workers, and aggregates results.
    """

    # Apply workload ratios
    wl_read, wl_exists, wl_delete, wl_ttl = WORKLOAD_DEFAULTS[workload_name]

    # Flush before each workload
    if not args.no_flush:
        if not quiet:
            print(f"[bench] Flushing server before '{workload_name}'...", flush=True)
        if args.transport == "rdma":
            from l3_client.rdma_client import RDMAClient
            _fc = RDMAClient(args.addr, port=args.port)
        else:
            from l3_client.client import CamaClient
            _fc = CamaClient(args.addr, port=args.port)
        _fc.flush()
        _fc.close()

    # Load-probe has its own two-phase logic
    if workload_name == "load-probe":
        # Temporarily patch args for load-probe
        orig_workload = args.workload
        args.workload = "load-probe"
        result = _run_load_probe(args, key_list, endpoints, quiet=quiet)
        args.workload = orig_workload
        return result

    # Pre-populate
    if not quiet:
        _prepopulate(args.transport, args.addr, args.port, key_list, args.value_size,
                     ttl_ms=wl_ttl)
    else:
        # Silent pre-populate
        from l3_client.sgl import SGL as _SGL
        if args.transport == "rdma":
            from l3_client.rdma_client import RDMAClient as _RC
            _pc = _RC(args.addr, port=args.port)
        else:
            from l3_client.client import CamaClient as _CC
            _pc = _CC(args.addr, port=args.port)
        buf = ctypes.create_string_buffer(os.urandom(args.value_size), args.value_size)
        ptr = ctypes.addressof(buf)
        sgl = _SGL(ptr=ptr, size=args.value_size)
        if args.transport == "rdma":
            handle = _pc.reg_memory(ptr, args.value_size, buf=buf)
            sgl = _SGL(ptr=ptr, size=args.value_size, reg_handle=handle)
        for key in key_list:
            try:
                _pc.set(key, sgl, ttl_ms=wl_ttl)
            except Exception:
                pass
        if args.transport == "rdma":
            try:
                _pc.dereg_memory(handle)
            except Exception:
                pass
        _pc.close()

    # Launch workers
    stop_event = threading.Event()
    barrier = threading.Barrier(args.clients + 1, timeout=120)
    results: list[dict] = [{} for _ in range(args.clients)]
    threads: list[threading.Thread] = []

    for i in range(args.clients):
        results[i] = {}
        ep = endpoints[i % len(endpoints)] if endpoints else None
        t = threading.Thread(
            target=_worker,
            args=(
                i,
                args.transport,
                args.addr,
                args.port,
                args.value_size,
                key_list,
                wl_read,
                args.distribution,
                zipf,
                stop_event,
                barrier,
                results[i],
                args.debug,
                ep,
                wl_exists,
                wl_delete,
                wl_ttl,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        return SweepResult(workload=workload_name, errors=1)

    start_time = time.monotonic()
    if not quiet:
        print(f"[bench] Running '{workload_name}' for {duration}s...", flush=True)

    time.sleep(duration)
    stop_event.set()

    for t in threads:
        t.join()

    elapsed = time.monotonic() - start_time

    # Aggregate
    total_ops = sum(r.get("ops", 0) for r in results)
    total_gets = sum(r.get("gets", 0) for r in results)
    total_hits = sum(r.get("hits", 0) for r in results)
    total_exists_ops = sum(r.get("exists_ops", 0) for r in results)
    total_exists_hits = sum(r.get("exists_hits", 0) for r in results)
    total_errors = sum(r.get("errors", 0) for r in results)
    total_bytes_read = sum(r.get("bytes_read", 0) for r in results)
    total_bytes_write = sum(r.get("bytes_write", 0) for r in results)

    ops_per_sec = total_ops / elapsed if elapsed > 0 else 0.0
    total_bw = (total_bytes_read + total_bytes_write) / elapsed / 1e9 if elapsed > 0 else 0.0

    # Hit rate: for exists-heavy workloads, use exists hit rate; otherwise GET hit rate
    if total_exists_ops > total_gets and total_exists_ops > 0:
        hit_rate = total_exists_hits / total_exists_ops * 100
    elif total_gets > 0:
        hit_rate = total_hits / total_gets * 100
    else:
        hit_rate = -1.0

    # Latency
    all_lats: list[float] = []
    for r in results:
        all_lats.extend(r.get("get_latencies", []))
        all_lats.extend(r.get("set_latencies", []))
        all_lats.extend(r.get("exists_latencies", []))
        all_lats.extend(r.get("del_latencies", []))

    p50 = 0.0
    p99 = 0.0
    if all_lats:
        all_lats.sort()
        p50 = _percentile(all_lats, 50) * 1000
        p99 = _percentile(all_lats, 99) * 1000

    return SweepResult(
        workload=workload_name,
        ops_per_sec=ops_per_sec,
        bandwidth_gbps=total_bw,
        hit_rate=hit_rate,
        p50_ms=p50,
        p99_ms=p99,
        errors=total_errors,
        total_ops=total_ops,
    )


# ---------------------------------------------------------------------------
# Sweep — run multiple workloads and print comparison table
# ---------------------------------------------------------------------------
def _print_sweep_table(sweep_results: list[SweepResult]) -> None:
    hdr = f"{'Workload':<16s} {'Ops/s':>10s} {'BW GB/s':>9s} {'Hit%':>7s} {'p50 ms':>9s} {'p99 ms':>9s} {'Errors':>8s}"
    print()
    print("SWEEP COMPARISON")
    print(hdr)
    print("-" * len(hdr))
    for r in sweep_results:
        hit_str = f"{r.hit_rate:5.1f}" if r.hit_rate >= 0 else "    -"
        print(f"{r.workload:<16s} {r.ops_per_sec:>10,.0f} {r.bandwidth_gbps:>9.2f} {hit_str:>7s} {r.p50_ms:>9.2f} {r.p99_ms:>9.2f} {r.errors:>8d}")
    print()


def _run_sweep(args, key_list: list[str], zipf: ZipfGenerator | None, endpoints) -> None:
    """Run each workload in sequence and print a comparison table."""
    if args.sweep_workloads:
        workloads = [w.strip() for w in args.sweep_workloads.split(",")]
        for w in workloads:
            if w not in WORKLOAD_DEFAULTS:
                print(f"[bench] ERROR: unknown workload '{w}'. "
                      f"Valid: {', '.join(WORKLOAD_DEFAULTS.keys())}", file=sys.stderr)
                sys.exit(1)
    else:
        workloads = ["mixed", "exists", "delete-heavy", "load-probe"]

    # Sweep uses shorter duration per workload when not explicitly set
    sweep_duration = args.duration

    sweep_results: list[SweepResult] = []
    for i, wl in enumerate(workloads):
        print(f"\n[sweep] === Running workload {i+1}/{len(workloads)}: {wl} ({sweep_duration}s) ===", flush=True)
        sr = _run_timed_workload(args, wl, key_list, zipf, endpoints, sweep_duration, quiet=True)
        sweep_results.append(sr)
        print(f"[sweep]   {wl}: {sr.ops_per_sec:,.0f} ops/s, {sr.bandwidth_gbps:.2f} GB/s", flush=True)

    _print_sweep_table(sweep_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAMA Python benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--addr", default="208.0.0.13", help="Server address")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (default: 18000 for TCP, 18001 for RDMA)")
    parser.add_argument("--transport", default="rdma", choices=["tcp", "rdma"],
                        help="Transport: tcp or rdma")
    parser.add_argument("--clients", type=int, default=32,
                        help="Worker threads")
    parser.add_argument("--duration", type=int, default=None,
                        help="Benchmark duration in seconds (default: 30, or 10 per workload in sweep mode)")
    parser.add_argument("--value-size", type=int, default=5_242_880,
                        help="Bytes per value (overridden by --preset)")
    parser.add_argument("--keys", type=int, default=100,
                        help="Unique key count (overridden by --preset)")
    parser.add_argument("--read-ratio", type=float, default=None,
                        help="Fraction of operations that are reads (default: per-workload)")
    parser.add_argument("--distribution", default="zipfian",
                        choices=["zipfian", "uniform"],
                        help="Key distribution")
    parser.add_argument("--preset", default=None,
                        choices=list(PRESETS.keys()),
                        help="Model/context preset overriding --value-size and --keys")
    parser.add_argument("--no-flush", action="store_true", default=False,
                        help="Skip sending FLUSH command to the server before starting")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging in client (per-op timing)")
    parser.add_argument("--no-multinic", action="store_true", default=False,
                        help="Disable multi-NIC RDMA endpoint discovery (multi-NIC is ON by default for RDMA)")
    parser.add_argument("--workload", default="mixed",
                        choices=list(WORKLOAD_DEFAULTS.keys()),
                        help="Workload mode")
    parser.add_argument("--ttl-ms", type=int, default=None,
                        help="TTL in ms for SET ops (default: 0 = none)")
    parser.add_argument("--exists-ratio", type=float, default=None,
                        help="Fraction of ops that are EXISTS")
    parser.add_argument("--delete-ratio", type=float, default=None,
                        help="Fraction of ops that are DELETEs")
    parser.set_defaults(post_stats=True)
    parser.add_argument("--no-post-stats", action="store_false", dest="post_stats",
                        help="Skip post-benchmark server stats")
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="Run multiple workloads and compare results")
    parser.add_argument("--sweep-workloads", type=str, default=None,
                        help="Comma-separated workloads for sweep (default: mixed,exists,delete-heavy,load-probe)")
    # Multiprocessing
    parser.add_argument("--engine", default=None, choices=["thread", "process"],
                        help="Execution engine (default: process for TCP, thread for RDMA)")
    # Saturation
    parser.add_argument("--saturate", action="store_true", default=False,
                        help="Auto-ramp client count until throughput plateaus")
    parser.add_argument("--saturate-max", type=int, default=64,
                        help="Max clients to test in saturation mode")
    parser.add_argument("--saturate-duration", type=int, default=10,
                        help="Seconds per step in saturation mode")
    parser.add_argument("--saturate-threshold", type=float, default=5.0,
                        help="Stop when improvement < this percent")
    # SGLang / Batch
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Pages per batch for sglang/batch workloads")
    parser.add_argument("--io-workers", type=int, default=16,
                        help="ThreadPoolExecutor size for sglang mode")
    parser.add_argument("--key-multiplier", type=int, default=2, choices=[1, 2],
                        help="Sub-keys per page: 1=MLA, 2=MHA")
    parser.add_argument("--compute-delay-ms", type=float, default=0.0,
                        help="Simulated GPU compute delay between get and set (ms)")
    parser.add_argument("--prefix-hit-rate", type=float, default=0.6,
                        help="Fraction of keys pre-populated for prefix matching")
    args = parser.parse_args()

    # Apply workload defaults, then let explicit flags override
    wl_read, wl_exists, wl_delete, wl_ttl = WORKLOAD_DEFAULTS[args.workload]
    if args.read_ratio is None:
        args.read_ratio = wl_read
    if args.exists_ratio is None:
        args.exists_ratio = wl_exists
    if args.delete_ratio is None:
        args.delete_ratio = wl_delete
    if args.ttl_ms is None:
        args.ttl_ms = wl_ttl

    # Validate ratios
    if args.delete_ratio + args.exists_ratio >= 1.0:
        parser.error("delete_ratio + exists_ratio must be < 1.0")

    # Apply preset only when explicitly given; otherwise keep --value-size / --keys
    if args.preset is not None:
        args.value_size, args.keys = PRESETS[args.preset]
        print(f"\nTIP: For static benchmarks, configure the server with:")
        print(f"  slab_preset = \"benchmark\"")
        print(f"  model_page_bytes = {args.value_size}")
        print()

    # Default duration: 10s for sweep, 30s otherwise
    if args.duration is None:
        args.duration = 10 if args.sweep else 30

    # Default port
    if args.port is None:
        args.port = 18001 if args.transport == "rdma" else 18000

    # Default engine: process for TCP (GIL ceiling), thread for RDMA (shared MR)
    if args.engine is None:
        args.engine = "thread" if args.transport == "rdma" else "process"

    # Validate saturation + sweep mutual exclusion
    if args.saturate and args.sweep:
        parser.error("--saturate and --sweep are mutually exclusive")

    # Check GIL release status for RDMA transport
    gil_released = None
    if args.transport == "rdma":
        try:
            import l3_client._cama_rdma as _rdma
            gil_released = getattr(_rdma, "GIL_RELEASED", False)
        except ImportError:
            gil_released = False

    # Multi-NIC endpoint discovery (RDMA only)
    endpoints = None
    if args.transport == "rdma" and not args.no_multinic:
        try:
            from l3_client.rdma_client import RDMAClient as _DiscoverClient
            _dc = _DiscoverClient(args.addr, port=args.port)
            _eps = _dc.rdma_endpoints()
            _dc.close()
            if len(_eps) > 1:
                endpoints = _eps
                print(f"[bench] Discovered {len(endpoints)} RDMA NICs:")
                for ep in endpoints:
                    print(f"[bench]   {ep.get('device', '?'):12s}  {ep['ip']}:{ep['port']}")
            elif len(_eps) == 1:
                print(f"[bench] Single RDMA endpoint — multi-NIC disabled")
            else:
                print(f"[bench] No RDMA endpoints discovered — using --addr")
        except Exception as e:
            print(f"[bench] Multi-NIC discovery failed ({e}) — using --addr")

    # Build key list
    key_list = [f"bench_key_{i:08d}" for i in range(args.keys)]

    # Build Zipf generator
    zipf = ZipfGenerator(args.keys) if args.distribution == "zipfian" else None

    value_mb = args.value_size / (1024 * 1024)

    print(f"[bench] CAMA Python Benchmark")
    print(f"[bench]   server:       {args.addr}:{args.port}")
    print(f"[bench]   transport:    {args.transport.upper()}")
    if gil_released is not None:
        print(f"[bench]   GIL released: {gil_released}")
        if not gil_released:
            print("[bench]   WARNING: C++ extension does NOT release GIL!")
            print("[bench]   Multi-threaded performance will be severely limited.")
            print("[bench]   Rebuild with: pip install -e . --no-build-isolation")
    if args.transport == "rdma":
        if endpoints is not None:
            print(f"[bench]   multi-NIC:    {len(endpoints)} NICs (round-robin)")
        elif args.no_multinic:
            print(f"[bench]   multi-NIC:    disabled (--no-multinic)")
        else:
            print(f"[bench]   multi-NIC:    single NIC")
    print(f"[bench]   engine:       {args.engine}")
    print(f"[bench]   clients:      {args.clients}")
    print(f"[bench]   duration:     {args.duration}s")
    print(f"[bench]   distribution: {args.distribution}")
    print(f"[bench]   value_size:   {args.value_size} bytes ({value_mb:.2f} MB)")
    print(f"[bench]   key_count:    {args.keys}")
    if args.sweep:
        wl_list = args.sweep_workloads if args.sweep_workloads else "mixed,exists,delete-heavy,load-probe"
        print(f"[bench]   mode:         sweep ({wl_list})")
    else:
        print(f"[bench]   workload:     {args.workload}")
    print(f"[bench]   read_ratio:   {args.read_ratio * 100:.1f}%")
    print(f"[bench]   exists_ratio: {args.exists_ratio * 100:.1f}%")
    print(f"[bench]   delete_ratio: {args.delete_ratio * 100:.1f}%")
    print(f"[bench]   ttl_ms:       {args.ttl_ms}")
    if args.workload in ("sglang", "sglang-stress", "batch"):
        print(f"[bench]   batch_size:   {args.batch_size}")
        if args.workload in ("sglang", "sglang-stress"):
            mha_str = "MHA" if args.key_multiplier == 2 else "MLA"
            print(f"[bench]   key_mult:     {args.key_multiplier} ({mha_str})")
            print(f"[bench]   prefix_hit:   {args.prefix_hit_rate * 100:.0f}%")
        if args.workload == "sglang":
            print(f"[bench]   io_workers:   {args.io_workers}")
            if args.compute_delay_ms > 0:
                print(f"[bench]   compute_ms:   {args.compute_delay_ms}")
        mem_mb = args.clients * args.batch_size * args.key_multiplier * args.value_size / (1024 * 1024)
        print(f"[bench]   buffer_mem:   {mem_mb:.0f} MB ({args.clients} ranks)")
    if args.saturate:
        print(f"[bench]   saturate:     max={args.saturate_max}, step={args.saturate_duration}s, threshold={args.saturate_threshold}%")
    if args.preset:
        print(f"[bench]   preset:       {args.preset}")
    if args.debug:
        print(f"[bench]   debug:        True")
    print(flush=True)

    # Sweep mode: run multiple workloads and compare
    if args.sweep:
        _run_sweep(args, key_list, zipf, endpoints)
        if args.post_stats:
            _post_stats(args)
        return

    # Saturation mode
    if args.saturate:
        from bench_worker import run_saturation
        run_saturation(args, key_list, endpoints)
        if args.post_stats:
            _post_stats(args)
        return

    # SGLang-realistic / SGLang-stress / Batch workloads
    if args.workload in ("sglang", "sglang-stress", "batch"):
        from bench_worker import (
            run_sglang_benchmark, run_batch_benchmark,
            print_sglang_results, print_batch_results,
        )
        if args.workload == "sglang":
            worker_results = run_sglang_benchmark(args, key_list, endpoints, stress=False)
            print_sglang_results(args, worker_results, stress=False)
        elif args.workload == "sglang-stress":
            worker_results = run_sglang_benchmark(args, key_list, endpoints, stress=True)
            print_sglang_results(args, worker_results, stress=True)
        else:
            worker_results = run_batch_benchmark(args, key_list, endpoints)
            print_batch_results(args, worker_results)
        if args.post_stats:
            _post_stats(args)
        return

    # Multiprocess engine for standard workloads
    if args.engine == "process" and args.workload != "load-probe":
        from bench_worker import run_multiprocess_benchmark, WorkerResult
        worker_results = run_multiprocess_benchmark(args, key_list, endpoints)
        agg = WorkerResult.aggregate(worker_results)

        elapsed = agg.get("elapsed", 1.0)
        total_ops = agg["total_ops"]
        ops_per_sec = agg["ops_per_sec"]
        total_bw = agg["bandwidth_gbps"]

        print()
        print("=== CAMA Python Benchmark Results ===")
        print(f"Transport:       {args.transport.upper()}")
        print(f"Engine:          process ({len(worker_results)} workers)")
        print(f"Workload:        {args.workload}")
        print(f"Duration:        {elapsed:.2f}s")
        print(f"Total ops:       {total_ops:,}")
        print(f"Throughput:      {ops_per_sec:,.0f} ops/s")
        print(f"GETs:            {agg['total_gets']:,}")
        print(f"SETs:            {agg['total_sets']:,}")
        hit_rate = agg.get("hit_rate", -1)
        if hit_rate >= 0:
            print(f"Hit rate:        {hit_rate:.1f}%")
        if agg["total_exists_ops"] > 0:
            ex_hr = agg["total_exists_hits"] / agg["total_exists_ops"] * 100 if agg["total_exists_ops"] > 0 else 0
            print(f"EXISTSes:        {agg['total_exists_ops']:,}  (hit_rate: {ex_hr:.1f}%)")
        if agg["total_deletes"] > 0:
            print(f"DELETEs:         {agg['total_deletes']:,}")
        print(f"Errors:          {agg['total_errors']:,}")
        read_bw = agg["total_bytes_read"] / elapsed / 1e9 if elapsed > 0 else 0
        write_bw = agg["total_bytes_write"] / elapsed / 1e9 if elapsed > 0 else 0
        print(f"Read bandwidth:  {read_bw:.2f} GB/s")
        print(f"Write bandwidth: {write_bw:.2f} GB/s")
        print(f"Total bandwidth: {total_bw:.2f} GB/s")
        print(f"Value size:      {args.value_size:,} bytes ({value_mb:.2f} MB)")
        print(f"Keys:            {args.keys}")

        # Per-worker latency
        for r in sorted(worker_results, key=lambda x: x.worker_id):
            if r.p99_ms > 0:
                print(f"  worker {r.worker_id}: avg={r.avg_ms:.2f}ms p50={r.p50_ms:.2f}ms "
                      f"p99={r.p99_ms:.2f}ms ops={r.ops:,}")

        # C++ stats
        if args.transport == "rdma":
            cpp_rt = [r.cpp_avg_roundtrip_us for r in worker_results if r.cpp_avg_roundtrip_us > 0]
            if cpp_rt:
                print(f"C++ roundtrip:   avg={sum(cpp_rt)/len(cpp_rt):.1f}us")

        print("=====================================")
        if args.post_stats:
            _post_stats(args)
        return

    # Flush server before starting (unless --no-flush)
    if not args.no_flush:
        print("[bench] Sending FLUSH to server...", flush=True)
        if args.transport == "rdma":
            from l3_client.rdma_client import RDMAClient
            _fc = RDMAClient(args.addr, port=args.port)
        else:
            from l3_client.client import CamaClient
            _fc = CamaClient(args.addr, port=args.port)
        _fc.flush()
        _fc.close()
        print("[bench] FLUSH complete", flush=True)

    # Load-probe mode: skip prepopulate + mixed loop, run two-phase instead
    if args.workload == "load-probe":
        _run_load_probe(args, key_list, endpoints)

        # Post-benchmark server stats (shared with normal path)
        if args.post_stats:
            _post_stats(args)
        return

    # Pre-populate
    _prepopulate(args.transport, args.addr, args.port, key_list, args.value_size,
                 ttl_ms=args.ttl_ms)

    # Launch workers with barrier for synchronized start
    stop_event = threading.Event()
    # +1 for the main thread participating in the barrier
    barrier = threading.Barrier(args.clients + 1, timeout=120)
    results: list[dict] = [{}] * args.clients
    threads: list[threading.Thread] = []

    connect_start = time.monotonic()

    for i in range(args.clients):
        results[i] = {}
        ep = endpoints[i % len(endpoints)] if endpoints else None
        t = threading.Thread(
            target=_worker,
            args=(
                i,
                args.transport,
                args.addr,
                args.port,
                args.value_size,
                key_list,
                args.read_ratio,
                args.distribution,
                zipf,
                stop_event,
                barrier,
                results[i],
                args.debug,
                ep,
                args.exists_ratio,
                args.delete_ratio,
                args.ttl_ms,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all workers to connect and register buffers
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        print("[bench] ERROR: barrier broken — workers failed to connect", file=sys.stderr)
        sys.exit(1)

    connect_elapsed = time.monotonic() - connect_start
    print(f"[bench] All {args.clients} workers connected in {connect_elapsed:.1f}s", flush=True)

    # Start timing AFTER all workers are ready
    start_time = time.monotonic()
    print(f"[bench] Running benchmark for {args.duration}s...", flush=True)

    # Progress ticker
    def _progress():
        tick = 0
        while not stop_event.is_set():
            time.sleep(5)
            tick += 5
            if stop_event.is_set():
                break
            elapsed = time.monotonic() - start_time
            ops_so_far = sum(r.get("ops", 0) for r in results)
            print(f"[bench] {elapsed:8.1f}s: {ops_so_far:10d} ops, "
                  f"{ops_so_far / elapsed:10.0f} ops/s", flush=True)

    progress_thread = threading.Thread(target=_progress, daemon=True)
    progress_thread.start()

    time.sleep(args.duration)
    stop_event.set()

    for t in threads:
        t.join()

    elapsed = time.monotonic() - start_time

    # Aggregate
    total_ops = sum(r.get("ops", 0) for r in results)
    total_gets = sum(r.get("gets", 0) for r in results)
    total_sets = sum(r.get("sets", 0) for r in results)
    total_hits = sum(r.get("hits", 0) for r in results)
    total_exists_ops = sum(r.get("exists_ops", 0) for r in results)
    total_exists_hits = sum(r.get("exists_hits", 0) for r in results)
    total_deletes = sum(r.get("deletes", 0) for r in results)
    total_errors = sum(r.get("errors", 0) for r in results)
    total_bytes_read = sum(r.get("bytes_read", 0) for r in results)
    total_bytes_write = sum(r.get("bytes_write", 0) for r in results)

    ops_per_sec = total_ops / elapsed if elapsed > 0 else 0.0
    hit_rate = (total_hits / total_gets * 100) if total_gets > 0 else 0.0
    read_bw = total_bytes_read / elapsed / 1e9 if elapsed > 0 else 0.0
    write_bw = total_bytes_write / elapsed / 1e9 if elapsed > 0 else 0.0
    total_bw = read_bw + write_bw

    print()
    print("=== CAMA Python Benchmark Results ===")
    print(f"Transport:       {args.transport.upper()}")
    print(f"Engine:          thread ({args.clients} workers)")
    if gil_released is not None:
        print(f"GIL released:    {gil_released}")
    print(f"Workload:        {args.workload}")
    print(f"Duration:        {elapsed:.2f}s")
    print(f"Total ops:       {total_ops:,}")
    print(f"Throughput:      {ops_per_sec:,.0f} ops/s")
    print(f"GETs:            {total_gets:,}")
    print(f"SETs:            {total_sets:,}")
    print(f"Hits:            {total_hits:,}  ({hit_rate:.1f}%)")
    if total_exists_ops > 0:
        exists_hit_rate = total_exists_hits / total_exists_ops * 100
        print(f"EXISTSes:        {total_exists_ops:,}  (hit_rate: {exists_hit_rate:.1f}%)")
    if total_deletes > 0:
        print(f"DELETEs:         {total_deletes:,}")
    print(f"Errors:          {total_errors:,}")
    print(f"Read bandwidth:  {read_bw:.2f} GB/s")
    print(f"Write bandwidth: {write_bw:.2f} GB/s")
    print(f"Total bandwidth: {total_bw:.2f} GB/s")
    print(f"Clients:         {args.clients}")
    print(f"Value size:      {args.value_size:,} bytes ({value_mb:.2f} MB)")
    print(f"Keys:            {args.keys}")
    print(f"Distribution:    {args.distribution}")

    # Per-NIC bandwidth breakdown
    if endpoints is not None:
        from collections import defaultdict
        nic_stats = defaultdict(lambda: {"bytes_read": 0, "bytes_write": 0, "workers": 0})
        for r in results:
            dev = r.get("nic_device")
            if dev:
                nic_stats[dev]["bytes_read"] += r.get("bytes_read", 0)
                nic_stats[dev]["bytes_write"] += r.get("bytes_write", 0)
                nic_stats[dev]["workers"] += 1
                nic_stats[dev]["ip"] = r.get("nic_ip", "")
                nic_stats[dev]["port"] = r.get("nic_port", 0)
        if nic_stats:
            print()
            print("--- Per-NIC Bandwidth ---")
            print(f"{'Device':<14s} {'IP':>15s}  {'Read GB/s':>10s}  {'Write GB/s':>10s}  {'Total GB/s':>10s}  {'Workers':>7s}")
            for dev in sorted(nic_stats):
                ns = nic_stats[dev]
                rd = ns["bytes_read"] / elapsed / 1e9 if elapsed > 0 else 0
                wr = ns["bytes_write"] / elapsed / 1e9 if elapsed > 0 else 0
                print(f"{dev:<14s} {ns['ip']:>15s}  {rd:>10.2f}  {wr:>10.2f}  {rd + wr:>10.2f}  {ns['workers']:>7d}")
            print()

    # Parallelism metric
    if args.clients > 1:
        sum_worker_time = sum(r.get("worker_elapsed", 0) for r in results)
        parallelism = sum_worker_time / elapsed if elapsed > 0 else 0
        print(f"Parallelism:     {parallelism:.1f}x (of {args.clients} threads)")

    # Latency stats
    all_get_lats = []
    all_set_lats = []
    all_exists_lats = []
    all_del_lats = []
    for r in results:
        all_get_lats.extend(r.get("get_latencies", []))
        all_set_lats.extend(r.get("set_latencies", []))
        all_exists_lats.extend(r.get("exists_latencies", []))
        all_del_lats.extend(r.get("del_latencies", []))

    if all_get_lats:
        all_get_lats.sort()
        avg_get = sum(all_get_lats) / len(all_get_lats) * 1000
        p50_get = _percentile(all_get_lats, 50) * 1000
        p99_get = _percentile(all_get_lats, 99) * 1000
        p999_get = _percentile(all_get_lats, 99.9) * 1000
        print(f"GET latency:     avg={avg_get:.2f}ms  p50={p50_get:.2f}ms  "
              f"p99={p99_get:.2f}ms  p99.9={p999_get:.2f}ms")

    if all_set_lats:
        all_set_lats.sort()
        avg_set = sum(all_set_lats) / len(all_set_lats) * 1000
        p50_set = _percentile(all_set_lats, 50) * 1000
        p99_set = _percentile(all_set_lats, 99) * 1000
        p999_set = _percentile(all_set_lats, 99.9) * 1000
        print(f"SET latency:     avg={avg_set:.2f}ms  p50={p50_set:.2f}ms  "
              f"p99={p99_set:.2f}ms  p99.9={p999_set:.2f}ms")

    if all_exists_lats:
        all_exists_lats.sort()
        avg_ex = sum(all_exists_lats) / len(all_exists_lats) * 1000
        p50_ex = _percentile(all_exists_lats, 50) * 1000
        p99_ex = _percentile(all_exists_lats, 99) * 1000
        p999_ex = _percentile(all_exists_lats, 99.9) * 1000
        print(f"EXISTS latency:  avg={avg_ex:.2f}ms  p50={p50_ex:.2f}ms  "
              f"p99={p99_ex:.2f}ms  p99.9={p999_ex:.2f}ms")

    if all_del_lats:
        all_del_lats.sort()
        avg_dl = sum(all_del_lats) / len(all_del_lats) * 1000
        p50_dl = _percentile(all_del_lats, 50) * 1000
        p99_dl = _percentile(all_del_lats, 99) * 1000
        p999_dl = _percentile(all_del_lats, 99.9) * 1000
        print(f"DELETE latency:  avg={avg_dl:.2f}ms  p50={p50_dl:.2f}ms  "
              f"p99={p99_dl:.2f}ms  p99.9={p999_dl:.2f}ms")

    # C++ transport stats (RDMA only)
    if args.transport == "rdma":
        cpp_roundtrip_us = []
        cpp_rdma_read_us = []
        for r in results:
            cs = r.get("cpp_stats")
            if cs:
                if cs.get("roundtrip_count", 0) > 0:
                    cpp_roundtrip_us.append(cs["avg_roundtrip_us"])
                if cs.get("rdma_read_count", 0) > 0:
                    cpp_rdma_read_us.append(cs["avg_rdma_read_us"])
        if cpp_roundtrip_us:
            avg_rt = sum(cpp_roundtrip_us) / len(cpp_roundtrip_us)
            print(f"C++ roundtrip:   avg={avg_rt:.1f}us")
        if cpp_rdma_read_us:
            avg_rd = sum(cpp_rdma_read_us) / len(cpp_rdma_read_us)
            print(f"C++ RDMA read:   avg={avg_rd:.1f}us")

    print("=====================================")

    # Post-benchmark server stats
    if args.post_stats:
        _post_stats(args)


if __name__ == "__main__":
    main()
