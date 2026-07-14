# C2 (#4625) implementation spec — per-NUMA-node weight replication + barrier-free decode schedule

> The remaining lever to take Qwen3.6-27B Q4_K CPU decode past the shipped worker-cap
> plateau (~1.4 tok/s) toward the ~5-7 tok/s node-local ceiling. Witnessed context:
> `experiments/qwen36/witness-epyc7742-dual-20260714/` and the NUMA-additivity witness.
> Reaching >=10 tok/s needs this PLUS one more lever (C5 #4628 sub-4-bit MLP, or a
> prefetch/non-temporal-store per-node kernel) — replication alone lands ~5-7, not ~14,
> because the decode kernel is partly compute-bound per node (NUMA-additivity witness).

# NUMA-aware CPU decode parallelism — design

Target: Qwen3.6-27B q4_k_m single-stream CPU decode, ~0.87 → ~10 tok/s on a
dual-socket EPYC 7742 (256 threads, **8 NUMA nodes** = NPS4, ~2 DDR4-3200
channels/node, ~40 GB/s achievable per node, ~320 GB/s aggregate). Decode
streams ~15 GB/token (Q4_K = 0.5625 B/param × 27e9). 10 tok/s ⇒ **150 GB/s
sustained** — impossible from one node (~40 GB/s ⇒ ~2.6 tok/s ceiling), so the
lever is aggregate multi-node bandwidth WITHOUT the parFor barrier collapse.

All file:line references are `C:\work\fak\internal\model\...` unless noted.

---

## 1. Current state — how decode parallelism works today

### Call chain (Qwen3.6-27B q4_k_m, Linux/amd64, no Metal)

The q4_k_m checkpoint loads its matmul majority as **resident raw Q4_K**
tensors, so decode runs the `sessionQ4KKernel` path:

- `Session.tokenHiddenQ` selects `mat = sessionQ4KKernel{s}` — `quant_forward.go:195-199`.
- Per layer it runs `blockStep` → `mulGroup` / `mat.mul` for q,k,v,o,gate,up,down.
- `sessionQ4KKernel.mul` — `quant_q4k.go:433-452` — routes a resident Q4_K weight to
  `s.q4kMatRowsDispatch(name, qt, xf)`.
- Pure-Go build: `q4kMatRowsDispatch` → `q4kMatRows` — `metal_q4k_off.go:39-41`.
- `q4kMatRows` → `q4kMatRowsInto` — `quant_q4k.go:119-144`.
- `q4kMatRowsInto` dispatches the row loop via **`parForRange(qt.out, qt.out*qt.in, …)`**
  — `quant_q4k.go:135` (int8 path) and `:138` (f32 path).
- `parForRange` — `parallel.go:295-301` — runs `parFor(n, **numWorkers**, body)`.
- The range kernel reads `qt.raw[o*rowBytes:]` — scalar `q4kMatRowsRange`
  `quant_q4k.go:150-170`, or the AVX2 f32 twin `q4kMatRowsRangeArch`
  `quant_amd64_q4k_f32.go:25-50` (`q4kRowDotF32AVX2`). the CPU server (EPYC-7742) is AVX2-only (no AVX-512).

`mulGroup` — `quant_q4k.go:460-473` — calls `s.q4kGroupDispatch(...)` which returns
`nil` in the pure-Go build (`metal_q4k_off.go:50-52`), so q/k/v are **three separate
`q4kMatRows` dispatches**, each its own `parFor`. Across a full token that is roughly
`7 × NumLayers + 1` (~450+) `parFor` dispatches, each synchronizing every worker.

### CRITICAL FINDING — the 16-cap does NOT bite the q4_k_m path

`numWorkers` defaults to `GOMAXPROCS(0)` = **256** on this box (no `FAK_WORKERS`/
`FAK_BUDGET` set) — `parallel.go:33-36`. The `q8DecodeWorkersFor` cap
(`budget.go:105-145`; amd64 branch `:134-143` returns `workers/8` clamped to 16 when
`workers≥64`) is applied **only** by the **Q8_0** decode entry points:
`qMatRowsInto` (`quant.go:235`) and `qMatRowsIntoMany` (`quant_gemv_many.go:43`).

The **resident Q4_K decode path never calls `q8DecodeWorkers()`** — it goes straight
to `parForRange` → `parFor(qt.out, numWorkers=256, …)`. So Qwen3.6-27B q4_k_m decode
today runs the GEMV at **all 256 workers, uncapped, across all 8 NUMA nodes** — the
exact all-core regime the `budget.go:122-133` comment warns about, but on the path the
comment's cap does not cover. This asymmetry (Q8 capped to 16, Q4_K uncapped to 256)
is itself worth fixing and is central to why q4_k_m sits at 0.87 tok/s.

### Why it collapses across NUMA — two independent pathologies

**(A) Weight placement (the bandwidth ceiling).** Each `q4kTensor.raw` is allocated
once at load by the loader goroutine (`quantizeQ4KFromRaw` → `pageAlignResidentBytes`,
`quant_q4k.go:481-516`; the loader hands payloads in at `ggufload/quant_q4k_loader.go:388`).
Under Linux default **first-touch**, every weight page physically lands on the single
node that ran the loader (node 0). All 256 workers — including the 224 on nodes 1-7 —
then stream all 15 GB from node 0's ~2 memory channels. Aggregate decode bandwidth is
therefore pinned near **one node (~40 GB/s)**, and 7/8 of workers additionally pay
cross-socket read latency. This alone floors decode at ~2.6 tok/s before any sync cost.

**(B) parFor barrier collapse (the sync cost).** `parFor` — `parallel.go:239-286` —
uses ONE shared work cursor `parNextChunk atomic.Int64` (`parallel.go:107`) that every
worker `fetch-add`s in `parGrab` (`parallel.go:137`: `parNextChunk.Add(1)`), plus ONE
shared `parActive atomic.Int64` (`:108`) that each worker decrements (`:159`) and the
caller **busy-waits** on (`:282-283`: `for parActive.Load() != 0 {}`). On a dual-socket
box these two cache lines ping-pong across the inter-socket link on every chunk grab and
every completion — a cross-node coherence transaction (~100-300 ns) per operation. For
the tiny decode matmuls (k/v are ~256–512 rows; chunkSize can be 1, `:253`) the fetch-add
storm and the cross-socket join dominate the actual dot-product work. The dispatch also
publishes 255 per-worker `seq` release-stores in a loop (`:270-279`) that must propagate
to remote sockets before workers start. Multiply by ~450 dispatches/token: sync, not
compute, sets the token time. (The spin-then-park comment `parallel.go:53-74` was tuned
on a 12-core single-socket M3 Pro; it never modeled 8 nodes.)

Net: (A) caps bandwidth at one node; (B) burns most of the remaining budget on
cross-socket synchronization. Naively raising the Q8 cap only worsens (B) without
touching (A), which is why the `budget.go` comment observes a regression.

---

## 2. The weight layout — and what replication takes

`q4kTensor` — `quant_q4k.go:61-64`:
```go
type q4kTensor struct { out, in, nblk int; raw []byte }
```
- **Per-tensor**, not one slab: each projection/layer is its own `q4kTensor`, stored in
  `m.q4kw map[string]*q4kTensor` (`quant_q4k.go:534-540`, built by
  `AddResidentQ4K` `:606-616`).
- `raw` is the **verbatim GGUF byte stream**, row-major: row `o` at
  `raw[o*rowBytes : …]`, `rowBytes = nblk*144`; each 256-weight super-block is 144 B
  (`q4kBlockBytes`, `:52`). No f32 co-residency in the lean loader.
- `raw` is either a slice of the mmap'd GGUF (`MAP_SHARED`, `mmap_unix.go:34`) or a
  page-aligned heap copy (`pageAlignResidentBytes`, `:497-516`).
- Total resident Q4_K ≈ **15 GiB** for 27B.

**To replicate per node** we need, for each `q4kTensor`, N byte-identical copies of
`raw` (N = online NUMA nodes), each physically bound to a distinct node. Because the
copies are byte-for-byte identical, **any replica produces bit-identical output** — the
only thing that changes is the physical address the kernel reads. Cost: N × 15 GiB =
**8 × 15 = 120 GiB**, well inside 945 GiB free. Store copies **off the Go heap**
(anonymous mmap) so 120 GiB is not GC-scanned.

---

## 3. Proposed design (Go-level, implementation-ready)

Two orthogonal changes: **replicate weights per node** (fixes A) and **replace the
shared-cursor decode dispatch with a static per-node schedule** (fixes B). New Linux-only
syscall glue lives in build-tagged files (`//go:build linux && amd64`) with no-op stubs
elsewhere — the exact pattern `mmap_unix.go` / `mmap_other.go` already use, stdlib-only
(no `x/sys`).

### 3a. Per-NUMA weight replication

New field + builder:
```go
// q4kTensor gains (nil ⇒ fall back to single-copy raw):
rawByNode [][]byte   // one byte-identical replica per NUMA node, node-bound
```
- `numaTopology()` (new, `numa_linux.go`): read online nodes from
  `/sys/devices/system/node/online` and per-node CPU lists from
  `/sys/devices/system/node/nodeN/cpulist`. Reuse `compute.parseNodeList`
  (`internal/compute/mempolicy.go:92`) — it already parses `"0-3,8"` masks.
- `buildQ4KReplicas(qt, topo)` (new): for each node `n`,
  1. `slab, _ := syscall.Mmap(-1, 0, len(raw), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON)`
     (off-heap; long-lived, immutable — ideal),
  2. `mbindSlab(slab, n)` — set `MPOL_BIND` to `{n}` via the raw `SYS_MBIND` syscall
     (`syscall.Syscall6`), so pages fault onto node `n`,
  3. copy `raw` into `slab` from a goroutine pinned to a CPU on node `n`
     (`pinThreadToNode`, below) as belt-and-suspenders first-touch,
  4. `mprotect` back to `PROT_READ`; `qt.rawByNode[n] = slab`.
- Call site: after `AddResidentQ4K`/`QuantizeQ4K` completes, walk `m.q4kw`.
- **Gate replication** on: N≥2 online nodes AND policy **unconstrained**
  (`compute.ReadHostMemStatus().Constrained == false`, `mempolicy.go:56-72`) AND
  free ≥ N×totalWeightBytes. Never replicate under `--membind=0` — it would OOM node 0
  (the very failure `mempolicy.go:10-18` documents). On any failure, leave `rawByNode`
  nil ⇒ existing path, correctness unchanged.

### 3b. Static, barrier-free per-node decode schedule

New pool (new file `numa_decode.go` + `numa_linux.go`), used **only** by the decode
GEMV; `parFor`/prefill/f32-exact paths are untouched (so `TestParallelMatchesSerial`
and the exact decode==prefill rungs, `parallel.go:53-67`, are unaffected):

```go
type numaWorker struct { seq, done atomic.Uint64; lo, hi int; _ [pad]byte } // 1/line
type nodeWorkers struct { node int; ws []numaWorker }                       // W_n workers
type numaPool struct { nodes []nodeWorkers }
```
- **One-time build**: for each node `n`, launch `W_n` goroutines; each calls
  `runtime.LockOSThread()` (already used at `guard/landlock_linux.go:190`,
  `compute/cuda_graph_test.go:51`) then `pinThreadToNode(n)` = `sched_setaffinity(2)`
  (raw `SYS_SCHED_SETAFFINITY`) to node `n`'s CPU mask. Now the thread only runs on node
  `n`, so its reads of `rawByNode[n]` and its node-local `x` scratch are LOCAL.
- **Schedule** `numaSchedule(out)` (cached; only a few distinct `out` dims exist):
  partition `[0,out)` into N contiguous **node-spans** (`out/N` rows each), then split
  each node-span into `W_n` fixed per-worker sub-ranges `[lo_w,hi_w)`. **No shared
  cursor** — each worker's range is pure arithmetic of (node, workerIdx, out).
- **Dispatch** `q4kMatRowsIntoNUMA(qt, x, y)`:
  1. copy `x` (hidden 5120 f32 = 20 KiB, negligible vs 15 GB) into each node's local
     scratch once,
  2. bump every worker's node-local `seq` (release),
  3. each worker runs `q4kMatRowsRangeNode(qt.rawByNode[node], x_local, y, lo, hi)` — the
     **existing kernel** (`q4kMatRowsRangeArch` refactored to take an explicit `raw
     []byte` instead of indexing `qt.raw`), then sets its `done` flag,
  4. caller joins by reading the **N per-node done-counters** (8 lines), not one global
     `parActive`. The only cross-socket traffic per GEMV is the single dispatch publish +
     the N-line join — the `parNextChunk` fetch-add storm is **gone**.
- Keep spin-then-park wake (`parallel.go:150-175`) per node so wakes stay node-local.
- `q4kMatRowsInto` branches: `if qt.rawByNode != nil { q4kMatRowsIntoNUMA(...) ; return }`
  else the current `parForRange` (`quant_q4k.go:135/138`).

Variant (lower-risk, keeps intra-node work-stealing for SMT/heterogeneous load): instead
of fully-static sub-ranges, give **each node its own cursor** (`parNextChunk` becomes
`[]atomic.Int64` indexed by node, each on node-local memory) and its own row-span +
replica. Removes cross-node cursor contention while preserving today's dynamic balance
within a node. **Recommended default**: per-node cursor; the fully-static form is the
fallback if even the node-local cursor line shows contention.

### 3c. Go's pinning limits (be honest)

Go goroutines are **not** pinned to OS threads and the scheduler migrates them, so a bare
goroutine gives NO NUMA locality guarantee. The guarantee requires
`runtime.LockOSThread()` **plus** `sched_setaffinity` on that thread — both stdlib. Costs:
locked threads don't return to the pool (fine — the pool is fixed, `Σ W_n ≤ 256`,
persistent); if `sched_setaffinity` fails (restrictive cpuset/cgroup), fall back to
interleave-only placement — still correct. Ensure `GOMAXPROCS ≥ Σ W_n` (default 256 fine).

### 3d. Worker budget

Unify decode budgeting: route BOTH q4k and q8 decode through the numaPool and pick
`W_n` per node from a knee measured on the box (§4). Reasonable start:
`W_n ≈ min(coresPerNode, 8)` so ~4–6 streams/node saturate ~2 channels without
re-introducing intra-node barrier cost — total ~32–48 streams across 8 nodes, each
reading local memory. Keep `FAK_WORKERS`/`FAK_BUDGET` as explicit overrides
(`budget.go:52-72`).

---

## 4. Cheapest first experiment (tonight, ZERO code)

Run the **existing** binary under three `numactl` placements and compare decode tok/s.
This decomposes the loss into {placement, cross-node-barrier, per-node-bandwidth} and
tells us whether replication is worth building — before writing a line.

| # | launch | what it isolates | predicted tok/s |
|---|--------|------------------|-----------------|
| 1 | `./fak …` (default) | today: weights on node 0, 256 all-node workers | ~0.87 (observed) |
| 2 | `numactl --interleave=all ./fak …` | weights striped over all 16 channels; 7/8 accesses remote but aggregate BW ↑ | **~3–8** (big jump) |
| 3 | `numactl --cpunodebind=0 --membind=0 FAK_WORKERS=32 ./fak …` | one node, all-local, no cross-socket | ~2–2.6 |

Reasoning / how to read it:
- **(2) ≫ (1)** ⇒ the bottleneck is **placement** (node-0 confinement): aggregating
  bandwidth across nodes is the win, which is exactly what replication buys (and
  replication beats interleave by making 8/8 accesses local instead of 1/8). **Green-light
  replication.** Predicted replication ceiling ≈ (3)×N bounded by ~320 GB/s ⇒ ~10–21 tok/s.
- **(3) ≫ (1)** but **(2) ≈ (1)** ⇒ the **cross-node barrier** dominates, not placement;
  prioritize the static per-node schedule (§3b) first, replication second.
- **(2) ≈ (3)** ⇒ single-node bandwidth is the wall; only more local nodes (replication)
  raise it.

Add a no-code **worker sweep** to find the parFor knee on this box:
`FAK_WORKERS ∈ {16,32,48,64,128,256}` under `--interleave=all`. The point where more
workers stop helping (or regress) is the barrier-collapse knee and directly sets `W_n`
per node for §3b. (Note: for the q4_k_m path this also *demonstrates* the uncapped-256
finding from §1 — `FAK_WORKERS=32` should already beat default 256 if the barrier is
real, because it forces the Q4_K path off all-core.)

Capture `DecodeRoofline` (`decode_roofline.go:59-89`) alongside each run for the honest
achieved-GB/s vs ceiling; note the ceiling helper `measureAggregateStreamGBps`
(`decode_roofline.go:96`) currently models an unreplicated aggregate, so >100% util after
replication is a witness artifact to fix (§5), not a numerics error.

---

## 5. Risk / faithfulness

- **Bit-identity (must hold).** Replicas are byte-for-byte copies of the same Q4_K bytes;
  the dot kernels (`q4kMatRowsRange` `:150-170`, `q4kRowDotF32AVX2`
  `quant_amd64_q4k_f32.go`) are unchanged and read the same super-blocks in the same
  order. The static partition changes only *which goroutine* computes a row and *from
  which physical copy* — never the per-row reduction, the same invariant the parallel lane
  already relies on (`parallel.go:53-67`). `TestQ4KMatRowsMatchesF32` and the
  greedy-continuation gate must stay green.
- **int8 path stays approximate & gated.** The int8 SDOT decode
  (`q4kMatRowsRangeInt8` `quant_q4k_int8.go:132`) is off by default
  (`q4kSDOTEnabled` / `FAK_KQ_INT8`, `quant_amd64_q4k.go:54-71`). Keep default = f32 exact;
  if int8 is used, its existing quality witness carries it — replication doesn't change
  that argument.
- **Memory safety.** 120 GiB replicas must fit under free AND under any confinement. Gate
  on `compute.ReadHostMemStatus()` (`mempolicy.go`): replicate only when
  `!Constrained` and free ≥ N×bytes. This reuses the existing OOM-under-membind guard
  rather than re-deriving it. Store replicas as anon mmap (off-heap) and `munmap` on model
  teardown; keep slices referenced so GC can't reclaim backing.
- **Portability.** `mbind` / `sched_setaffinity` are `//go:build linux && amd64` raw
  syscalls with no-op stubs elsewhere (mirror `mmap_unix.go`/`mmap_other.go`); non-Linux
  ⇒ `rawByNode == nil` ⇒ current `parForRange` path, unchanged.
- **Affinity failure ⇒ graceful degrade.** If `sched_setaffinity`/`mbind` fails
  (cgroup/cpuset), fall back to interleave placement + existing dispatch. Never hard-fail
  decode on a NUMA error.
- **Prefill untouched.** Prefill (`q4kGemm` `:187`, batched, compute-bound, reuses each
  weight across P tokens) does not need replication; leave it on `parFor`/`numWorkers`. It
  may read `rawByNode[0]` transparently.
- **Witness honesty.** Update `measureAggregateStreamGBps` (`decode_roofline.go:96`) to
  model N node-local streams so `BWUtilPct` stays ≤100% after replication — a
  reporting-only change, no effect on numerics.
- **Don't regress small boxes.** Everything is gated on N≥2 nodes; single-socket/desktop
  amd64 keeps today's exact behavior (the `budget.go:132-133` "only ≥64 capped" spirit).
