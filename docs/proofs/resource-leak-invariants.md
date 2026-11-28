---
title: "fak proof: resource-leak invariants (L1–L3)"
description: "Inductive state-machine proofs for global registry quiescence, bounded map/secondary-index conservation, and allocator/handle conservation."
---

# A · resource-leak-invariants

This document formalizes three inductive state-machine proof obligations for resource-leak freedom across `ctxmmu`, `gateway`, `compute`, `engine`, and `ggufload`. Each obligation is in **Regime A** (algebraic / structural conservation, bijection, boundedness, and idempotence) and is bound to deterministic `go test` witnesses.

| Obligation | Scope | Core Invariant | Primary Witness |
|---|---|---|---|
| **L1** | `ctxmmu.MMU`, `gateway.Server`, `compute.WatchdogPrefillScheduler` | Registry bijection & idempotent `Close()` quiescence | `TestProofLeakL1_CtxMMURegistryStateMachine`, `TestServerCloseReleasesGlobalMapsAndEvictor`, `TestWatchdogPrefillSchedulerClose` |
| **L2** | `gateway` A2A stores, `engine.ContinuousBatcher` | `\|primary\| <= Cap`, `\|secondary\| <= Cap`, no orphan growth | `TestA2APushConfigsBoundedAcrossEvictions`, `TestContinuousBatcherCompletedMapBounded*` |
| **L3** | `compute.DirectSlabAllocator` / CUDA idle pool, `ggufload` | `liveBytes + freeBytes == capacity`, zero leaked blocks/FDs on success & error | `TestVRAMSoakDirectSlabAllocator`, `TestVRAMSoakCUDAIdlePool`, `TestGGUFLoadNoFDLeak` |

---

## L1 — Global Registry Bijection & Idempotent Quiescence

THEOREM  For each managed resource type `T ∈ {ctxmmu.MMU, gateway.Server, compute.WatchdogPrefillScheduler}` and its package-global registries `R` (and per-instance background worker set `W`):
1. **Injective registration:** Every constructor or arm transition on an instance `x` adds at most one entry keyed by instance identity `x` in each registry `r ∈ R`.
2. **Total, idempotent quiescence:** `x.Close()` is total (including on `nil` receivers), idempotent (`Close(); Close()` is a no-op on `R` and `W`), stops every active background worker `w ∈ W(x)`, and removes `x` from every `r ∈ R`.
3. **Cardinality restoration:** Across any finite trace starting from baseline registry cardinality `|R_before|` in which every constructed/armed instance is closed, `|R_after| == |R_before|` and `|W_after| == |W_before|`.

REGIME  A — algebraic / structural (bijection between live instances and global registry entries; idempotence of `Close()`).

PROOF  By structural induction over the transition system `(Init, Arm, Close)` for each of the three types:
1. **`ctxmmu.MMU`** (`activeMMUs`, `mmuCOWPageTables`, `mmuForkManagers`, `mmuKVPools`):
   - `registerActiveMMU` (`internal/ctxmmu/mmu.go:119-123`) appends a newly constructed `*MMU` once under `activeMMUsMu`; lazy per-MMU tables (`COWPageTable()`, `ForkManager()`, `KVPool()`) key at most one entry by `*MMU`.
   - `(*MMU).Close()` (`internal/ctxmmu/mmu.go:149-175`) returns immediately on `m == nil`, removes `m` from `activeMMUs` under `activeMMUsMu`, and executes `delete(..., m)` on `mmuCOWPageTables`, `mmuForkManagers`, and `mmuKVPools` under their respective mutexes. Deleting an absent key or scanning a slice without `m` is a no-op, so `Close()` is total and idempotent.
2. **`gateway.Server`** (`reuseSeams`, `serverControlManagers`, `sessionClientRuntimes`, `kvEvictor`):
   - `reuseSeams` (`internal/gateway/toolproc_reuse.go:125,140`), `serverControlManagers` (`internal/gateway/http_control.go:13,78`), and `sessionClientRuntimes` (`internal/gateway/session_client.go:125,133`) are `sync.Map` tables keyed by `*Server`, so `Store`/`LoadOrStore` add at most one entry per server identity. `SetIdleGatedEvictor` (`internal/gateway/kv_tier.go:449-451`) attaches at most one `*IdleGatedEvictor` to `s.kvEvictor`.
   - `(*Server).Close()` (`internal/gateway/coherence.go:170-204`) guards `s == nil`, atomically detaches `s.kvEvictor` under `s.kvEvictorMu` and calls `ev.Stop()` if non-nil, calls `s.DisarmToolprocReuse()` (which calls `reuseSeams.Delete(s)` at `internal/gateway/toolproc_reuse.go:155`), and deletes `s` from `serverControlManagers` and `sessionClientRuntimes` (`coherence.go:202-203`). Re-running `Close()` sees `s.kvEvictor == nil` and absent `sync.Map` keys, preserving quiescence.
3. **`compute.WatchdogPrefillScheduler`** (`schedulerFeedbackControllers`):
   - `EnablePhysicalFeedback` (`internal/compute/watchdog_prefill_scheduler.go:1282-1288`) either deletes `s` (when `c == nil`) or stores a single entry keyed by `*WatchdogPrefillScheduler` in `schedulerFeedbackControllers` (`sync.Map`).
   - `(*WatchdogPrefillScheduler).Close()` (`watchdog_prefill_scheduler.go:1292-1298`) guards `s == nil`, calls `schedulerFeedbackControllers.Delete(s)`, and returns `nil`, making `Close()` total and idempotent.

WITNESS
```bash
go test -count=1 ./internal/ctxmmu -run 'TestProofLeakL1_CtxMMURegistryStateMachine|TestMMUCloseReleasesGlobalRegistries'
go test -count=1 ./internal/gateway -run 'TestServerCloseReleasesGlobalMapsAndEvictor'
go test -count=1 ./internal/compute -run 'TestWatchdogPrefillSchedulerClose'
```
- `TestProofLeakL1_CtxMMURegistryStateMachine` (`internal/ctxmmu/leak_proofs_witness_test.go:9-135`): fixed-seed (`2026112801`) 1000-step state-machine fuzzer over `New`, `TouchCOW`, `TouchFork`, `TouchKVPool`, `Close`, and duplicate/nil `Close` asserting exact cardinality equality against a ghost model at every step.
- `TestMMUCloseReleasesGlobalRegistries` (`internal/ctxmmu/close_leak_test.go:8-68`): verifies nil-safety, 64-instance cleanup, and idempotent second `Close()`.
- `TestServerCloseReleasesGlobalMapsAndEvictor` (`internal/gateway/leak_test.go:152-182`): arms all three `sync.Map` registries plus a running `kvEvictor`, calls `srv.Close()`, asserts all three maps drop `srv`, `ev.IsRunning() == false`, and `leakcheck.Stable` goroutine count remains bounded.
- `TestWatchdogPrefillSchedulerClose` (`internal/compute/watchdog_prefill_scheduler_test.go:789-810`): verifies `EnablePhysicalFeedback` registration and `Close()` removal from `schedulerFeedbackControllers`.

VERDICT  **PROVEN** (2026-11-28, `go test -count=1` green on all three packages: `internal/ctxmmu` 0.106s, `internal/gateway` 2.591s, `internal/compute` 1.873s).

DOS  bound at ship.

---

## L2 — Bounded Map & Secondary-Index Conservation

THEOREM  For any finite trace of `Insert`, `Admit`, `Complete`, `Cancel`, and `Evict` transitions:
1. **`gateway` A2A stores** (`primary = a2aTaskStore.tasks`, `secondary = a2aPushStore.configs`, `Cap = a2aMaxTasks = 4096`): `|primary| <= Cap`, `|secondary| <= Cap`, and any eviction of `victim` from `primary` prunes `victim` from `secondary` (no orphan webhook config growth).
2. **`engine.ContinuousBatcher`** (`primary = completedMap`, `secondary = completedOrder`, `Cap = completedLimit()`): `|primary| <= Cap`, and `|secondary|` remains bounded (`<= 2*Cap` before compaction, `== |primary|` after compaction), with stale/evicted entries pruned so neither map nor order queue grows unboundedly.

REGIME  A — algebraic / structural (bounded capacity invariant and primary-to-secondary pruning).

PROOF
1. **`gateway` A2A stores:**
   - `insertLocked` (`internal/gateway/a2a.go:40-48`) checks `len(s.tasks) >= a2aMaxTasks` (`a2aMaxTasks = 4096`, `a2a.go:25`) before inserting `taskID`. When at capacity, it selects `victim := s.evictionVictimLocked()` (`a2a.go:53-68`), deletes `victim` from `s.tasks`, and immediately calls `deleteA2APushConfigIfInitialized(victim)` (`a2a.go:43-44`; defined at `internal/gateway/a2a_pushnotify.go:180-190`), which deletes `victim` from `ps.configs` if the push store is initialized. Thus `|s.tasks|` never exceeds `a2aMaxTasks` and primary eviction always prunes the secondary index.
   - Independently, `(*a2aPushStore).setConfig` (`internal/gateway/a2a_pushnotify.go:159-176`) checks `!exists && len(ps.configs) >= a2aMaxTasks` and evicts the oldest registration before insertion, so `|ps.configs| <= a2aMaxTasks` holds unconditionally even if configs are registered out-of-band.
2. **`engine.ContinuousBatcher`:**
   - `recordCompletedSlotLocked` (`internal/engine/continuous_batcher.go:407-436`) inserts `(sessionID, slot)` into `cb.completedMap` and appends `{sessionID, submissionSeq}` to `cb.completedOrder`. While `len(cb.completedMap) > limit && cb.completedHead < len(cb.completedOrder)`, it pops `oldest` from `cb.completedOrder[cb.completedHead]`, zeroes the popped slot, advances `cb.completedHead`, and deletes `oldest.sessionID` from `cb.completedMap` when its `SubmissionSeq` matches (`continuous_batcher.go:421-428`).
   - When `cb.completedHead == len(cb.completedOrder)`, the slice is reset to `[:0]`; when `cb.completedHead >= limit || len(cb.completedOrder) > 2*limit`, `compactCompletedOrderLocked()` (`continuous_batcher.go:438-450`) filters `cb.completedOrder[cb.completedHead:]` to only entries still live in `cb.completedMap` with matching `SubmissionSeq`, zeroes the tail, and resets `completedHead = 0`. Re-admission (`internal/engine/continuous_batcher_admission.go:88`) and ack (`internal/engine/continuous_batcher_wire.go:463-464`) delete from `cb.completedMap`, and compaction drops those stale entries from `cb.completedOrder`.

WITNESS
```bash
go test -count=1 ./internal/gateway -run 'TestA2APushConfigsBoundedAcrossEvictions'
go test -count=1 ./internal/engine -run 'TestContinuousBatcherCompletedMapBounded'
```
- `TestA2APushConfigsBoundedAcrossEvictions` (`internal/gateway/leak_test.go:91-150`): inserts `a2aMaxTasks + 64` tasks and matching push configs, asserting `len(store.tasks) <= a2aMaxTasks`, `len(ps.configs) <= a2aMaxTasks`, and the evicted `task-00000` is absent from `ps.configs`.
- `TestContinuousBatcherCompletedMapBounded` (`internal/engine/continuous_batcher_leak_test.go:10-64`): submits and steps `defaultMaxCompletedSlots + 104` requests, asserting `len(completedMap) == defaultMaxCompletedSlots` and oldest entries are evicted.
- `TestContinuousBatcherCompletedMapBoundedOnCancelAndReAdmit` (`internal/engine/continuous_batcher_leak_test.go:67-128`): exercises interleaved `Submit`, `Cancel`, and `Step` with `maxCompletedSlots = 4`, asserting `len(completedMap) == 4` and `len(completedOrder) <= 8`.

VERDICT  **PROVEN** (2026-11-28, `go test -count=1` green: `internal/gateway` 2.329s, `internal/engine` 0.065s).

DOS  bound at ship.

---

## L3 — Allocator & Handle Conservation

THEOREM
1. **`compute.DirectSlabAllocator` / CUDA idle pool:** For any valid `Allocate`/`Free`/`Reset`/`Close` sequence on a slab of capacity `C = d.totalCapacity`, `liveBytes + freeBytes == C` (`d.Used() + d.Available() == C`, with `0 <= d.Used() <= C`), `Free` coalesces adjacent free blocks without losing span, `Reset` restores `d.Used() == 0` and `d.Available() == C`, and `Close` retires all active tracking and backing pins with zero leaked blocks. For the CUDA idle pool, pooled bytes remain capped and all allocations return cleanly on free/evict cycles.
2. **`ggufload` open/close handle conservation:** Every `Open`, `OpenHeaders`, `OpenWeights`, and `OpenWeightsMapped` call — on both valid checkpoints (after `Close()` on `WeightSource`) and malformed/missing-shard error paths — closes all OS file descriptors and shard handles, leaving open FD count equal to baseline.

REGIME  A — algebraic / structural (conservation of bytes/blocks and file-descriptor handle balance).

PROOF
1. **`compute.DirectSlabAllocator`:**
   - `Allocate` (`internal/compute/direct_slab.go:175-251`) checks `size > d.totalCapacity-d.usedBytes` before alignment, splits a free `slabBlock` if `chosen.length > alignedSize` preserving total block span, records `d.active[alloc] = chosen`, and increments `d.usedBytes += size`.
   - `Free` (`direct_slab.go:254-301`) rejects double-free (`alloc.freed`) or unknown allocations, deletes from `d.active`, decrements `d.usedBytes -= alloc.size`, marks `block.free = true`, and coalesces with `block.next` and `block.prev` if free. `Available()` (`direct_slab.go:304-308`) returns `d.totalCapacity - d.usedBytes`, so `Used() + Available() == TotalBytes()` is an invariant.
   - `Reset()` (`direct_slab.go:333-356`) marks all entries in `d.active` freed, replaces `d.head` with a single free block of length `d.totalCapacity`, and sets `d.usedBytes = 0`. `Close()` (`direct_slab.go:361-395`) marks `d.closed = true`, releases any provider owner, marks `d.active` freed, clears `d.active` and `d.head`, unpins `d.pinner`, and rejects subsequent `Allocate` calls (`direct_slab.go:183-185`).
2. **`ggufload` handle conservation:**
   - `Open` (`internal/ggufload/gguf_open.go:15-22`) opens `f` and unconditionally `defer f.Close()` before `Read(f)`, covering both success and parse-error returns.
   - `openWeights` (`gguf_open.go:42-51`) closes the backbone `WeightSource` via `errors.Join(err, ws.Close())` if `attachDeepSeek41EngramCompanion` fails.
   - `(*WeightSource).Close()` (`internal/ggufload/gguf_weightsource.go:297-333`) serializes under `life.closeMu`, drains active borrowers, claims `s.closers`, and closes every shard reader in order (returning the first error while still closing remaining shards), with subsequent `Close()` calls returning `nil` (or `errWeightSourceCloseIncomplete` if an earlier close was interrupted).

WITNESS
```bash
go test -count=1 ./internal/compute -run 'TestVRAMSoakDirectSlabAllocator|TestVRAMSoakCUDAIdlePool'
go test -count=1 ./internal/ggufload -run 'TestGGUFLoadNoFDLeak'
```
- `TestVRAMSoakDirectSlabAllocator` (`internal/compute/vram_soak_test.go:14-82`): runs 10 rounds × 32 multi-size allocate/free cycles (varying request order to force coalescing) plus `Reset()` on both clean and unfreed states, asserting `Used() == 0`, `Available() == totalBytes`, and post-`Close()` `Allocate` failure.
- `TestVRAMSoakCUDAIdlePool` (`internal/compute/vram_soak_test.go:87-119`): compiles and runs the production CUDA idle-pool functions (`fcuda_pool_*`, `fcuda_malloc_tagged`, `fcuda_free`) across 4 multi-phase allocation/reuse/eviction/teardown cycles.
- `TestGGUFLoadNoFDLeak` (`internal/ggufload/fd_leak_test.go:23-109`): runs 64 iterations of `Open`, `OpenHeaders`, `OpenWeights`, `OpenWeightsMapped` on valid, malformed, and missing-shard GGUF fixtures, asserting zero FD growth via `/proc/self/fd` on Linux.

VERDICT  **PROVEN** (2026-11-28, `go test -count=1` green: `internal/compute` 1.873s, `internal/ggufload` 0.231s; note: `TestGGUFLoadNoFDLeak` exercises all open/error/close paths on all platforms and asserts `/proc/self/fd` count on Linux).

DOS  bound at ship.
