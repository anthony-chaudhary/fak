# Scope: structural sharing for the legacy Metal host prefix KV (2026-09-14, #12856)

Parent: #12683. Coordinates with: #12528, #12550, #12831, #12400.

## Verdict

Verdict: GO, scoped and dispatchable. The audit the ticket demanded ("inventory the backend-nil
Qwen Metal readers and writers and choose a bounded compatibility boundary") is complete.
No safe one-to-three-file slice existed before this note; one exists now, with a fail-closed
rollback that keeps the current flat `[][]float32` layout for every unsupported path.

TL;DR: the audit is complete, one bounded three-file slice now exists, and no sharing is
implemented here. This note implements no sharing, makes no zero-copy claim, and qualifies
no end-user performance. Physical qualification remains a separate follow-on per the ticket.

## For / Problem / Today / Better because / Witness

- For: developers forking local Qwen3.8 agent sessions from one long shared prompt on
  Apple Silicon through the backend-nil legacy Metal path.
- Problem: `PrefixSnapshot.Clone` and `KVCache.Clone` copy every host `K`, `Kraw`, and
  `V` float slice before a reused session can append its divergent suffix. The captured
  OpenCode prompt is 34,127 tokens; adding the planned 1,024-token output reserve yields
  35,151 rows. From the current F32 K/Kraw/V geometry that prefix payload is 6,910,967,808
  bytes (≈6.436 GiB) per eager clone. This is a source-derived byte count. It is not a
  measured allocation, copy, resident memory, or completed full native run.
- Today: `internal/model/kv.go` calls `Cache.Clone()` from both snapshot creation and
  snapshot cloning; `internal/model/kvcache.go` deep-copies all three planes.
- Better because: an eventual backend-nil ownership design can share immutable prefix
  storage and isolate only divergent writes while keeping the original prefix exact.
- Witness: a software regression. It restores two sessions from one real prefix snapshot.
  It appends different suffixes through the legacy path. It proves the source snapshot plus
  both continuations remain exact. It reports copied prefix bytes and physical memory, not
  clone-call latency. Package-only command below.

## Geometry (the byte figure, verified)

`Config` for Qwen3.8 is 64 layers. Only 16 are `full_attention`. The other 48 are
linear-attention recurrent state (`internal/model/contextsize_test.go:88-136`,
`qwen38ContextPlanConfig()`: `NumLayers=64`, `NumKVHeads=4`, `HeadDim=256`,
`LayerTypes[i%4==3]="full_attention"` → 16 full layers). The ticket's figure is exactly:

```
35,151 rows × 16 KV-layers × 3 planes (K, Kraw, V) × 4 KVHeads × 256 HeadDim × 4 B
= 35,151 × 16 × 3 × 4 × 256 × 4
= 6,910,967,808 bytes  ✓
```

Charging all 64 layers (as `planCacheGeometry`, `internal/model/cache_geometry.go:78`, does
before the hybrid discount) overstates this by 4×. A sharing slice must decide explicitly
which layer count it charges; today the budget and the payload disagree.

## Boundary: what this owns vs its siblings

- #12528 (CLOSED) owns `internal/radixkv/paged_binding.go` only. It binds ctxmmu page tables
  to Metal PagedAttention, and it does not touch `internal/model`.
- #12550 + #12831 own the GDN/recurrent device COW: exactly `internal/model/qwen35_hal.go`,
  `internal/model/kv.go`, `internal/model/qwen35_backend_contract_test.go` (plus the #12831
  verifier files `qwen35_verify_panel_device.go` / `_test.go`). #12831's non-goals draw this
  ticket's fence verbatim: *"Prefix snapshot host/byte reader synchronization, … performance
  claims, hardware execution, or parent closure."* They do not change the host softmax
  `K/Kraw/V` planes used when `Backend == nil`.
- #12400 owns `internal/compute/vulkan_kv.go` only (contiguous-segment COW).
- #12191 / #12313 own the ctxmmu `KVPool` + COW/telemetry substrate.
- **This issue (#12856)** owns the missing contract for the **legacy backend-nil host-KV
  owner**: not the device COW, not the paged pool, but the flat contiguous host prefix that
  `internal/model` attention actually reads.

`#12550`'s stalled COW composition lives in scratch (`_scratch/goals/local-wip-12550-*.md`);
its identified verifier defect at `qwen35_verify_panel_device.go:146` **is repaired at HEAD**
(the call is now wrapped in `s.qwen35HAL.mutateSequence(...)` setting `request.States` inside
the callback; see `internal/model/qwen35_verify_panel_device.go:146-149`). Its residual,
snapshot host/byte reader synchronization and reentrant-`Close` deadlock, is the surface
this ticket scopes, and it stays disjoint from #12550's three files.

## The incompatibility set (who requires contiguity)

The `KVCache` layout is `K/Kraw/V [][]float32` indexed flat as `K[l][j*w : (j+1)*w]` with
`w = NumKVHeads*HeadDim` (`internal/model/kvcache.go:11-34,92`). A structural share is only
safe if every consumer tolerates an indirection. The inventory says most readers do; the
**writers do not**, and they are not confined to the device path.

Sites that reassign the per-layer slice (append, reserve, or eviction compaction) are the
hard incompatibility: structural sharing must either copy-on-write first or refuse, typed
fail-closed. 42 `Cache.{K,Kraw,V}[...] = append(...)` sites exist in non-test
`internal/model` alone, plus whole-slice reassignment in `kvcache.go` (`evictSupported`
`kvcache.go:165-212`, `Reserve` `:307-333`), `glm_dsa_session.go`, `kv_tree_compaction.go`,
`spanserialize.go`, and the `paged_*` gather/materialize bridges.

Critically, these append sites run on the **backend-nil legacy path, including Metal**:

- `internal/model/kv.go:1018,1024,1108` (`blockStep` Kraw append, `appendKV`, `ropeRowQK`; `Backend == nil` branch)
- `internal/model/metal_prefill.go:195-197,281-295` (Metal host slices)
- `internal/model/metal_prefill_hybrid.go:588-590,880-882,1092-1094`
- `internal/model/metal_prefill_hybrid_core.go:375-386`
- `internal/model/metal_decode.go:162-163` (the `copy` at :142-143 is a read)
- `internal/model/qwen35_batch_decode_metal.go:97-99`
- `internal/model/metal_qwen35_resident_decode.go:566-593`
- `internal/model/qwen35_prefill.go:375-388`, `qwen35_prefill_q4k.go:1015-1028`
- `internal/model/prefill_q4k.go:173`, `quant_forward.go:722`, `kvcache_q8.go:275-303`

So the ticket's warning holds: an append seam alone is not evidence that `PagedKV.Fork`
can replace flat `KVCache.Clone`. Any share assumption that "only the device path appends"
is false on this architecture.

Read-only consumers index contiguously but never mutate, so they could tolerate an
indirection that presents a contiguous view: the attention loops (`kv.go:1056,1072`;
`batch_attn.go`), `rescore.go:107`, `reposition_witness.go`, `kv_cache_bytes.go`,
`prefix_snapshot_bytes.go`, and `verify.go`.

Transforms that rebuild whole planes (`spanserialize.go`, `paged_materialize.go`,
`qwen35_paged_swap.go`, `glm_dsa_session.go`) must run on, or materialize to, a private flat
copy. `PrefixSnapshot` currently does the opposite: `Session.PrefixSnapshot()` deep-clones at
`kv.go:80`, `PrefixSnapshot.Clone()` deep-clones again at `kv.go:115`, and `Restore` merely
moves the pointer at `kv.go:174-175`. `PrefixSnapshot.Close` (`kv.go:186-202`) nils `Cache`
with no plane-specific release.

## Prior art in tree (reuse, do not reinvent)

- `PagedKV` / `PagedKV.Fork` (`internal/model/pagedkv.go:44-59,166-170,278-284`) already
  implements refcounted block sharing + `ensureOwned` COW, with `gather` materializing a
  contiguous per-layer slice. Its own header is honest about scope (`pagedkv.go:26-34`): it
  is *"NOT a device-side paged-attention kernel … and it does NOT replace the default direct
  []float32 Session."* It is the data-structure template for this ticket's eventual leaf.
- `PagedPrefixSession.Fork` (`internal/model/paged_prefix_cow.go:886`) and
  `PagedKV.ForkMeasured` (`internal/model/kvshare_receipt.go:27`) prove zero-copy fork is
  already measured.
- `internal/ctxmmu/cow_block.go`, `fork.go`, `cow_metrics.go`; `internal/radixkv/paged_binding.go`;
  `internal/compute/vulkan_kv.go` as sibling device/page-table COW, all disjoint from this lane.

The gap is integration, not invention: no existing primitive is installed as the owner of the
legacy host `K/Kraw/V` prefix.

## Bounded compatibility boundary (owned files)

The future implementation leaf owns exactly three files:

1. `internal/model/kvcache.go`, the layout owner. Add an optional shared-prefix
   representation behind `Clone`/`CloneWithReserve`, plus the copy-on-write guard that
   `Truncate` (`:394`), `Reserve` (`:307`), and `evictSupported` (`:165`) must pass before
   mutating a shared plane.
2. `internal/model/kv.go`, the ownership funnel: the `Backend == nil` branches of
   `PrefixSnapshot` (`:86-87`), `PrefixSnapshot.Clone` (`:120-121`), `Restore`
   (`:137-183`), and `Close` (`:186-202`) where the shared prefix lifetime is pinned and
   released exactly once. (This file is shared with #12550/#12831; land after their lease
   clears, or coordinate a disjoint hunk.)
3. `internal/model/kv_cache_bytes.go` (+ `prefix_snapshot_bytes.go`), which must report shared
   prefix bytes once, or a structural share silently double-counts into the memory budget.

Rollback and flat-layout preservation. Keep the current `[][]float32` layout as the
default and gate sharing behind an explicit capability, exactly as `FAK_PAGED_KV` gates
`pagedkv.go` / `paged_hal.go`. Every site that reassigns `c.K[l]` must either (a) be refused
on a shared cache with a typed fail-closed verdict (mirroring
`RecurrentEvictUnsupportedError`, `kvcache.go:104-126`), or (b) be preceded by a
copy-on-write of just the mutated layer. Reverting the leaf to `Clone()`'s deep copy restores
byte-for-byte today's behavior with no consumer change.

## Correctness witness (software)

The regression must exercise the actual session contract, not an idle `Clone` microbenchmark:

1. Build a non-empty backend-nil Qwen session and capture a `PrefixSnapshot`.
2. Clone/restore two independently owned sessions.
3. Append distinct suffix tokens through the production legacy path.
4. Prove the original prefix remains byte-exact in `K`, `Kraw`, `V`, positions, lineage, and logits.
5. Prove each branch matches an independently prefetched full prompt plus its suffix after divergence.
6. Prove close/eviction releases shared ownership exactly once.
7. Report prefix bytes copied before divergence and on first divergent append.

Baseline (green today, no GPU required):

```bash
go test ./internal/model -run 'TestTokenLineage|TestSessionFromPrefix|TestPrefixSnapshot' -count=1
```

Anchors: `internal/model/token_lineage_test.go:40` (`TestTokenLineageLegacyWriteShiftRollbackAndPrefixReuse`,
`m.SessionFromPrefix(s.Cache)` at `:61`), `:96`
(`TestTokenLineageQwen38SequencePrefillSnapshotAndMetadataReceipt`, `s.PrefixSnapshot()` at `:113`),
`gemma4_prefix_reuse_test.go:89`, `prefix_snapshot_bytes_test.go:9`,
`qwen35_backend_contract_test.go:514`. Legacy Metal tests are gated by
`//go:build darwin && arm64 && cgo` (not `fakmetal`); the "MatchesCPU" arms need a real
device, the decline paths do not (`metal_prefill_hybrid_test.go:46`).

## Non-goals

- GDN recurrent-handle COW (#12550/#12831).
- Vulkan device KV (#12400).
- A new generic KV-cache epic.
- An idle clone-only benchmark.
- Relaxing admission or memory-pressure guards.
- Claiming current `PagedKV` is already compatible with contiguous attention readers.
- Any zero-copy or end-user performance claim from this audit.

## Physical qualification remains separate

Software ownership completion does not qualify end-user performance. Preserve the existing
35,151-row / full-task native Qwen3.8 workflow as a separate physical promotion witness with
exact artifact, source, prompt/task, context, engine/backend/no-fallback identity, peak
footprint, paging, correctness, and task completion. This is not a new required criterion for
the current prefix-reuse value proof under #12683.