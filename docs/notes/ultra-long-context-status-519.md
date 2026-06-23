---
title: "fak ultra-long-context (#519): code-grounded status of the levers"
description: "A maintainer status decomposition of epic #519 (ultra-long-context first-class support): which levers on top of the >100k work floor already shipped, which are partial, and which are gated — grounded in on-disk code at HEAD, not the issue text."
---

# Ultra-long-context epic (#519): code-grounded status decomposition

_Tracking epic: **[#519](https://github.com/anthony-chaudhary/fak/issues/519)** — "ultra-long-context
(>100k) first-class support — levers on top of the work floor."_

**Grounded at:** `HEAD = 7bbbac8` (2026-06-23). Every `file:line` below was opened and read while
writing this doc; nothing here is a restatement of the issue text or of the framing note
([`RESEARCH-ultra-long-context-levels-and-naming-2026-06-22.md`](RESEARCH-ultra-long-context-levels-and-naming-2026-06-22.md)).
This is a **status artifact for a maintainer**, not an implementation of any lever — no code changed
(`go build ./...` green; `go vet ./...` clean).

---

## TL;DR — the epic's "five missing levers" thesis is stale; three shipped since the framing note

The framing note (2026-06-22) decomposed the epic into the >100k **work floor** (shipped:
`longctxbench` / `turnbench.RunLongContextLadder`) plus **five missing levers** on top of it. On disk
at HEAD **three of those five are no longer missing**, and a fourth is partial. The work floor itself
also gained the idle-fraction param (`#520`) that the ragged-decode lever needed to be priced.

Counts by lever (the five the note seeded, plus the shipped bases they lean on):

| Lever (note §2/§5) | Status at HEAD | Owning issue |
|---|---|---|
| **L1** Persistent per-agent KV (reread elision) | **shipped** (base) | `internal/model` Session/KVCache |
| **L2** Cross-agent prefix share (prefix once + clone) | **shipped** (base) | `NewBatchFromPrefix`, `KVCache.Clone` |
| Bit-exact middle-span KV eviction | **shipped** (base) | `KVCache.Evict` |
| Sliding-window attention (O(window) read-mask) | **shipped** (base) | `windowForLayer` / `windowLoContig` |
| RoPE scaling / longrope (reach 100k positions) | **shipped** (base) | `longrope.go`, `rope_scaling.go` |
| RadixAttention prefix-tree + policy evict | **shipped** (base) | `internal/radixkv` |
| KV residency tiers + witnessed exact-span evict | **shipped** (base) | `internal/cachemeta`, `internal/enginecache` |
| Result-admit gate + page-out + poison re-screen | **shipped** (base) | `internal/ctxmmu`, `internal/recall` |
| **A. Ragged batched decode / idle-lane skip** | **shipped** | **#520** (work-floor idle-fraction param landed) |
| **B. First-class context-residency query** | **shipped** | `internal/ctxresidency` (+ sibling `internal/contextq`, #514 / epic #437) |
| **C. `Session.TrimToWindow` + kernel-mediated compaction** | **partial** — trim API shipped; external span-swap not wired | residual below |
| **D. KV compression on the eviction path** | **partial** — compressed-KV view + tier-demotion shipped; compress-and-demote on evict not wired | residual below |
| **E. Live wall-clock anchor at >100k** | **gated** (hardware) | live-anchor issue; [SIMULATED] in `CLAIMS.md:141` |

**The load-bearing correction:** the framing note's §2 inventory (written 2026-06-22) lists A, B, and C
as **MISSING**. At HEAD A and B have fully shipped and C's safe-trim API has shipped — the same kind of
stale-thesis drift the [#487 decomposition](model-arch-seam-status-487.md) corrected for the model-arch
seam. This doc re-grounds each lever against the code that actually landed in the day since the note.

**The permanent residual the epic carries forever:** the **live wall-clock anchor at >100k** (E) needs a
model resident on a bench node — it is explicitly not runnable on the build box and is fenced
`[SIMULATED]` in the honesty ledger (`CLAIMS.md:141`). The floor's *work* ratios are exact arithmetic
and stand independently; the absolute wall-clock at this regime is a separate, hardware-gated
measurement. Closing E is not reachable from this environment.

---

## Method & honest caveats

- **Two sources, reconciled.** The framing note is the *why*; this doc is the *verified status at
  HEAD*. Where they disagree (A/B/C), the code wins and the drift is recorded inline — exactly the
  discipline of the `#487` / `#492` decompositions.
- **Every citation is real** and was read at `HEAD = 7bbbac8`. The witness test named for each lever was
  confirmed on disk (not run here — native `go test` is blocked by an OS Application-Control policy on
  this Windows host per `AGENTS.md`; run the suite under WSL via `./test.ps1`).
- **One lever is hardware-gated, not code-gated.** E cannot be closed by any change to this tree from a
  build box; it is flagged, not hand-waved.

---

## Sibling-issue map

| Issue | State | Owns | On-disk landing |
|---|---|---|---|
| **#519** | OPEN | the epic itself | this doc |
| **#520** | shipped | A — ragged decode (`StepBatchActive`) + the work-floor `IdleFraction` param | `internal/model/batch.go`, `internal/turnbench/longcontext.go` |
| **#514** | shipped (child of epic #437) | sibling — model-scoped KV view / materialization gate (`contextq`) | `internal/contextq/` |
| live-anchor | OPEN | E — wall-clock validation of the floor's ratios at >100k | gated (bench node + resident model) |

---

## A — Ragged batched decode / idle-lane skip ✅ shipped

**The framing note (§2) called this MISSING: "*`model.StepBatch` decodes all C lanes every step.*" That
is no longer true.** `StepBatchActive` decodes only the active lanes and compacts the batch dimension,
so a fleet with K of C lanes idle this turn does `(C−K)/C` of the full batch's decode work.

- `StepBatchActive(ids, active []bool)` — `internal/model/batch.go:1199`. Idle lanes are left untouched
  (no KV append, no position advance, no logits); a lane blocked on a tool, finished (EOS), or idle can
  be reactivated on a later step without paying for work it did not need. The compaction gathers the
  active lanes into a contiguous sub-panel and runs the **exact `stepBatchF32`/`stepBatchQ` machinery**
  over it, so for every active lane the result is bit-for-bit identical to a full `StepBatch` with the
  idle lanes discarded (the same property that makes `StepBatch` bit-identical to serial `Step`).
- `LastStepMACs()` — `batch.go:1164` — reports the exact B-proportional projection MAC count, scaling
  with the **active** batch size, so the work-elimination is witnessed as an exact integer ratio.
- Witnesses: `TestRaggedBatchIdleLaneSkip` `batch_test.go:607`, `TestRaggedBatchIdleLaneSkipQ8`
  `batch_test.go:734`.
- Priced in the work floor: `SessionShape.IdleFraction` `internal/turnbench/longcontext.go:176` scales
  the decode FLOPs of every arm by `(1−IdleFraction)`; the headline `P=100k C=40` regime sets
  `IdleFraction: 0.5` (`longcontext.go:416`). Witness: `TestLongContextIdleFractionFloor`
  `longcontext_test.go:199` (every arm scales identically; the floor never does more work),
  `TestLongContextIdleFractionClamped` `:265`.

**Residual:** `StepBatchActive` is the model-leaf decode step; whether the live **serving/gateway** loop
(the multi-user path in `internal/gateway`) actually drives a heterogeneous fleet through it, vs. the
all-active `StepBatch`, is the wiring residual — the lever's *mechanism* is shipped, its *deployment on
the hot serving path* is the open finish item.

---

## B — First-class context-residency query ✅ shipped

**The framing note (§2/§3) called this MISSING: "*ctxmmu query API is observability-only
(`Held`/`HeldLen`/`Evicted`)*" and proposed a "context-residency query" over the resident/evictable/held
span ledger with eviction blast-radius. That query now exists as `internal/ctxresidency`.**

- `ctxresidency.Query(c *kvmmu.Context, mmu *ctxmmu.MMU) Snapshot` — `internal/ctxresidency/ctxresidency.go:95`.
  Per-span read with exactly the three states the note asked for: `StateResident` `:20`,
  `StateEvictable` `:24`, `StateHeld` `:27` (held = quarantined; K/V span evicted, references live).
- `EvictBlastRadius{Tokens, DependentEntries}` — `ctxresidency.go:36` — what evicting a resident/
  evictable span would cost (the K/V positions plus the live `cachemeta` dependents an `Evict` would
  drop). This is the "eviction blast-radius" the note named.
- Reconciles with both kernels' own counters (the note's "observability-only" floor, now the
  cross-check): `Snapshot.ResidentTokens == kvmmu.Context.CacheLen`, `HeldSpans == kvmmu.Evicted`,
  `ByteHeld == ctxmmu.HeldLen` (`ctxresidency.go:73-83`, docstring `:51`). Witness:
  `TestQueryReconcilesWithCounters` `ctxresidency_test.go:49`.
- Read-only by construction: `TestQueryIsReadOnlyNoPoisonLaundering` `ctxresidency_test.go:173` — a query
  cannot change cache length, held count, or cleared count; blast-radius split witnessed by
  `TestResidentVsEvictableBlastRadius` `:124`.

**Sibling (do not conflate):** `internal/contextq` (`Query` `contextq.go:270`, model-scoped KV view
`kvview.go:110`, the `#514` child of epic #437) is the *materialization* layer — a typed, fail-closed
read that materializes a working set from the CDB image with a `GateKVView` quality gate. `ctxresidency`
is the *coherence-state* ledger read the note asked for; `contextq` is the adjacent materialization
sibling. Both shipped; B is closed.

---

## C — `Session.TrimToWindow` + kernel-mediated external compaction ⚠️ partial

**The framing note (§4) called this MISSING: "*`TrimToWindow` designed in SLIDING-WINDOW doc, no API.*"
Half of it shipped: the safe-trim API exists.** The residual is the *external-compaction* half.

- `Session.TrimToWindow(slack) int` — `internal/model/swa.go:48` — bounds the session's KV to O(window)
  for a stream of any length. When the cache grows past `MaxWindow()+slack` it evicts the oldest
  positions back down to `MaxWindow()` through the proven `KVCache.Evict` (which renumbers and re-RoPEs
  survivors; because RoPE is relative, the within-window attention scores are preserved). No-op when no
  window is configured, so non-SWA (Llama/Qwen) models are byte-identical to before.
- `Config.MaxWindow()` — `swa.go:25` — the keep-count (widest per-layer window when every layer
  configures one; 0 sentinel = unbounded / not safely trimmable).
- Witness: `TestBoundedWindowMatchesFullWindow` `internal/model/trimwindow_test.go:24` (a trimmed
  windowed decode is argmax-identical to the same decode over the full cache).

**Residual (the open half):** what the note calls **kernel-mediated external compaction** — a harness
(e.g. Claude Code's auto-compaction) emits a summary span; the kernel `Evict`s the summarized range and
ingests the summary *under the same result-admit gate* (so a poisoned summary cannot launder back in),
i.e. compaction as a coherence-checked span swap. The safe-drop mechanism (`TrimToWindow` + the
result-admit gate in `internal/recall`, `reScreen` `recall.go:413`) exists as parts; the **wired span-swap
on the live agent loop** (`internal/agent/loop.go`, before the per-turn decode) is not. The note's
"compaction" is C-trim shipped + external-swap not-started.

---

## D — KV compression wired into eviction ⚠️ partial (modeled + tier-demotion shipped; compress-on-evict not)

**The framing note (§4): "*`cachemeta.MatCompressedKV` designed, not on the evict path.*" Still true —
and the demotion machinery around it shipped.**

- `MatCompressedKV MaterializationView = "compressed_kv"` — `internal/cachemeta/materialization.go:37`
  — an approximate / quantized KV span, lossy; `IsApproximate()` `:55`. Quality-gated:
  `MaterializeVerdict` refuses an unmeasured or over-bound compressed view. Witness:
  `TestCompressedKVRequiresQualityEvidence` `materialization_test.go:142`.
- Tier-demotion-under-pressure **shipped**: `PlacementAction = "demote"` `placement.go:44`,
  `PlanPlacement` relocates a hot span to a colder *attendable* tier instead of dropping it; tier ladder
  `localTierLadder = {HBM, DRAM, NUMA-far, CXL, Disk}` `hardware.go:166`. Witness:
  `TestDemoteBeatsEvictUnderPressure` `placement_test.go:32`, `TestDemoteSkipsFullTiersToCXL` `:47`.
- `MatCompressedKV` is consumed today as a `contextq` **KVView** (`internal/contextq/kvview.go`,
  `GateKVView` `:110`), not as an evict-path step.

**Residual:** the note's specific seam — *when a span would be evicted under residency pressure (L3),
compress-and-demote it (lossy quantize → demote) instead of dropping it, admitting the compressed span
only when the measured quality delta is under bound* — is **not wired**. The quality gate exists, the
demotion exists, the compressed view exists; the evict-path **compress-then-demote step** that composes
them is the open finish item.

---

## E — Live wall-clock anchor at >100k ⛔ gated (hardware) — not reachable from this environment

**The framing note (§5) and the honesty ledger agree: this needs a model resident on a bench node and
is not run on the build box.** It is the one lever this decomposition cannot close.

- `[SIMULATED]` row — `CLAIMS.md:141`: "*The live WALL-CLOCK validation of the floor's ratios at >100k …
  needs a model resident on a bench node and is not run on the build box; the floor itself is exact
  arithmetic, but the absolute wall-clock at this regime is a separate, gated measurement.*"
- The floor↔wall-clock loop already closes at **small** scale via `cmd/sessionbench -validate`; the
  >100k regime is gated because the naive arm's `O(T²)` re-prefill is intractable to run live, so the
  anchor needs a resident model on a bench node to validate the floor's ratios end-to-end.

**Why this is a gate, not a missing lever:** no change to this tree from a build box closes E. The
floor's *work* numbers (token + O(L²)-aware FLOP, `CLAIMS.md:140` `[SHIPPED]`) stand independently of
the wall-clock; E only validates their live ratios. Tracked in the live-anchor issue.

---

## Recommended epic checkbox updates (#519)

On-disk evidence at HEAD proves the following should move from the note's "MISSING" column to their
real status:

- [x] **A** — ragged batched decode / idle-lane skip — **shipped** (`StepBatchActive` `batch.go:1199`,
  `#520`; priced via `IdleFraction` in the work floor). Residual: serving-path deployment.
- [x] **B** — first-class context-residency query — **shipped** (`ctxresidency.Query` `ctxresidency.go:95`,
  resident/evictable/held + blast-radius, counter-reconciled, read-only).
- [~] **C** — compaction — **partial**: `TrimToWindow` `swa.go:48` shipped (safe bounded trim); the
  kernel-mediated external span-swap on the live loop is not wired.
- [~] **D** — KV compression on evict — **partial**: `MatCompressedKV` `materialization.go:37` +
  quality gate + tier-demotion (`placement.go:44`) shipped; compress-and-demote on the evict path is
  not wired.
- [ ] **E** — live wall-clock anchor at >100k — **gated** (`CLAIMS.md:141` `[SIMULATED]`; needs a bench
  node + resident model). Not reachable from a build box.

The epic's "levers on top of the work floor" are now statused: two shipped clean (A, B), two partial
with a precise finish item each (C, D), one hardware-gated forever on this environment (E). The
>100k work floor itself (`longctxbench`, `CLAIMS.md:140` `[SHIPPED]`) remains the load-bearing
delivered proof the levers lean on.

---

_Generated as the #519 deliverable: a grounded decomposition for a maintainer to act on. All `file:line`
citations were read at `HEAD = 7bbbac8`. No code changed; `go build ./...` green, `go vet ./...` clean._
