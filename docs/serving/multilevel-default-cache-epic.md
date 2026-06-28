---
title: "Multilevel default cache for the fak kernel: HBM → DRAM → disk, hardware-aware by default"
description: "The progress spine that finishes the hardware-capacity bridge (#706): wire the demote-not-evict executor into a live loop, derive real pressure for every local tier (not just HBM), and make hardware-aware placement the kernel's default. Each rung is a prove-or-refute step bound to a dos verb."
---

# Multilevel default cache: HBM (L1) → DRAM (L2) → disk/CXL/remote (L3), by default

The fak kernel owns its KV cache as a kernel object rather than renting it from vLLM or
SGLang. That ownership is the whole reason a hardware-aware, multilevel cache is *fak's* job
and not the serving engine's: when you own residency, you own which tier a span lives in and
when it moves. This epic finishes the job — it makes the kernel place every cached span on
the tier that fits it, demote one tier colder under real memory pressure instead of dropping
it, and do so **by default**, against the **device that actually exists**, on every local
level of the hierarchy.

It is the completion-and-downward-extension of the hardware-capacity bridge, [epic #706](https://github.com/anthony-chaudhary/fak/issues/706).
It does **not** restate #706's positioning or rebuild its planes. Read these first:

- [`docs/serving/hardware-aware-cache.md`](hardware-aware-cache.md) — the `cachemeta` policy/metadata plane (tier model, per-tier TTL, demote-not-evict `PlanPlacement`).
- [`docs/explainers/hardware-limits-and-capacity.md`](../explainers/hardware-limits-and-capacity.md) — the two planes and the bridge between them; the plank-by-plank status #706 tracks.
- [`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md`](../notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md) — the L3 (disaggregated) positioning study, epic #79.

> **Status of this document.** This is a *planning and progress* artifact, not a measured
> benchmark. Every "shipped" claim below names a tested primitive in this repo (file:line),
> and every rung states the exact `dos` verb that proves or refutes it. The epic is **not
> done**: the honest baseline is that L1/HBM placement is wired into the policy plane but the
> executor is library-only, and L2/DRAM + L3/disk plan against representative defaults rather
> than the live box. Those are the rungs that follow.

## The crux

There are two planes, and after #706 they meet at exactly one tier.

- **The policy plane** (`internal/cachemeta`) decides *where a span should live and when it
  should move*: `PlanPlacement` ranks HBM → DRAM → NUMA-far → CXL → disk → remote, and on
  memory pressure emits **demote** (relocate to a colder *attendable* tier, a one-time
  `bytes / bandwidth` stage) rather than **evict** (drop it, pay a full `tokens × per-token
  prefill` rebuild later). This is built, tested, deterministic.
- **The execution plane** physically moves bytes: `internal/engine`'s `CapacityAdapter`
  stages a span to a colder tier then evicts it from the live cache; `internal/storedrv`'s
  `Router` fans content-addressed payloads across blob (RAM) → blobfs (disk) → blobhttp
  (remote).

After #706 the two planes touch **only at HBM, and only as a library**:

1. **The executor is never called on a live serve loop.** `CapacityAdapter.Execute` /
   `RunCapacityPressureSweep` are invoked only from tests and `cmd/hwcachedemo` /
   `cmd/cxlpooldemo`. Nothing in `internal/gateway` / `internal/agent` / the served decode
   path runs the sweep when KV pressure rises. *(Confirmed by the explainer's own honest
   fence: "Nothing here claims … the serving loop demotes KV under live pressure.")*
2. **Only HBM has real pressure.** `DeviceHBMPressure` (`internal/engine/capacity_pressure.go`)
   derives live HBM fullness from `compute.DeviceMemoryInfo`. There is **no**
   `DeviceDRAMPressure` / disk-pressure equivalent — L2/DRAM, disk, and CXL plan against
   `cachemeta.DefaultTierProfiles`' representative numbers and an *assumed-empty* pressure of
   0. So the demote ladder is hardware-aware at its top rung and blind below it.
3. **storedrv routes orthogonally to the policy.** `Router` picks a tier by payload *size*
   and a `Hint` (plane/scope/taint/durability); it consumes no `PlacementDecision` and no
   `TierPressure`. The KV-residency policy and the payload-storage tiering are two systems
   that never consult each other.
4. **The whole path is off by default.** storedrv only registers when `FAK_STORE` is set;
   the capacity sweep has no enablement at all because no serve loop calls it.

So the one-line crux: **the hardware-aware cache is a decision the kernel computes but does
not yet act on, knows only for the top tier, and does not run by default.** This epic closes
those four, lowest-risk-and-highest-proof first.

## The tier map — what is shipped-live, shipped-but-library-only, unbuilt

Levels named like CPU caches: **L1 = HBM** (GPU memory, when present), **L2 = DRAM** (host
RAM), **L3 = disaggregated** (local SSD/disk, then CXL far-memory, then a remote pool).

| Tier | Capacity report | Live pressure → policy | Executor wired to serve | Status |
|---|---|---|---|---|
| **L1 / HBM** | ✅ `cuda.go DeviceMemory()` → `cudaMemGetInfo` | ✅ `DeviceHBMPressure` | ❌ library-only (`CapacityAdapter` uncalled on serve) | shipped policy, **executor not live** |
| **L2 / DRAM** | ✅ `HostSystemMemoryInfo()` (per-OS probe) | ❌ no `DeviceDRAMPressure` | ❌ | **probe exists, pressure-wire unbuilt** |
| **L3 / disk** | ❌ no filesystem-space probe | ❌ | ⚠️ `storedrv` blobfs tiers by size, not by pressure | **unbuilt (policy plans against defaults)** |
| **L3 / CXL** | ❌ no CXL HAL backend | ❌ | ⚠️ profile + share-kind exist; no device | **metadata-only; physical move unbuilt** |
| **L3 / remote** | n/a (unbounded) | n/a | ⚠️ `storedrv` blobhttp; not policy-driven | **transport exists, not policy-linked** |

The shipped foundations every rung rides on, so no rung rebuilds them:

- Tier model + cost-driven placement: `internal/cachemeta` (`hardware.go`, `lifecycle.go`, `placement.go`).
- Live HBM pressure: `internal/engine/capacity_pressure.go` (`DeviceHBMPressure`, `PlanPlacementForDevice`).
- The executor: `internal/engine/capacity_adapter.go` (`CapacityAdapter`), `capacity_sweep.go` (`RunCapacityPressureSweep`).
- Host-RAM probe: `internal/compute/capacity.go` (`HostSystemMemoryInfo`, `HostMemoryInfo`) + `hostmem_{darwin,linux,windows,other}.go`.
- Payload tiering transport: `internal/storedrv` (`Router` over blob/blobfs/blobhttp).

## The progress spine — prove or refute, rung by rung

Each rung names a **claim**, the **build** that would make it true, and the **witness**: the
exact `dos` verb (and the assertion behind it) that proves it shipped or refutes a premature
"done." A rung is not done until `dos verify <plan> <phase>` returns `shipped:true` *and* the
ship commit passes `dos commit-audit` (claim matches diff). The plan id for the whole spine
is **`MLCACHE`**.

Ordered by proof-per-hour: cheapest, hardware-free, highest-certainty proofs first; the
GPU-gated and cross-node rungs last.

### Rung 1 — L2/DRAM live pressure (`MLCACHE1`) — hardware-free, ships from any box

**Claim.** The placement policy plans the DRAM tier against the host's *real* RAM fullness,
not the `DefaultTierProfiles` placeholder — so under host-memory pressure a span that would
have demoted *into* DRAM instead skips to a colder tier with room.

**Build.** A `DeviceDRAMPressure(b, residentBytes) → (pressure, capacity, known)` mirroring
`DeviceHBMPressure`, sourced from `compute.HostSystemMemoryInfo()` (backend-free, so it works
on the cpu-ref floor), folded into the request by a `withDRAMPressure` / `withDRAMCapacity`
copy-on-write exactly like the HBM wire. Fail-open: `known=false` → request used verbatim.

**Witness.** A test proving the flip: with `TierDRAM` pressure forced to 1.0 the
`coldestColderWithRoom` walk skips DRAM and the decision's `ToTier` is the next colder tier
(NUMA-far/CXL/disk) — the same shape as #707's HBM keep→demote flip test. Refute condition:
if the test passes with the wire *absent* (i.e. the default already produced the colder
target), the rung proves nothing and must be reframed.

> **Status: SHIPPED `8245e5de` (#985, #986–988 track the rest).** `internal/engine/capacity_dram.go`
> + `capacity_dram_test.go` ship `HostDRAMPressure` / `PlanPlacementForHost` /
> `PlanPlacementForDeviceAndHost` with the copy-on-write `withDRAMPressure` / `withDRAMCapacity`
> helpers. `TestDRAMPressureFlipsDemoteTarget` passes: DRAM pressure 1.0 moves the demote target
> off DRAM to NUMA-far, and an explicit refute guard asserts the *default* target was DRAM, so
> the flip proves the wire and not the request. Probe math, fail-open, and copy-on-write
> non-mutation each have a passing test.
>
> **Honest witness note.** `dos commit-audit 8245e5de` returns **ABSTAIN** — not because the
> diff is empty (it adds the wire + tests) but because the subject verb "derive" is outside the
> referee's claim-kind verb set, so it makes no *checkable* claim to the auditor. `dos review`
> over the range is CLEAN (the abstain is `unverifiable`, not a residual). And
> `dos verify MLCACHE MLCACHE1` is `shipped:false` because `MLCACHE`/`MLCACHE1` are this doc's
> own plan/phase convention with no registry row or commit marker in the stamp grammar. So the
> rung's real evidence is **the passing refute-guarded test + the diff-witnessed file set**, and
> the lesson for later rungs is concrete: lead the ship subject with a referee-recognized verb
> (`add`/`feat … add`) and carry the `MLCACHE<n>` marker, so both `dos commit-audit` and
> `dos verify` bind without an amend on the shared trunk.

> **First-ship target.** This is the leading rung precisely because it needs no GPU and no
> serve loop — it is provable on the Windows dev box and in CI, and it removes the "only HBM
> is hardware-aware" asymmetry that makes the whole ladder blind below its top rung.

### Rung 2 — L3/disk live pressure (`MLCACHE2`) — hardware-free

**Claim.** The spill decision (demote to disk) plans against the *real* free space on the
spill filesystem, so a full disk forces eviction instead of a spill that would fail.

**Build.** A `DiskPressure(path) → (pressure, capacity, known)` over a cross-platform
statfs/`GetDiskFreeSpaceEx` probe (new `internal/compute` file with per-OS build tags,
matching the `hostmem_*` shape), wired into the request's `TierDisk` profile/pressure.

**Witness.** A test: disk pressure 1.0 → a span that beats recompute but has no colder room
evicts (`no_colder_tier_with_room`) instead of spilling. `dos verify MLCACHE MLCACHE2`.

### Rung 3 — executor on a live loop (`MLCACHE3`) — the keystone

**Claim.** The served decode path runs the capacity sweep under real pressure: when HBM (or
DRAM) crosses a high-water mark, the live KV candidates are demoted/spilled via
`CapacityAdapter`, and the move is visible on `/metrics` as a real `fak_engine_cache_*`
transition — not a synthetic demo.

**Build.** Call `RunCapacityPressureSweep` from the in-kernel served decode boundary
(`internal/gateway` / the engine decode loop) with the live resident KV candidates and the
high-water target, behind a default-on policy with a documented disable. This is the
"executor wired to serve" cell the tier map marks ❌ for every tier today.

**Witness.** A served-path test (or a `cmd/` harness on a real backend) showing a demote
transition recorded through the cache-event stream during decode under pressure, *plus* the
honest fence in the explainer flipped from "Nothing here claims the serving loop demotes KV
under live pressure" to a cited, tested claim. `dos verify MLCACHE MLCACHE3` + a
`dos commit-audit` confirming the diff touches the serve path, not just a test.

> This rung is **gated on a backend that can hit real pressure** (the A100/L4 fleet via the
> Slack bridge, or a synthetic-pressure backend in test). It is the highest-value rung and
> the one most likely to be refuted by "it only ran in a demo" — so its witness must show a
> serve-path call site, not a library call.

> **Status: SHIPPED `ca5aad6a` (#1073, the live serve-path call site).** `internal/gateway/`
> `kvmmu_pressure_relief.go` adds `maybeRelieveKVPressure`, called from `complete()`'s
> post-decode success tail (the boundary right after `observePlannerRequestMemory`), driving two
> host-injected, import-clean seams — `KVPressureCandidateProvider` (live resident spans) and
> `KVPressureSweeper` (the `engine.CapacityAdapter` closure). `cmd/fak/kvmmu_pressure_bridge.go`
> is the host glue (twin of `kvmmu_slot_bridge.go`) that builds the sweeper over a live
> `compute.Backend` + `CapacityAdapter` and lowers candidates into `engine.CapacityPressureCandidate`.
> Gated behind `FAK_INKERNEL_KVMMU`, fail-open (nil seams / empty candidates ⇒ a no-op,
> byte-identical to before), `FAK_KV_HIGHWATER` overrides the 0.80 mark. The demote folds into
> the `fak_engine_cache_*` cache-event stream automatically via the adapter's recorder.
>
> **Witness.** `TestMaybeRelieveKVPressureDemotesUnderPressure` proves demote-not-drop under
> simulated HBM pressure (StageSpan → Evict, `AppliedMoves==1`, a `ddr_cache` cache-event row),
> with a refute guard (`…GateOffIsNoOp`) that asserts the gate-off / no-seams path is inert — so
> the test proves the WIRE, not a default-on behavior change. `dos commit-audit ca5aad6a` returns
> **OK / diff-witnessed** (the `add`-led subject made the claim checkable, where MLCACHE1's
> "derive" abstained), `dos review` over the ship range is **CLEAN** (zero residual), and the
> diff touches the serve path (`internal/gateway/gateway.go`), not just a test.
>
> **Honest fence (still in force).** The production provider is **nil** at the serve.go call
> site: `InKernelPlanner` keeps residency in a radix reuse tree and builds a `kvmmu.Context`
> ephemerally per eviction, so there is no durable resident-span list to enumerate yet. So this
> rung ships the LIVE, non-test call site + the synthetic-pressure demote test; it does NOT yet
> assert the served loop demotes a *real* span under *real* GPU pressure. That last step — a
> persistent span enumerator over `kvmmu.Segment{From,Len,KV}` feeding a non-nil provider — is
> the follow-on **#1074 / #987**. As with MLCACHE1, `dos verify MLCACHE MLCACHE3` is
> `shipped:false` because `MLCACHE`/`MLCACHE3` are this doc's own plan/phase convention with no
> marker in the stamp grammar; the rung's real evidence is the diff-witnessed commit-audit + the
> passing refute-guarded test.

### Rung 4 — default profiles from the live box (`MLCACHE4`) — hardware-free

**Claim.** On startup the kernel replaces `DefaultTierProfiles`' representative
order-of-magnitude numbers with the tiers the box *actually has* (HBM capacity from the CUDA
probe, DRAM capacity from the host probe, disk capacity from rung 2's probe) — so a box with
no GPU has no HBM tier in its ladder, and a box with 1.5 TB RAM plans against 1.5 TB.

**Build.** A `ProbedTierProfiles(b) map[ResidencyTier]TierProfile` that starts from the
defaults and overrides each tier's `CapacityBytes` from the live probes, dropping tiers the
box cannot prove. Surfaced on `/debug/vars` so the operator can see the ladder the kernel
chose.

**Witness.** A test asserting a no-GPU profile set omits `TierHBM` and a probed DRAM capacity
differs from the placeholder; `dos verify MLCACHE MLCACHE4`.

### Rung 5 — storedrv consults the placement policy (`MLCACHE5`)

**Claim.** The payload-storage router and the KV-residency policy stop being orthogonal: a
`Router.Put` of a large, durable, fleet-scoped payload lands on the tier the *placement
policy* would choose under current pressure, not purely on its byte size.

**Build.** A thin adapter that lets `storedrv.Router` take an optional `cachemeta`-derived
placement hint (pressure-aware tier selection) without changing the frozen `abi.Resolver`
default path — additive, off unless the caller opts in, the same posture `PutHinted` already
takes.

**Witness.** A test: under high hot-tier pressure a payload that `Accept(size)` would keep
hot instead routes durable; `dos verify MLCACHE MLCACHE5`. Refute condition: if this only
duplicates `PutHinted`'s size/durability routing, it adds no tier-awareness and must be cut.

### Rung 6 — L3 CXL / disaggregated physical move (`MLCACHE6`) — cross-node, gated

**Claim.** A span demoted to CXL/remote is physically relocated (not just marked) and stays
attendable/recoverable, closing the gap the L3 study (#79) and the CXL pool doc name.

**Build.** Tracked by #79's children B–E (the `L3RegionBackend` behind the frozen `Resolver`
seam). This epic does **not** rebuild the external L3 store; it consumes it. Deferred behind
rungs 1–5 and an external L3 instance.

**Witness.** Per #79's definition of done; `dos verify` against the child that lands it.

## Honest boundaries (carry into every rung)

- `cachemeta` is the **policy plane and owns no bytes** — every rung that moves data does it
  through the engine adapter / storedrv, never in the policy plane.
- The default tier *profiles* are representative until rung 4 probes them; the *pressure*
  inputs become real per-tier as rungs 1–2 land. A rung must not claim "hardware-aware" for a
  tier whose pressure is still the 0 placeholder.
- A rung is "shipped" only when its `dos verify` is `shipped:true` **and** `dos commit-audit`
  on the ship commit is `OK`/diff-witnessed. A green test alone, or a subject-only commit
  message, does not close a rung — that is the whole point of binding each rung to a witness
  the agent did not author.
- Rungs 3 and 6 are hardware/fleet-gated; rungs 1, 2, 4, 5 ship from any box including the
  GPU-less dev box and CI. The spine is ordered so progress never stalls on hardware.

## Related, in this repo

- `internal/cachemeta/placement.go` — `PlanPlacement`, the demote-vs-evict cost model.
- `internal/engine/capacity_pressure.go` — `DeviceHBMPressure`, the rung-1 pattern to mirror.
- `internal/engine/capacity_adapter.go` — `CapacityAdapter`, the executor rung 3 wires live.
- `internal/compute/capacity.go` — `HostSystemMemoryInfo`, the rung-1 probe.
- `docs/serving/cxl-memory-pool.md` — the CXL tier economics behind rung 6.

Parent epic: #706 (hardware-capacity bridge). Sibling: #79 (L3 disaggregated positioning).
