# feat(residency): tensor-class residency mapping with measured NVMe row-read latency floor (fak ggufload)

GitHub issue: [#14094](https://github.com/anthony-chaudhary/fak/issues/14094).

Process cause: none

```routing
lane: gsq-rco-followons
paths: ["internal/ggufload/gguf_residency_linux.go", "internal/ggufload/gguf_mmap.go", "internal/ggufload/deepseek41_engram_serve.go", "docs/tickets/gsq-rco-followons-20261028/INDEX.md"]
expected_steps: 4
```

## Current state

Verified against the three seam files (SW-VERIFIED source reads, 2026-10-28):

- `gguf_mmap.go` — opt-in mmap seam (`FAK_GGUF_MMAP`, Tier-3 SSD-offload foundation #2726/#2722). `openShardReader(path, mmap bool)` picks ONE reader per shard: read-only memory map or historical `os.Open` + `ReadAt`; `retainShardReader` upgrades the parse-only open; selection is one process-wide boolean — no tensor-class dimension anywhere in the path.
- `gguf_residency_linux.go` (linux build tag) — `fileResidentBytes` maps a file read-only and counts mincore-resident pages: a post-hoc census of how much of a mapped shard became resident. A measurement primitive, not a planner input.
- `deepseek41_engram_serve.go` — shipped DS4.1 Engram serve-load attachment (public #13662, leaf of #12640): declared-layer n-gram tables opened through row sources behind verified artifact bindings, bounded resident row cache (`v41EngramServeBudgetRows = 4`). The ONLY class-specific residency decision in the tree today — hardcoded to the Engram stage, not a general classification.

Baseline: every non-Engram weight is one tier chosen at shard-open time. No per-class split yet; the serving planner has no vocabulary to express "lookup-only pageable, matmul resident".

## Problem frame

- Centrality: Enabling — consumer-envelope serving of 125B-class MoE requires weight-tiering by tensor class; without the mapping the planner cannot even name the two tiers, so the serving envelope does not exist yet.
- P1 Context: advanced — extends the shipped mmap seam, cites the shipped Engram attachment as prior art; no new transport, authority, or policy.
- P2 Net value: advanced — the 'pageable' class gets an evidence-based admission (measured NVMe row-read floor) instead of an assumption; the design becomes a contract the planner lane consumes.
- P3 Adaptation: preserved — INSPIRE-ONLY clean-room design input, NOT a port; reuses the `openShardReader`/`mmapReaderAt` transport and the Engram row-source + verified-binding idiom.
- P4 Operations: preserved — design + host-measurement ticket; no serving-path behavior change.

## Core through-line

Step 1: the mapping table — tensor class → residency tier (lookup-only/pageable: mmap, fault-on-read; matmul-resident: VRAM/RAM), class predicates expressed in terms the existing seam can evaluate (row-addressed sparse lookup vs dense matmul weight; per-token access pattern: one-row-per-token vs every-token-every-layer). Step 2: the measurement — NVMe random row-read latency floor on this desktop, sealed with a timestamp. Step 3: classification predicate stubs in `ggufload` admitting the 'pageable' class ONLY against the measured floor (numeric threshold TENTATIVE until the record lands). Step 4: campaign INDEX + doc/table constant. Extend the CLASSIFICATION; don't rebuild the transport.

## Scope

In:
- Design: the tensor-class → residency mapping table (two tiers; predicates; worked example on the studied 51.2B-param n-gram PLE table — shard 2, fixed IQ4_NL, 28.8 GB, one-row-per-token sparse lookup).
- Measurement: NVMe random row-read latency floor on this host — page-cache-cold reads, row sizes matching PLE-table rows (model config when derivable; else a bracket set spanning plausible n-gram row sizes), >=10k reads warm-up excluded, p50/p95/p99, recorded with sealing timestamp and host identity. Probe in native Go (test or small verb); no loose scripts.
- Admission rule: 'pageable' admitted ONLY if the measured floor clears a named budget as a function of the decode tok/s target — budget(T) = alpha · (1/T_target), alpha = share of the inter-token interval one row-read may consume (alpha TENTATIVE, start 0.25; numeric threshold TENTATIVE until the record lands).
- Stubs: classification predicate functions in `ggufload` consuming the table; unit-tested against a fixture inventory.
- Docs: campaign INDEX (`docs/tickets/gsq-rco-followons-20261028/INDEX.md`) + the mapping table as a doc or table constant.

Out:
- No tiering implementation in the serving planner — named follow-on, consuming lane qwen38-flash-next bringup.
- No changes to the DS4.1 Engram path (shipped behavior; cited as prior art, not modified).
- No shard download, no HW run, no perf claim.

## Gold-plating boundary

One mapping table, one sealed measurement record, one predicate-stub site. No planner wiring, no new transport, no autotune ladder, no multi-host sweep.

## Metric moved

- Metric: consumer-envelope memory footprint for 125B-class MoE serving (enabler metric — without it the serving envelope does not exist).
- Baseline: current single-class residency — all weights one tier chosen at shard-open time (`FAK_GGUF_MMAP` boolean in `gguf_mmap.go`); no per-class split yet (verified against the three seam files; see Current state).
- Direction: down (resident footprint) — ENABLED by this ticket, not achieved; the achievement belongs to the planner follow-on.
- Reader: mapping table + sealed latency record now; planner resident-footprint output later.

## Dependencies

- Blocked by: nothing — both deliverables are host-testable NOW on this desktop.
- Blocks: the serving-planner tiering follow-on and the qwen38-flash-next bringup lane (consuming lane; not started yet).

## Parent context

Denominator parent: #3613 (GSQ-RCO follow-on campaign tracker, lane
gsq-rco-followons).

Design input from the upstream release: HF repo `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (GSQ arXiv 2604.18556, RCO arXiv 2605.00649) — INSPIRE-ONLY clean-room input, not a port. Live public prior art: the shipped DS4.1 Engram attachment (public #13662, leaf of #12640) opens its n-gram tables mmap-only by design. Campaign index: `docs/tickets/gsq-rco-followons-20261028/INDEX.md` (created alongside).

## Why this is next

Weight-tiering by tensor class is the INSPIRE-ONLY design input named by the upstream GSQ-RCO release analysis; its next-checkable step is "Define tensor-class -> residency mapping; verify row-read latency floor on NVMe before committing to the pattern". Both halves are host-cheap and unblocked; the consuming lane (qwen38-flash-next bringup) is not started yet, so landing design + floor FIRST gives the planner work a settled contract instead of mid-flight re-derivation.

## Evidence

- [SW-VERIFIED] `gguf_mmap.go`: `openShardReader(path, mmap bool)` — one boolean per shard; `retainShardReader` upgrades the parse-only open; no class dimension in either path.
- [SW-VERIFIED] `gguf_residency_linux.go`: `fileResidentBytes` (mincore over a read-only map, linux only) — measures residency after the fact; usable as a post-hoc witness for the 'pageable' tier, not a planner input.
- [SW-VERIFIED] `deepseek41_engram_serve.go`: Engram attachment with `v41EngramServeBudgetRows = 4` bounded row cache, row-source + verified-binding admission (public #13662) — prior art for class-specific residency, hardcoded to one stage.
- [source-only] HF repo `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (allocation manifests fetched 2026-10-28): shard 2 = `per_layer_token_embd`, 51.2B params fixed IQ4_NL, 28.8 GB, one-row-per-token sparse lookup; card recommends `-lm mmap --lazy-mode on` so shard 2 stays mmap'd on SSD and resident footprint = matmul shard only.
- [source-only] Upstream card metadata for the sibling DeepSeek-V4.1-Flash-1017 GSQ-RCO build: ~110.60 GB Engram table shipped mmap-only — independent corroboration across a second artifact (no local reproduction).

## Next checkable step

Run the latency-floor probe on this host: write the scratch extent, take page-cache-cold random row-reads at the PLE row-size set (>=10k reads), record p50/p95/p99 + sealing timestamp + host identity into `_scratch/gsq-rco-followons/nvme-row-read-floor-YYYYMMDD.json`, then compare the p99 against budget(T) for the named decode target and mark the 'pageable' class admitted or refused, with the number in the record.

## Definition of done

- [ ] [SW] Mapping table landed (doc or table constant, TENTATIVE name) with the two tiers and class predicates evaluable by the existing seam; worked example on the studied PLE table.
- [ ] [SW] Sealed latency record: methodology (cold-read technique, row sizes, iteration count), p50/p95/p99, sealing timestamp, host identity.
- [ ] [SW] Admission rule: 'pageable' admitted ONLY against budget(T) = alpha · (1/T_target); alpha and the numeric threshold marked TENTATIVE until the record lands.
- [ ] [SW] Predicate stubs in `ggufload` compile and pass a fixture-inventory unit test (lookup-only vs matmul assignment).
- [ ] [SW] DS4.1 Engram path untouched — diff audit shows zero behavioral hunks; cited, not modified.
- [ ] [SW] Campaign INDEX created with this ticket's status; gofmt/vet clean.

## Working spine

Design first (table + predicates checked against the three seam files), then the host measurement (cheap, deterministic, no approval needed), then stubs + hermetic test, then INDEX + doc. No hardware, no downloads, no serving-path change at any step.

## Acceptance gate

The sealed record exists, the mapping table carries a named budget function for the 'pageable' class, and the predicate stubs are green under the focused test below.

## Witness

- `go test ./internal/ggufload/ -run TestResidencyClass -count=1` (test name TENTATIVE until the stub lands)
- `_scratch/gsq-rco-followons/nvme-row-read-floor-YYYYMMDD.json` (sealed; timestamp in the record)
- Diff audit: zero hunks touching DS4.1 Engram behavior

## Likely files

- internal/ggufload/gguf_residency_linux.go
- internal/ggufload/gguf_mmap.go
- internal/ggufload/deepseek41_engram_serve.go
- internal/ggufload/<residency-classification doc or table constant> (TENTATIVE name, e.g. `residency_class.go`)
- docs/tickets/gsq-rco-followons-20261028/INDEX.md

## Closure binding

The closing commit cites this ticket, its GitHub mirror, and `(fak ggufload)`.

## Work estimate

Estimate: 2 points (mapping table + latency-floor measurement record + classification predicate stubs).

## Overall completion contribution

Contribution: 2/8 points of the gsq-rco-followons set (set total ESTIMATED; siblings tracked in `docs/tickets/gsq-rco-followons-20261028/INDEX.md`, denominator parent #3613).

## Completion standard

Research (design + measurement deliverable; implementation is a named follow-on).

## Target operating envelope

- test packages: n/a (research ticket; predicate-stub tests land with the stubs, not at filing)
- sealed latency records: n/a (measurement record lands in step 2, not at filing)

## Witnessed operating envelope

- test packages: 0 packages
- sealed latency records: 0 measurements

github_issue: anthony-chaudhary/fak#14094

github_issue: anthony-chaudhary/fak#14094
