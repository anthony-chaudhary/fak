# Bench: evidence mechanisms worth adapting, not a benchmark contract to import

**Status:** pinned exhaustive study complete; three clean-room follow-ons filed  
**Observed:** 2026-08-29  
**Source:** `anthony-chaudhary/Benchmark@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`

## Verdict

Bench has three mechanisms worth adapting into FAK: watcher-safe sequence-addressed evidence snapshots, receipted workload-event/telemetry correlation with rejected-row accounting, and parser-completeness sidecars bound to claim eligibility. FAK already has stronger or equivalent contracts for typed metric mapping, sinks, raw artifact integrity, benchmark preflight, publication admission, and execution/export separation, so those are confirmations rather than new work. Bench does **not** prove a universal net-true matched benchmark envelope or an integrated semantic-quality gate.

Because `pyproject.toml:71` declares MIT but the pinned tree has no root `LICENSE` and GitHub reports no detected license, every accepted borrow is a clean-room conceptual adaptation. No Bench source is copied.

## Value frame

- **Centrality:** Enabling.
- **For:** FAK maintainers building benchmark, campaign, ingestion, and evidence surfaces.
- **Problem:** useful mechanisms in a sibling benchmark repository were not mapped to FAK with pinned provenance, overlap checks, and adoption risk.
- **Today:** knowledge was implicit in a local checkout and could be lost, duplicated, or overstated.
- **Better because:** one durable study separates PRESENT, PARTIAL, and excluded mechanisms and gives each surviving gap an independently dispatchable issue.
- **Witness:** this note, the exhaustive inventory, immutable receipt `study_36e9161928991db39b7aebd463809d8ca37e9ac9d66ba074888c4bd45edd6882`, and issues #10302-#10304.

The P1-P4 checks resolve as: **P1** benchmark/evidence maintainers are the user; **P2** duplicate or incomplete evidence machinery is the problem; **P3** ad hoc source inspection and transcript-only notes are the next-best alternative; **P4** pinned source anchors, exact FAK witnesses, explicit exclusions, and small tracked follow-ons make the result more durable and falsifiable.

## Source identity and license posture

- Source checkout: operator-local; exact local location retained only in the private companion repository.
- Canonical repository: `https://github.com/anthony-chaudhary/Benchmark`.
- Pinned revision: `b60f2f05e23ec38ae4f50ca19717c2f615ed419a`; the checkout was clean on `main...origin/main` when inspected.
- Repository identity at observation: 842 commits, 309 tags, 14 closed issues, zero open issues, zero pull requests, discussions disabled, zero stars, zero forks, and last push at `2026-08-19T00:33:36Z`.
- License evidence conflicts in sufficiency, not text: `pyproject.toml:71` says `license = "MIT"`, but there is no root license file and GitHub license metadata is null. The conservative disposition is clean-room conceptual adaptation only.

## Scope, method, and completeness

The pass covered committed source and documentation, architecture, runtime paths, tests and fixtures, history/tags/releases, forge state, roadmap and unfinished-work-marker evidence, license provenance, FAK self-query, candidate disposition, deduplication, issue filing, and durable receipt registration. Architecture and methodology reads were delegated as bounded sidecars and independently reconciled against the pinned checkout by the coordinator.

The mechanical inventory at `docs/research/inventory/anthony-chaudhary-benchmark.json` reports 9,516 files, 986 directories, 4,862,125,342 bytes, 2,758 runtime files, 1,102 test files, 1,282 documentation files, and 1,847,800 text lines across 44 immediate subsystem groups. Those totals are a **mechanical local-tree denominator, not project-owned source totals**: the checkout includes `.venv-test` and other broad material, while enumerated dependency/cache/control directories are skipped. Forge state, FAK self-query, the candidate matrix, and issue tracking are attached through the monitored-repository evidence rather than inferred from the filesystem census.

Honest boundaries: this study did not execute private production deployments, authenticate artifact producers, audit every dependency license, or establish comparative benchmark effectiveness. Bench has no public adoption signal, so decisions rest on pinned source and tests rather than popularity.

## Architecture and control/data flow

```text
CLI
  -> command handler
    -> typed config/models
      -> CapacityTest orchestration
        -> server managers + workload/telemetry phases
          -> raw and structured results
            -> loaders, sinks, charts, reports, publishing
```

- Parser construction and dispatch live at `benchmark.py:493-519,650-768@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- Single runs (`commands/_run.py:310-545@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`) and sweeps (`commands/_sweep.py:75-465@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`) converge on `CapacityTest`.
- Campaign expansion and typed campaign models remain separate at `commands/_campaign.py:96-312@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` and `models/campaign.py:95-149@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- `CapacityTest` composes phase, metrics, warmup, classification, telemetry, raw-capture, and display behavior around `engine/_orchestrator.py:54-68@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`; its lifecycle is connect, launch, readiness, workload/telemetry, graceful stop, and close (`engine/_orchestrator.py:683-735@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`).
- Server ownership is abstracted by `server/_base.py:19-112@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` and implemented across `server/_managers.py:209-1655@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- Output ownership stays separate through `sinks/base.py:25-67@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, `sinks/__init__.py:1-15@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, loaders, charts, reports, export, and publishing.

The useful architectural lesson is composition: execution has one orchestrator, while server lifecycle, evidence transport, analysis, rendering, and publication remain replaceable seams.

## Methodology strengths

- The random seed is recorded at `engine/_orchestrator.py:84@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- Warmup is explicit and excluded from measured loads at `engine/_orchestrator.py:146,710,715@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`; loaders ignore `is_warmup` rows.
- Wall-clock measurement starts before setup/loading at `engine/_orchestrator.py:547@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, avoiding a narrow request-only clock.
- Fairness analysis is isolated in `loader/_fairness.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- Replay has a dedicated implementation and tests: `monitoring/replay_trace.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` and `tests/test_replay_trace.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- Raw capture and integrity are explicit in `engine/_raw_archiver.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` with `tests/test_raw_metrics_archiver.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- Raw parser auditing is isolated in `engine/_raw_auditor.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` with `tests/test_raw_metrics_auditor.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- Typed preflight lives in `engine/preflight.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` with `tests/test_preflight.py@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.

## Methodology limits

- No universal net-true lifecycle contract proves that every reported number includes setup, warmup, recovery, verification, teardown, export, and publishing.
- No machine-enforced matched comparison envelope binds model, prompts, seed, concurrency, generation settings, hardware, and quality.
- Campaign warmup discipline is configurable, so misuse remains possible.
- There is no integrated semantic-quality gate.
- Some evidence paths fail soft; replay skips malformed rows rather than making loss part of an admission receipt.
- Nearest-time joins are ambiguous under clock skew, concurrency, duplicate timestamps, and reordering.
- Publication admission is Confluence-specific and chiefly metadata-oriented.
- Hashes detect mutation but neither authenticate producers nor prove upstream completeness.
- Parser auditing checks syntax/family coverage, not semantic plausibility.

These limits are why the study borrows bounded evidence mechanisms rather than treating Bench as a benchmark-authority replacement.

## Candidate dispositions

| Borrow | Source at pinned revision | Axis | Why it matters in Bench's worldview | FAK witness / status | Disposition | Issue |
|---|---|---|---|---|---|---|
| Typed external metric mapping | `monitoring/schema.py:16-70@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | Metric semantics | Normalizes heterogeneous monitor names before downstream analysis. | `internal/cacheobs/metricspec.go:77-169` already types metric specs and coverage; **PRESENT**. | Exclude; no generic remap abstraction. | - |
| Pluggable sink contract | `sinks/base.py:25-67@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | Evidence transport | Keeps execution independent from console, file, and future destinations. | `internal/findingsink/findingsink.go:137-168` plus `internal/auditreceipt`; **PRESENT**. | Exclude; architectural confirmation only. | - |
| Watcher-safe sequence snapshots | `sinks/snapshot_dir.py:37-75@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | Live evidence publication | A stable latest pointer plus immutable sequence files lets readers observe progress without racing rewrites. | Existing findings output lacks the complete benchmark snapshot contract; **PARTIAL**. | Clean-room optional module. | [#10302](https://github.com/anthony-chaudhary/fak/issues/10302) |
| Workload-event/telemetry correlation receipt | `monitoring/replay_trace.py:44-179@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | External telemetry attribution | Aligns device observations with workload events for replay and diagnosis. | Correlation seams exist, but ingestion lacks a bounded receipt with rejected-row debt; **PARTIAL**. | Clean-room optional module. | [#10303](https://github.com/anthony-chaudhary/fak/issues/10303) |
| Raw metrics archiver | `engine/_raw_archiver.py:55-235@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | Artifact integrity | Preserves source-separated raw evidence with sequence, digest, and byte count. | `internal/benchmarkartifact` and `internal/armbench/manifest.go`; **PRESENT**. | Exclude; FAK already owns the integrity envelope. | - |
| Raw parser-completeness receipt | `engine/_raw_auditor.py:42-205@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | Claim eligibility | Makes parser-family coverage visible before trusting derived metrics. | `internal/benchcatalog/ingest.go:168-314` and `internal/benchauthority/validate.go:133-178` do not emit the proposed sidecar; **PARTIAL**. | Clean-room optional module. | [#10304](https://github.com/anthony-chaudhary/fak/issues/10304) |
| Typed benchmark preflight | `engine/preflight.py:45-203@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | Admission | Refuses unsupported or incoherent runs before expensive execution. | `internal/bench/ctxadmission.go:341-490` and `internal/bench/setupparity.go:510-517`; **PRESENT**. | Exclude; FAK's admission contract is stronger on this axis. | - |
| Publish admission | `publish/_gate.py:42-160@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | Evidence publication | Prevents incomplete metadata from reaching the reporting destination. | `internal/benchauthority/validate.go:23-178`; **PRESENT**. | Exclude; avoid a Confluence-shaped generic gate. | - |
| Campaign/export separation | `commands/_campaign.py:96-312@b60f2f05e23ec38ae4f50ca19717c2f615ed419a` | Ownership boundaries | Lets campaign expansion, run execution, export, reporting, and publication evolve independently. | `cmd/fak/benchpost.go:32-318` and `internal/benchcatalog/ingest.go:253-314`; **PRESENT**. | Exclude; retain as architectural confirmation. | - |

## Accepted borrows

1. **#10302 - watcher-safe sequence-addressed benchmark evidence snapshots.** Preserve immutable sequence identity, an atomic/stable reader target, and concurrency tests; do not generalize it into a new sink framework.
2. **#10303 - external workload-event/telemetry correlation receipts.** Make accepted, rejected, ambiguous, skewed, and unmatched rows explicit; a nearest-time match without rejected-row accounting is not sufficient evidence.
3. **#10304 - raw parser-completeness receipts.** Emit family/row coverage and bind incompleteness to benchmark claim eligibility; syntax coverage must not be represented as semantic plausibility.

Each child is open, labeled `class:dev` and `gen/next`, cites parent #10263, carries the pinned source/license constraint and immutable study receipt, and names a first test.

## Explicit exclusions

- No matched-envelope issue is filed from this study: FAK already has substantial benchmark artifact, setup-parity, admission, and authority contracts, and no newly verified exact gap survived deduplication.
- No generic atomic-write abstraction, generic sink framework, generic policy layer, raw-archiver duplicate, or Confluence-shaped publish gate is filed.
- Bench's server managers and orchestration are not imported wholesale; the useful result is seam confirmation, not a second execution stack.
- No source-level integration is allowed under the observed license posture.

## Risks and refresh triggers

Refresh this study if Bench gains a root license, GitHub begins detecting a license, its pinned mechanisms materially change, a release or public adoption signal appears, or FAK lands #10302-#10304 and needs overlap reconciliation. Re-open methodology review if Bench adds a machine-enforced matched envelope, net-true lifecycle accounting, authenticated producer provenance, or semantic-quality admission.

Primary risks in any adaptation are overstating timestamp correlation, treating parser coverage as semantic validity, publishing a mutable latest file without immutable sequence identity, and allowing optional evidence failures to remain invisible. The child issues preserve those failure modes as explicit receipt fields and tests.

## Completion evidence

- Pinned source and clean checkout: `anthony-chaudhary/Benchmark@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`.
- Exhaustive mechanical inventory: `docs/research/inventory/anthony-chaudhary-benchmark.json`.
- Immutable receipt: `study_36e9161928991db39b7aebd463809d8ca37e9ac9d66ba074888c4bd45edd6882`, read back through `fak study search --limit 5 "Bench benchmark evidence"`.
- Durable tracker: [#10263](https://github.com/anthony-chaudhary/fak/issues/10263).
- Independently dispatchable follow-ons: [#10302](https://github.com/anthony-chaudhary/fak/issues/10302), [#10303](https://github.com/anthony-chaudhary/fak/issues/10303), and [#10304](https://github.com/anthony-chaudhary/fak/issues/10304).
- Candidate decision denominator: nine mechanisms; three PARTIAL adaptations filed, six PRESENT mechanisms excluded.

## Companions

- Workflow: `field-borrow`
- Parent: #10263
- Receipt: `study_36e9161928991db39b7aebd463809d8ca37e9ac9d66ba074888c4bd45edd6882`
- Inventory: `docs/research/inventory/anthony-chaudhary-benchmark.json`
