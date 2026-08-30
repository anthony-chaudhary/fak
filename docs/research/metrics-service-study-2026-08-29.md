# metrics-service study — aligned evidence and operator observability

**Observed:** 2026-08-29  
**Tracker:** #10287  
**Receipt:** `study_7079004a7801dd62685c9f5ff323f40650d4cf2c52daf83a96d2c76103501f93`  
**Source:** `anthony-chaudhary/metrics-service@df58aad21de603992fc7b4900ac814610092c6c3` (`v0.21.16`)  
**Disposition:** study and clean-room adaptation only; no project license grant was found at the pinned revision.

## Verdict

`metrics-service` is a coherent Go observability runtime whose useful spine is:

> JSON configuration → backend adapters → deadline-aligned collection → normalized and validated snapshot → concurrent sink fan-out → bounded query and operator surfaces.

The study retained three independent follow-ons for FAK:

1. **Aligned fleet snapshots with typed partial outcomes** — #10312.
2. **Typed component readiness with copyable recovery actions** — #10314.
3. **Registry-driven dashboard and alert drift auditing** — #10313.

Other mechanisms were marked present, watch-only, or excluded rather than filing a broad adoption epic. In particular, direct Grafana push is excluded as a default because it expands credential and external-write authority, and raw source code cannot be copied because the repository carries no license grant.

## Source identity and evidence quality

- Pinned commit: `df58aad21de603992fc7b4900ac814610092c6c3`; immutable tree `ecbe7a5eaae575df22c3c18f9ad2ec639b069aa7`.
- Annotated tag: `v0.21.16`; the tag has no cryptographic signature.
- Repository: private, 83 commits, 37 reachable tags, one reported Git author across all commits.
- Remote metadata observed on 2026-08-29: no GitHub issues, three historical pull requests, discussions disabled, GitHub Releases stopping at `v0.19.0` while tags continue through `v0.21.16`.
- Tree size: 273 tracked files, including 154 Go files, 52 `_test.go` files, and 9 tracked testdata fixtures.
- Local verification: `go test ./...` passed at the pinned checkout.
- Working-tree isolation: pre-existing untracked `_make_value_pdf.py` and `metrics-service-value-overview.pdf` were excluded and left untouched.

The test corpus spans unit, property, fuzz, contract, smoke, and pipeline E2E styles. High-value witnesses include `cmd/metrics-service/pipeline_e2e_test.go:25@df58aad…`, `internal/api/handler_diagnostics_contract_test.go:32,100@df58aad…`, and the directory-tail cases at `internal/backend/dirtail/dirtail_test.go:133-281@df58aad…`.

## Architecture studied

### Configuration and composition

`internal/config/config.go:250-335@df58aad…` constructs defaults and overlays JSON, while `internal/config/config.go:377-430@df58aad…` separates strict errors from warnings. `cmd/metrics-service/main.go:20-46@df58aad…` is an auditable composition root for configuration, process cancellation, pipeline, run manager, observer, Grafana, HTTP serving, and shutdown.

The mechanism is useful, but the loader does not visibly use unknown-field rejection. FAK should retain its stronger validation posture rather than copy that weakness.

### Collection, partial results, and publication

`internal/collector/alignment.go:20-67@df58aad…` gives matched pull targets one collection window and launches them concurrently. `internal/collector/collector_test.go:142-199@df58aad…` distinguishes partial backend failure from total failure. `internal/collector/collector.go:339-370@df58aad…` validates once before concurrent sink publication, waits for sink attempts, then removes rich raw Prometheus families before long-lived storage.

This yields three transferable principles:

- compare distributed evidence from one logical observation window;
- retain independently successful evidence when another source fails;
- validate the canonical object once at the publication seam, then compact rich transient detail after derivation.

### Storage and lifecycle

`internal/sink/memory/store.go:15-82@df58aad…` is a fixed-capacity chronological ring. The run lifetime is hierarchical: process context, run-duration context, then per-scrape timeout at `internal/run/run.go:113,223@df58aad…`. `cmd/metrics-service/server.go:188-216@df58aad…` bounds HTTP shutdown and closes sinks afterward.

The ring is cheap and predictable, but it relies on an immutability-after-publication rule because snapshots are stored by pointer. FAK should not generalize that into a single cross-kernel event store without preserving each receipt domain's existing ownership, provenance, and retention contract.

### Operator surface

`internal/ui/page_status.go:20-80,181-248,279-291@df58aad…` aggregates uptime, sinks, backend state, schemas, mappings, runs, diagnostics, and compact error details. `internal/ui/ui.go:119-155@df58aad…` attaches diagnostics and masks secrets. `internal/ui/sse.go:12-90@df58aad…` emits a deliberately bounded live payload only when the snapshot timestamp changes.

The product worldview is strong: native status should answer what is working, what is degraded, what workload is affected, and what detail remains available without requiring Grafana. Its main evidence gap is visual: no `_test.go` files exist under `internal/ui/`, so FAK's adaptation requires a captured render witness rather than inheriting that gap.

### Metrics, dashboards, and drift

`internal/sink/prometheus/prometheus.go:20-176@df58aad…` owns stable canonical metrics while `internal/sink/prometheus/raw.go:11-196@df58aad…` optionally preserves backend-specific families. `internal/grafana/generate.go:20-48,206-460@df58aad…` derives dashboards from schema; `internal/grafana/audit.go:15-89@df58aad…` finds missing coverage and deprecated references; `internal/grafana/generate_test.go:93-224@df58aad…` exercises real-schema generation and real-dashboard audit.

The canonical/raw split helps integration velocity, but the raw path lacks an obvious cardinality budget, family allowlist, or label-value cap. FAK should default to canonical metrics and keep any forensic path bounded, scrubbed, provenance-typed, and explicitly enabled.

## Candidate decisions

### 1. Aligned fleet observations with typed partial outcomes — PARTIAL / DEFAULT

**For:** operators and benchmark authors comparing a fleet at one logical instant.  
**Problem:** serial probes mix time windows and a single aggregate error obscures successful peers.  
**Today:** `internal/metrics/device_spine_scrape.go:122-175@7ce62aaf2` scrapes peers serially; `internal/metrics/device_spine_federate.go:55-72@7ce62aaf2` federates available samples without an explicit completeness contract.  
**Better because:** one bounded window makes evidence comparable and typed outcomes preserve independent successes.  
**Witness:** tests for shared deadline/timestamp, mixed success, total failure, cancellation, concurrency bound, and stable ordering.

- **Centrality:** Core.
- **P1 managed context:** neutral.
- **P2 net-true efficiency:** measure wall time plus request/goroutine overhead.
- **P3 bounded adaptation:** positive; one result seam with caller-owned limits.
- **P4 integrated operations:** positive; fleet diagnosis becomes explicit.
- **Issue:** #10312.

### 2. Component readiness with recovery actions — PARTIAL / DEFAULT

**For:** operators starting or diagnosing fak without an external dashboard.  
**Problem:** process-alive and server-ready do not express which enforcement, receipt, provider, engine, cache, or optional integration component failed.  
**Today:** `internal/serverproduct/contract.go:97-235,311-345@7ce62aaf2` and `internal/serverproduct/encoding.go:16-92@7ce62aaf2` provide a strong authored-spec and ready-receipt contract, but not a general degraded-state matrix.  
**Better because:** critical failure is distinguishable from optional degradation, with exact next actions.  
**Witness:** contract tests plus a captured operator render showing scrubbed states, criticality, last success, retry behavior, and recovery commands.

- **Centrality:** Enabling.
- **P1:** positive when cache/context state is visible without reconstructing setup.
- **P2:** require reduced diagnosis time without a mandatory sidecar.
- **P3:** positive through typed extensible rows.
- **P4:** directly positive; status and recovery stay on the real server path.
- **Issue:** #10314.

### 3. Metric registry and dashboard/alert drift audit — PARTIAL / DEFAULT

**For:** maintainers and operators evolving metrics and visualization together.  
**Problem:** dashboards can reference retired metrics, omit required families, violate cardinality contracts, misuse counters, or lose recovery runbooks.  
**Today:** `internal/metrics/openmetrics.go:13-344@7ce62aaf2` validates and renders OpenMetrics, and `internal/metrics/parity_dashboard.go:198-370@7ce62aaf2` builds one native dashboard, but there is no general registry-to-dashboard/alert audit.  
**Better because:** the operational contract becomes deterministic and release-gated while native operation remains independent from Grafana.  
**Witness:** negative fixtures for each finding class and a real bundled-asset pass.

- **Centrality:** Stewardship.
- **P1:** neutral.
- **P2:** positive if the audit remains cheap and avoids operational rework.
- **P3:** positive; new metrics become bounded registry entries.
- **P4:** directly positive; assets stay coupled to kernel semantics.
- **Issue:** #10313.

### Excluded or watch-only

- **Strict default-overlay config:** PRESENT / EXCLUDE. FAK already has mature typed configuration, preflight, doctor, policy, and refusal/recovery surfaces; the source's unknown-field posture is weaker.
- **Rich transient data then compact storage:** PARTIAL / WATCH. Useful principle, but a generic event store risks conflating established receipt domains.
- **Direct Grafana push:** ABSENT / EXCLUDE as a default. It adds credentials and external mutation. Generation, download, and audit should precede any separately guarded push feature.
- **Full raw backend metrics:** ABSENT / EXCLUDE as an unbounded default. Any forensic path needs family and label allowlists, budgets, expiry, dropped-series counters, and strict secret/private-data exclusion.

## Provenance and licensing gate

A tracked-tree search found no `LICENSE`, `LICENCE`, `NOTICE`, `COPYING`, or `COPYRIGHT`, and GitHub reported no license metadata. Direct dependencies appear to carry upstream licenses in the local module cache, but that does not grant a license to this repository's source.

Therefore:

- no source was copied;
- all follow-ons require clean-room design from behavior and requirements;
- redistribution or incorporation remains blocked without an explicit grant;
- this note is technical evidence, not legal advice.

## Completeness audit

Opened or inventoried:

- `README.md`, overview, design decisions, deployment docs, audit, changelog, version, module files;
- `cmd/metrics-service/` composition, config, pipeline, server, smoke and E2E tests;
- all `internal/` package families, with deep reads in config, collector, run, memory/Prometheus/InfluxDB sinks, API diagnostics, UI, Grafana, ingest, directory-tail, field mapping, schema, observer, and assistant wiring;
- test inventory and fixtures;
- `design/`, `docs/`, `observability/`, `grafana/`, `integration/`, `reference/`, `plans/`, and `pr/` source classes;
- complete commit/tag history, recent release progression, PR/issue/discussion/release metadata, license/vendor/submodule/generated-artifact searches.

Intentionally not treated as current implementation proof:

- old broad plans, `pr/`, `reference/`, and early observability documents when recent code contradicted or superseded them;
- exhaustive panel-by-panel dashboard JSON review;
- live Grafana/InfluxDB/DGX deployment, production soak, or network mutation;
- historical snapshot/archive binaries;
- the two pre-existing untracked target files.

No submodules or `vendor/` tree were present. Swagger JSON/YAML are generated release artifacts per `.claude/skills/release/skill.md:44-54@df58aad…`; they lack a visible generated header. The monitor registry command `fak dev study-monitor --due-days 14` could not complete because a pre-existing row uses unsupported source class `paper_source`; this study does not alter that unrelated schema debt.

## Durable outputs

- Receipt input: `docs/research/inventory/anthony-chaudhary-metrics-service.json`.
- Receipt: `study_7079004a7801dd62685c9f5ff323f40650d4cf2c52daf83a96d2c76103501f93`.
- Monitor row: `docs/research/monitored-repositories.json`.
- Tracker: #10287.
- Follow-ons: #10312, #10313, #10314.