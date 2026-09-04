---
title: "Dev-process private boundary: public engine vs private factory"
description: "The source of truth for partitioning fak's public open-core runtime from its private autonomous development process modules, and the architectural contracts that govern their interaction."
---

# Dev-Process Private Boundary

This document defines the architectural boundary between **public `fak`** (the open-source agent kernel and model runtime) and **private `fak-private`** (the proprietary serving product and autonomous development factory).

> **Context:** As detailed in `NOTES-serving-product-economics-and-hardware-architecture-2026-09-03.md` (in `fak-private`), our autonomous development flywheel ("The Factory") is as much a competitive advantage as the kernel itself ("The Engine"). This document establishes the boundary, encapsulation rules, and migration phases for moving development process modules into `fak-private`.
>
> For GPU hardware boundaries, see [`docs/gpu-server-private-boundary.md`](gpu-server-private-boundary.md). For public fleet status, see [`docs/fleet.md`](fleet.md). For multi-repo workspace topology and synchronization mechanics, see [`docs/notes/2026-09-03-dual-repo-workspace-and-safe-sync.md`](notes/2026-09-03-dual-repo-workspace-and-safe-sync.md).

---

## 1. The Core Split: The Engine vs. The Factory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PUBLIC REPOSITORY (fak)                           │
│                            "The Open Source Core"                           │
│                                                                             │
│  - Core Go runtime: cmd/fak, internal/engine, internal/abi, internal/ctxmmu│
│  - Client drop-in proxy & safety gate: fak guard, fak preflight, fak serve  │
│  - Exported SDK & Public API: pkg/abi, pkg/scorecard, pkg/fakclient         │
│  - Open-source benchmark runners: livecodebench, radixbench                 │
│  - Standard diagnostic probes: cfgprobe, gpucheck                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Imports via go.work / pkg/*
                                       │ CLI process boundary via fak dev
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRIVATE REPOSITORY (fak-private)                      │
│                      "The Commercial Serving & Factory"                     │
│                                                                             │
│  1. THE SERVING PRODUCT                                                     │
│     - Multi-tenant KV-MMU memory multiplexer & scheduler                    │
│     - Speculative branch rollback & zero-recompute engine                   │
│     - Multi-tenant OpenAI / Anthropic-compatible metering proxy             │
│     - Hardware tiering: GDS / AMD Smart Access Storage daemons              │
│                                                                             │
│  2. THE AUTONOMOUS FACTORY ("How We Develop fak")                           │
│     - Autonomous wave dispatchers & queueing: platform/dispatch/            │
│     - Paced contract leases: refs/fak/locks/contract-*                      │
│     - Metric scorecards & regression ratchets: platform/scorecards/         │
│     - Session watchdogs & self-healing reapers: platform/watchdogs/         │
│     - Hardware control bridge: tools/dgxbridge, dgxsh                       │
│     - Durable agent memory archives & sync hooks                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Public Tree (`fak`)

The public repository owns the open-source kernel, developer SDKs, runtime capabilities, and verifiable public benchmarks.

### What stays in public `fak`:
1. **Core Kernel Runtime**:
   - Model serving, tokenizers, KV-MMU, and execution engines (`internal/engine`, `internal/model`, `internal/compute`, `internal/kvmmu`).
   - Context MMU, capability floors, policy evaluation, and tool adjudication (`internal/adjudicator`, `internal/policy`, `internal/ctxmmu`).
   - The primary CLI entry point (`cmd/fak`).
2. **Exported Public Seam (`pkg/`)**:
   - `pkg/abi`: Frozen ABI verdicts, capabilities, and tool envelopes.
   - `pkg/scorecard`: Public scorecard primitives, grading, and rendering.
   - `pkg/fakclient`: Agent-RPC client and event streaming wrappers.
   - `pkg/harnesskit` & `pkg/harnesssidecar`: Public harness integration abstractions.
3. **Open Benchmarks & Probes**:
   - Standard, reproducible benchmark suites (`livecodebench`, `radixbench`, `tokensim`).
   - Hardware detection and readiness probes (`cfgprobe`, `gpucheck`).
4. **Transition Process Boundary**:
   - `cmd/fak-dev`: Dedicated CLI for repo maintainers during migration.
   - `cmd/fak/dev.go`: Compatibility process handoff redirecting `fak dev <verb>` to `fak-dev` as a separate child process.

---

## 3. Private Tree (`fak-private`)

The private repository owns commercial serving components and the autonomous development process engine that powers continuous development sweeps.

### What lives in `fak-private`:
1. **The Autonomous Factory Modules**:
   - **`platform/dispatch/`**: Wave dispatchers, ticket DAG resolution, and worker life-cycle orchestrators (migrating `tools/issue_dispatch.py`, `tools/worker_worktree.py`, `cmd/dispatchworker`).
   - **Contract Leases**: Distributed, CAS-guarded contract leases (`refs/fak/locks/contract-<id>`) governing token/IO pacing across autonomous agents.
   - **`platform/scorecards/`**: Autonomous development scorecards, control panes, and regression ratchets (migrating `tools/scorecard_control_pane.py` and proprietary scorecard scripts).
   - **`platform/watchdogs/`**: Fleet and session watchdogs, background autohealers, and reapers (migrating `tools/fleet_resume_watchdog.*` and session monitor daemons).
   - **Hardware Control**: Private lab bridges (`tools/dgxbridge`), remote cluster fabrics, and unredacted GPU execution runbooks.
   - **Durable Memory Store**: Historical agent memory archives (`agent-memory/fak/`) and sync infrastructure.

---

## 4. Architectural Contracts Across the Seam

### 4.1 Go Workspace & Module Encapsulation Rule
The Go compiler enforces that packages under `internal/` within module `github.com/anthony-chaudhary/fak` **cannot be imported** by an external module such as `github.com/anthony-chaudhary/fak-private`.

Therefore, the Go-level boundary adheres strictly to:
- **`go.work` Workspace Setup**: `fak-private` contains a root `go.work` referencing `.` and `../fak`, enabling unified local resolution.
- **`pkg/` Export Rule**: Any Go types or abstractions required by both the public runtime and private factory must live in `fak/pkg/*` (e.g., `pkg/abi`, `pkg/scorecard`, `pkg/fakclient`), NEVER in `internal/*`.
- **Zero Runtime Import of Private Code**: Public `fak` must never reference or import any module or package in `fak-private`.

### 4.2 Process Boundary Seam (`fak dev`)
For complex maintainer operations, `fak` and `fak-private` decouple through the CLI process boundary:
- `fak dev <verb>` in runtime `fak` does not compile dev tooling into `cmd/fak`. Instead, it locates and executes `fak-dev` as a subprocess via standard I/O streams.
- Private factory tools in `fak-private` invoke public `fak` or `fak-dev` via CLI subcommands with `--json` outputs, ensuring complete binary separation.

### 4.3 Ref-Based and JSON Storage Seams
- **Git Refs**: Autonomous workers coordinate state through namespaced git references (`refs/fak/locks/contract-*` for contracts, `refs/fak/locks/<lane>` for file trees).
- **Scrubbed JSON**: Public status readers (e.g., `fleetctl`) consume scrubbed status manifests produced across the boundary without exposing raw hostnames, private IPs, or API keys.

---

## 5. Asymmetric Leak Gate & Scrub Policy

Because `fak` is open source and `fak-private` contains proprietary flywheels and credentials:
- **Git Hooks**: `tools/check_committed_files.py` and `tools/githooks/pre-commit` run `tools/scrub_public_copy.py --audit-staged` on every commit in `fak`.
- **Forbidden Needles**: GPU server hostnames, Slack tokens, private paths, private repo URLs, and raw execution logs are blocked from entering `fak`.
- **Boundary Refusal**: Any attempt to commit private platform code directly into the public `internal/` tree is blocked under `FILE_ADMISSION`.

---

## 6. Migration Roadmap

The migration of dev process modules to `fak-private` proceeds in five sequential phases:

| Phase | Milestone | Deliverables |
|---|---|---|
| **Phase 1** | **Groundwork & Boundary Contract** *(Current)* | `docs/dev-process-private-boundary.md`, `go.work` integration in `fak-private`, clean workspace compilation, issue #11166. |
| **Phase 2** | **Private Platform Scaffolding** | Scaffold `platform/dispatch/`, `platform/scorecards/`, and `platform/watchdogs/` in `fak-private` with clean `pkg/*` imports. |
| **Phase 3** | **Autonomous Dispatch & Contract Leases** | Implement contract lease queueing (`refs/fak/locks/contract-*`) and dispatch workers in `fak-private/platform/dispatch/`. |
| **Phase 4** | **Scorecards & Watchdogs Relocation** | Migrate scorecard control panes and session recovery watchdogs to `fak-private/platform/`. |
| **Phase 5** | **Public Deprecation & Handoff** | Deprecate legacy in-tree dev scripts in `fak/tools/`, leaving thin process handoffs. |
