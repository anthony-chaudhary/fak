---
title: "Public-harness vs private-factory contract and isolation rules"
description: "Architectural specification governing module encapsulation, secret quarantine, execution models, and artifact distribution across public fak and private fak-private repositories."
---

# Public-Harness vs. Private-Factory Contract and Isolation Rules

> **Status:** Approved Architectural Specification  
> **Issue:** #11611  
> **Scope:** Public SDK (`pkg/harnesskit`), Architecture Invariants (`internal/architest`), Dual-Repo Workspace Boundary  
> **Date:** September 2026  
> **Authors:** fak Architecture & Infrastructure Guild  

---

## 1. Executive Summary & Problem Statement

The **fak** ecosystem is architected across two complementary, decoupled repositories:

1. **Public `fak` (`github.com/anthony-chaudhary/fak`)**: The open-source core containing the model inference engine, context-MMU, drop-in safety proxy (`fak guard`), CLI (`cmd/fak`), and exported developer SDKs (`pkg/*`).
2. **Private `fak-private` (`github.com/anthony-chaudhary/fak-private`)**: The proprietary commercial serving product and autonomous development factory ("The Factory"), housing autonomous wave dispatchers, paced contract leases, hardware control bridges (`dgxbridge`), and internal test benches.

When engineers or autonomous agents build agent harnesses or platform tooling that operate across this dual-repo boundary, critical questions arise:
- How do private factory modules consume public harness types without violating Go compiler module rules or causing spec drift?
- Why do high-throughput private factory workers bypass interactive `fak guard` supervisor wrappers?
- How is secret quarantine preserved so private credentials and lab identifiers never leak into public checkouts?

This document defines the normative architectural contracts, encapsulation invariants, execution models, and distribution protocols governing public harnesses and private factory systems.

---

## 2. Strategic Alignment with Core Tenets (P1–P4)

- **P1 (Managed Context):** Memory stores and session archives remain strictly partitioned: public agent memories live in `$CLAUDE_CONFIG_DIR/projects/C--work-fak/memory/`, while private factory histories live in `agent-memory/fak-private/`. Private agents overlay public facts; public runs never ingest private state.
- **P2 (Net-True Efficiency):** High-throughput factory workers in `fak-private/platform/dispatch/` import compiled `pkg/` libraries directly under `go.work`, eliminating subprocess spawning overhead and JSON re-serialization during bulk issue-resolution waves.
- **P3 (Bounded Adaptation):** Module boundaries are fail-closed: the Go compiler rejects any attempt by `fak-private` to import `fak/internal/*`. Public interfaces under `pkg/` are frozen and protected by static AST verification in `internal/architest`.
- **P4 (Integrated Operations):** Artifacts migrate across the boundary via air-gapped `.fakpack` OCI bundles signed with Cosign simple-signing (`application/vnd.dev.cosign.simplesigning.v1+json`), guaranteeing cryptographic provenance without coupling Git histories.

---

## 3. The Core Boundary Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PUBLIC REPOSITORY (fak)                           │
│                            "The Open Source Core"                           │
│                                                                             │
│  - Core Go Runtime: cmd/fak, internal/engine, internal/abi, internal/ctxmmu│
│  - Drop-in Safety Gate: fak guard, fak preflight, fak serve                 │
│  - Exported Public SDK: pkg/harnesskit, pkg/abi, pkg/scorecard, pkg/fakclient│
│  - Open Benchmarks & Probes: livecodebench, radixbench, gpucheck            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ 1. go.work resolution of pkg/* only
                                       │ 2. Subprocess CLI calls: fak dev --json
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRIVATE REPOSITORY (fak-private)                      │
│                      "The Commercial Serving & Factory"                     │
│                                                                             │
│  1. THE COMMERCIAL SERVING PRODUCT                                          │
│     - Multi-tenant KV-MMU memory multiplexer & continuous batch scheduler   │
│     - Speculative branch rollback & zero-recompute engine                   │
│     - Hardware tiering: GDS / AMD Smart Access Storage daemons              │
│                                                                             │
│  2. THE AUTONOMOUS FACTORY ("How We Develop fak")                           │
│     - Autonomous wave dispatchers & queueing: platform/dispatch/            │
│     - Paced contract leases: refs/fak/locks/contract-*                      │
│     - Session watchdogs & self-healing reapers: platform/watchdogs/         │
│     - Lab hardware control bridge: tools/dgxbridge, dgxsh                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Contract 1: The `pkg/` Export Invariant (Go Module Encapsulation)

### Rule
**Under no circumstances may `fak-private` (or any external module) import packages located in `fak/internal/*`.**

The Go compiler strictly enforces internal package encapsulation: code outside `github.com/anthony-chaudhary/fak` attempting to import `.../internal/...` triggers a compile error (`use of internal package not allowed`).

### Export Seam
All shared data structures, protocols, and interfaces required by both the public runtime and the private factory must reside under `fak/pkg/`:

| Shared Concern | Public Package | Private Factory Consumer | Purpose |
|---|---|---|---|
| **Harness Lock v2** | `pkg/harnesskit/lockv2` | `fak-private/platform/dispatch` | Programmatically parse, validate, and compute RFC 8785 canonical IDs for `harness.lock.json`. |
| **Harness Public SDK** | `pkg/harnesskit` | `fak-private/platform/harness` | Common builder contracts, capability negotiation, and lifecycle streams. |
| **Frozen ABI** | `pkg/abi` | `fak-private/serving/adjudicator` | Binary-stable tool call envelopes, decision verdicts, and execution receipts. |
| **Scorecard Models** | `pkg/scorecard` | `fak-private/platform/scorecards` | Public scorecard KPIs, evaluation schemas, and markdown rendering. |
| **Agent RPC Client** | `pkg/fakclient` | `fak-private/tools/benchrunners` | Typed client for streaming tool events and gateway requests. |

### Enforcement via Static AST Gates
Repository architectural tests in `internal/architest/` continuously enforce this invariant at compile-time and AST-parse time:
1. **Zero Internal Imports in `lockv2` (`TestHarnessKitLockV2Export` in `internal/architest/harnesskit_lockv2_test.go`):** Statically parses all AST nodes in `pkg/harnesskit/lockv2`, asserts that zero `internal/*` packages are imported, and compiles an external test Go module against the exported package to compute RFC 8785 canonical IDs and validate secret contracts.
2. **Encapsulation Compile Barrier (`TestHarnesskitExternalImportBoundary` in `internal/architest/harnesskit_external_test.go`):** Verifies that an isolated external module compiles cleanly against `pkg/harnesskit`, while any attempt to import `fak/internal/*` (such as `internal/abi`) fails immediately with a compiler error (`use of internal package ... not allowed`).
3. **External Upgrade Contract (`TestHarnesskitUpgradeContractFromCleanModule` in `internal/architest/harnesskit_upgrade_external_test.go`):** Asserts that external modules can execute upgrade lifecycles against clean exported types without dependency drift.

---

## 5. Contract 2: Secret Quarantine & Air-Gap Invariants

### 1. Private Identifiers Never Enter the Public Tree
- **Lab Infrastructure:** Hostnames, IP addresses, cluster fabrics, SSH keys, and unredacted GPU execution runbooks (`dgxbridge`, `da33`, `dgxsh`) belong exclusively in `fak-private`.
- **Public Reference:** Public docs reference private hardware only via the scrubbed stub [`docs/private-comms-channel.md`](../private-comms-channel.md) and scrubbed trial receipts in `docs/_witnesses/`.
- **Git Commit Filter:** The repository pre-commit hook (`tools/githooks/pre-commit`) and `FILE_ADMISSION` check refuse any staged paths matching `*dgx*`, private shell scripts, or raw credential strings.

### 2. Opaque Secret Contracts in `harness.lock.json`
To allow harnesses to execute across both environments without leaking credentials:
- Lockfiles must **never** store plaintext secrets.
- Secrets are declared as opaque reference contracts:
  ```json
  {
    "kind": "secret",
    "id": "anthropic_key",
    "ref": "env:ANTHROPIC_API_KEY",
    "source": "env"
  }
  ```
- `lockv2.ValidateSecretContracts()` enforces that any asset of kind `secret` that specifies a non-empty `value` field immediately fails validation with `SECRET_PLAINTEXT_LEAK`.

### 3. Air-Gap Integrity & Offline Validation Contract
To ensure that harnesses and models can be deployed into strictly air-gapped sovereign environments without leaking internal topologies:
- **Hermetic Asset Resolution:** Air-gapped deployments and public harness distributions cannot rely on live outbound networks or unverified endpoints. All assets and components must be packaged locally or addressed by immutable SHA-256 layer digests.
- **Outbound URL Gate:** Any asset or component reference declaring an unpinned external `http://` or `https://` URL is rejected with `AIRGAP_URL_FORBIDDEN` during air-gapped bundle verification.
- **Internal Infrastructure Boundary:** Factory tools running on internal lab fabrics must not embed intranet URLs, cluster DNS names (`*.internal`, `*.cluster.local`), or private S3/GCS bucket paths into exported lockfiles or public receipts.

---

## 6. Contract 3: Factory Execution Pattern (Why Factory Workers Bypass `fak guard`)

A common architectural misconception is that autonomous factory workers running in `fak-private/platform/dispatch/` should each be wrapped in an interactive `fak guard` process. 

### Why Factory Workers Bypass `fak guard` Wrappers:
1. **High-Frequency Concurrency:** Autonomous waves launch 30 to 100+ concurrent micro-agents targeting disjoint lane leases (`refs/fak/locks/<lane>`). Spawning 100 long-lived interactive `fak guard` gateway processes would exhaust loopback sockets, consume gigabytes of RSS memory, and introduce port contention.
2. **Process Lifecycle Overheads:** `fak guard` initializes interactive TUI watchers, account rotation daemons, and terminal host recovery. For headless batch workers whose lifetime is bounded to a single task packet, this startup overhead wastes critical latency.
3. **Direct Contract Interaction:** Factory workers in `fak-private` need direct programmatic access to:
   - Git ref-based contract leases (`refs/fak/locks/contract-*`).
   - DOS lane arbitration (`dos arbitrate`).
   - Fast JSON data extraction.
   
   Wrapping these micro-operations in an interactive proxy adds serialization latency without providing additional security (since the factory worker is already bounded by repository lane leases and Git branch transaction guards).

### The Sanctioned Factory Interaction Patterns:
- **In-Process Go Compilation:** When `fak-private` tools require data structures (like lock files or ABI schemas), they import `github.com/anthony-chaudhary/fak/pkg/*` directly under the shared `go.work` workspace.
- **Subprocess CLI (`fak dev --json`):** When executing runtime operations, factory scripts invoke `fak` or `fak-dev` as a short-lived subprocess via standard input/output streams with `--json`, ensuring clean binary separation.

---

## 7. Contract 4: Cross-Repo Bundle Distribution (`.fakpack`)

When proprietary models, specialized policies, and compiled binaries from `fak-private` must be distributed to public or edge environments, they are packaged into hermetic, air-gapped `.fakpack` OCI bundles:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            .fakpack OCI BUNDLE                              │
│                                                                             │
│  ├── harness.lock.json      (RFC 8785 Canonical JCS Lock)                   │
│  ├── manifest.json          (OCI Image Index Manifest)                      │
│  ├── blobs/sha256/...       (Layers: binaries, model weights, instructions) │
│  └── signature.cosign       (Cosign SimpleSigning Ed25519 Payload)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Packaging:** Produced in `fak-private` via `fak pack create --lock harness.lock.json --bin ./bin --assets ./assets`.
2. **Cosign Signing:** The bundle manifest digest is signed using standard Go cryptography via `application/vnd.dev.cosign.simplesigning.v1+json`.
3. **Public Verification:** Public `fak` runtimes verify bundle integrity and signature before unpack:
   ```bash
   fak pack verify --bundle bundle.fakpack --public-key key.pub
   ```
   Verification strictly enforces a four-phase gate:
   - **Hermetic content digests:** SHA-256 validation of all layer blobs and manifest descriptors (`BUNDLE_DIGEST_MISMATCH` / `BUNDLE_CORRUPT`).
   - **Harness completeness:** Checks that all components and assets declared in `harness.lock.json` are present in the archive.
   - **Air-gap safety gate:** Immediate rejection of any outbound `http://` or `https://` URLs in asset or component references (`AIRGAP_URL_FORBIDDEN`).
   - **Cosign signature verification:** Cryptographic verification of the embedded signature against the operator's pinned public key (`BUNDLE_SIGNATURE_INVALID`).

---

## 8. Summary of Isolation Invariants

| Boundary | public `fak` | private `fak-private` | Governing Contract |
|---|---|---|---|
| **Git Repositories** | Independent public repo. | Independent private companion repo. | Never submoduled; sibling checkouts linked by `go.work`. |
| **Go Code Imports** | Exports clean APIs in `pkg/*`. Zero imports of `fak-private`. | Imports `fak/pkg/*`. Zero imports of `fak/internal/*`. | Compiler-enforced + `internal/architest`. |
| **Process Interfacing** | Exposes CLI verbs with `--json`. | Invokes `fak` / `fak-dev` CLI or imports `pkg/*`. | Subprocess IPC stream boundary. |
| **Credentials & Lab State** | Zero private credentials. Scrubbed witnesses only. | Owns `dgxbridge`, lab hardware tunnels, private keys. | `FILE_ADMISSION` + pre-commit secret scanner. |
| **Harness Execution** | Interactive agents use `fak guard` or `pkg/harnesskit`. | High-throughput factory workers use direct `pkg/*` and CLI `--json`. | Execution Profile Specification. |

---

## 9. Related Documents

- **Issue #11611:** Public-harness vs private-factory contract and isolation rules across `fak` and `fak-private`.
- **Issue #11386:** Export v2 lock parser and secret validator to `pkg/` for `fak-private` consumption.
- **Architectural Specification:** [`docs/architecture/when-to-use-fak-guard-for-harnesses.md`](when-to-use-fak-guard-for-harnesses.md)
- **Dev-Process Private Boundary:** [`docs/dev-process-private-boundary.md`](../dev-process-private-boundary.md)
- **GPU-Server Private Boundary:** [`docs/gpu-server-private-boundary.md`](../gpu-server-private-boundary.md)
- **Multi-Repo Sync Guide:** [`docs/notes/2026-09-03-dual-repo-workspace-and-safe-sync.md`](../notes/2026-09-03-dual-repo-workspace-and-safe-sync.md)
