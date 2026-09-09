---
title: "The Companion-Aware Dev Harness Doctrine and Asymmetric Companion Paradigm"
description: "Architectural specification codifying the Asymmetric Companion Paradigm across public fak and private fak-private, retiring the legacy 'Running from fak-private is strictly better' dichotomy, and establishing first-class native development in fak with automatic companion governance."
---

# The Companion-Aware Dev Harness Doctrine and Asymmetric Companion Paradigm

> **Status:** Approved Canonical Architectural Specification  
> **Issue:** fak-private#793 (Ticket TICKET-04) / fak#12643  
> **Scope:** Dev Tooling (`cmd/fak-dev`, `.githooks/`), 5-Gate Boundary Invariants, Context Provenance, Multi-Repo Architecture  
> **Date:** September 2026  
> **Authors:** fak Architecture & Infrastructure Guild  

---

## 1. Executive Summary & Problem Statement

The `fak` software ecosystem divides intellectual property into two companion repositories linked at local build time via `go.work`:
1. **Public `fak` (`github.com/anthony-chaudhary/fak`)**: The open-source core engine containing model inference runtime (`fak serve`, `fak up`), hardware HALs (Metal, ROCm, Vulkan, AVX-512), model weight loaders (GGUF, SafeTensors), context MMU (`internal/ctxmmu`), drop-in tool capability floor (`fak guard`), frozen ABI (`pkg/abi`), and client SDKs (`pkg/*`).
2. **Private `fak-private` (`github.com/anthony-chaudhary/fak-private`)**: The proprietary commercial serving product and autonomous development factory ("The Factory"), housing multi-tenant KV multiplexing, continuous batch schedulers, autonomous issue dispatchers (`platform/dispatch`), contract lease locks (`platform/leaseref`), session watchdogs (`platform/watchdogs`), test scorecards (`platform/scorecards`), and hardware cluster bridges (`tools/dgxbridge`).

### The Historical Pathology: "Running from fak-private is strictly better"
Historically, developer tooling, 5-gate boundary checkers (`cmd/fak-boundary`), secret leak auditing (`tools/scrub_public_copy.py`), context provenance minting (`cmd/fak-sync provenance`), and ticket specifications were homed exclusively in `fak-private`. As a consequence:
- Engineers and autonomous coding agents operating natively inside public `fak` lacked local pre-commit enforcement and CLI access to boundary validation and leak detection.
- A prevailing operational heuristic emerged: *"Running from fak-private is strictly better"*. Agents and contributors were routinely instructed to change working directory (CWD) to `fak-private` even when working on purely open-source core runtime features.
- When agents remained in `fak`, they risked committing boundary violations (such as importing private packages or internal leaks) or unscrubbed secrets that only failed later during private CI or cross-repo audit passes.
- External open-source contributors without access to `fak-private` faced a confusing contributor experience if tooling assumed private factory infrastructure was always present.

### The Resolution: The Companion-Aware Dev Harness
This specification codifies the **Asymmetric Companion Paradigm** and establishes the **Companion-Aware Dev Harness**:
1. **Public `fak` is the open-core engine standard**: It builds, vets, and tests standalone with zero dependencies on private infrastructure. External contributors and public CI enjoy a completely clean, hermetic development experience.
2. **Automatic Companion Discovery & Governance**: When `fak-private` is present side-by-side in internal environments, `fak` automatically discovers it and activates the complete validation harness (5-gate boundary checks, secret leak audits, context provenance minting, and ticket inspection) by default.
3. **Retirement of the "Strictly Better" Dichotomy**: Operating natively from `fak` now provides the exact same safety floor, verification gates, and quality ratchets as `fak-private`. Native CWD development in `fak` is first-class. CWD is chosen strictly based on architectural domain (public engine vs. private platform/factory), never because of a disparity in tooling or safety.

---

## 2. The Asymmetric Companion Paradigm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PUBLIC REPOSITORY (fak)                           │
│                      "The Open Core Engine Standard"                        │
│                                                                             │
│  STANDALONE OSS MODE                        COMPANION-AWARE INTERNAL MODE    │
│  - Zero private dependencies                - Auto-discovers ../fak-private │
│  - Hermetic build & test (go test ./...)    - Pre-commit: 5-Gate checks     │
│  - Public linters & AST rules               - Pre-commit: Secret leak audit │
│  - Clean for external contributors          - CLI: fak-dev gate / provenance│
│  - Standalone GitHub Actions CI             - CLI: fak-dev ticket reader    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Auto-Discovery (../fak-private, go.work)
                                       │ Zero-Import Subprocess Bridge (exec.Command)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRIVATE COMPANION (fak-private)                       │
│                      "The Commercial Serving & Factory"                     │
│                                                                             │
│  - 5-Gate Boundary Invariant Engine: cmd/fak-boundary, platform/boundary/    │
│  - Secret Scrubbing & Leak Auditor: tools/scrub_public_copy.py              │
│  - Context Provenance Minting Engine: platform/provenance/, cmd/fak-sync    │
│  - Autonomous Factory Tickets & Specs: docs/tickets/                        │
│  - Commercial Serving & Multi-Tenant Scheduling: platform/l3store, gateway  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Mode 1: Public Standalone Standard (External OSS & Public CI)
Public `fak` is designed as an open-core standard. External open-source contributors, community packagers, and public GitHub Actions workflows must never be coupled to private infrastructure:
- **Zero Required Private Variables:** `fak` builds and executes tests without requiring `FAK_PRIVATE_ROOT`, private cluster credentials, or internal tokens.
- **No Git Submodules or Private Subtrees:** The repository tree contains no dangling or inaccessible submodules.
- **Graceful Tooling Degradation:** If companion-aware tools (`fak-dev gate`, `fak-dev ticket`, or `.githooks/pre-commit`) run without `fak-private` present, they gracefully report that the companion is absent and exit cleanly (code 0) or fall back to public checks (`gofmt`, public linters). External contributors are never blocked by missing private repositories.

### 2.2 Mode 2: Companion-Aware Internal Development (Side-by-Side)
In internal engineering and autonomous agent environments where `fak-private` resides alongside `fak` (the standard maintainer and agent workspace layout):
- **Automatic Companion Discovery:** `fak` tooling dynamically resolves the companion workspace root via an ordered search hierarchy (`FAK_PRIVATE_ROOT` $\to$ `go.work` $\to$ sibling `../fak-private`).
- **Default-On 5-Gate Boundary Checks:** The git pre-commit hook automatically executes `go run ./cmd/fak-boundary check --staged` against the companion engine before allowing any staged commit in `fak`.
- **Default-On Secret Leak Audits:** Pre-commit hooks run `python tools/scrub_public_copy.py --audit-staged --root .` to ensure no lab hostnames, cluster IPs, or private tokens leak into public git history.
- **Native Context Provenance:** Developers and agents authoring dual implementation/test commits can mint and audit cryptographic context provenance trailers directly via `fak-dev provenance`.
- **In-Tree Ticket Inspection:** Autonomous agents working in `fak` can discover, list, and read task specifications residing in `fak-private/docs/tickets/` via `fak-dev ticket` without changing directories.

### 2.3 Retirement of the "Running from fak-private is strictly better" Dichotomy
The legacy advice that *"Running from fak-private is strictly better"* reflected an early gap in tooling availability: private verification tools could only be invoked from within `fak-private`. This led to friction, confusion over repository ownership, and unnecessary cross-repo context switching.

Under the Companion-Aware Dev Harness:
- **Native `fak` development is fully governed and protected:** Working in `fak` automatically invokes the exact same boundary, leak, and provenance gates as working in `fak-private`.
- **Neither CWD is superior:** Both repositories provide first-class developer and agent ergonomics.
- **CWD Selection Rule:** Working directory is determined solely by what you are modifying:
  - If modifying inference engine, compute HAL, model loaders, context MMU, tool capability floors, or public SDKs $\to$ **operate natively in `fak`**.
  - If modifying autonomous dispatch, worktree flow runners, contract leasing, commercial serving policies, or cluster bridges $\to$ **operate natively in `fak-private`**.

---

## 3. Technical Architecture & Discovery Mechanism

### 3.1 Companion Resolution Hierarchy
Companion-aware tooling in `fak` resolves the location of `fak-private` using a deterministic, 4-tier discovery order:

```
1. Environment Override:  FAK_PRIVATE_ROOT
      │ (Non-empty and directory exists?)
      ├─► YES: Use FAK_PRIVATE_ROOT
      └─► NO:
2. Go Workspace Inspect:  go.work
      │ (Parses use directives for path ending in "fak-private")
      ├─► YES: Resolve relative to go.work directory
      └─► NO:
3. Sibling Path Check:    ../fak-private
      │ (Directory exists relative to fak root?)
      ├─► YES: Use sibling ../fak-private
      └─► NO:
4. Graceful Fallback:     COMPANION_ABSENT
      └─► Enter Standalone OSS Mode (bypass private gates, log advisory notice)
```

This resolution order guarantees that custom container mounts, CI runner layouts, standard sibling checkouts, and standalone contributor clones are all supported without manual reconfiguration.

### 3.2 Zero-Import Subprocess Bridge Pattern
Under **Gate 2 (`GATE_PUBLIC_IMPORT_PRIVATE`)** and the Go compiler's internal package rules, public `fak` code is **strictly forbidden from importing `fak-private/*` Go packages**.

To provide native CLI access to private tools without creating compiler or module dependencies, `fak-dev` employs the **Zero-Import Subprocess Bridge Pattern** (`internal/devcmd/companion_bridge.go`):
- CLI commands in `fak-dev` resolve the companion root at runtime.
- The bridge invokes companion Go commands via `os/exec.Command` (e.g., `go run ./cmd/fak-boundary check --staged` or `go run ./cmd/fak-sync provenance audit`).
- Standard input, standard output, and standard error streams are attached directly to the parent process.
- Subprocess exit codes are propagated verbatim, ensuring that failure in a companion gate returns a non-zero exit code to the invoking terminal or git hook.

### 3.3 Native Companion Verbs in `fak-dev`
Developers and agents working inside `fak` invoke companion tooling directly through native `fak-dev` commands:

| Command | Target Companion Tool | Function | Standalone Fallback |
| :--- | :--- | :--- | :--- |
| `fak-dev gate check [--staged\|--all]` | `cmd/fak-boundary check` | Verifies 5-gate import encapsulation and placement rules across repositories. | Skips private gate, runs public syntax/import linter. |
| `fak-dev gate query "<concept>"` | `cmd/fak-boundary query` | Queries the 5-gate IP taxonomy catalog to determine correct repository and gate. | Reports concept query requires companion repository. |
| `fak-dev gate explain <path>` | `cmd/fak-boundary explain` | Explains placement rules, permitted imports, and gate classification for a file path. | Reports path explanation requires companion repository. |
| `fak-dev audit-leak [--staged]` | `tools/scrub_public_copy.py` | Scans staged or modified files for private cluster IPs, credentials, or internal tokens. | Advisory warning; passes if no private secret patterns declared. |
| `fak-dev provenance mint` | `cmd/fak-sync provenance mint` | Mints cryptographic SHA-256 context provenance trailers (`Impl-Context`, `Test-Context`). | Generates standalone unverified trailer. |
| `fak-dev provenance audit` | `cmd/fak-sync provenance audit` | Audits commit context separation trailers to verify dual-context authoring compliance. | Advisory warning; skips private audit. |
| `fak-dev ticket list` | `docs/tickets/` | Lists available task tickets across capability tranches in `fak-private`. | Reports companion tickets unavailable in standalone checkout. |
| `fak-dev ticket show <id>` | `docs/tickets/**/TICKET-<id>.md` | Displays full specification, routing metadata, and acceptance criteria for a ticket. | Reports ticket unavailable in standalone checkout. |
| `fak-dev ticket next` | `docs/tickets/` | Discovers the next actionable, unblocked ticket based on priority and dependencies. | Reports ticket queue unavailable in standalone checkout. |

### 3.4 Companion-Aware Git Pre-Commit Hooks
Public `fak` maintains companion-aware pre-commit hooks:
- **POSIX:** `.githooks/pre-commit` (for Linux, macOS, and Git Bash)
- **PowerShell:** `.githooks/pre-commit.ps1` (for native Windows PowerShell)

#### Execution Lifecycle:
1. **Gate 0 (Scratch & Temp File Hygiene):** Inspects `git diff --cached --name-only`. Immediately rejects staged temporary files (`.tmp`, `.gotmp`, `.gocache`, `.bak`, `~`).
2. **Companion Discovery:** Probes for `fak-private` via `FAK_PRIVATE_ROOT`, `go.work`, or `../fak-private`.
3. **If Companion Present:**
   - Runs `go run ./cmd/fak-boundary check --staged` inside `fak-private`. If any 5-gate boundary violation is detected, the commit is aborted.
   - Runs `python tools/scrub_public_copy.py --audit-staged --root .` from `fak-private`. If any secret or private cluster IP is staged, the commit is aborted.
4. **If Companion Absent:**
   - Emits an informational notice: `[pre-commit] Companion fak-private not found; running public standalone checks.`
   - Executes public `gofmt` verification and public AST boundary checks.
   - Exits 0, allowing external contributors to commit cleanly.

---

## 4. Boundary & Provenance Invariants

### 4.1 The 5-Gate Invariant Summary
Every staged change in `fak` or `fak-private` is validated against the 5-Gate IP Taxonomy:
- **Gate 1 (`GATE_PRIVATE_IMPORT_PUBLIC_INTERNAL`):** Private factory code may import `fak/pkg/*`, but **NEVER** `fak/internal/*`.
- **Gate 2 (`GATE_PUBLIC_IMPORT_PRIVATE`):** Public `fak` code must **NEVER** import `fak-private/*`.
- **Gate 3 (`GATE_PROPRIETARY_PLATFORM_PLACEMENT`):** Proprietary platform and factory code must not be placed in public `fak`. Public engine code must be vendor-neutral and open.
- **Gate 4 (`GATE_PUBLIC_SDK_PLACEMENT`):** Compute HALs and model loaders belong in public `fak`; public SDK packages (`pkg/*`) must not be defined directly in `fak-private`.
- **Gate 5 (`GATE_NON_SDK_PUBLIC_IMPORT`):** Private operational tools (`tools/dgxbridge`) may only import public `fak/pkg/*` packages, never internal runtime packages.

### 4.2 Context Provenance & Dual-Context Test Separation
When a commit modifies both implementation code and test code, the **Dual-Context Test Separation Invariant** applies. To prevent single-context self-testing pathologies (confirmation bias, testing accidental implementation artifacts):
- An independent subagent or separate context window authors the tests.
- Cryptographic SHA-256 session fingerprints track authorship:
  ```text
  Impl-Context: sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069 (agent:worker/session-812)
  Test-Context: sha256:cb22484f29c37bcf14b5f712b79de100c14fdb82d8865b37380402e12f37e35b (agent:cross-validator/session-813)
  Separation-Verdict: SEPARATED
  ```
- When operating in `fak`, mint trailers using `fak-dev provenance mint` and audit commits using `fak-dev provenance audit --commit HEAD`.

### 4.3 Explicit Path Staging Mandate
Blanket staging (`git add -A`, `git add .`, or `git commit -a`) is **strictly forbidden** across both repositories. Every staged path must be explicitly named:
```bash
git add internal/compute/strix/kernel.go internal/compute/strix/kernel_test.go
git commit -s -m "feat(compute): implement wave32 wmma tile load (fak compute)"
```

---

## 5. Operating Regimes Compared

The table below summarizes the operational differences across environments:

| Feature / Gate | Standalone OSS (`fak`) | Companion-Aware Internal (`fak`) | Commercial Serving & Factory (`fak-private`) |
| :--- | :--- | :--- | :--- |
| **Primary Workload** | Inference runtime, HALs, loaders, public SDKs | Inference runtime, HALs, loaders, public SDKs | Autonomous factory, dispatch, commercial serving, SRE |
| **Target Directory** | `C:\...\fak` (or Linux/macOS clone) | `C:\...\fak` | `C:\...\fak-private` |
| **Companion Detected** | No | Yes (`../fak-private`) | Self (Root) |
| **5-Gate Boundary Checks** | Skipped / Public AST only | Active (`fak-dev gate`, pre-commit) | Active (`fak-boundary`, pre-commit) |
| **Secret Leak Scrubbing** | Public gitignore only | Active (`fak-dev audit-leak`, pre-commit) | Active (`scrub_public_copy.py`, pre-commit) |
| **Context Provenance** | Optional / Unattested | Active (`fak-dev provenance`) | Active (`fak-sync provenance`) |
| **Ticket Inspection** | N/A (uses public GitHub issues) | Active (`fak-dev ticket`) | Active (`fak-ticket`, `docs/tickets/`) |
| **Pre-Commit Enforcement** | Public `gofmt` & scratch hygiene | Full 5-Gate + Leak + Scratch hygiene | Full 5-Gate + Leak + Scratch hygiene |
| **Trunk Convergence** | Public `main` | Dual-repo synced `main` | Dual-repo synced `main` |

---

## 6. Guidance for Autonomous Coding Agents

When autonomous agents are dispatched to work on tasks:
1. **Respect Working Directory Intent:**
   - If dispatched into `fak`, remain in `fak`. Do not execute `cd ../fak-private` simply to run boundary or leak checks. Use `fak-dev gate` and `fak-dev audit-leak`.
   - If dispatched into `fak-private`, remain in `fak-private`.
2. **Never Import Private Packages in Public Code:**
   - Verify all imports in public code resolve within `github.com/anthony-chaudhary/fak`.
   - If shared functionality is needed by both repositories, place interfaces and data models in `fak/pkg/*`.
3. **Run Companion Verification Before Landing:**
   - Before committing in `fak`, run `fak-dev gate check --staged` and `fak-dev audit-leak --staged`.
   - Ensure the commit message includes DCO signoff (`-s`), a conventional type/scope, and the mandatory `(fak <leaf>)` trailer.
4. **Treat Both Repositories as First-Class Citizens:**
   - The legacy belief that one must "run from `fak-private`" is obsolete.
   - The Companion-Aware Dev Harness brings the full power of the autonomous factory directly to the public open-core runtime.

---

## 7. Authoritative References

- **Boundary Specification:** [`BOUNDARY.md`](../../fak-private/BOUNDARY.md) (in `fak-private`)
- **Public/Private Harness Boundary:** [`docs/architecture/public-private-harness-boundary.md`](public-private-harness-boundary.md)
- **Core Engine Public / Factory Private Doctrine:** [`NOTE-2026-09-09-CORE-ENGINE-PUBLIC-FACTORY-PRIVATE`](../../fak-private/docs/notes/2026-09-09-core-engine-public-factory-private-doctrine.md)
- **Companion Pre-Commit Hooks Ticket:** `docs/tickets/companion-dev-harness/TICKET-01-companion-aware-precommit-hooks.md` (Issue #790)
- **Companion Dev Bridge Verbs Ticket:** `docs/tickets/companion-dev-harness/TICKET-02-companion-dev-bridge-verbs.md` (Issue #791)
- **Companion Ticket Reader Ticket:** `docs/tickets/companion-dev-harness/TICKET-03-companion-ticket-reader.md` (Issue #792)
- **Codify Companion Dev Harness Doctrine:** `docs/tickets/companion-dev-harness/TICKET-04-codify-companion-dev-harness-doctrine.md` (Issue #793)
