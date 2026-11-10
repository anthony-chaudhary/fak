---
title: "First day in the fak repository"
description: "The engineer's day-one guide for a fresh clone of the fak kernel: prerequisites, the 10-minute sanity loop, where things live, platform rules, which checkout you have, test conventions, first tasks, and how to land your change."
---

# First day in the fak repository

> **Audience:** an engineer who cloned this repository to *change* code or docs — not a
> product user. If you want to run fak rather than build it, start at
> [`GETTING-STARTED.md`](../GETTING-STARTED.md) instead. Everything below assumes the
> repository root is your working directory.

## 0. Prerequisites

- **Go 1.26+** (`go version`). `go.mod` pins `toolchain go1.26.10`, so with
  `GOTOOLCHAIN=auto` (the default) Go fetches that toolchain automatically on the first
  build.
- **git** — any recent version.
- **make** on macOS/Linux, or PowerShell plus `scripts/ci.ps1` on Windows (the `make ci`
  equivalent: build + vet + test).
- **python3** — only for the repository gates (`tools/install_trunk_guard.py`,
  `tools/docs_scorecard.py`), not for building.

The external dependency set is deliberately tiny: four pinned `golang.org/x` modules
(`x/image`, `x/sys`, `x/term`, `x/text`) in an 8-line `go.sum`. Adding a new root-module
dependency requires the dependency-budget review described in
[CONTRIBUTING.md](../CONTRIBUTING.md) ("Go sub-modules, applications, and tooling
preference") — the trunk guard pins the require set and `go.sum`, so an intentional
change updates the allowlist in the same reviewed change.

## 1. The 10-minute sanity loop

Build the binary and watch the capability floor refuse and allow a tool call — no model,
no key, no GPU.

### macOS / Linux

```bash
go build -o fak ./cmd/fak
./fak version
./fak agent --offline
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
```

Expected: `fak agent --offline` books a task offline against an injection fixture;
`refund_payment` returns **DENY** (`POLICY_BLOCK`) and `search_kb` returns **ALLOW**.

### Windows (native build)

```powershell
go build -o fak.exe .\cmd\fak
.\fak.exe version
.\fak.exe agent --offline
.\fak.exe preflight --policy examples\customer-support-readonly-policy.json --tool refund_payment --args "{}"
.\fak.exe preflight --policy examples\customer-support-readonly-policy.json --tool search_kb --args "{}"
```

Same expected verdicts. Test *execution* on Windows differs — see section 3.

**Witness:** you just watched the capability floor refuse and allow, with no model.

## 2. Where things live (5 min)

| Path | What it is |
|---|---|
| `cmd/fak/` | The `fak` binary — every runtime verb (`preflight`, `serve`, `agent`, …). |
| `cmd/fak-dev/` | Repository-development tooling (`fak-dev buildcheck`, `windows-setup`, …). |
| `internal/<subsystem>/` | Kernel subsystems: `adjudicator`, `policy`, `vdso`, `engine`, `gateway`, `ctxmmu`, `model`, … |
| `examples/` | Policy manifests **and** runnable demos. |
| `docs/` | Explainers, integration guides, benchmark methodology, proofs. |
| `tools/<name>/` | Quarantined Go leaf modules (repository gates, scorecards). |
| `deploy/` | Deployment assets. |

Every verb is listed in [`docs/generated/verb-surface.md`](docs/generated/verb-surface.md)
(generated from the source tree by `go run ./cmd/verbsdoc`; do not edit it by hand).

## 3. Platform rules (the part that bites)

### Windows

- **Build, vet, and run natively.** `go build`, `go vet`, and running `fak.exe` all work
  on the host.
- **Run test execution under WSL**, not natively: `.\test.ps1 ./internal/<pkg>/` from the
  repo root (or `fak test <pkg>`). Microsoft Defender Antivirus real-time ML detections
  repeatedly quarantine the freshly compiled per-package test executables; the
  [2026-08-25 audit for issue #8919](docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md)
  supersedes the older "Application Control policy" diagnosis still quoted in some docs.
  Do not disable Defender or add broad exclusions.
- **One-time setup (optional):** `go run ./cmd/fak-dev windows-setup --apply` enables
  native `go test` plus the other local-machine exceptions; dry-run without `--apply`
  first. Details:
  [CONTRIBUTING.md, "One-time Windows developer setup"](../CONTRIBUTING.md#one-time-windows-developer-setup).
- **No long-lived local serves** on the shared box — use the fleet/GPU nodes for real
  model serves, and kill any temporary serve when done.

### macOS / Linux

Everything runs natively. Your inner loop is `make test-fast` (build + vet + short tests,
seconds); the full gate is `make ci` before you commit.

## 4. Which checkout are you?

| You are… | Mechanics |
|---|---|
| **Outside contributor** (fork → branch → PR) | Branch freely on your fork; no trunk guard, no `fak sync`; CI runs the full gate on the PR. See [CONTRIBUTING.md, "Fork and open a pull request"](../CONTRIBUTING.md#fork-and-open-a-pull-request). |
| **Maintainer / in-repo agent** (shared `main` checkout) | Work directly on `main`; install the guards once; commit by explicit path; push with `fak sync push`. |

For the shared-checkout row, the one-time setup:

```bash
python tools/install_trunk_guard.py   # arms core.hooksPath -> tools/githooks (once per clone)
go install ./cmd/fak-dev              # lands in your Go bin dir (%USERPROFILE%\go\bin or $HOME/go/bin); make sure it is on PATH
fak-dev buildcheck --vet              # compile check that hides peer WIP
fak sync check                        # confirm you can converge with origin/main safely
```

Trunk law in one breath: work on `main` (side branches trip `OFF_TRUNK`), never
`git add -A`, never stage a peer's uncommitted files, and a dirty shared tree is normal —
not a reason to branch. Full rules: [CONTRIBUTING.md](../CONTRIBUTING.md) and
[AGENTS.md](../AGENTS.md).

## 5. Test conventions (90 seconds)

| Command | Use |
|---|---|
| `fak test affected` | Changed packages + transitive importers — the default inner loop. |
| `fak test --list` | Inventory the runner tiers. |
| `make test-fast` | Fast gate: build + vet + `go test -short ./...` (skips the model witnesses). |
| `make ci` | Full gate (includes the ~538 MB model oracle; CI budgets `-timeout=25m`). Plan tens of minutes. |
| `.\test.ps1 ./internal/<pkg>/` | One package under WSL on Windows. |

New cost-shaped tests must declare their wall-clock cost above the test function:

```go
// fak-test:runtime slow est=90s lane=optin
func TestSomethingExpensive(t *testing.T) { ... }
```

`<class>` is `fast | medium | slow | integration`; `lane` defaults to `default`
(`short | optin | race` also exist). The companion `fak-testruntime` gate refuses a
cost-shaped test without the marker — full contract in
[AGENTS.md](../AGENTS.md) ("Calibrate test breadth").

## 6. Pick your first task

- **GitHub-side:** `gh issue list --label "good first issue" --state open` is the
  maintained newcomer queue (a GitHub label, so its state lives on GitHub, not in this
  repo).
- **Doc-only work, no ticket needed:** the 12-task board in
  [`docs/adoption/good-first-tasks.md`](docs/adoption/good-first-tasks.md) — each entry
  names the exact file to edit and its difficulty.

## 7. Land it

- Subject: Conventional Commits plus a `(fak <leaf>)` trailer naming the lane, e.g.
  `fix(gateway): treat same-tick ready as positive timeToReady (fak gateway)`; use
  `docs(scope): …` for doc-only changes.
- DCO sign-off on every commit (enforced by hook).
- Lint the subject first: `fak commit --preview`.
- Land by explicit path: `fak commit --path <p> -m "<subject> (fak <leaf>)"` — never
  `git add -A`, never a peer's files.
- Push: `fak sync push` (retries transient non-fast-forward races; stops safely on genuine
  divergence). Never force-push.
- A guard refused you? Read the token, then `fak recover <TOKEN>` or
  `dos man wedge <TOKEN> --explain` — recover, don't fight it.

## 8. Learn next

- [`LEARNING-PATH.md`](../LEARNING-PATH.md) — the prerequisite-ordered course; the
  contributor route starts at FAK 207.
- [`EXTENDING.md`](../EXTENDING.md) — plug in a leaf, prove it correct, prove it faster.
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — the kernel subsystem contracts.
