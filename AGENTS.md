# AGENTS.md — orientation for coding agents

> You are an autonomous agent working in this repo. This file is the machine-read entry
> point (the [agents.md](https://agents.md) convention). It is intentionally
> command-dense and free of philosophy. For the *why*, read [`README.md`](README.md);
> for a curated doc map, read [`llms.txt`](llms.txt). Humans: see [`START-HERE.md`](START-HERE.md).

## What this project is

**fak** is an *agent kernel*: one Go binary that sits between an AI agent and the tools
it calls, and adjudicates every tool call before it runs — deny by structure, repair
malformed calls, quarantine poisoned results. It is both a **security gate** (a
default-deny capability floor the model can't talk past) and a **performance gate** (do
the shared setup work once, not every turn).

## Repo layout (where things live)

| Path | What it is |
|---|---|
| `go.mod` · `cmd/` · `internal/` | **The Go module is the repository root** (the kernel + the `fak` CLI). |
| `cmd/fak/` | The `fak` binary (every verb: `preflight`, `serve`, `agent`, `policy`, `bench`, …). |
| `internal/` | Kernel subsystems: `adjudicator`, `policy`, `vdso`, `engine`, `gateway`, `ctxmmu`, `model`, … |
| `examples/` | Policy manifests **and** runnable demos (`adjudication-demo/`, `agentdojo-redteam/`, `mcp/`). |
| `docs/` | Explainers, integration guides (`docs/integrations/`), benchmark methodology, proofs. |

## Build / test / run

> **The Go module is the repository root** — run `go` commands from the clone root.
> Needs Go 1.26+ (`GOTOOLCHAIN=auto` self-fetches). Zero external deps, so no `go.sum`.

```bash
go build ./cmd/fak        # -> ./fak  (fak.exe on Windows).
make test-fast            # ~2s smoke gate: build + vet + `go test -short ./...`
make test                 # full suite incl. the weight-backed model witnesses
make ci                   # the full gate: build + vet + test + claims-lint  (Windows: scripts/ci.ps1)
```

Or install the released binary directly — the module is at the repo root, so this resolves:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Windows:** `go build` / `go vet` / `go run` work natively, but native `go test` is
> blocked by an OS Application-Control policy on the freshly-compiled test binaries — run
> the suite under WSL: `./test.ps1` from the repo root. This is an OS quirk, not a code failure.

## The 60-second proof (no key, no model, no GPU — verified)

This is the canonical first command. Run it before anything else:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK): refused by structure, no model in the loop
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW: not a blanket block
go run ./cmd/fak agent --offline                                                                                       # -> injection-in-context YES->no, destructive-op YES->no, task still booked
```

The first `go run` compiles the binary (~30–60s, plus a one-time Go-1.26 toolchain fetch);
later runs are instant. Full walkthrough: [`docs/repro-packet.md`](docs/repro-packet.md).

## Hard rules (these WILL bite an agent — they are enforced below the agent layer)

- **Work directly on the trunk (`main`). Never open a feature branch or new worktree.**
  The trunk guard *refuses* off-trunk commits (the `OFF_TRUNK` law). A dirty/diverged
  tree means pull/merge in place or STOP — never escape into a side branch.
- **Commit by explicit path** — `git commit -- <paths>`, never `git add -A`. This is a
  shared multi-session tree; never stage a peer's uncommitted files.
- **Sign off every commit** — `git commit -s` (DCO). Use a Conventional-Commits subject
  with a `(fak <leaf>)` trailer; a docs-only change uses a `docs(scope):` subject.
- **Every claim carries a tag.** Each `- [` line in [`fak/CLAIMS.md`](CLAIMS.md) must
  carry exactly one of `[SHIPPED]` / `[SIMULATED]` / `[STUB]` (lint-enforced by
  `make claims-lint`). Don't overclaim; the repo keeps an honesty ledger.
- **Add a feature as a leaf, not a core edit.** `python tools/new_leaf.py <name> --tier
  <tier> [--register]` stamps a conforming skeleton; the frozen ABI (`fak/internal/abi`)
  is additive-only and human-owned. `internal/architest` fails the build on a bad import.

Check your setup first: `python tools/extend_preflight.py`. Full contributor contract:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Where to go next

| If you want to… | Read |
|---|---|
| Every CLI verb + what's shipped | [`docs/cli-reference.md`](docs/cli-reference.md) |
| Install / run tiers (offline → gateway → in-kernel model) | [`fak/GETTING-STARTED.md`](GETTING-STARTED.md) |
| Put fak in front of *your* agent (Claude Code / Cursor / MCP) | [`docs/integrations/`](docs/integrations/) · [`fak/examples/mcp/`](examples/mcp/) |
| The deployable capability floor (policy manifests) | [`fak/POLICY.md`](POLICY.md) · [`fak/examples/README.md`](examples/README.md) |
| Extend the kernel (plug in → prove correct → prove faster) | [`fak/EXTENDING.md`](EXTENDING.md) · [`fak/ARCHITECTURE.md`](ARCHITECTURE.md) |
| What's real vs simulated vs stub | [`fak/CLAIMS.md`](CLAIMS.md) · [`fak/STATUS.md`](STATUS.md) |
| Every benchmark number (single source of truth) | [`fak/BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) |
| A curated map of all the docs | [`llms.txt`](llms.txt) |

License: [Apache-2.0](LICENSE).
