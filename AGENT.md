# AGENT.md — fak

> **Not the human contributor guide.** This file is operating instructions for *automated*
> contributors working inside the maintainers' shared checkout. Humans want
> [`README.md`](README.md) for what fak is and [`CONTRIBUTING.md`](CONTRIBUTING.md) for how
> to contribute; nothing on this page applies to you.

> Amp and other harnesses adopting the single-file `AGENT.md` convention auto-load this
> file. OpenCode and Codex auto-load [`AGENTS.md`](AGENTS.md), which is the canonical,
> fuller entry point — read that for the complete build/test/run, repo map, and rules. Doc map: [`llms.txt`](llms.txt). Contributor contract: [`CONTRIBUTING.md`](CONTRIBUTING.md).

This repo is **fak**, an agent kernel: one Go binary that sits between an AI agent and the
tools it calls and adjudicates every call before it runs — deny by structure, repair
malformed calls, quarantine poisoned results. Its MCP server is wired in `.mcp.json`.

## Build / test / run

- `go build -o fak ./cmd/fak` — compile the binary.
- `make test-fast` — the ~2s smoke gate; `make ci` — the full green gate (on native
  Windows run the suite under WSL with `./test.ps1`).
- `./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"`
  — the no-key 60-second proof (returns DENY by structure).

## Must-know rules (enforced below the agent layer)

- Work directly on the trunk (`main`); never open a feature branch or worktree — the
  trunk guard refuses off-trunk commits (`OFF_TRUNK`). The one sanctioned exception: a
  **detached** per-worker worktree that lands on `main` via `fak worktree worker
  prepare|land|reap` (build isolation, #1334) — detached and lane-serialized, it is not a
  branch and never trips `OFF_TRUNK`. Feature branches and off-trunk commits stay forbidden.
- Commit by explicit path (`git commit -- <paths>`, never `git add -A`); sign off with
  `git commit -s` (DCO).
- Subdivide work and fail-to-abstain — smaller or worker models must keep tasks
  bounded to atomic S0/S1 leaf units (1–3 files, single package, one witness) and
  abstain with a structured reason from high-difficulty aspects (concurrency, ABI,
  kernel/memory layout) instead of speculating.
- The Go module is the repository root — run `go` commands from the clone root.

To run your harness's own model behind the kernel, pick your recipe under
[`docs/integrations/`](docs/integrations/README.md) (Amp users: [`amp.md`](docs/integrations/amp.md);
OpenCode users: [`opencode.md`](docs/integrations/opencode.md)).
