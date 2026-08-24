# Gemini CLI context — fak

> **Not the human contributor guide.** This file is operating instructions for *automated*
> contributors working inside the maintainers' shared checkout. Humans want
> [`README.md`](README.md) for what fak is and [`CONTRIBUTING.md`](CONTRIBUTING.md) for how
> to contribute; nothing on this page applies to you.

The canonical agent instructions for this repo are in [`AGENTS.md`](AGENTS.md) — read it
first for build/test/run, the repo map, and the rules. Curated doc map:
[`llms.txt`](llms.txt). Contributor contract: [`CONTRIBUTING.md`](CONTRIBUTING.md).

This repo is **fak**, an agent kernel: one Go binary that sits between an AI agent and the
tools it calls and adjudicates every call before it runs — deny by structure, repair
malformed calls, quarantine poisoned results. Its MCP server is wired in `.mcp.json`.

Must-know rules (enforced below the agent layer):

- **Delegate substantive work and keep the coordinator context clean** — use guarded
  headless agents or equivalent isolated workers for investigation, implementation, tests,
  and review. Keep only decisions and compact witnessed evidence in the primary context;
  independently verify worker effects before landing or reporting them. Direct work is for
  lightweight coordination and truly trivial tasks. See [`AGENTS.md`](AGENTS.md).
- Work directly on the trunk (`main`); never open a feature branch or worktree — the
  trunk guard refuses off-trunk commits (`OFF_TRUNK`). The one sanctioned exception: a
  **detached** per-worker worktree that lands on `main` via `fak worktree worker
  prepare|land|reap` (build isolation, #1334) — detached and lane-serialized, it is not a
  branch and never trips `OFF_TRUNK`. Feature branches and off-trunk commits stay forbidden.
- Commit by explicit path (`git commit -- <paths>`, never `git add -A`); sign off with
  `git commit -s` (DCO).
- The Go module is the repository root — run `go` commands from the clone root
  (`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` resolves directly).

To run the Gemini CLI behind the kernel (governed tool calls via MCP / an
OpenAI-compatible gateway), follow [`docs/integrations/gemini-cli.md`](docs/integrations/gemini-cli.md).
