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

## Operational sharp edges & model guidance (Gemini 3.8 Flash & peers)

When operating as or delegating to Gemini 3.8 Flash, observe these behavioral fences:

- **Enforce strict token and diff economy**: 3.8 Flash tends toward verbosity and output inflation (~2×–4× output tokens vs peer models). Keep commentary brief (<3 lines in CLI mode), avoid conversational filler, and never introduce unasked scaffolding, multi-panel apps, or unsolicited helper files. Keep diffs strictly minimal.
- **Circuit-break tool error loops**: If a command or tool call fails or is denied by a kernel guard, never immediately retry the same call. Read the error, change tack, or emit a structured refusal.
- **Rely on external verification, not self-narration**: Never declare a task complete based on internal reasoning alone. Ground every claim in an observed tool execution receipt or test result (`go test`, `fak validate`).
- **Prefer discrete file tools**: 3.8 Flash exhibits lower reliability on raw shell pipelines (TerminalBench regressions). Use structured tools (`Read`, `Edit`, `Glob`, `Grep`) over complex piped bash commands.
- **Fail-to-abstain on high-difficulty subsystems**: Concurrency lock invariants, frozen ABI (`internal/abi`), and low-level SIMD/CUDA mechanics exceed safe Flash reasoning. Emit an `ABSTAIN` verdict with diagnostic context rather than guessing.
- Full analysis & evaluation data: [`docs/notes/2026-09-03-gemini-3.8-flash-initial-feedback-and-guidance.md`](docs/notes/2026-09-03-gemini-3.8-flash-initial-feedback-and-guidance.md).

To run the Gemini CLI behind the kernel (governed tool calls via MCP / an
OpenAI-compatible gateway), follow [`docs/integrations/gemini-cli.md`](docs/integrations/gemini-cli.md).
