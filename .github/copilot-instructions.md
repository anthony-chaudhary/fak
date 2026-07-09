# Copilot instructions

The canonical agent instructions for this repo are in [`AGENTS.md`](../AGENTS.md) — read
it first for build/test/run, the repo map, and the rules. Curated doc map:
[`llms.txt`](../llms.txt). Contributor contract: [`CONTRIBUTING.md`](../CONTRIBUTING.md).

Must-know rules (enforced below the agent layer):

- Work directly on the trunk (`main`); never open a feature branch or worktree — the
  trunk guard refuses off-trunk commits (`OFF_TRUNK`). The one sanctioned exception: a
  **detached** per-worker worktree that lands on `main` via `fak worktree worker
  prepare|land|reap` (build isolation, #1334) — detached and lane-serialized, it is not a
  branch and never trips `OFF_TRUNK`. Feature branches and off-trunk commits stay forbidden.
- Commit by explicit path (`git commit -- <paths>`, never `git add -A`); sign off with
  `git commit -s` (DCO).
- The Go module is the repository root — run `go` commands from the clone root
  (`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` resolves directly).
