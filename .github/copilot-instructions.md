# Copilot instructions

The canonical agent instructions for this repo are in [`AGENTS.md`](../AGENTS.md) — read
it first for build/test/run, the repo map, and the rules. Curated doc map:
[`llms.txt`](../llms.txt). Contributor contract: [`CONTRIBUTING.md`](../CONTRIBUTING.md).

Must-know rules (enforced below the agent layer):

- **Delegate substantive work and keep the coordinator context clean** — use guarded
  headless agents or equivalent isolated workers for investigation, implementation, tests,
  and review. Keep only decisions and compact witnessed evidence in the primary context;
  independently verify worker effects before landing or reporting them. Direct work is for
  lightweight coordination and truly trivial tasks. See [`AGENTS.md`](../AGENTS.md).
- **Default is to ship** — when the tree is green (`make ci`), commit AND push unprompted;
  decide from the work in front of you, you don't wait to be asked. The rules below gate
  that default (stay on trunk, commit by path, never force-push); if a guard refuses
  (`OFF_TRUNK`) or a peer's merge is mid-flight, reconcile in place or stop.
- Work directly on the trunk (`main`); never open a feature branch or worktree — the
  trunk guard refuses off-trunk commits (`OFF_TRUNK`). The one sanctioned exception: a
  **detached** per-worker worktree that lands on `main` via `fak worktree worker
  prepare|land|reap` (build isolation, #1334) — detached and lane-serialized, it is not a
  branch and never trips `OFF_TRUNK`. Feature branches and off-trunk commits stay forbidden.
- Commit by explicit path (`git commit -- <paths>`, never `git add -A`); sign off with
  `git commit -s` (DCO).
- Keep code comments succinct: omit comments that restate syntax or narrate each step. Preserve
  comments that explain non-obvious rationale, invariants, safety, concurrency, compatibility, or
  exported APIs; prefer clearer code and put durable tutorials in docs.
- The Go module is the repository root — run `go` commands from the clone root
  (`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` resolves directly).
