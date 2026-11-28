# Copilot instructions

Canonical agent rules, build/test/run, and repo map live in [`AGENTS.md`](../AGENTS.md); doc map: [`llms.txt`](../llms.txt); contributor contract: [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Copilot delta rules

- **Delegate substantive work and keep the coordinator context clean**: use guarded headless agents or isolated workers for investigation, implementation, tests, and review; keep only decisions and compact witnessed evidence in the primary context, and verify worker effects independently.
- **Ship green work unprompted**: when `make ci` is green, commit explicit paths via `fak commit --path <p> -m "<subject> (fak <leaf>)"` (or `fak sweep --apply`) and push via `fak sync push`.
- Work directly on the trunk (`main`); use detached worker worktrees (`fak worktree worker prepare|land|reap`) only for build isolation, and keep `go` commands rooted at the repo clone.
- Keep comments concise (invariants/rationale only, never syntax narration) and frame diagnostics around the next checkable step.
