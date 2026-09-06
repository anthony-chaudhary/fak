# CLAUDE.md

> **Not the human contributor guide.** This file is operating instructions for *automated*
> contributors working inside the maintainers' shared checkout. Humans want
> [`README.md`](README.md) for what fak is and [`CONTRIBUTING.md`](CONTRIBUTING.md) for how
> to contribute; nothing on this page applies to you.

## The canonical instructions

The canonical agent instructions for this repo are in **[`AGENTS.md`](AGENTS.md)** —
read it first for build/test/run, the repo map, and the rules.

## The five hard rules

The five that will bite you if you skip them:

- **Work directly on the trunk (`main`). Never open a feature branch or new worktree** —
  the trunk guard *refuses* off-trunk commits (`OFF_TRUNK`). The *one* sanctioned
  exception is a **detached** per-worker worktree that lands its diff back on `main` via
  `fak worktree worker prepare|land|reap` (build isolation, #1334 / epic #3165): because
  it is detached (never a branch) and commits only through the serialized land under the
  worker's lane lease, it is not off-trunk and never trips `OFF_TRUNK`. Feature branches
  and any other off-trunk commit stay forbidden. (Details in [`AGENTS.md`](AGENTS.md).)
- **Commit via `fak commit` or `fak sweep` by explicit path** — all commits must be made via
  `fak commit --path <p> -m "<subject> (fak <leaf>)"` or `fak sweep --apply --lane <lane> -m "<subject>"`,
  never `git add -A` or uncoordinated raw `git commit` (shared multi-session tree).
  `fak commit` locks the lane, stages exactly your paths, provides automatic DCO sign-off (with `-s` accepted for compatibility),
  and verifies that no peer files were raced in (`PATHSPEC_RACE`). End every ship commit's Conventional-Commits
  subject with a bindable `(fak <leaf>)` trailer so the `dos verify` referee can bind it — e.g.
  `fix(gateway): treat same-tick ready as positive (fak gateway)`. A bare un-stamped subject stays NOT_SHIPPED.
  Emergency raw git fallback is permitted ONLY when the `fak` binary is unbuilt (`git commit -s -m "<subject> (fak <leaf>)" -- <paths>`).
  The [`/commit-clean`](.claude/skills/commit-clean/SKILL.md) skill mechanizes this rule end to end. (Full convention in [`AGENTS.md`](AGENTS.md).)
- **Default is to ship green work and sync safely unprompted** — do more work by default:
  pre-flight sync with trunk (`fak sync check`, `fak sync reconcile --apply`, or `fak sync apply`),
  verify on-device (`fak validate --mine <paths>`, `go test ./internal/<pkg>/...`), commit by explicit path
  (`fak commit --path <p> -m "<subject> (fak <leaf>)"` or `fak sweep --apply --lane <lane> -m "<subject>"`), and push unprompted via `fak sync push` or `fak commit --push`.
  "Green" requires shift-left proof: for changes touching executable CLI verbs, gateway adapters, or runtime logic,
  execute real paths in dogfood or integration tests rather than relying on mock-only or shallow tests. Stay on the trunk,
  never force-push, defer to the guard (`OFF_TRUNK` / a peer merge in flight), and reconcile trunk divergence in place with `fak sync apply`. Full default + verify command
  in [`AGENTS.md`](AGENTS.md).
- **Divide and conquer: delegate substantive work and keep this coordinator context clean; enforce capability-aware scoping and persistence** —
  decompose substantive or multi-part requests into atomic single-concern units and launch specialized subagents
  concurrently for independent components. Use guarded headless agents or equivalent isolated workers for investigation,
  implementation, tests, and review. Constrain smaller models and workers to atomic S0/S1 leaf units (1–3 files,
  single package, exactly one witness). Scope abstention strictly to bounded high-difficulty aspects
  (concurrency, frozen ABI, kernel memory layout, security gates): emit a structured `ABSTAIN`
  for the escalated boundary while executing all independent, safe, solvable sub-components (such as
  reproduction tests or diagnostics). Treat guard refusals as actionable feedback rather than session stops:
  query `fak recover <TOKEN>`, adapt execution or wait out transient locks, and maintain momentum on the
  objective without repeating failing calls. Keep only decisions and compact witnessed
  evidence here; independently verify worker effects before landing or reporting them. Reserve
  direct work for lightweight coordination and truly trivial tasks. Full contract in [`AGENTS.md`](AGENTS.md).
- **The Go module is the repository root** — run `go` commands from the clone root;
  `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` resolves directly.

## Doc map

Doc map for humans and agents: [`llms.txt`](llms.txt). Full contributor contract:
[`CONTRIBUTING.md`](CONTRIBUTING.md).
