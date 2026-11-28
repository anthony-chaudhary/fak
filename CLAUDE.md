# CLAUDE.md

> Automated-contributor overlay for Claude Code. Canonical rules, build/test/run, and repo map live in [`AGENTS.md`](AGENTS.md); humans read [`README.md`](README.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Claude Code coordinator & worker deltas

- **Delegate substantive work and keep this coordinator context clean** — dispatch pairwise tree-disjoint worker triads (researcher -> worker -> cross-validator) for multi-part work, keep depth-1 leaf workers within 1–3 files and package-scoped tests with no nested `task` calls, and independently verify
  worker effects before landing or reporting.
- **Land and sync unprompted**: run `fak sync check` / `fak sync apply`, verify real paths (`fak validate --mine <paths> --smoke`, `make mac-perf`, `make cuda-test`), commit one issue per commit via `fak commit --path <p> -m "<subject> (fak <leaf>)"` or `fak sweep --apply` (see [`/commit-clean`](.claude/skills/commit-clean/SKILL.md)), and push via `fak sync push`.
- **Trunk and script rules**: stay on `main` (use `fak worktree worker prepare|land|reap` only for detached worker isolation), use native Go programs (`cmd/*`, `fak` leaves) instead of new `.ps1`/`.sh`/`.bat` scripts, and qualify private issue citations as `fak-private#<num>`.
- **Forward momentum**: treat guard refusals (`fak recover <TOKEN>`) and unmet targets as data; name the next checkable lever.

Doc map: [`llms.txt`](llms.txt) · Contributor contract: [`CONTRIBUTING.md`](CONTRIBUTING.md).
