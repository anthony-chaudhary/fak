# CLAUDE.md

The canonical agent instructions for this repo are in **[`AGENTS.md`](AGENTS.md)** —
read it first for build/test/run, the repo map, and the rules.

The three that will bite you if you skip them:

- **Work directly on the trunk (`main`). Never open a feature branch or new worktree** —
  the trunk guard *refuses* off-trunk commits (`OFF_TRUNK`).
- **Commit by explicit path** — `git commit -- <paths>`, never `git add -A` (shared
  multi-session tree). Sign off with `git commit -s` (DCO). Use a Conventional-Commits
  subject and end every ship commit with a `(fak <leaf>)` trailer so the `dos verify`
  referee can bind it — e.g. `fix(gateway): treat same-tick ready as positive (fak gateway)`.
  A bare un-stamped subject stays NOT_SHIPPED. (Full convention in [`AGENTS.md`](AGENTS.md).)
- **Default is to ship** — once the tree is green (`make ci`), commit AND push; don't wait
  to be asked. Stay on the trunk, never force-push, defer to the guard (`OFF_TRUNK` / a peer
  merge in flight). Full default + verify command in [`AGENTS.md`](AGENTS.md).
- **The Go module is the repository root** — run `go` commands from the clone root;
  `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` resolves directly.

Doc map for humans and agents: [`llms.txt`](llms.txt). Full contributor contract:
[`CONTRIBUTING.md`](CONTRIBUTING.md).
