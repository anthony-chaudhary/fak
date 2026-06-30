# fak — repo rules for Continue

> Continue auto-loads rules from `.continue/rules/`. The canonical, fuller agent
> instructions are in [`AGENTS.md`](../../AGENTS.md) — read it first for build/test/run,
> the repo map, and the rules. Doc map: [`llms.txt`](../../llms.txt). Contributor
> contract: [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

This repo is **fak**, an agent kernel: one Go binary that sits between an AI agent and the
tools it calls and adjudicates every call before it runs — deny by structure, repair
malformed calls, quarantine poisoned results. Its MCP server is wired in `.mcp.json`.

Must-know rules (enforced below the agent layer):

- Work directly on the trunk (`main`); never open a feature branch or worktree — the
  trunk guard refuses off-trunk commits.
- Commit by explicit path (`git commit -- <paths>`, never `git add -A`); sign off with
  `git commit -s` (DCO).
- The Go module is the repository root — run `go` commands from the clone root.

To run Continue's own model behind the kernel (governed tool calls, quarantine), follow
[`docs/integrations/continue.md`](../../docs/integrations/continue.md).
