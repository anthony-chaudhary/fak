# Gemini CLI context — fak

The canonical agent instructions for this repo are in [`AGENTS.md`](AGENTS.md) — read it
first for build/test/run, the repo map, and the rules. Curated doc map:
[`llms.txt`](llms.txt). Contributor contract: [`CONTRIBUTING.md`](CONTRIBUTING.md).

This repo is **fak**, an agent kernel: one Go binary that sits between an AI agent and the
tools it calls and adjudicates every call before it runs — deny by structure, repair
malformed calls, quarantine poisoned results. Its MCP server is wired in `.mcp.json`.

Must-know rules (enforced below the agent layer):

- Work directly on the trunk (`main`); never open a feature branch or worktree — the
  trunk guard refuses off-trunk commits.
- Commit by explicit path (`git commit -- <paths>`, never `git add -A`); sign off with
  `git commit -s` (DCO).
- The Go module is the repository root — run `go` commands from the clone root
  (`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` resolves directly).

To run the Gemini CLI behind the kernel (governed tool calls via MCP / an
OpenAI-compatible gateway), follow [`docs/integrations/gemini-cli.md`](docs/integrations/gemini-cli.md).
