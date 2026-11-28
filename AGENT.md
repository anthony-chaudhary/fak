# AGENT.md — fak

> Single-file `AGENT.md` overlay for Amp and peer harnesses. Canonical rules, full build/test/run, and repo map live in [`AGENTS.md`](AGENTS.md); humans read [`README.md`](README.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

**fak** is an agent kernel (Go module at repo root; MCP wired in `.mcp.json`) that adjudicates tool calls before execution.

## Quick commands & delta rules

- Build and proof: `go build -o fak ./cmd/fak`, `make test-fast` (or `./test.ps1` under WSL on Windows), `make ci`, and `./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"`.
- Stay on `main` (use detached `fak worktree worker prepare|land|reap` only for worker isolation) and commit explicit paths via `fak commit --path <p> -m "<subject> (fak <leaf>)"` or `fak sweep --apply`.
- Keep worker tasks atomic (1–3 files, single package, one witness), abstain on isolated high-difficulty boundaries, and close findings with the next checkable step.

Recipes: [`docs/integrations/`](docs/integrations/README.md) ([Amp](docs/integrations/amp.md), [OpenCode](docs/integrations/opencode.md)) · Doc map: [`llms.txt`](llms.txt).
