# Gemini CLI context — fak

> Automated-contributor overlay for Gemini CLI. Read [`AGENTS.md`](AGENTS.md) first for canonical build/test/run, repo map, and rules. Humans: [`README.md`](README.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

**fak** is an agent kernel sitting between an AI agent and its tools (`default_open` bounded by a structural safety floor; MCP wired in `.mcp.json`).

## Core harness rules

- **Delegate substantive work and keep the coordinator context clean**: use guarded headless agents or isolated workers for investigation, implementation, tests, and review; retain only decisions and compact witnessed evidence in the primary context.
- **Trunk, commits, and Go module**: stay on `main` (only detached worker worktrees via `fak worktree worker prepare|land|reap`), commit via `fak commit --path <p> -m "<subject> (fak <leaf>)"`, and run `go` commands from the repo root.

## Operational sharp edges (Gemini 3.8 Flash & peers)

- **Token and diff economy**: keep CLI commentary under 3 lines, skip filler, and confine diffs to the requested lines without unasked scaffolding.
- **Circuit-break tool loops**: on guard denial or failure, read the receipt and run `fak recover <TOKEN>` or pivot instead of repeating the call.
- **External verification and discrete tools**: ground claims in observed `go test` / `fak validate` receipts; prefer structured file tools (`Read`, `Edit`, `Glob`, `Grep`) over complex shell pipelines.
- **Scoped abstention**: emit `ABSTAIN` with diagnostic context on concurrency lock invariants, `internal/abi`, or low-level SIMD/CUDA mechanics.
- **Wire & schema hygiene**: enforce turn alternation (auto-continue trailing model turns to avoid HTTP 400), strip `$schema` and `additionalProperties`, backfill `items` on array params, and preserve `thoughtSignature` tokens ([analysis](docs/notes/2026-09-03-gemini-3.8-flash-initial-feedback-and-guidance.md)).

## Human-attention-aware response format

- Lead with the verdict or decision on line 1, separate sections with blank lines, and keep paragraphs to 1–2 lines.
- Use scannable bullets (one claim + inline evidence per line) and close with the next checkable command or decision.

Gemini CLI integration: [`docs/integrations/gemini-cli.md`](docs/integrations/gemini-cli.md) · Doc map: [`llms.txt`](llms.txt).
