# marketdemo — inert extension descriptor conformance

Run the real extension-market conformance witness with no network, model, or durable state:

```bash
go run ./cmd/marketdemo -selfcheck
```

The demo validates one inert descriptor against the production `internal/market` parser and prints a deterministic PASS witness. After Go compilation, the selfcheck normally completes in under one second. Use `-json` to inspect the descriptor catalog.

## What this demo does not claim

This demo does not claim to load or execute extension code, prove artifact provenance beyond the fixture digest, or establish compatibility with an external marketplace. It proves only local descriptor parsing and conformance. See [`../../CLAIMS.md`](../../CLAIMS.md) for the project honesty ledger.

The command creates no files or external state, so no cleanup is required.
