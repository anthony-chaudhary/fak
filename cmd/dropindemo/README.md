# Drop-In Demo

`dropindemo` shows the adoption path for putting fak in front of the coding agent you
already run. It renders the same wiring rules `fak guard` uses for Claude Code, Codex,
OpenCode, and Aider: provider detection, upstream base URL, and the child-only environment
variables that point the agent at the gateway.

Run the browser gallery:

```bash
go run ./cmd/dropindemo -addr 127.0.0.1:8154
```

Open `http://127.0.0.1:8154`.

Run it behind a preserved path prefix:

```bash
FAK_DEMO_BASE_PATH=/dropin go run ./cmd/dropindemo
```

Run the headless witness:

```bash
go run ./cmd/dropindemo -selfcheck
```

The self-check completes in seconds after the Go build cache is warm. It is deterministic:
it reads only the in-repo `internal/dropin` wiring table and does not call a model,
provider API, browser, or network service.

Print the terminal gallery:

```bash
go run ./cmd/dropindemo -print
```

Inspect one agent's dry-run wiring:

```bash
go run ./cmd/dropindemo -agent codex
```

This demo does not claim that a model request was sent, that a provider credential works,
or that the safety floor blocked a dangerous tool. It is the zero-key wiring proof for the
drop-in entry point. Use `cmd/guarddemo` for the safety-floor behavior proof.
