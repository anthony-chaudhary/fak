# Synthetic fault plugin

This fixture is a deliberately small newline-JSON subprocess used to prove the
`internal/extensionfault` containment spine. Select a mode with `--mode`:

- `healthy`: answers calls.
- `crash`: exits before its readiness frame.
- `startup-hang`: never emits its readiness frame.
- `hang`: becomes unresponsive after readiness.
- `leak`: allocates and spins in background goroutines, then hangs on a call.
- `malformed`: emits an invalid response frame.

The supervisor owns startup and call deadlines, kills the failed process,
restarts only within a fixed budget, and quarantines that extension while other
extensions remain usable. This is not a general plugin protocol, broker,
sandbox, resource-usage meter, descendant-process jail, or live `fak` MCP
registration surface. OS-level CPU/memory limits and durable quarantine state
remain future work.
