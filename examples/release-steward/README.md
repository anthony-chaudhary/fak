# Six-month release steward

A no-key, offline demo of long-horizon session identity: one stable macro identity
advanced through three baseline sessions, two restarts, mailbox delivery, a delegated
child, a micro-fleet operation, selective state promotion, and terminal
export/retirement. Its JSON receipt makes the session/sub-agent/micro-operation
boundaries inspectable and proves raw child history is not durable by default.

## Run

```sh
go run ./examples/release-steward
```

Requires: Go 1.26+ (the repository toolchain). No API key, network, model, or GPU.
The run completes in under a second once the build cache is warm; the first `go run`
pays a one-time compile.

## What you'll see

A captured run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md): a six-line lifecycle
summary followed by one JSON receipt (`fak.release-steward-demo/1`) carrying the
month-by-month receipts, the two promoted durable facts, and
`"raw_child_history_retained": false`. The run is deterministic — re-running prints
byte-identical output and exits 0.

## Scope — what this does not claim

This demo simulates the *boundaries* in one process; what it does not claim: it does
not run real models (`model` fields name a class, not an engine), it does not survive
an actual machine restart, and it does not exercise the durable-memory subsystem it
illustrates — the `durable_memory` entries are fixture strings carried by the demo.
The receipt is a demo artifact for inspecting the identity/boundary shape, not the
production release-steward ledger, and nothing here is a performance claim.
