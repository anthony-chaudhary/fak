# Micro-context bounded-fan-out demo

`microcontextdemo` exercises many logical agent contexts over a bounded physical-worker pool
while keeping one immutable base context installed. Start here with the compact synthetic spine
in `captured_selfcheck.go`; `main.go` exposes the larger research and live-witness modes.

## Quickstart

Prerequisite: Go 1.26 or newer. From the repository root, run:

```sh
go run ./cmd/microcontextdemo -captured-selfcheck
```

After a warm Go build cache the selfcheck completes in under one second; an initial run may take
longer while Go compiles the package. The command uses a local deterministic fixture: successful
output is byte-identical across re-runs and is locked to [EXAMPLE-OUTPUT.md](EXAMPLE-OUTPUT.md) by
`TestCapturedSelfcheckMatchesExampleOutput`.

## What you see

The selfcheck asserts that 32 logical contexts complete exactly once, share one base installation,
and never exceed four concurrent physical workers. It prints stable contract facts and leaves
timing and host-resource observations to the full JSON mode.

## Scope

This fixture demonstrates bounded fan-out and shared-base accounting. It does **not claim** model
quality, token throughput, KV-cache residency, GPU behavior, or live-provider performance. See the
[micro-context research status](../../docs/research/micro-context-fabrics.md) for the broader staged
evidence and remaining limits.
