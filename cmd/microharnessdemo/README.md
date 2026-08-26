# Bounded microagent harness-construction demo

`microharnessdemo` shows bounded fak-native microagents selecting tools and proof requirements,
then returning compact receipts to a root context. Start here with the fixture-backed selfcheck;
the live flags exercise the same receipt boundary through a configured provider endpoint.

## Quickstart

Prerequisite: Go 1.26 or newer. From the repository root, run:

```sh
go run ./cmd/microharnessdemo -selfcheck -ledger off
```

With a warm Go build cache this completes in under one second; an initial run may take longer
while Go compiles the package. `-ledger off` keeps the documentation witness self-contained.
The fixture report and rendered stdout are deterministic and byte-identical for the same goal;
`TestMicroharnessDeterminism` repeats the report 100 times, and
`TestExampleOutputMatchesSelfcheck` locks the render to [EXAMPLE-OUTPUT.md](EXAMPLE-OUTPUT.md).

## What you see

The output shows three admitted task classes, bounded depth and turns, the compact receipts kept
by the root, and a controlled comparison of deterministic accounting units. `PASS` means the
receipt, task-class, recursion, quality-coverage, and context-reduction assertions all held.

## Scope

This controlled fixture demonstrates receipt-bound context isolation and bounded recursive
delegation. It does **not claim** live-model quality, provider latency, billed-token savings, or
universal superiority over monolithic agents. The benchmark's time, token, cache, and cost fields
are deterministic fixture units, not provider telemetry. See the
[bounded-microagent claim record](../../docs/claims/bounded-microagents-construct-harnesses-cmd-microharnessdemo.md)
for the deeper evidence boundary.
