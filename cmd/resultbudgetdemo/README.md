# Tool-result budget demo

A deterministic no-model witness for shaping a tool request down to a declared result budget while preserving exhaustive intent, observe-only mode, and unknown-tool fail-safe behavior.

## Quickstart

From the repository root:

```console
go run ./cmd/resultbudgetdemo -selfcheck -pretty
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- enforce mode reduces the known tool request from 500 items to the declared maximum of 10;
- observe mode reports the proposed change without changing effective arguments;
- exhaustive intent and unknown tool contracts pass through without silent truncation.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 3 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim that an external tool will honor the requested cardinality or that the fixture item counts equal production token savings. It proves deterministic request shaping and receipts at the adapter boundary.

The implementation and its deeper contract live in [`cmd/resultbudgetdemo/main.go`](main.go) and the broader truth ledger in [`CLAIMS.md`](../../CLAIMS.md).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
