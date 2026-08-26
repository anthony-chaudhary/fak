# Harness stack preflight demo

A deterministic offline witness that combines stack resolution, workload fit, and support evidence into an allow path and a refused path with ranked alternatives.

## Quickstart

From the repository root:

```console
go run ./cmd/stackpreflightdemo -selfcheck
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- the minimum supported path is allowed while recommendations remain warnings;
- a mandatory unsupported T4 dependency blocks launch;
- the refusal includes ranked alternatives instead of a bare dead end.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 5 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim to probe live hardware, download model artifacts, or certify that a deployment will work outside the frozen fixtures. It proves the preflight decision contract over checked-in evidence.

The implementation and its deeper contract live in [`internal/stackpreflight`](../../internal/stackpreflight/) and [`internal/supportgraph`](../../internal/supportgraph/).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
