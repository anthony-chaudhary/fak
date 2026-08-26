# Session intent demo

A deterministic no-model witness for keeping minimum active effort, target active effort, and maximum elapsed time as distinct session-control semantics.

## Quickstart

From the repository root:

```console
go run ./cmd/sessionintentdemo -selfcheck
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- one hour of active work remains in the `continue` state;
- two hours of active work becomes normally `eligible` to stop;
- ten hours of elapsed time produces the forced `timeout` decision.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 5 seconds; it does not wait for the illustrated two-hour or ten-hour durations.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim to run a long-lived agent, sleep for the encoded durations, or enforce an operating-system scheduler. It evaluates frozen progress snapshots against the session-intent contract.

The implementation and its deeper contract live in [`internal/sessionintent`](../../internal/sessionintent/) and [`docs/long-session-value.md`](../../docs/long-session-value.md).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
