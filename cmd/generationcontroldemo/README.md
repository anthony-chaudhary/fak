# Continuous generation control demo

A deterministic no-model witness for interrupting an unsafe streamed tool call, checkpointing the accepted trajectory, and resuming it under a replacement worker epoch.

## Quickstart

From the repository root:

```console
go run ./cmd/generationcontroldemo -selfcheck
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- a streamed `Remove-Item` call is redirected before the destructive action is accepted;
- the checkpoint carries the accepted trajectory text across the handoff;
- the resumed epoch names the replacement worker and ends with `SELF_CHECK_PASS`.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 2 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim live provider token generation, operating-system process migration, or execution on the named CPU/GPU devices. The worker, model, and device fields are deterministic fixture identities for the control contract.

The implementation and its deeper contract live in [`internal/generationctl`](../../internal/generationctl/) and [`internal/streamrules`](../../internal/streamrules/).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
