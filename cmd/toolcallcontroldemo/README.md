# Tool-call control demo

A deterministic no-model witness for classifying repeated, batchable, low-information, and necessary tool proposals before execution, with a control-versus-prefilter ablation receipt.

## Quickstart

From the repository root:

```console
go run ./cmd/toolcallcontroldemo -selfcheck -pretty
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- an unchanged repeated read is reused, compatible searches are batched, and a low-information browse is deferred;
- the necessary write remains allowed instead of being optimized away;
- the emitted ablation compares the permissive control arm with the deterministic prefilter arm.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 3 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim to execute the proposed tools, call a model, or treat its fixture token counts as a production cost measurement. It proves the local decision and accounting shape.

The implementation and its deeper contract live in [`internal/toolcallcontrol`](../../internal/toolcallcontrol/) and the claim boundaries in [`CLAIMS.md`](../../CLAIMS.md).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
