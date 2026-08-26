# Harness stack resolution demo

A deterministic offline witness for resolving a satisfiable harness stack and preserving the transitive reason when an unsatisfied hardware requirement forces refusal.

## Quickstart

From the repository root:

```console
go run ./cmd/stackresolvedemo -selfcheck
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- the satisfiable fixture resolves to an `ALLOW` receipt;
- the incompatible fixture resolves to a `REFUSE` receipt naming `device.cuda.sm80`;
- the selfcheck exits zero only after both the positive and negative paths match their contracts.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 5 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim to install the resolved components, probe the local accelerator, or prove live inference compatibility. It resolves checked-in manifests and providers only.

The implementation and its deeper contract live in [`internal/stackresolve`](../../internal/stackresolve/) and its [`testdata`](../../internal/stackresolve/testdata/).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
