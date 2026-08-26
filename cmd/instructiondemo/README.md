# Dynamic instruction composition demo

A deterministic no-model witness showing that turn-scoped instructions can change while the cacheable stable prefix remains unchanged, and that an invalid stable-prefix mutation is denied.

## Quickstart

From the repository root:

```console
go run ./cmd/instructiondemo -selfcheck
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- two turns retain the same stable-prefix digest while their full instruction digests differ;
- the provider-policy attempt to replace the stable prefix is classified as a denied contract outcome;
- the JSON receipt carries the public instruction-composition contract and outcome counts.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 10 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim a provider-specific prompt-cache hit, billed token savings, or live remote instruction delivery. It exercises the local composition contract with frozen in-process fragments.

The implementation and its deeper contract live in [`internal/harnessinstructions`](../../internal/harnessinstructions/) and [`pkg/harnesskit`](../../pkg/harnesskit/).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
