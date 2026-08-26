# Support graph demo

A deterministic offline witness for keeping supported, unsupported, stale, and unknown support states distinct when querying a frozen hardware/model tuple graph.

## Quickstart

From the repository root:

```console
go run ./cmd/supportgraphdemo -selfcheck
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- the exact L4 tuple is supported by decisive evidence;
- the exact T4 tuple is refused with a fallback and penalty;
- old evidence becomes stale and an unseen layout remains unknown rather than guessed.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 3 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim to probe a live GPU, driver, kernel, or artifact registry. It demonstrates query semantics against the pinned `awq.json` fixture.

The implementation and its deeper contract live in [`internal/supportgraph`](../../internal/supportgraph/) and the pinned [`awq.json`](../../internal/supportgraph/testdata/awq.json).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
