# Workload-fit selection demo

A deterministic offline witness for selecting different harness candidates for coding and legal-review workloads while keeping compatibility separate from purpose-specific fitness.

## Quickstart

From the repository root:

```console
go run ./cmd/workloadfitdemo -selfcheck
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the full witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- the coding fixture selects `ponytail@r8`;
- the legal-review fixture selects the purpose-built legal harness;
- the rejected coding harness carries unmet legal-review requirements and their evidence sources.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 3 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete output with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim to benchmark candidate quality, certify legal advice, or recommend a production harness. It demonstrates deterministic selection over a frozen fixture matrix.

The implementation and its deeper contract live in [`internal/workloadfit`](../../internal/workloadfit/) and the honesty boundaries in [`CLAIMS.md`](../../CLAIMS.md).

## What you see

A passing run ends with an explicit PASS/selfcheck verdict and includes the positive and negative evidence summarized above. See the full checked-in capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
