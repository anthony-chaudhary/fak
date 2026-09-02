# Local hardware benchmark receipts

`fak bench local` promotes the v1 receipt workflow introduced by issue #10421
(commit `cb55a247ddce88186aac114b69509fdc8cfaaa31`) into the first-class CLI.
The implementation is shared from `internal/localbench`; this example remains a
thin compatibility wrapper rather than a second receipt implementation.

## Discover and run

```console
fak bench local inventory
fak bench local run --benchmark modelbench --engine fak-native --out receipt.json -- <operator-selected command> [args...]
fak bench local verify receipt.json
fak bench local submit receipt.json
```

`inventory` prints normalized local hardware plus the existing fak benchmark
catalog. Select a catalog `name`, choose and label the engine explicitly, and
provide the exact child command after `--`. The catalog's `run` field is a recipe
for discovery only; `fak bench local run` does not execute it automatically.

The command is run exactly as supplied. There is **no automatic fallback** to
llama.cpp, Ollama, LM Studio, or any other external engine. The receipt preserves
the operator-selected benchmark, engine label, and scrubbed command.

## What you'll see

A captured run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md): the `inventory`
hardware block plus catalog excerpt, a sealed receipt produced by a trivial
`/bin/echo hello` child, the `VERIFIED` line from `verify`, and the no-upload
`submit` packet. On a warm build cache, `inventory` completes in under a second
and each of `run`/`verify` completes in under a second; the first `go run` of the
example pays a one-time compile. A real benchmark child takes however long that
child takes — the receipt records its duration either way.

## Scope — what this does not claim

This example demonstrates the receipt *workflow*, not engine performance: it does
not claim any speed or quality number for the capture host, and the captured
receipt runs a trivial child (`/bin/echo hello`) — deliberately not a model
benchmark — so its duration and digest fields demonstrate the sealed-receipt
mechanics only. The catalog `run` recipes are shown for discovery and are not
executed automatically. What this does not claim either: that a submitted receipt
was reviewed by anyone but its operator — the scrubber covers common secret
shapes and cannot identify every application-specific secret.

The legacy example entry point delegates to the same implementation:

```console
cd examples/local-hardware-benchmark
go run . inventory
go run . run --benchmark modelbench --engine fak-native --out receipt.json -- <command> [args...]
go run . verify receipt.json
go run . submit receipt.json
```

## Receipt and privacy contract

Schema: `fak.local-hardware-benchmark.receipt/v1`.

The receipt records:

- catalog benchmark name and explicit engine label for new runs;
- exact command argv after credential and home-path scrubbing;
- UTC start/end timestamps, duration, exit status, and raw-output digest/size;
- bounded scrubbed child output;
- normalized OS, architecture, CPU, memory, accelerator, and toolchain data;
- fak, repository, and Go provenance;
- a SHA-256 integrity seal over the canonical receipt.

Existing #10421 v1 receipts without the newer optional `benchmark` and `engine`
fields remain verifiable. Verification fails closed on unknown schema versions,
unknown JSON fields, trailing JSON data, and integrity mismatches.

Scrubbing covers common token/password/authorization assignments, bearer tokens,
home-directory paths, host/user/machine identifiers, and secret-valued command
flags. Review the receipt and generated submission packet before sharing; no
scrubber can identify every application-specific secret.

## Submission is no-upload

`submit` verifies the receipt and deterministically prints:

1. a Markdown issue body; and
2. a prefilled GitHub issue URL.

It never opens the URL or uploads the receipt. The operator must review the
packet and explicitly submit it.

## Child failures

A failed or missing child command still produces a sealed receipt when the output
path is writable. The CLI then returns failure and reports the recorded exit
status (`-1` when the process could not start), preserving the failure as evidence.
