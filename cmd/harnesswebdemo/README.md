# Local harness web UI demo

A deterministic offline selfcheck for fak's loopback operator home, including its protocol events, approvals, typed failures, reconnect cursor, overview projections, and captured HTML receipt.

## Quickstart

From the repository root:

```console
go run ./cmd/harnesswebdemo --selfcheck
```

The command needs Go 1.26 or newer. The selfcheck is offline and needs no API key, model download, network access, or GPU. Exit code 0 means the complete render-and-protocol witness passed; a nonzero exit is a failed or invalid run.

## What the selfcheck proves

- the `fak.harness.run/v1` protocol renders the normal, approval, and typed-failure scenarios;
- reconnect resumes from an exclusive event cursor instead of replaying the full run;
- the operator overview projects runs, goals, dashboards, and two visual skins;
- the rendered HTML is reduced to a stable SHA-256 receipt.

## Expected runtime and repeatability

On a warm Go build cache the selfcheck completes in under 2 seconds; a cold toolchain build can take longer.

The fixture path is deterministic and safe to re-run. Two consecutive runs emit byte-identical output, and `main_test.go` compares the complete receipt with [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) so drift fails the captured regression test.

## Scope

This demo does not claim a remotely exposed production service, a complete coding-harness replacement, or live model and tool execution. The selfcheck drives the local offline product entirely in process; authenticated non-loopback deployment and live adapters are separate operating modes.

The full operating contract, interactive launch instructions, and live-workspace mode are documented in [`docs/harness-web-demo.md`](../../docs/harness-web-demo.md). The implementation lives in [`internal/harnessweb`](../../internal/harnessweb/).

## What you see

A passing run emits one `HARNESS_WEB_SELFCHECK ok` receipt with the protocol version, scenario event counts, overview counts, and captured HTML digest. See the exact checked-in receipt in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).
