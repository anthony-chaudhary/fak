# Independent server receipt handoff

This example proves that server lifecycle and harness consumption stay separate. It starts a local fixture through the production server lifecycle, gives a sibling harness product only the immutable receipt path, performs one chat call, tears down the receipt-owned process, and then rereads every artifact after both products have exited.

## Quickstart

Start here from the repository root. Go 1.26+ is the only prerequisite; the
repository toolchain setting can fetch it automatically.

```sh
go run ./examples/independent-server -selfcheck
```

The command needs no key, model download, external network, container, or GPU. On
a warm Go build cache it completes in under 10 seconds; the first run can take
longer while Go fetches the declared toolchain and compiles the package.

## What you see

The JSON reports two clean product roots, artifact and adapter identity, readiness
evidence, one harness chat, ownership-checked teardown, per-phase elapsed
milliseconds, and zero harness lifecycle calls. A stable projection from an actual
passing run is captured in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md). The assertion
test in `selfcheck_test.go` also rereads the receipt, binding, state, and event log
after both products exit instead of trusting the printed verdict.

The fixture requests, event counts, and pass/fail invariants are deterministic and
safe to rerun. The complete JSON is deliberately not byte-identical: temporary
process identities, build-dependent executable digests, receipt digests, and
elapsed milliseconds vary between runs.

Current platform limitation: the captured selfcheck and package test pass under
Linux/WSL. On the Windows control host used for issue #9245, the same run reached
owned teardown but `serverlifecycle` reported `STOP_TIMEOUT` after killing the
fixture process. Use WSL for this witness until the production Windows lifecycle
wait is repaired; this demo does not mask or reinterpret that refusal.

## What this does not claim

This demo does not claim model quality, remote-server reliability, or production
throughput. Its loopback `llama-server` fixture proves the ownership and immutable
receipt handoff contract without a real model. See the
[independent-server integration guide](../../docs/integrations/independent-server.md)
for the corresponding real-server sequence and credential boundary.
