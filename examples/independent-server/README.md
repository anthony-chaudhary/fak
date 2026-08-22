# Independent server receipt handoff

This example proves that server lifecycle and harness consumption stay separate. It starts a local fixture through the production server lifecycle, gives a sibling harness product only the immutable receipt path, performs one chat call, tears down the receipt-owned process, and then rereads every artifact after both products have exited.

From the repository root:

```sh
go run ./examples/independent-server -selfcheck
```

The command needs no key, download, external network, container, or GPU. Its JSON reports the two clean product roots, artifact and adapter identity, readiness evidence, the harness chat, ownership-checked teardown, per-phase elapsed milliseconds, and zero harness lifecycle calls.
