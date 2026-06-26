# tokendemo quickstart

Run the deterministic smoke first:

```bash
go run ./cmd/tokendemo -selfcheck
```

That command exercises the three documented ledger fixtures through the real
kernel path and should finish in a few seconds with `OK - all suites reproduced
the documented ledger invariants.` Read `main.go` next for the CLI entrypoints;
the parallel and served modes are larger witnesses over the same core ledger.
