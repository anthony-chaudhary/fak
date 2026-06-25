# causalbench - causal invalidation witness

`causalbench` is a no-model, deterministic witness for external-write invalidation.
It admits two cached read results under two different world-state witnesses, refutes one
witness, and proves that only the dependent read is evicted while the sibling stays warm.

## Prerequisites

Requires Go only from the repository root. It does not need a model, GPU, API key,
network service, or files outside this repo. The run completes in a few seconds and
returns exit code 0 only if all causal-invalidation invariants hold.

## Run it

```bash
go run ./cmd/causalbench -selfcheck
go run ./cmd/causalbench -selfcheck -out causalbench-witness.json
```

The optional `-out` path writes the machine-readable witness record. The default run
writes no files.

## What you see

```text
== fak causal-invalidation demo ==
ADMIT: read#1 + read#2 both serve byte-exact from cache (hit==fresh call).
WRITE: external write refutes commit:aaaaaaaaaaaa -> Revoke evicted 1 entry.
EXACT: read#1 now MISSES; read#2 stays a byte-identical hit (max|Delta|=0).
REFUSE: a re-fill under the refuted witness does not repopulate read#1.
OK - external write causally evicted exactly the dependent read.
```

The Go test path exercises the same `run()` function, so `go test ./cmd/causalbench`
guards the invariants without depending on terminal output.

## Scope

This demo does not claim a distributed cache deployment, cross-host transport, or a
performance number. It proves the structural vDSO witness ledger behavior over fixed
in-process calls. For the benchmark framing, see [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
