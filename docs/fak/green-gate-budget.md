# Green-Gate Budget

> **Audience.** Anyone tuning the incremental verify loop — by the end you'll know the tracked time budgets, how `fak affected` enforces them, and how to override a budget for one run.

`fak affected` is the incremental verify-loop gate for a path-scoped change. It runs
`go test` on the changed package plus every package that imports it, then compares the
measured wall time with a tracked budget. A run that passes tests but exceeds the budget
returns `GATE_LATENCY_REGRESSION`; the full `make ci` suite remains the merge/CI backstop.

## Tracked Budget

| Cell | Command | Budget | Gate |
|---|---|---:|---|
| incremental warm | `make test-affected` | 30s | hard fail via `fak affected --budget` |
| incremental cold | `go clean -testcache && VERIFY_LOOP_BUDGET=60s make test-affected` | 60s | hard fail via `fak affected --budget` |
| full warm | `make ci` | observe | authoritative backstop, not shortened |
| full cold | `go clean -cache -testcache && make ci` | observe | authoritative backstop, not shortened |

The default budget is committed in `Makefile` as `VERIFY_LOOP_BUDGET ?= 30s`, and
operators can tighten or loosen it explicitly for one run without editing the tree:

```bash
VERIFY_LOOP_BUDGET=45s make test-affected
```

## Report

The `make test-affected` budgeted run writes `.fak/verify-loop-affected.json` with
schema `fak.verify_loop.v1`. The report records the selected packages, total package
count, the exact `go test` command, elapsed milliseconds from changed-file discovery
through package selection and test completion, budget milliseconds, and one verdict:

| Verdict | Meaning |
|---|---|
| `OK` | tests passed and elapsed time stayed within budget |
| `TEST_FAILED` | the real tests failed; latency is secondary |
| `GATE_LATENCY_REGRESSION` | tests passed, but the verify loop exceeded budget |
| `NOOP` | no Go package was affected by the selected paths |

Representative one-file measurement:

```bash
go run ./cmd/fak affected --file internal/affectedtests/affectedtests.go --budget 30s --report .fak/verify-loop-affected.json
```

This is a net-true inner-loop win only within its scope: it reruns the import-graph
closure for the changed Go files and does not replace `make ci`. Non-Go runtime data
dependencies, build-tag-specific behavior, and cross-repo services still ride the full
gate.

## Read next

- [loop-tool-map](loop-tool-map.md) — which verb to reach for at the verify stage this gate sits in.
- [Engineering is building loops](../explainers/engineering-is-building-loops.md) — the loops doctrine behind a tracked inner loop.
