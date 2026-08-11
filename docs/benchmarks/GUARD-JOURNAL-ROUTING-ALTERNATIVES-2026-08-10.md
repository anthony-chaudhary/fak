# Guard-journal action-routing alternatives â€” 2026-08-10

Status: **INCOMPLETE**. Issue [#6349](https://github.com/anthony-chaudhary/fak/issues/6349) tracks real policy/rule-system witnesses.

`internal/guardroute.Decide` routes the same empty, structural-anomaly, below-threshold reason, and at-threshold reason folds to exact no-op, pickable-finding, or issue actions.

| Arm | Class | Status |
|---|---|---:|
| fak native typed router | native | available |
| count-threshold-only | tuned baseline | available; misses structural routes |
| fak + DOS decisions | integration | write/read-back seam witnessed; benchmark row incomplete |
| OPA | external | unavailable |
| Cedar | external | unavailable |
| Drools | external | unavailable |
| Alertmanager | external | unavailable |

Completion requires action/severity accuracy, false/missed routes, latency/throughput, CPU/RSS, network/storage, setup/operator time, and total cost from pinned commands and independent read-back. Local tests lock inventory and zero unavailable measurements. Three Windows/amd64 samples were 1,418, 987.2, and 1,013 ns/op (median 1,013 ns/op; 881 B/op; 11 allocs/op); this is not a cross-system ranking, and no external ranking exists yet.

## DOS decisions continuation witness — 2026-08-10

Prerequisite [#6397](https://github.com/anthony-chaudhary/fak/issues/6397) is closed and commit
`414bec7d7be6269f98db95e6ff7bad3366718f0b` is independently contained by `origin/main`.
Against a fresh private temporary git workspace carrying only `dos.toml`, the shipped
`cmd/fak-dos` seam created exactly one host decision with key `bench-6349-blank`, action
`OPEN_ISSUE`, severity `P1`, and payload `{"case":"blank-reason"}`. A separate
`decisions list --json` process returned exactly one matching row and exact
`HOST_QUEUE_ITEM` / `HUMAN` metadata; direct parsing also confirmed that `source_path`
remained under the temporary workspace. `decisions remove` returned `removed:true`, a
final independent list returned `[]`, and bounded cleanup removed only the verified
temporary workspace and temporary executable.

This clears the former DOS writer blocker, but does **not** complete #6349. Rechecking the
seven-arm acceptance matrix against the independently posted continuation evidence leaves
these exact gates:

- OPA, Cedar, Drools, and Alertmanager still need independently emitted/read-back severity,
  not severity inferred from action-policy configuration.
- Drools still needs repeated-batch p50 and p95 latency.
- Every external arm still needs production-representative network/storage,
  setup/operator-time, and total-cost accounting; warm local zero-network and zero
  incremental-software-cost observations do not satisfy those fields.
- The DOS arm still needs the full common five-fold benchmark row (accuracy,
  false/missed routes, latency/throughput, CPU/RSS, network/storage, operator time, and
  total cost); this continuation proves only the formerly blocked structured queue seam.

No cross-system ranking or completion claim is made until every field above has an
independent witness.
