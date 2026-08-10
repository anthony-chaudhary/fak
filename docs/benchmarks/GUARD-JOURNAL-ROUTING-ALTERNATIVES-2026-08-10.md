# Guard-journal action-routing alternatives â€” 2026-08-10

Status: **INCOMPLETE**. Issue [#6349](https://github.com/anthony-chaudhary/fak/issues/6349) tracks real policy/rule-system witnesses.

`internal/guardroute.Decide` routes the same empty, structural-anomaly, below-threshold reason, and at-threshold reason folds to exact no-op, pickable-finding, or issue actions.

| Arm | Class | Status |
|---|---|---:|
| fak native typed router | native | available |
| count-threshold-only | tuned baseline | available; misses structural routes |
| fak + DOS decisions | integration | unavailable |
| OPA | external | unavailable |
| Cedar | external | unavailable |
| Drools | external | unavailable |
| Alertmanager | external | unavailable |

Completion requires action/severity accuracy, false/missed routes, latency/throughput, CPU/RSS, network/storage, setup/operator time, and total cost from pinned commands and independent read-back. Local tests lock inventory and zero unavailable measurements. Three Windows/amd64 samples were 1,418, 987.2, and 1,013 ns/op (median 1,013 ns/op; 881 B/op; 11 allocs/op); this is not a cross-system ranking, and no external ranking exists yet.
