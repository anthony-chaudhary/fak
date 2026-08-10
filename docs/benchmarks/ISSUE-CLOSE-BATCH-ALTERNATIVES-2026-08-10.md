# Budgeted issue-close batching alternatives â€” 2026-08-10

Status: **INCOMPLETE**. Issue [#6364](https://github.com/anthony-chaudhary/fak/issues/6364) tracks real hosted-system witnesses.

The same seven issues are planned in batches of three under a five-call budget and one-call reserve, with exact allowed/held batches, request costs, and rollback commands. Arms: fak native; fixed chunks; fak + GitHub Issues; gh close loop; GitHub GraphQL batching; Jira bulk transition; Linear bulk update. Completion requires plan/cost/rollback accuracy, latency, CPU/RSS/network, setup/operator time, total cost, pinned versions/configuration, commands, and independent read-back. Three Windows/amd64 samples were 1,120, 1,049, and 1,332 ns/op (median 1,120 ns/op; 1,538 B/op; 21 allocs/op). Unavailable arms stay measurement-zero; no cross-system ranking exists yet.
