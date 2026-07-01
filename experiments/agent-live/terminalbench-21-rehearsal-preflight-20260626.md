# Terminal-Bench 2.1 Rehearsal Preflight

- Generated: `2026-06-26T23:16:10Z`
- Benchmark: `Terminal-Bench 2.1 raw/fak rehearsal preflight`
- Issue: `#900`
- Dataset: `terminal-bench/terminal-bench-2-1`
- Status: `BLOCKED_PREFLIGHT`
- Evidence class: `REHEARSAL_PREFLIGHT`
- Result claim allowed: `false`
- Oracle smoke ready: `false`
- Raw paid smoke ready: `false`
- fak paid smoke ready: `false`
- Boundary: Host-readiness preflight only: probes whether this host can attempt the Terminal-Bench 2.1 raw/fak rehearsal (Harbor, Docker engine, OPENAI_API_KEY, fak gateway). It is never a benchmark result; result_claim_allowed stays false. The oracle smoke and the credentialed raw-vs-fak compare remain the result-bearing artifacts.

## Host gates

| Gate | OK | Detail |
|---|:---:|---|
| `harbor_present` | yes | harbor 0.15.0 |
| `docker_engine_up` | no | docker engine not reachable |
| `oracle_smoke_artifact` | yes | not required (pass --oracle-artifact to enforce oracle-before-paid) |
| `openai_api_key_present` | no | OPENAI_API_KEY not set in this shell |
| `fak_gateway_reachable` | no | not probed (http://localhost:8080/v1); pass --probe-gateway to check |

## Blocking reasons

- `DOCKER_ENGINE_DOWN`
- `OPENAI_API_KEY_MISSING`

- Next action: start the Docker engine, then re-run the preflight and the official oracle smoke before any paid run
- Official-run contract: `experiments/agent-live/terminalbench-official-run-contract-20260626.json`
- Submission packet: `docs/benchmarks/TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md`
- Candidate task ids: `go-cli-help-benign, python-config-fix-danger-after-tests`

## Known dependencies before any result claim

- a running Docker engine for every Harbor task container (oracle, raw, and fak arms)
- OPENAI_API_KEY in the rehearsal shell for the credentialed raw baseline (#900)
- a running, reachable fak gateway so the fak arm routes through it (the client-facing /v1/responses inbound route shipped in #925; the remaining requirement is that the gateway is up on this host)
- explicit paid-spend authority before the credentialed raw/fak smoke pair
- the official Harbor grader output as the sole authority for any pass-rate number
