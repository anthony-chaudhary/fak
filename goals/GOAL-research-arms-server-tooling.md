---
loop: goal
goal_slug: research-arms-server-tooling
witness: "go test -v ./internal/researcharm/... && go test -v ./cmd/fak -run TestArms"
budget: { max_iters: 15 }
lane: researcharm
---
# Objective
Build Go tooling to support concurrent research project arms on the same machine hitting the fak server, providing request origin attribution (answering where requests are coming from), concurrency limiting, arm-level leasing (shared and exclusive), and CLI inspection tools (`fak arms status`, `fak arms who`, `fak arms lease`, `fak arms limit`).

# Non-Goals
- Do not modify frozen ABI (`internal/abi`).
- Do not disrupt active non-arm requests or existing gateway semantics when arms tooling is default/unconfigured.
- Do not introduce non-Go scripts (no Python, shell, or PowerShell).
- Do not break existing gateway tests or API contracts.

# Plan
- [x] 1. Define `internal/researcharm` package with request attribution, loopback socket/PID origin lookup, arm registry, lease management (shared/exclusive), and concurrency limiter.
- [x] 2. Write unit tests for `internal/researcharm` ensuring full coverage of attribution, leasing, concurrency throttling, and refusal reasons.
- [x] 3. Wire `researcharm` into `internal/gateway` HTTP middleware and add endpoints (`GET /v1/fak/arms`, `GET /v1/fak/arms/traffic`, `POST /v1/fak/arms/lease`, `POST /v1/fak/arms/limits`).
- [x] 4. Implement CLI verb `fak arms` in `cmd/fak/arms.go` with commands `status`, `who`/`traffic`, `lease acquire|release|list`, and `limit set`.
- [x] 5. Wire CLI dispatch in `cmd/fak/main.go` and add CLI tests in `cmd/fak/arms_test.go`.
- [x] 6. Run verification suite (`go test ./internal/researcharm/...` and affected tests) and run live query against the running server.

# Results and Verification Evidence
1. `internal/researcharm`: Implemented `types.go`, `origin.go`, `coordinator.go`, with 6 passing unit tests covering explicit headers, trace inference, user-agent heuristics, concurrency limiting, shared/exclusive leasing, and lease enforcement.
2. `internal/gateway`: Added `/v1/fak/arms`, `/v1/fak/arms/traffic`, `/v1/fak/arms/lease`, `/v1/fak/arms/limits` endpoints, request admission hook, and lease accounting in `debitServedSessionTurn`. Verified with `TestGatewayArmsEndpoints`, `TestGatewayArmsThrottling`, and `TestGatewayArmsExclusiveLease`.
3. `cmd/fak/arms.go`: Implemented `fak arms status`, `fak arms who`, `fak arms lease acquire|release|list`, and `fak arms limit`, with live socket inspection fallback to directly identify caller PIDs and commands hitting the server. Verified with `TestArmsCLIFullLifecycle`.
4. Architecture and layer conformity: Registered `"researcharm": 2` in `internal/architest/architest_test.go`. `TestEveryPackageDeclaresTier` and `TestNoUpwardImports` pass.
5. Live execution: Verified with `./fak arms status` and `./fak arms who`, directly identifying local caller processes (PID 4741 `./fak chat`, PID 7497 `prometheus`) and 15 research project arms across 80 active sessions.

# Scratch / last-refusal
