# End-to-end value-chain selfcheck witness — 2026-08-13

## Command

```powershell
fak micro --selfcheck --json
```

## Captured receipt

```json
{"verdict":"PASS","path":"kernel->session-gateway->scheduler->microagents","kernel":"in-process-http/mock","agents":2,"done":2,"http_count":2,"provider_tokens":58,"stopped":2,"concurrency_cap":1,"offline":true}
```

## What independently gates PASS

The command starts the real `internal/gateway` HTTP handler in-process, counts requests at
`/v1/chat/completions`, reads provider-shaped token usage from the HTTP planner response, and
reads retired agent state back from the shared `session.Table`. PASS requires all two agents,
requests, and sessions plus nonzero usage; a worker's return string cannot satisfy the gate.

## Scope

This is a deterministic **architecture witness** requiring no key, network, external model, or
GPU. `offline:true` is deliberate. It does not establish quality, cost, or latency
advantage; issue #6520 owns that paired tuned-baseline proof.
