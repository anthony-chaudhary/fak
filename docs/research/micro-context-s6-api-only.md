---
title: "Micro-context S6: API-only shared-base adapter"
description: "A process-local provider admission seam declaring only observable controls — requests, tokens, concurrency, spend, and cache telemetry — around the endpoint."
---

# Micro-context S6: API-only shared-base adapter

**Status:** witnessed on 2026-08-06 with a keyed billing credential. The adapter makes no controlled-kernel claim.

S6 adds a process-local provider admission seam around the existing OpenAI-compatible micro-context endpoint. `APIProviderShape` declares only observable provider controls: RPM, TPM, concurrency, spend estimation, cache-control shape, and cache-telemetry shape. A bounded concurrency queue waits rather than dropping excess local workers; conservative token/spend reservations reconcile to provider-reported usage; cancellation exits the wait; and `Retry-After` accepts both delta seconds and HTTP dates.

Offline contracts cover two provider shapes:

- **explicit cache:** provider cache control plus billed cached-token telemetry;
- **opaque:** byte-identical shared prefix, with `not-observable` as the strongest cache claim.

## Captured live run

The live run used a keyed Groq billing credential (not OAuth), four logical contexts and four orchestration workers over one admitted provider request at a time:

| Measure | Observed |
|---|---:|
| Submitted / completed / failed | 4 / 4 / 0 |
| Orchestration workers / provider concurrency cap | 4 / 1 |
| Wall time | 1.023 s |
| TTFT p50 / p95 / max | 0.201 / 0.221 / 0.385 s |
| Prompt / completion tokens | 2,876 / 32 |
| Provider-billed cached prompt tokens | 512 |
| Usage-bearing responses | 4 / 4 |

The earlier unadmitted 100-context probe hit provider TPM limits after 15 completions. With local admission active, four workers queued through one provider slot and all four completed. The live artifact scrubs the credential and endpoint.

```powershell
go test ./internal/microagent -run TestAPIAdmission
go run ./cmd/microcontextdemo -verify-api-only experiments/microcontext/s6-groq-api-only-4-pass-2026-08-06.json
```

## Claim boundary

`512` cached prompt tokens is provider billing telemetry and therefore a valid API-observable cache result. It is not direct evidence of provider-internal KV blocks, cache residency, or prefill tokens physically skipped. Provider prices were not pinned for this run, so spend gating is contract-tested but no dollar-savings claim is made. The adapter exposes provider constraints; it does not claim kernel scheduling, GPU occupancy, server-side batching, or controlled-kernel cache behavior.

