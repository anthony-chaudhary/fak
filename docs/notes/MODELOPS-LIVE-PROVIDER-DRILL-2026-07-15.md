---
title: "Exact-model live provider rollback drill"
description: "A bounded loopback proxy returned a provider-shaped 403 for one exact model on the live seam: rollback and recovery witnessed, production promotion still HOLD."
---

# Exact-model live provider rollback drill — 2026-07-15

Status: **ROLLBACK and recovery witnessed; production remains HOLD.**

Issue: #4832. Parent: #4634.

## Method

A temporary loopback proxy sat on the real Claude provider invocation seam for one bounded drill. It parsed only the request's exact `model` field:

- requests for `claude-opus-4-8` received a deterministic provider-shaped HTTP 403 with drill request ID `req_drill_opus_4832`;
- requests for any other exact model were forwarded unchanged to the configured upstream provider;
- request bodies, credentials, authorization headers, account identity, raw provider payloads, session IDs, UUIDs, and costs are not retained here.

The proxy was created under the OS temporary directory, listened only on loopback, and was stopped after the two bounded invocations. This is a transport-level fault injection around a real provider seam, not fabricated model output and not production traffic.

## Exact-ID read-back

| Arm | Requested exact ID | Provider-seam result | Bound |
|---|---|---|---|
| Candidate fault | `claude-opus-4-8` | exit 1; HTTP 403; `is_error=true`; `terminal_reason=api_error`; 188 ms | one non-sensitive prompt, no tools |
| Fallback recovery | `claude-sonnet-4-6` | exit 0; two forwarded HTTP 200 responses; provider metadata named `claude-sonnet-4-6`; result `DRILL_SONNET_RECOVERED`; 3,007 ms wall / 3,869 ms API | one non-sensitive prompt, no tools |

The duplicate proxy events per arm are the provider client's bounded request behavior; both events carried the same exact model. No Haiku request was made.

## Gate evaluation

The sanitized exact-ID observations were evaluated by a fresh `fak` build from the committed trunk:

```text
candidate claude-opus-4-8:
  samples=1 success_rate=0 provider_error_rate=1 fallback_rate=1
fallback claude-sonnet-4-6:
  samples=1 success_rate=1 provider_error_rate=0 fallback_rate=0
required_tier=1
```

Result:

- exit `3`, action `ROLLBACK`, selected `claude-sonnet-4-6`;
- exact candidate breaches: success, provider-error, and fallback-rate thresholds;
- exact-ID outcome counters: one Opus ROLLBACK and one Sonnet PROMOTE;
- emitted alert contract: owner `modelops-oncall`, route `model-provider-incidents`, acknowledgement SLA 10 minutes, runbook `docs/model-production-readiness-inventory.md#reliability--rollback`.

A separate no-safe-fallback control marked the capability-safe Sonnet fallback unhealthy while leaving Haiku at capability tier 2. The same gate returned exit `4`, action `HOLD`, and reason `no healthy capability-safe fallback; hold for operator escalation`. It did not select Haiku for the tier-1 request.

## Scrubbed witness hashes

The private temporary directory retained the scrubbed summary and decisions for independent local read-back. Their SHA-256 values are:

- `scrubbed-summary.json`: `5832eff79dde3593f82641e1eaea6987f1620d2927597e70e7a0674a6d96a0f5`
- `rollback-decision.json`: `a73962edaed951c43e7ecea8ece73cfd0bbc40614d3f98a5a66b09c9da48b361`
- `hold-decision.json`: `35608fc8eb6a7d902988dfc199c46c76d2ad83ed0557657b7d7884cc3d7f9169`

## Verdict

The drill proves exact-model fault attribution, capability-safe traffic drain to Sonnet, successful live fallback recovery, structured alert ownership, and fail-closed HOLD when no safe fallback is healthy. It does **not** override the broader 40-run capability campaign's HOLD or promote production traffic.
