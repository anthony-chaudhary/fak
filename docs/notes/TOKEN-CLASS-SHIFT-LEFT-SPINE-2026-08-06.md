# Token classes as a guard/server duty — spine and research note

Date: 2026-08-06

## Thesis

A scalar `total_tokens` is too late and too lossy for admission. The same 102k-token request can be cheap but prefill-heavy, expensive but decode-light, or mostly a provider cache read. The guard should classify the request **before spend and scheduling**, while the server should reconcile the forecast with provider-observed usage afterward.

The first stable vocabulary is deliberately small:

| Class | Known when | Economic profile | Scheduling profile | Shift-left lever |
|---|---|---|---|---|
| `input.uncached` | input known; cache status forecast | normal input price | prefill/KV allocation | stabilize/cache prefix, route prefill capacity |
| `input.cached` | input known; cache status forecast | discounted provider read | cache lookup/transfer and residual prefill | preserve affinity, verify cache expectation |
| `output.reserved` | maximum known, actual unknown | expensive upper bound | serial decode reservation | cap output, route decode capacity |

This is a **two-axis** classification: USD and scheduler units are intentionally separate. A class can dominate one without dominating the other. Provider-specific details (for example cache-write tiers, reasoning tokens, audio/image tokens, accepted/rejected prediction tokens) should map into a versioned superset rather than being collapsed into prompt/completion totals.

## Minimal working spine

`fak token-profile` accepts counts, per-million prices, and scheduler weights and emits `fak-token-profile/1` before a request runs. `--halo` is the captured, keyless example:

```text
HALO: 102k total tokens is not one load: a cache-heavy request has three independently actionable classes.
TOKEN PROFILE  phase=preflight total=102000 worst_case_usd=$0.077000 scheduler_units=40500
  input.uncached   tokens=10000    cost=$0.030000 load=10000 certainty=forecast
  input.cached     tokens=90000    cost=$0.027000 load=22500 certainty=forecast-cache-expectation
  output.reserved  tokens=2000     cost=$0.020000 load=8000 certainty=reservation-upper-bound
DOMINANCE cost=input.uncached load=input.cached
SHIFT LEFT: preserve cache affinity and verify the expected cache hit before admission
BOUNDARY: forecast only; reconcile provider-observed usage after completion
```

The point is not the default weights; they are explicit policy inputs. The value is making the vector available at the guard checkpoint, before a scalar budget or generic queue makes the wrong decision.

## Contract direction

1. **Forecast at guard/admission:** tokenizer-derived input, cache expectation + confidence/provenance, max output reservation, modality/tenant/model/route.
2. **Reserve at server/scheduler:** enforce separate prefill, cache-bandwidth/KV, and decode envelopes; attach the profile to route decisions.
3. **Observe at provider boundary:** preserve native usage fields and map them to canonical classes without losing raw provenance.
4. **Reconcile:** compare forecast/reservation/actual by class; release unused output reservation and score cache-prediction error.
5. **Expose:** one request ID across admission logs, `/debug/vars`, usage ledger, traces, and operator query.

## Explicit limits of the spine

- Counts are supplied, not yet extracted from a live request body.
- Cached input is an expectation, not a claim that the provider hit its cache.
- Scheduler weights are illustrative configurable units, not benchmarked hardware truth.
- `output.reserved` is an upper bound, not observed output.
- The spine does not yet gate or route a live request; those integrations are separately dispatchable issues.

These boundaries prevent a preflight estimate from masquerading as a billed or hardware-observed fact.
