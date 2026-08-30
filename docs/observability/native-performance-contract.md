---
title: "fak-native performance observability contract"
description: "Frozen observability contract for fak-native performance metrics, labels, correlation identifiers, evidence handling, and compatibility."
---

# fak-native performance observability contract

Status: frozen contract v1 (2026-08-28)  
Machine-readable source: `internal/nativeperfobscontract.Frozen()`  
Validator: `go test ./internal/nativeperfobscontract`

This contract answers one bounded question: **does every performance-relevant stage of a fak-native model run have a stable exported signal, or an explicit justified exclusion?** The validator treats `UNKNOWN` as debt and fails closed. It also fixes the engine to `fak-native`; this contract cannot silently relabel llama.cpp execution as native evidence.

New evidence defaults to **Qwen3.8**. Qwen3.6 remains valid only for explicit regression, compatibility, historical comparison, or hardware/artifact constraints, and historical Qwen3.6 artifacts keep their original identity.

## Contract rules

- Metric names are stable, lowercase, and prefixed `fak_native_`.
- Types are limited to `counter`, `gauge`, and `histogram`; units are explicit.
- Every metric carries the bounded `engine` label. The only valid value for this contract is `fak-native`.
- Labels describe receipt-bounded enums. Request IDs, prompts, paths, hosts, and arbitrary model strings are forbidden. `model_family` is a curated bounded value, not an artifact path or free-form model name.
- `cardinality_budget` is the maximum intended live series count for the metric. Raising it is a contract change.
- Provenance names the existing fak receipt or producer. This contract does not create a parallel source of truth.
- Freshness states when the producer must update the signal. Missing observations are unsupported; they are never converted to zero.
- A surface is complete only with `status: exported` plus a contracted metric, or `status: excluded` plus a justification. Any other status, including `UNKNOWN`, fails validation.

## Native performance surface inventory

| Surface | Existing producer / truth | Contracted signal | Freshness |
|---|---|---|---|
| Model loading | `internal/modelengine/lifecycle.go` | `fak_native_model_load_seconds` | Each load attempt |
| Admission and queueing | `internal/modelengine/nativesched.go` | `fak_native_admission_wait_seconds` | Each admission/rejection |
| Tokenization | `internal/tokenizer/tokenizer.go` | `fak_native_tokenization_seconds` | Each encode/decode operation |
| Prefill | `internal/nativeperf/receipt.go` | `fak_native_phase_seconds{phase="prefill"}` and `fak_native_tokens_total{phase="prefill"}` | Phase receipt close |
| Decode | `internal/nativeperf/receipt.go` | `fak_native_phase_seconds{phase="decode"}` and `fak_native_tokens_total{phase="decode"}` | Phase receipt close |
| KV capacity/use | `internal/modelperfobs/kv_capacity.go`, `internal/xenginekv/` | `fak_native_kv_bytes` | Allocation, release, observation snapshot |
| Cache results | `internal/modelperfobs/cache_state.go`, native receipts | `fak_native_cache_events_total` | Each lookup result |
| Transfers | `internal/modelperfobs/bandwidth.go` | `fak_native_transfer_bytes_total` | Each completed native transfer |
| Kernel launches | `internal/metalgemm/execution_events.go` | `fak_native_kernel_launches_total` | Each observed fak-owned launch |
| Synchronization | native backend execution events | `fak_native_synchronization_seconds` | Each observed synchronization |
| Memory | `internal/modelperfobs/observation.go` | `fak_native_memory_bytes` | Lifecycle and observation boundaries |
| Output | `internal/modelengine/pipeline.go` | `fak_native_output_seconds` | Each completed output stage |
| Benchmark gates | `internal/nativebench/nativebench.go`, `internal/nativeperf/gate.go` | `fak_native_benchmark_gate_total` | Each gate decision |
| Promotion / revert | native benchmark gate receipt | `fak_native_promotion_total` | Each promote, hold, or revert decision |

The complete names, types, units, label sets, cardinality budgets, provenance, freshness, and unsupported semantics live in the machine-readable Go contract. Keep dashboards and exporters downstream of that contract instead of copying this table into a second schema.

## Existing receipts and dashboards

The contract cross-references the existing truth surfaces rather than replacing them:

- `internal/nativeperf` owns native run receipts, profiles, graph attribution, and performance gates.
- `internal/modelperfobs` owns memory, bandwidth, cache, KV-capacity, topology, and pressure observations.
- `internal/modelengine` owns fak-native lifecycle, pipeline, prefill/decode separation, and scheduling state.
- `internal/nativebench` owns deterministic benchmark samples and comparisons.
- `docs/benchmarks/model-observability.md` documents the model observability receipt and dashboard interpretation.
- `docs/_witnesses/native-performance-profile/README.md` indexes captured native performance profile evidence.
- `docs/native-inference-goal.md` is authoritative for fak-native identity, matched envelopes, and the no-silent-fallback rule.

## Deterministic proof

Run:

```bash
go test ./internal/nativeperfobscontract
```

`TestFrozenContractIsValidAndMachineReadable` emits and decodes the deterministic JSON coverage matrix in-process. The negative witnesses prove that validation fails when:

1. a required producer has neither a metric nor an exclusion;
2. the engine changes from `fak-native` (including a llama.cpp fallback); or
3. a high-cardinality label such as `request_id` enters the contract.

Call `nativeperfobscontract.JSON(nativeperfobscontract.Frozen())` from audits or dashboards that need the canonical machine-readable matrix. Validation happens before JSON is returned.
