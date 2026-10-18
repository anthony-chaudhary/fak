---
title: "fak-native performance observability contract"
description: "Retired inventory of the frozen fak-native performance metric catalog: none of its names is exported, so dashboards must not query them as live signals."
---

# fak-native performance observability contract

Status: retired. The Go catalog (`fak-native-performance-observability-contract/1`) and its validator were removed because nothing invoked them; this page keeps the retired inventory and rules.

The contract asked one bounded question: **does every performance-relevant stage of a fak-native model run have a stable exported signal, or an explicit justified exclusion?** It fixed the engine to `fak-native`, so llama.cpp execution could not be relabeled as native evidence.

All 14 surfaces were `excluded` with an explicit `RETIRED` reason and no `metric` field. None of the 14 catalog metric names has an invoked Prometheus exporter; catalog entries and receipt source references never established runtime emission. `fak_native_tokens_total` is a catalog entry covered by both prefill and decode exclusions. The existing `fak_native_phase_seconds_total` export is a different metric family and does not implement `fak_native_phase_seconds`.

New evidence defaults to **Qwen3.8**. Qwen3.6 remains valid only for explicit regression, compatibility, historical comparison, or hardware/artifact constraints, and historical Qwen3.6 artifacts keep their original identity.

## Contract rules

- Metric names are stable, lowercase, and prefixed `fak_native_`.
- Types are limited to `counter`, `gauge`, and `histogram`; units are explicit.
- Every metric carries the bounded `engine` label. The only valid value for this contract is `fak-native`.
- Labels describe receipt-bounded enums. Request IDs, prompts, paths, hosts, and arbitrary model strings are forbidden. `model_family` is a curated bounded value, not an artifact path or free-form model name.
- `cardinality_budget` is the maximum intended live series count for the metric. Raising it is a contract change.
- Provenance identifies the referenced fak receipt or source; it does not prove an exporter invokes it.
- Freshness describes the catalog's intended update boundary, not a current runtime guarantee. Missing observations are unsupported; they are never converted to zero.
- A surface is complete only with `status: exported` plus a contracted metric, or `status: excluded` plus a justification. Any other status, including `UNKNOWN`, fails validation.

## Retired native performance export inventory

Every row below is excluded from exported coverage. The names and intended freshness remain catalog definitions.

| Surface | Referenced source / receipt | Retired catalog signal | Intended freshness |
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

Dashboards must not query retired names as live signals. A surface may claim `exported` only after an actual invoked exporter and emitted-series witness exist.

## Existing receipts and dashboards

The contract cross-references the existing truth surfaces rather than replacing them:

- `internal/nativeperf` owns native run receipts, profiles, graph attribution, and performance gates.
- `internal/modelperfobs` owns memory, bandwidth, cache, KV-capacity, topology, and pressure observations.
- `internal/modelengine` owns fak-native lifecycle, pipeline, prefill/decode separation, and scheduling state.
- `internal/nativebench` owns deterministic benchmark samples and comparisons.
- `docs/benchmarks/model-observability.md` documents the model observability receipt and dashboard interpretation.
- `docs/_witnesses/native-performance-profile/README.md` indexes captured native performance profile evidence.
- `docs/native-inference-goal.md` is authoritative for fak-native identity, matched envelopes, and the no-silent-fallback rule.
