# Quantized loader conversions in startup diagnostics

Date: 2026-11-01

The profiled Q4K GGUF loader already records resident-versus-dequantized tensors by source quantization and dense/expert class. Normal serving startup sets AlertWriter while leaving Progress unset. The old conversion summary used Progress, so these records did not produce startup conversion notices.

The loader now emits one bounded warning per converted quant/class row after collection. It reuses the existing LoadAlert and LoadProfile.Alerts path, which serving forwards to startup messages. Each warning reports source_quant, class, dequant_tensors and source_bytes. It describes temporary F32 conversion during loading; it does not infer the conversion's cause, final resident precision or selected kernel.

Float sources (F32/F16/BF16) and entirely packed-resident rows do not produce quantized-conversion warnings. Profiles retain alerts even when no output writer is configured. Existing progress output remains optional.

## Measured detection coverage

The fixed real-loader fixture contains Q4_K embedding and Q6_K output weights. Its residency options exercise the existing conversion path, with three Q4_K dense matrices remaining packed.

| Source quant/class | Converted tensors | Source bytes | Startup notices before | Startup notices after |
| --- | ---: | ---: | ---: | ---: |
| Q4_K / dense embedding | 1 | 36,864 | 0 | 1 |
| Q6_K / dense output | 1 | 53,760 | 0 | 1 |

The pre-edit regression ran and failed with converted_classes=2, alerted_classes=0, silent_classes=2. After the change it reports converted_classes=2, alerted_classes=2, silent_classes=0. Silent rows fell 2 -> 0: a 100% reduction in this fixed two-row cohort, exceeding the 50% target.

Four integration cases pass: startup warning output, entirely packed-resident control, durable alerts without an output writer, and an actual F32-source control. The controls produce zero quantized-conversion warnings. JSON round-trip assertions preserve both conversion rows and alerts.

This is software detection coverage. Conversion behavior and costs are unchanged. Throughput and selected-kernel performance are unmeasured. The result does not establish coverage of all engine fallback paths.

## Wiring and acceptance

- internal/ggufload/gguf_profile.go aggregates warnings from existing LoadPaths.
- internal/ggufload/quant_q4k_loader.go invokes finalization after the serial collector.
- cmd/fak/serve_load_helpers.go constructs the ordinary warnings-enabled profiler.
- cmd/fak/serve_model_load.go forwards saved alerts into startup messages.
- internal/ggufload/gguf_profile_fallback_test.go exercises the real loader and fixed cohort.

From the public checkout, select its standalone workspace for these commands (PowerShell: $env:GOWORK='off'):

```text
go test ./internal/ggufload -run '^TestLoadQuantFallbackAlerts$' -count=1 -v -timeout=5m
go test ./internal/ggufload -count=1 -timeout=5m
go vet ./internal/ggufload
```

CLAIM_TEST_GREEN: the focused regression passed all four cases (0.325s package execution). Final full package tests passed in 56.907s; targeted vet completed with exit 0. Independent intent-versus-diff review accepted the production wiring, fixed-denominator result and false-positive controls.

The first full package run also exposed an older HTTP diagnostics test that rejected the existing nested page_cache object. The adjacent one-file repair validates its exact ten field names and types, preserves unknown-field checks, qualifies modeled source I/O, and compares served values with the model's source counters. The existing 31-field legacy V41 phase contract remains unchanged.

## Remaining scope

HAL dense-projection and output-head staging can widen weights after backend capability checks. The existing PlannedWeightRequants implementation inventories IQ types; the separate Q6_K fallback accounting gap remains source-inspected, with no fresh counter baseline measured in this change. Its follow-up must capture a real before receipt and add planned/observed ledger assertions before acceptance.
