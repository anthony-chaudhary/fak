---
title: "Quantization support and delegation matrix"
description: "This page is fak's neutral, versioned quantization interoperability contract. It does not rank methods,"
---
# Quantization support and delegation matrix

This page is fak's neutral, versioned quantization interoperability contract. It does **not** rank methods, promise that two similarly named encodings are interchangeable, or claim model quality or performance. A row applies only when its capability ID, artifact version, and runtime ID all match.

The claims are deliberately separate:

- **Artifact** says what bytes or tensor state can be identified.
- **Recipe** says whether fak creates or calibrates that artifact.
- **Runtime** says whether execution is native, delegated, research-only, or refused.
- **Hardware envelope** bounds the claim and says what this contract witness actually exercised.

| Capability ID | Status | Artifact version | Artifact claim | Recipe claim | Runtime ID | Runtime claim | Hardware envelope | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quant.gguf-q4-k.cpu.v1 | supported | gguf-v3 | GGUF v3 tensors encoded as Q4_K | Consumes an already-quantized artifact; fak does not claim to produce or calibrate it | fak-native-cpu | Native fak GGUF loader and CPU Q4_K dequantization; no external runtime | Implementation: CPU path. Contract witness: linux/amd64 under WSL; no throughput or quality measurement | [source](../internal/ggufload/quant_q4k_loader.go) |
| quant.gptq.external.v1 | delegated | gptq-model | GPTQ-quantized model in the external runtime's accepted artifact layout | Calibration and quantization are owned by the artifact producer, not fak | external-runtime | Handled by an operator-selected GPTQ-capable runtime; fak has no native GPTQ loader claim | Defined by the selected external runtime; not measured or inferred by fak | [engine contract](../docs/supported/engines.md) |
| quant.fp8.research.v1 | research-only | fak-fp8-e4m3fn | Internal E4M3FN block data used by fak's experimental FP8 compute path; not a portable checkpoint format | Experimental runtime conversion with per-block scaling; no training or calibration quality claim | fak-research-fp8 | Research-only fak FP8 path; not a public serving compatibility promise | Implementation includes software/reference paths and optional accelerators; this matrix witness measures none | [source](../internal/compute/fp8.go) |
| quant.bitsandbytes.in-process.v1 | unsupported | bnb-4bit | bitsandbytes 4-bit module/checkpoint state | No fak-native bitsandbytes conversion or calibration recipe | fak-native | Unsupported in-process: fak does not embed the bitsandbytes runtime or silently reinterpret its state | No supported fak-native hardware envelope | [model contract](../docs/supported/models.md) |

## Typed decisions

Consumers should call `quantmatrix.Adjudicate` rather than infer support from a format name. The public outcomes and reason codes are:

| Situation | Outcome | Reason code |
| --- | --- | --- |
| Registered native combination | `allow` | `QUANT_NATIVE` |
| Registered delegated combination | `delegate` | `QUANT_EXTERNAL_RUNTIME` |
| Registered research-only combination | `abstain` | `QUANT_RESEARCH_ONLY` |
| Unknown capability ID | `abstain` | `QUANT_UNKNOWN_ID` |
| Unknown artifact version | `abstain` | `QUANT_UNKNOWN_ARTIFACT_VERSION` |
| Known unsupported status or runtime combination | `refuse` | `QUANT_UNAVAILABLE_COMBINATION` |

There is no name-based fallback. In particular, unknown GGUF revisions, alternative Q4_K layouts, and runtime IDs not named above are not silently treated as a registered row.

## Witness boundary

`internal/quantmatrix/quantmatrix_test.go` independently reads this Markdown table, resolves every row through the Go registry, verifies every local evidence link, and requires all four status classes. Its WSL `linux/amd64` run is a contract/docs witness over four fixtures; it does not load weights, benchmark a model, establish quantization quality, or validate accelerator behavior.
