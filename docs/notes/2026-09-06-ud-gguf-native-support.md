# UD GGUF native CPU support — September 6, 2026

Work: [#11957](https://github.com/anthony-chaudhary/fak/issues/11957),
within [#9820](https://github.com/anthony-chaudhary/fak/issues/9820).

`UD` names an Unsloth Dynamic model-level quantization mixture. Unsloth explains
that quantization choices vary by layer and model in its
[Dynamic 2.0 description](https://unsloth.ai/blog/dynamic-v2).
The constituent encodings are standard GGML types; the pinned
[GGML enum](https://github.com/ggml-org/llama.cpp/blob/6fe74980162af0ed5e559870d5deccafaa034e7c/ggml/include/ggml.h)
defines Q3_K and IQ3_S, with no separate UD tensor type. A filename alone does
not establish a universal fixed inventory or authenticate a quantization recipe.

## Evidence boundaries

The committed [header fixture provenance](../../internal/ggufload/testdata/qwen38_ud_q2kxl_provenance.json)
pins `unsloth/Qwen3.8-27B-GGUF` revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, file
`Qwen3.8-27B-UD-Q2_K_XL.gguf`. Its 866 tensors have this inventory:

| Encoding | Tensors | Encoding | Tensors | Encoding | Tensors |
|---|---:|---|---:|---|---:|
| F32 | 360 | Q2_K | 16 | Q3_K | 5 |
| Q4_K | 21 | Q5_K | 2 | Q6_K | 6 |
| Q8_0 | 98 | IQ2_XXS | 48 | IQ2_XS | 34 |
| IQ3_XXS | 112 | IQ1_S | 20 | IQ3_S | 57 |
| IQ2_S | 67 | IQ4_XS | 19 | IQ1_M | 1 |

`TestQwen38UDQ2KXLPinnedHeader` checks that real header and configuration.
The fixture contains no weight payloads and proves no forward execution.

`TestUDMixedGGUFNativeForwardParity` generates one complete three-layer llama
GGUF containing those same 15 encodings. It checks raw matrix bytes, native
CPU prefill and decode, and logits against the fully dequantized F32 forward
path. Its small synthetic weights isolate format dispatch; they do not prove
Qwen architecture correctness or real-model quality. Q3_K/IQ3_S decoder tests
also compare against independent pinned upstream C decoder digests.

## Native path and its limits

The resident loader retains eligible Q3_K and IQ3_S matrices as their original
110-byte blocks of 256 weights. Native CPU GEMV/GEMM decodes those blocks during
execution. Session dispatch also supports resident K-quant models containing
no Q4_K tensors; use `LoadModelQ4K` with `Session.Q4K=true` for that API path.

Eligibility remains architecture- and tensor-dependent. Weights requiring
normalization or layout conversion, including affected Q/K projections, retain
the existing dequantize/normalize/Q8 route. `WithDenseKQuantResident(false)`
preserves the device-loading fence. This change adds no packed Q3_K/IQ3_S GPU
kernel and does not make every model tensor raw resident.

For the pinned mixture, which contains Q4_K, the existing CPU CLI selection is:

```bash
FAK_Q4K=1 fak serve --gguf /path/to/Qwen3.8-27B-UD-Q2_K_XL.gguf
```

This command is source-verified selection guidance, not a captured full-model
serve receipt. Omit `--backend` to select the CPU reference backend. The CLI
currently requires Q4_K in the artifact inventory to select this resident arm;
the no-Q4_K API execution witness does not remove that CLI condition. The default
CPU serve path otherwise uses lean Q8. See
[serve selection](../../cmd/fak/serve_load_helpers.go) and
[flag definitions](../../cmd/fak/serve.go).

## Reproduction and remaining work

On an isolated Linux checkout, including WSL on Windows, run:

```bash
go test ./internal/model -run '^TestQ3KIQ3SNativeReference$' -count=1
go test ./internal/ggufload -run '^(TestQ3KIQ3SResidentGGUFExecution|TestUDMixedGGUFNativeForwardParity|TestQwen38UDQ2KXLPinnedHeader)$' -count=1
```

Run `fak validate --mine <owned-path>...` for the scoped build, vet, and test
receipt. The mixed test fails on the committed baseline because Q3_K becomes
Q8 instead of remaining resident. These commands describe reproducible checks;
the issue records their observed outcomes and final commit identity.

The uncommitted recipe-validation API discussed in
[#11956](https://github.com/anthony-chaudhary/fak/issues/11956) is not required
by this loader or execution support. Remaining work includes
[CUDA Q2/IQ2 acceleration](https://github.com/anthony-chaudhary/fak/issues/11222),
[matched UD quant performance](https://github.com/anthony-chaudhary/fak/issues/11223),
and [Qwen chat/tool/grammar parity](https://github.com/anthony-chaudhary/fak/issues/11948).
No whole-model speed, physical-memory saving, or quality gain is claimed here.
