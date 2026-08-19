# Qwen3.8-27B exact FP8 TP2 witness

This directory freezes the 2026-08-19 exact-checkpoint acceptance for
`Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` on two
A100-SXM4-40GB devices with vLLM tensor parallelism behind fak.

Reproduce the folds from the repository root:

```console
go run ./cmd/fak model acceptance-gate \
  --input docs/_witnesses/qwen38-27b-2026-08-19/fp8-tp2-acceptance-input.json
go run ./cmd/fak model readiness-inventory \
  --input docs/_witnesses/qwen38-27b-2026-08-19/fp8-tp2-acceptance-input.json \
  --artifact _scratch/qwen38-8011/qwen38-fp8-witness-v6.tgz \
  --artifact-revision internal/modelaccept@r9+g6ebc983c91 \
  --expected-corpus qwen38-hardware-acceptance-2026-08-19 \
  --as-of 2026-08-19T07:30:00Z
```

The raw archive SHA-256 is
`c81ead67b1510222f5adc2a26a0add4b49b267e808d1c82f387e9b81e3c7fc2e`.
The one-device 40GB OOM remains an honest HOLD; this witness establishes TP2.


## Native CUDA Q4_K_M

The same corpus also freezes native fak CUDA acceptance for
`unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` on one
A100-SXM4-40GB device. The exact 17,106,775,008-byte Q4_K_M file has SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

```console
go run ./cmd/fak model acceptance-gate \
  --input docs/_witnesses/qwen38-27b-2026-08-19/cuda-gguf-acceptance-input.json
go run ./cmd/fak model readiness-inventory \
  --input docs/_witnesses/qwen38-27b-2026-08-19/cuda-gguf-acceptance-input.json \
  --artifact _scratch/qwen38-8011/qwen38-a100-v10.tgz \
  --artifact-revision internal/model@r444+g607a8a2924 \
  --expected-corpus qwen38-hardware-acceptance-2026-08-19 \
  --as-of 2026-08-19T05:20:00Z
```

The raw archive SHA-256 is
`865b27e33909cd7c294f9cdbdec65e60d9d8889b52ad27f6b46de2b700477255`.
It captures exact checkpoint identity, startup, device residency, native-forward markers,
responses, tool admission, and `PASS_NATIVE_CUDA_GGUF`.

The generic `fak.modelaccept.report/2` rows encode the three observed capability probes;
they do not encode hardware topology. Hardware and checkpoint provenance therefore remain
bound to the raw archives, their hashes, and this README rather than inferred from the
reports alone.
