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

