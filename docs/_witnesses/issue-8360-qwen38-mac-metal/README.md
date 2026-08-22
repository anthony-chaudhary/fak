# Issue #8360 — Qwen3.8-27B Q4_K_M MacBook Metal campaign

`summary.json` is the compact readout. `campaign-report.json` and `archive.json` are the
validator-clean campaign pair produced by `qwen38campaign`; `platform.json`
independently captures the live M3 Pro process, 36 GiB memory, swap, model
metadata, and cleanup-era platform state.

Result: **PROMOTE**. All 18 frozen trials passed (six workload families, three
repetitions each) on llama.cpp 9828 Metal with the exact Q4_K_M artifact. The
campaign witnessed restart readiness, resident/no-fallback operation, and
successful cleanup. The 32K-token long-context cold trial took 400.679 s;
its two cache-warm repetitions took 9.280 s and 9.300 s, so cold and warm
latency must not be conflated.

Reproduction command (MacBook):

```bash
qwen38campaign \
  --config config.json \
  --corpus docs/benchmarks/qwen38-quant/corpus.json \
  --report campaign-report.json \
  --archive archive.json
```

The operator config pinned:

- `unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
- artifact SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- llama.cpp runtime revision `ebd048fc5`
- `internal/qwen38quantrun@r8+gb0ce51b59`
- `--ctx-size 65536 --parallel 1 -ngl 99`

The config itself is omitted because it embeds machine-local absolute paths;
the complete argv and immutable identities remain in `campaign-report.json`.
