# Qwen3.6 Perf Gate

- Verdict: PASS
- Minimum ratio: 1x

| case | metric | fak tok/s | llama.cpp tok/s | ratio | verdict |
|---|---:|---:|---:|---:|---|
| amd-p16-64-256 | `prefill_P16` | 14.86 | 5.20 | 2.86x | PASS |
| amd-p16-64-256 | `prefill_P64` | 27.46 | 14.57 | 1.88x | PASS |
| amd-p16-64-256 | `prefill_P256` | 31.34 | 9.95 | 3.15x | PASS |
| amd-p16-64-256 | `decode` | 1.24 | 0.99 | 1.25x | PASS |
| amd-p512-1024 | `prefill_P512` | 30.28 | 9.32 | 3.25x | PASS |
| amd-p512-1024 | `prefill_P1024` | 29.67 | 9.31 | 3.19x | PASS |
| amd-p512-1024 | `decode` | 1.15 | 0.99 | 1.16x | PASS |
