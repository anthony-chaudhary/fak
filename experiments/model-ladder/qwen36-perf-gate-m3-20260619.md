# Qwen3.6 Perf Gate

- Verdict: FAIL
- Minimum ratio: 1x

| case | metric | fak tok/s | llama.cpp tok/s | ratio | verdict |
|---|---:|---:|---:|---:|---|
| m3-resident-q4k-vs-metal | `prefill_P22` | 0.44 | 51.55 | 0.01x | FAIL |
| m3-resident-q4k-vs-metal | `decode` | 0.90 | 7.29 | 0.12x | FAIL |

Failures:
- m3-resident-q4k-vs-metal P22: fak 0.44 < 1x llama 51.55
- m3-resident-q4k-vs-metal decode: fak 0.9 < 1x llama 7.29
