# Qwen3.6 Perf Gate

- Verdict: FAIL
- Minimum ratio: 0.5x

| case | metric | fak tok/s | llama.cpp tok/s | ratio | verdict |
|---|---:|---:|---:|---:|---|
| metal-q4k-27b | `decode` | 1.20 | 7.29 | 0.16x | FAIL |

Bar provenance (the llama.cpp-Metal bar is observed-external, not a fak witness):
- metal-q4k-27b: llama.cpp-Metal SOTA bar. Provenance-caveated: #459 (build/version + exact llama.cpp flags) and #452 (bar measurement conditions) must be respected. This is a RELAYED external number, not a fak-controlled witness.

Failures:
- metal-q4k-27b decode: fak 1.2 < 0.5x llama 7.29
