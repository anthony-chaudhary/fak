---
title: "Qwen3.8-27B latest performance"
description: "One-page index of the newest accepted Qwen3.8-27B performance evidence, its status, and the comparable Qwen3.6/llama.cpp bars."
---

# Qwen3.8-27B latest performance

**Latest accepted Mac result:** native fak Metal runs Qwen3.8-27B Q4_K_M at
**2.3–2.9 decode tok/s** and **3.2–8.4 full-prefill tok/s** on the 18-GPU-core,
36 GiB Apple M3 Pro. This is an acceptance result, not parity with llama.cpp.

Source of truth: [`metal-native-run-summary.json`](../_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json),
observed 2026-08-20. The exact model is the 17,106,775,008-byte
`unsloth/Qwen3.8-27B-GGUF` Q4_K_M artifact at revision
`f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`, SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

## Current numbers

| Measure | Result | Scope |
|---|---:|---|
| Decode | **2.3 tok/s** | text after full prefill |
| Decode | **2.9 tok/s** | JSON after full prefill |
| Fully cached decode | **0.4–1.3 tok/s** | observed range; do not combine with the full-prefill rows |
| Full prefill | **3.2 / 5.9 / 8.4 tok/s** | text / JSON / tool probes; prompt lengths differ |
| Time to ready | **103.858 s cold / 34.897 s warm-file-cache** | native Metal service startup |
| Max RSS | **18.92–19.56 GB** | two captured runs |
| macOS peak footprint | **42.54–45.63 GB** | OS footprint is not RSS; zero swaps recorded |
| Functional acceptance | **PASS** | exact text, strict JSON, and admitted tool probes |

The same witness records **0.03 tok/s before the decode fix**, so the accepted path is
roughly **77–97×** faster than that broken baseline. That ratio does not establish
parity with another engine.

## What changed relative to Qwen3.6

The premise that native fak Qwen3.6 had reached near parity with llama.cpp on this Mac
is not supported by the accepted page. The [Qwen3.6 parity bar](QWEN36-PARITY-RESULTS.md)
records:

| Qwen3.6-27B Q4_K_M path | Prefill | Decode |
|---|---:|---:|
| llama.cpp Metal b9707 | **51.55 tok/s** | **7.29 tok/s** |
| fak resident-Q4_K Metal | **2.6 tok/s at P=27; 7.3 at P=940** | **1.2 tok/s** |

Against fak's last accepted Qwen3.6 Metal decode, Qwen3.8's 2.3–2.9 tok/s is
**1.92–2.42× higher**. Against the older Qwen3.6 llama.cpp bar, it is only
**32–40%** as fast. These are directional comparisons, not model-to-model parity:
the generation, prompt/corpus, code revision, and measurement date differ, and there
is no accepted same-artifact Qwen3.8 llama.cpp row yet.

The main delta is implementation maturity, not a demonstrated architectural windfall:

- Qwen3.6 established the hybrid Gated DeltaNet + full-attention path, resident Q4_K
  weights, and initial Metal kernels, but remained launch-bound and explicitly below
  llama.cpp speed parity.
- Qwen3.8 reuses that `metal/qwen35-hybrid-session-v1` forward path and adds streamed
  Q4_K residency, memory/lifetime fixes, warmup serialization, and newer Metal GEMM
  work. The accepted witness therefore measures a more mature fak path.
- The newest landed Q4_K 4–8-vector specialization is a **kernel microbenchmark**, not
  a new end-to-end model number: [`issue-8326-metal-q4k-multivector.json`](../_witnesses/issue-8326-metal-q4k-multivector.json)
  reports **1.57–1.71×** over repeated GEMV for the measured shapes, with parity checks. Qwen3.8 must be
  remeasured end to end before attributing that gain to tokens/sec.

## Evidence status and next comparison

- **Accepted Mac result:** the 2026-08-20 native-Metal witness linked above.
- **Not a newer Mac speed result:** open work on resident Metal decode and bounded host
  materialization (#8324 and #8101) has not produced a later accepted tokens/sec witness.
- **Not comparable:** the current BF16 and Q5_K_M campaign artifacts are still open-work
  evidence; Q5_K_M is explicitly `INVALID_API_CONTRACT`, and BF16 is an A100 campaign,
  not the Apple-Metal parity row.
- **Required parity closure:** run the exact Qwen3.8 artifact and frozen prompts through
  native fak Metal and current llama.cpp Metal on the same M3 Pro, report prompt and
  decode separately, and retain hashes, memory, fallback identity, and repeated trials.

For benchmark contract requirements, see [`BENCHMARK-CONTRACT-MAP.md`](BENCHMARK-CONTRACT-MAP.md).
