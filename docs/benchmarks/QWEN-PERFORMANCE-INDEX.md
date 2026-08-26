---
title: "Qwen performance index"
description: "Canonical cross-hardware index and publishing route for accepted Qwen performance results."
---

# Qwen performance index

**This is the one current index for Qwen performance updates.** Hardware workers publish the full receipt under `docs/_witnesses/`, then add or update one row here. Detailed model pages and `BENCHMARK-AUTHORITY.md` remain evidence and methodology sources; worker logs and issue comments are not the current result surface.

## Current highlights

Numbers in different rows are not interchangeable. Quote the model, artifact/precision, engine, hardware, metric, and date with the number.

| Model / hardware | Current accepted highlight | Status and comparison | Authoritative evidence |
|---|---|---|---|
| **Qwen3.8-27B BF16, two datacenter GPUs** | Thinking off: **3/3 correct, p95 376.18 ms** to first correct arithmetic answer; thinking on: **3/3, p95 3378.02 ms**. | `PASS` for the frozen arithmetic/TP2 envelope; not a tokens/sec, GGUF, or production-readiness claim. | [#8623 receipt](../_witnesses/issue-8623-qwen38-27b/README.md), 2026-08-24 |
| **Qwen3.8-27B Q4_K_M, A100-class CUDA, fak-native** | Cold unique decode **11.8–12.1 tok/s**, **5/5 exact**. Confirmed cache hits were about **0.2 tok/s** and **0/5 exact** in the attribution rerun. | `HOLD_CACHE_RESTORE_REGRESSION`; native identity is `cuda/qwen35-gdn-ssm-decode-v1`. Do not report the cold row as cache or serving parity. | [#8819 cache attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md), 2026-08-25 |
| **Qwen3.8-27B Q4_K_M, A100-class CUDA, matched reference** | Pinned llama.cpp median decode **36.55 tok/s** versus cached fak-native median **0.3 tok/s** in the overnight campaign. | `HOLD_BELOW_PARITY`; the reference is permitted for parity diagnosis only and is not the fak execution path. | [#8848 campaign](../_witnesses/issue-8848-qwen38-overnight/README.md), 2026-08-25 |
| **Qwen3.8-27B, A100 CUDA prompt attention** | Exact-artifact prefill **195.7 → 217.5 tok/s** (**+11.1%**) at 2060 tokens, with full-model state parity. | Accepted kernel-path gain from #8643; HTTP remains timing/parity-only and the default remains below llama.cpp. | [#8643 issue receipt](https://github.com/anthony-chaudhary/fak/issues/8643), commit `2b54497aa0` |
| **Qwen3.8-27B Q4_K_M, Apple M3 Pro Metal, fak-native** | Accepted full-run decode **2.3–2.9 tok/s**; full prefill **3.2–8.4 tok/s** depending on probe. | Functional `PASS`, below parity. A later issue reports a **3.3 tok/s** P32/T64 planning point, but the frozen multi-probe page remains the accepted cross-probe result until an equivalent receipt replaces it. | [current Metal detail](QWEN38-27B-LATEST.md) and [run receipt](../_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json); later point [#8697](https://github.com/anthony-chaudhary/fak/issues/8697) |
| **Qwen3.6-27B Q4_K_M, Apple M3 Pro** | llama.cpp Metal: **51.55 prefill / 7.29 decode tok/s**. fak resident-Q4_K Metal: **2.6 at P=27 / 7.3 at P=940 prefill, 1.2 decode tok/s**. | Historical parity bar and architecture predecessor; do not present it as Qwen3.8 performance. | [Qwen3.6 parity results](QWEN36-PARITY-RESULTS.md), 2026-06-26 |

**No accepted current Qwen3.8 performance row is indexed for AMD/Vulkan or CPU-only hardware.** A worker result on either remains unpublished until it satisfies the route below; absence is not a zero or a failed benchmark.

## Publishing route for every hardware worker

1. **Capture the receipt first.** Retain immutable raw/summary evidence under `docs/_witnesses/issue-<id>-<slug>/`; scrub private control-plane details. An issue comment or worker log alone is not durable evidence.
2. **Bind the operating envelope.** Record model ID and immutable revision/hash, quantization/precision, fak-native engine and forward-path identity, hardware and topology, OS/driver/runtime, commit or `module@rev`, prompt/corpus and token counts, cache state, repetitions/statistic, quality result, memory, and observation date.
3. **Classify the result.** Mark `PASS`, `KEEP`, `HOLD`, `EXCLUDE`, or `PROXY`; state whether it is end-to-end, kernel-only, reference-runtime, or planning evidence. Native/performance rows must name the fak-native engine. Never silently route execution through llama.cpp.
4. **Update this index in the same landing.** Replace the applicable current row or add a genuinely distinct hardware/model envelope. Link the retained receipt and preserve the prior row in its detailed page or witness; do not overwrite incomparable numbers.
5. **Update downstream summaries only after indexing.** `README.md`, release notes, plans, and issue comments may quote an indexed row. `BENCHMARK-AUTHORITY.md` owns general benchmark method and broader non-Qwen results; this page owns Qwen result discovery.

A row is **not accepted** when its artifact or engine identity is ambiguous, quality failed but the speed is presented as a gain, setup/recovery costs are hidden, the source exists only in transient worker output, or unlike hardware/model/cache envelopes are collapsed into one headline.

## Detail and campaign routes

- [Qwen3.8-27B Metal detail and Qwen3.6 delta](QWEN38-27B-LATEST.md)
- [Qwen3.8 ladder contract](qwen38-ladder/README.md)
- [Qwen3.8 native overnight campaign](../_witnesses/issue-8848-qwen38-overnight/README.md)
- [Qwen3.8 cache attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md)
- [Benchmark authority and comparison doctrine](../../BENCHMARK-AUTHORITY.md)
- [Sanctioned hardware routes](../fleet-compute-nodes.md)
