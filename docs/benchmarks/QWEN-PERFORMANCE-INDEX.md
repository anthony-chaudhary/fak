---
title: "Qwen performance index"
description: "Canonical cross-hardware index and publishing route for accepted Qwen performance results."
---

# Qwen performance index

**This is the one current index for Qwen performance updates.** Hardware workers publish the full receipt under `docs/_witnesses/`, then update this page in the same landing. Detailed model pages and `BENCHMARK-AUTHORITY.md` remain evidence and methodology sources; worker logs and issue comments are not the current result surface.

<!-- qwen38-frontdoor:begin -->
## Generated front-door readout

This block is derived by `fak native-performance --frontdoor-md`; classifications cannot be spliced across envelopes.

- **ACCEPTED:** fak-native Metal Q4_K_M delivered **2.3-2.9 decode tok/s** with functional `PASS` in the frozen M3 Pro full-run envelope. [Receipt](../_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json).
- **APPROXIMATE:** the closest near-matched observation is **3.3 vs 6.966061 tok/s (~47%)**. It is not accepted parity: P31/T64 native versus P32/T64 llama.cpp, with no joint quality-complete receipt. [Issue #8697](https://github.com/anthony-chaudhary/fak/issues/8697).
- **DIAGNOSTIC:** the separate A100 cache-restore arm measured **~0.2 tok/s with 0/5 exact**. Failed quality keeps it out of accepted and approximate comparison headlines. [Attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md).
<!-- qwen38-frontdoor:end -->

## Read this first: rows are envelopes, not a timeline

A newer result replaces an older row **only when its envelope key matches**: model + artifact/precision + engine/path + hardware/topology + workload/cache mode. A Metal kernel improvement does not supersede an A100 CUDA result, and a kernel microbenchmark does not supersede an end-to-end tokens/sec result.

Read the first two columns before the number:

- **CURRENT** — newest accepted, quality-passing result for that exact envelope key. There may be one CURRENT row per key.
- **DIAGNOSTIC** — retained because it explains a current hold or parity gap; never read it as a fak-native headline.
- **HISTORICAL** — predecessor context only. It stays out of current-result comparisons.
- **AWAITING REMEASURE** — newer code exists, but no comparable accepted run exists. It cannot replace a CURRENT row.

Every CURRENT row has both an **observed** date and a **review-by** date. Review does not mean rerun blindly: it either promotes a comparable receipt, advances the review date with evidence that no replacement exists, or removes the row from this index. Superseded numbers remain in their immutable witness/detail page, not as a second current row here.

## Current results by envelope

Numbers in different envelope keys are not interchangeable. Quote the key, model, artifact/precision, engine, hardware, metric, and observed date with the number.

| Lifecycle | Envelope key | Current accepted highlight | Status and comparison | Evidence and freshness |
|---|---|---|---|---|
| **CURRENT** | `q38-bf16-tp2-arithmetic-ttfc` | Thinking off: **3/3 correct, p95 376.18 ms** to first correct arithmetic answer; thinking on: **3/3, p95 3378.02 ms**. | `PASS` for the frozen arithmetic/TP2 envelope; not a tokens/sec, GGUF, or production-readiness claim. | [#8623 receipt](../_witnesses/issue-8623-qwen38-27b/README.md); observed **2026-08-24**; review by **2026-09-07**. |
| **CURRENT** | `q38-q4km-native-cuda-a100-cold-decode` | Cold unique decode **11.8–12.1 tok/s**, **5/5 exact**. | Native `cuda/qwen35-gdn-ssm-decode-v1`. The cache arm is held; do not report this cold row as cache or serving parity. | [#8819 cache attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md); observed **2026-08-25**; review by **2026-09-01**. |
| **DIAGNOSTIC** | `q38-q4km-cuda-a100-cache-parity` | Cache hits were about **0.2–0.3 tok/s**; the attribution rerun was **0/5 exact**. Pinned llama.cpp reference median was **36.55 tok/s**. | `HOLD_CACHE_RESTORE_REGRESSION` and `HOLD_BELOW_PARITY`. The reference is parity diagnosis only, not the fak execution path. | [#8819 attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md) and [#8848 campaign](../_witnesses/issue-8848-qwen38-overnight/README.md); observed **2026-08-25**; review by **2026-09-01**. |
| **CURRENT** | `q38-a100-p2060-prompt-attention` | Exact-artifact prefill **195.7 → 217.5 tok/s** (**+11.1%**) at 2060 tokens, with full-model state parity. | Accepted component-path gain from #8643; it is not the whole serving path. HTTP remains timing/parity-only and default serving remains below llama.cpp. | [#8643 issue receipt](https://github.com/anthony-chaudhary/fak/issues/8643), commit `2b54497aa0`; observed **2026-08-24**; review by **2026-09-07**. |
| **CURRENT** | `q38-q4km-native-metal-m3pro-fullrun` | Accepted full-run decode **2.3–2.9 tok/s**; full prefill **3.2–8.4 tok/s** depending on probe. | Functional `PASS`, below parity. A later **3.3 tok/s** P32/T64 point is a different workload shape and does not replace this row. | [Metal detail](QWEN38-27B-LATEST.md) and [run receipt](../_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json); observed **2026-08-20**; review by **2026-08-27**. |
| **HISTORICAL** | `q36-q4km-metal-m3pro-parity-bar` | llama.cpp Metal: **51.55 prefill / 7.29 decode tok/s**. fak resident-Q4_K Metal: **2.6 at P=27 / 7.3 at P=940 prefill, 1.2 decode tok/s**. | Architecture predecessor only; do not present it as current Qwen3.8 performance. | [Qwen3.6 parity results](QWEN36-PARITY-RESULTS.md); observed **2026-06-26**; retained as predecessor context. |

**No accepted current Qwen3.8 performance row is indexed for AMD/Vulkan or CPU-only hardware.** Absence is not a zero or a failed benchmark.

## Newer code awaiting comparable remeasurement

This section prevents landed optimizations from masquerading as newer result rows. Entries are reaped when a comparable receipt is promoted above, the work is rejected, or the stated review date passes without a renewed reason to retain it.

| Lifecycle | Candidate | Why it has not replaced a row | Reap condition |
|---|---|---|---|
| **AWAITING REMEASURE** | Recent Metal Q4_K/GDN/projection work (#8833, #9096, #9097, #9102) | Landed implementation and focused parity/kernel evidence do not provide a comparable M3 Pro full-run receipt. One attempted #9102 run was invalidated by swap pressure, so it carries no speed claim. | Promote a same-key `q38-q4km-native-metal-m3pro-fullrun` receipt or remove at review on **2026-08-27**. |
| **AWAITING REMEASURE** | Later M3 Pro P32/T64 point (#8697) | **3.3 tok/s** is useful planning evidence, but its prompt/output shape does not match the frozen multi-probe row. | Promote after the frozen probes are rerun under the same artifact and engine identity, or remove at review on **2026-08-27**. |

## Shift-left update and reap process

This happens in the result-producing change, not in a later documentation cleanup:

1. **Choose the envelope key before running.** Reuse an existing key only if model, artifact/precision, engine/path, hardware/topology, workload, and cache mode match. Otherwise add a distinct key.
2. **Capture the receipt first.** Retain immutable raw/summary evidence under `docs/_witnesses/issue-<id>-<slug>/`; scrub private control-plane details. An issue comment or worker log alone is not durable evidence.
3. **Bind the operating envelope.** Record model revision/hash and quantization. Native/performance rows must name the fak-native engine and forward-path identity, hardware/topology, OS/driver/runtime, commit or `module@rev`, workload/token counts, cache state, repetitions/statistic, quality, memory, and observation date.
4. **Classify before quoting.** A quality-passing comparable run can be CURRENT. Reference-only or failed-quality evidence is DIAGNOSTIC. New code without a comparable run is AWAITING REMEASURE. Older model context is HISTORICAL.
5. **Replace atomically.** When promoting a result, delete the old CURRENT row with the same envelope key in the same commit. Its witness remains immutable. Never append a second CURRENT row for that key.
6. **Set review/reap state now.** Every CURRENT, DIAGNOSTIC, and AWAITING REMEASURE entry gets an observed/review date or explicit durable retention reason. At review, promote, renew with evidence, or remove it.
7. **Update downstream summaries last.** `README.md`, release notes, plans, and issue comments may quote only the resulting CURRENT row. `BENCHMARK-AUTHORITY.md` owns general method; this page owns Qwen result discovery.

The deterministic test rejects duplicate CURRENT envelope keys and active rows without lifecycle freshness metadata. A row is also not accepted when artifact/engine identity is ambiguous, quality failed but speed is presented as a gain, setup/recovery costs are hidden, or unlike envelopes are collapsed into one headline.

## Detail and campaign routes

- [Qwen3.8-27B Metal detail and Qwen3.6 delta](QWEN38-27B-LATEST.md)
- [Qwen3.8 ladder contract](qwen38-ladder/README.md)
- [Qwen3.8 native overnight campaign](../_witnesses/issue-8848-qwen38-overnight/README.md)
- [Qwen3.8 cache attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md)
- [Benchmark authority and comparison doctrine](../../BENCHMARK-AUTHORITY.md)
- [Sanctioned hardware routes](../fleet-compute-nodes.md)
