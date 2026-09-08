---
title: "Qwen performance index"
description: "Canonical cross-hardware index and publishing route for accepted Qwen performance results."
---

# Qwen performance index

**This is the one current index for Qwen performance updates.** Hardware workers publish the full receipt under `docs/_witnesses/`, then update this page in the same landing. Detailed model pages and `BENCHMARK-AUTHORITY.md` remain evidence and methodology sources; worker logs and issue comments are not the current result surface.

<!-- qwen38-frontdoor:begin -->
## Generated front-door readout

This block is derived by `fak native-performance --frontdoor-md`; classifications cannot be spliced across envelopes.

_3 reviewed row(s) are reaped from active presentation; immutable witnesses remain._
<!-- qwen38-frontdoor:end -->

## Read this first: rows are envelopes, not a timeline

A newer result replaces an older row **only when its envelope key matches**: model + artifact/precision + engine/path + hardware/topology + workload/cache mode. A Metal kernel improvement does not supersede an A100 CUDA result, and a kernel microbenchmark does not supersede an end-to-end tokens/sec result.

Read the first two columns before the number:

- **CURRENT** — newest accepted, quality-passing result for that exact envelope key. There may be one CURRENT row per key.
- **DIAGNOSTIC** — retained because it explains a current hold or parity gap; never read it as a fak-native headline.
- **HISTORICAL** — predecessor context only. It stays out of current-result comparisons.
- **AWAITING REMEASURE** — newer code exists, but no comparable accepted run exists. It cannot replace a CURRENT row.

Every CURRENT row has both an **observed** date and a **review-by** date. Review does not mean rerun blindly: it either promotes a comparable receipt, advances the review date with evidence that no replacement exists, or removes the row from this index. Superseded numbers remain in their immutable witness/detail page, not as a second current row here. Receipt classifications such as **ACCEPTED** and **APPROXIMATE** remain visible in historical evidence, but only CURRENT rows belong in the active presentation.

## Current results by envelope

Numbers in different envelope keys are not interchangeable. Quote the key, model, artifact/precision, engine, hardware, metric, and observed date with the number.

| Lifecycle | Envelope key | Current accepted highlight | Status and comparison | Evidence and freshness |
|---|---|---|---|---|
| **CURRENT** | `q38-bf16-tp2-arithmetic-ttfc` | Thinking off: **3/3 correct, p95 376.18 ms** to first correct arithmetic answer; thinking on: **3/3, p95 3378.02 ms**. | `PASS` for the frozen arithmetic/TP2 envelope; not a tokens/sec, GGUF, or production-readiness claim. | [#8623 receipt](../_witnesses/issue-8623-qwen38-27b/README.md); observed **2026-08-24**; review by **2026-09-07**. |
| **CURRENT** | `q38-q4km-native-cuda-a100-cold-decode` | Cold unique decode **11.8–12.1 tok/s**, **5/5 exact**. | Native `cuda/qwen35-gdn-ssm-decode-v1`. The cache arm is held; do not report this cold row as cache or serving parity. | [#8819 cache attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md); observed **2026-08-25**; review by **2026-09-01**. |
| **DIAGNOSTIC** | `q38-q4km-cuda-a100-cache-parity` | Cache hits were about **0.2–0.3 tok/s**; the attribution rerun was **0/5 exact**. Pinned llama.cpp reference median was **36.55 tok/s**. | `HOLD_CACHE_RESTORE_REGRESSION` and `HOLD_BELOW_PARITY`. The reference is parity diagnosis only, not the fak execution path. | [#8819 attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md) and [#8848 campaign](../_witnesses/issue-8848-qwen38-overnight/README.md); observed **2026-08-25**; review by **2026-09-01**. |
| **CURRENT** | `q38-a100-p2060-prompt-attention` | Exact-artifact prefill **195.7 → 217.5 tok/s** (**+11.1%**) at 2060 tokens, with full-model state parity. | Accepted component-path gain from #8643; it is not the whole serving path. HTTP remains timing/parity-only and default serving remains below llama.cpp. | [#8643 issue receipt](https://github.com/anthony-chaudhary/fak/issues/8643), commit `2b54497aa0`; observed **2026-08-24**; review by **2026-09-07**. |
| **CURRENT** | `q38-q4km-native-metal-m3pro-sequence-prefill` | Forward-owned sequence prefill: **43.8% faster prefill** (10,284.5 vs 18,304.9 ms), **1 command buffer vs 192**, 0 fallbacks. | `PASS` for M2 sequence boundary; closes #9230/#9525. | [#9525 receipt](../_witnesses/issue-9525-qwen38-sequence-prefill/README.md); observed **2026-09-03**; review by **2026-09-17**. |
| **HISTORICAL** | `q38-q4km-native-metal-m3pro-fullrun` | Accepted full-run decode **2.3–2.9 tok/s**; full prefill **3.2–8.4 tok/s** depending on probe. | Functional `PASS`, below parity. The review window passed without a comparable renewal, so this row is retained only as dated evidence. | [Metal detail](QWEN38-27B-LATEST.md) and [run receipt](../_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json); observed **2026-08-20**; reviewed through **2026-08-27**. |
| **HISTORICAL** | `q36-q4km-metal-m3pro-parity-bar` | llama.cpp Metal: **51.55 prefill / 7.29 decode tok/s**. fak resident-Q4_K Metal: **2.6 at P=27 / 7.3 at P=940 prefill, 1.2 decode tok/s**. | Architecture predecessor only; do not present it as current Qwen3.8 performance. | [Qwen3.6 parity results](QWEN36-PARITY-RESULTS.md); observed **2026-06-26**; retained as predecessor context. |

**No accepted current Qwen3.8 performance row is indexed for AMD/Vulkan or CPU-only hardware.** Absence is not a zero or a failed benchmark.

<!-- strix-halo-comparison-ladder:begin -->
## Strix Halo external comparison ladder

Source: [`strix-halo-comparison-ladder.json`](strix-halo-comparison-ladder.json) · status: **NO_COMPARABLE_LOCAL_RESULT** · observed **2026-09-08**.

These are frozen external challenge points, not accepted fak results. `CONTEXT_ONLY` rows emit no ratio; `EXCLUDE` rows cannot become a performance bar. A published absolute point can be exceeded without proving engine superiority. That stronger claim requires a paired local reference run with the complete artifact, workload, cache, power, quality, and engine envelope.

| Disposition | Source row | Frozen point | Why no ratio is emitted |
|---|---|---:|---|
| **CONTEXT_ONLY** | `nabe-qwen38-27b-vulkan-mtp-c1-c4-c8` | c1 **16.65**, c4 **24.13**, c8 **29.42** aggregate output tok/s | Closest dense Qwen3.8-27B Q4_K_M + MTP bars, but measured on 128 GiB with artifact hash, cache state, power, and repeated-run statistics incomplete. |
| **CONTEXT_ONLY** | `yandaq-lmstudio-q4km-mtp-c1` | **26.47** mean decode tok/s across prose, code, and JSON | Different MTP-capable Q4_K_M bytes and 128 GiB warm sampled workload; artifact hash, frozen prompt packet, held-out quality, and measured power are missing. |
| **CONTEXT_ONLY** | `mikeveerman-q6-mtp-c1-q8-plain-c4` | Q6+MTP c1 **18.72** decode tok/s; prefill **246→210** tok/s; Q8 plain c4 **21.75** aggregate tok/s | Q6/Q8 plus optional MTP are different quant/speculation envelopes; exact hashes, held-out quality, and measured power are missing. |
| **CONTEXT_ONLY** | `thetom-ggml-q4km-plain-and-mtp-c1` | ggml Q4_K_M c1 plain **11.26**; MTP **33.03** tok/s with **31.11%** worst-case spread | Different main/draft GGUF bytes, incomplete power and prompt identity, and unstable non-deterministic MTP output make both points context only. |
| **CONTEXT_ONLY** | `agention-dflash2-rocmfp4-adaptive-c1` | bare **14.0**; adaptive DFlash2 **65.6** structured and **26.1** prose tok/s | WATCH: specialized ROCmFP4+DFlash2 predictable-output ceiling with an unpinned engine and missing exact hashes, repeated quality replay, and power evidence. |
| **CONTEXT_ONLY** | `halogen-flash-next-http-c1` | HTTP mean **43.6**; 32k served MTP mean **41.7** tok/s | Custom sparse-MoE Flash-Next/HGN runtime on 128 GiB is cross-model context; artifact, corpus, cache, and power envelopes do not match dense local Q4_K_M. |
| **CONTEXT_ONLY** | `q38rocm-rocmfp4-mtp` | **30.56–36.04** output tok/s across 512–4096-token contexts | Different ROCmFP4/backend envelope; exact engine provenance and clean long-output integrity still need independent reproduction. |
| **CONTEXT_ONLY** | `flash-next-vulkan-real-agent-median` | **33.1** warm-cache median output tok/s over 10 agent conversations | Sparse-MoE/PLE Flash-Next, mixed quantization, and 128 GiB are cross-model context, never a dense-Qwen matched win. |
| **EXCLUDE** | `nabe-qwen38-27b-hip-corrupt-concurrency` | no admissible point | HIP c8/c16/c32 emitted corrupt output with and without MTP; garbage-token throughput is not a benchmark. |

The reachable target is the distinct **64 GiB** Ryzen AI Max+ 395 / Radeon 8060S appliance. The default fair gate is a plain, speculation-off local A/B on the exact Unsloth Q4_K_M artifact (`sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`); only that paired run can establish engine superiority. MTP is a separately re-pinned optional path because the 26–33 tok/s MTP rows use different GGUF bytes carrying a built-in or sidecar draft head. A future candidate remains non-comparable until #12034 binds the physical receipt to the tested source and the receipt reports `engine=fak-native`, its executed backend/path, `fallback_count=0`, accepted output-token counts, power/thermal state, and quality evidence. Ngram or repetitive-output rates remain excluded from general-throughput claims.
<!-- strix-halo-comparison-ladder:end -->

## Newer code awaiting comparable remeasurement

This section prevents landed optimizations from masquerading as newer result rows. Entries are reaped when a comparable receipt is promoted above, the work is rejected, or the stated review date passes without a renewed reason to retain it.

| Lifecycle | Candidate | Why it has not replaced a row | Reap condition |
|---|---|---|---|
| **HISTORICAL** | Metal Q4_K/GDN/projection work (#8833, #9096, #9097, #9102) | Landed implementation and focused parity/kernel evidence did not provide a comparable M3 Pro full-run receipt. One attempted #9102 run was invalidated by swap pressure, so it carries no speed claim. | Reaped at review on **2026-08-27**; retain only as implementation history. |
| **HISTORICAL** | M3 Pro near-match (#8697) | The former **APPROXIMATE** observation was **3.3 vs 6.966061 tok/s (~47%)**, native P31/T64 versus llama.cpp P32/T64. | Reaped at review on **2026-08-27**; retain only as non-comparable planning context because it lacks a joint quality-complete receipt. |
| **AWAITING REMEASURE** | M3 Pro Metal sequence prefill & kernel microbenchmarks (post-#1085) | The September 3, 2026 witness ([`qwen38-m3pro-metal-benchmarks-2026-09-03.json`](../_witnesses/qwen38-m3pro-metal-benchmarks-2026-09-03.json)) captures M2 sequence prefill at **7.80× speedup** (19.73 vs 153.91 ms), fused MLP at **1.62×**, and resident Qwen3.5-0.8B Metal decode at **11.42 tok/s** with zero swap growth. These are kernel/sequence microbenchmarks; full Qwen3.8-27B end-to-end remeasurement remains awaiting a comparable receipt. | Promote only after an end-to-end quality-complete, comparable Qwen3.8-27B full-run receipt; review by **2026-09-17**. |
| **AWAITING REMEASURE** | Vulkan Qwen GDN route/decode and host-visible Q4_K staging (#9693, #9680, #9730) | The native Vulkan model path now has GDN routing and decode plus host-visible Q4_K weight staging. Canonical artifact pinning and native receipt/readmit lineage tests also improved run identity, but none of this is an accepted AMD/Vulkan performance receipt. | Promote only after a native full-model, quality-complete, comparable AMD/Vulkan receipt; review by **2026-09-11**. |

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
- [Strix Halo external comparison ladder](strix-halo-comparison-ladder.json)
- [Benchmark authority and comparison doctrine](../../BENCHMARK-AUTHORITY.md)
- [Sanctioned hardware routes](../fleet-compute-nodes.md)
