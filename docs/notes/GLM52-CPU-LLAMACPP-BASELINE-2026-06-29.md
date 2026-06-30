---
title: "GLM-5.2 on a CPU-only box (CPU server): the witnessed llama.cpp pure-CPU baseline, and why fak-native is still RAM-gated"
description: "The long-open apples-to-apples CPU witness, finally captured on CPU server (EPYC 7742, AVX2-only, no GPU): the real 753B GLM-5.2 UD-Q4_K_M serves on pure CPU via llama.cpp mmap at pp512=16.1 tok/s prefill, tg64=1.47 tok/s decode. Records the operational levers that made it measurable (sequential page-cache pre-warm vs random mmap-fault death; plain mmap vs --numa-distribute thrash), the fidelity caveat (llama.cpp drops GLM-5.2's DSA indexer + MTP head), and the witnessed root cause that still blocks fak-native here: q4 all-resident OOMs the fit gate, and q3/IQ3_XXS dequants to f32 and OOM-kills the box."
---

# GLM-5.2 on CPU server (CPU-only): the llama.cpp pure-CPU baseline + the fak-native RAM wall

_2026-06-29._ Companion to
[GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md)
(the §4.3 **B3 (CPU host)** row) and
[GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md)
(matrix row C). It closes the multi-session-open **"GLM-5.2 CPU throughput"** witness with a
real, measured number on a **CPU-only** host, and records — with a dmesg OOM witness — why the
**fak-native** CPU serve is still blocked on the same box.

Host class is scrubbed to its hardware: a 2-socket **AMD EPYC 7742** (Rome/Zen2), 256 threads,
~1 TB RAM (**~462 GiB free** alongside a protected long-running co-tenant eval), **AVX2+FMA only
(no AVX-512/VNNI)**, no GPU. The lab host/channel stay private.

## 1. The result — llama.cpp pure-CPU, real 753B GLM-5.2

`llama-bench` (build `9d5d882d8`), GLM-5.2 **UD-Q4_K_M** (433.82 GiB, 753.86 B params),
`-ngl 0` (pure CPU), `-t 128`, **plain mmap**, page cache pre-warmed:

| test | tok/s |
|---|---|
| **pp512** (prefill) | **16.10 ± 2.91** |
| **tg64** (decode) | **1.47 ± 0.01** |

So the real 753B MoE **decodes at ~1.5 tok/s on a CPU-only AVX2 box** — usable for async/batch
agent work, not interactive. Prefill is ~11× faster (batched, compute-bound, scales with cores).

**Real-usage cross-check (`llama-server` + a live chat completion).** A `/v1/chat/completions`
request to the same model produced a coherent, correct answer (GLM-5.2 is a reasoning model, so
the text arrived in `reasoning_content` — a faithful explanation of MoE sparse activation),
proving end-to-end *usage*, not just a synthetic bench. The server-side `timings`
**independently confirm the bench**: `predicted_per_second = 1.45 tok/s` decode (≈ the `tg64=1.47`
above) over 160 generated tokens, and `prompt_per_second = 0.87` for a *cold single 45-token*
prompt (single-prompt prefill is far below the batched `pp512=16.1`, as expected). The server
became ready ~9 min after launch on warm cache — note `/health` returns 200 *before* the model
finishes loading, so a real-readiness probe must retry the chat through the early `503 "Loading
model"`.

**Honest scope / category boundary** (carry these with the number):
- This is the **real** 433 GB checkpoint, real serving — comparable to other *real-serving* CPU
  numbers, NOT to the synthetic reduced-layer `glmdsatput` kernel micro-number.
- **Fidelity caveat:** this llama.cpp build logs `model has unused tensor blk.N.indexer.* /
  blk.N.nextn.* — ignoring`, i.e. it **drops GLM-5.2's DSA sparse-attention indexer and the
  multi-token-prediction head**. It is not running the true DSA forward (fak's correctness
  differentiator). The tok/s is an honest *throughput* baseline, not an architecture-faithful one.
- Different host from the GPU-server llama.cpp figure (2.62 tok/s single / 4.84 agg@2 via
  `--n-cpu-moe` hybrid on an 8-GPU box). This is **pure CPU, no GPU at all**, older Zen2 AVX2.

## 2. The operational levers that made it measurable

The prior sessions could not get a CPU number because the load died slow. Two levers fixed it:

- **Pre-warm the page cache sequentially.** A cold `mmap` serve of a 434 GB MoE demand-faults
  expert pages in *random* order (a 512-token prefill routes through most experts), measured at
  **~0.09 GiB/s** — ~80 min just to touch the model once. Pre-reading the shards sequentially
  (`numactl --interleave=all cat *.gguf >/dev/null`) fills page cache at **~1.5–2.8 GB/s** (~30×),
  after which `llama-bench`'s mmap faults hit warm RAM. This is the same NVMe-first lesson as the
  load-speed note, generalized to "warm the cache before you serve."
- **Plain mmap, NOT `--numa distribute`.** With `--numa distribute`, llama.cpp allocated **~238 GiB
  anonymous** RAM, evicted the model's page cache (436→320 GiB), and re-faulted — thrashing for
  20+ min with no result. Plain mmap keeps weights as evictable file-backed pages + tiny anon;
  RAM held steady (avail ~462 GiB throughout) and the bench completed in minutes.

## 3. fak-native is still RAM-gated here (with a dmesg witness)

fak loads weights **all-resident** (no mmap/demand-paged path, #974), so on a box where the model
≈ free RAM it cannot both keep the read cache warm and hold the resident copy:

- **q4 (UD-Q4_K_M, ~458 GiB resident).** Refused by the host fit gate before load:
  `serveGGUFHostHeadroom = 0.15` ⇒ needs `weights ≤ MemAvailable × 0.85 ≈ 393 GiB`; ~458 > 393 ⇒
  typed `FitTooBig`. Correct refusal (parity with the device path) — but it means no number.
- **q3 (UD-Q3_K_M, 335 GiB on disk, IQ3_XXS experts).** The fit gate **passes** (it estimates a
  raw-resident ~350 GiB). But `fak-bin-iq3` (HEAD `7129b6a`, #344 "add IQ3_XXS dequant") **dequants
  IQ3_XXS → f32 at load**, so the resident footprint blows up ~6×: it reached **anon-rss ~464–487 GB**
  by ~15–20 % of the load and was **OOM-killed** — witnessed twice in dmesg
  (`Out of memory: Killed process … (fak-bin-iq3) … anon-rss:486577040kB`). This is the concrete
  root cause behind the prior `RESULT_Q3 = SERVE_DIED`. The global OOM killer correctly took the
  fak process, not the protected co-tenant eval (good-neighbor invariant held).

**So the fak-native CPU decode tok/s on CPU server remains `not yet`.** The missing wiring is a
**raw-resident IQ3 path** (the trunk loader has `quant_q4k_loader.go case TensorIQ3_XXS`, not active
in the staged `fak-bin-iq3`) or the **mmap/demand-paged weight load** (#974-B). Either lets a
≤ ~390 GiB-resident GLM quant load without the dequant blow-up. Next checkable step: rebuild fak from
trunk for raw-resident IQ3 and re-run the q3 serve with the same sequential pre-warm.

## 4. Reproduce

```sh
# Pre-warm page cache (sequential), then pure-CPU llama.cpp bench:
numactl --interleave=all cat <private-nvme>/glm52-q4/UD-Q4_K_M/*.gguf > /dev/null
/projects/llama.cpp/build-cpu/bin/llama-bench \
  -m <private-nvme>/glm52-q4/UD-Q4_K_M/GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf \
  -ngl 0 -t 128 -p 512 -n 64 -r 2 -o md
# NB: use build-cpu/ (the build/bin/ llama-server is a broken 17 KB CUDA stub); do NOT pass
# --numa distribute on this memory-constrained host (anon blow-up + cache thrash).
```
