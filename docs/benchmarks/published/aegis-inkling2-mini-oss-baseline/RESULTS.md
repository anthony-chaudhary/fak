---
title: "Campaign record: Inkling-2-Mini OSS engine baseline on Aegis RTX 5080 16 GB"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak aegis-inkling2-mini-oss-baseline (2026-11-22): Inkling-2-Mini OSS engine baseline on Aegis RTX 5080 16 GB."
---

# RESULTS — Inkling-2-Mini OSS engine baseline on Aegis RTX 5080 16 GB (2026-11-22)

**[HW-WITNESSED]** aegis-fak (RTX 5080 16 GB, sm_120, driver 615.78.08, kernel 7.2.11-arch1-1),
measure lease `hwl-66dbb68b85999389` (released). Eight short exclusive windows (11:34Z–12:21Z),
each engine alone on the GPU with process-group teardown and a GPU-clean check between stages.
The serving unit `fak-llama-qwen27b.service` was already down when the window opened. A peer's
DS4.1 lease had stopped it. It was restarted and confirmed healthy after the last window.
Machine-readable: [`perf-receipt-v2.json`](perf-receipt-v2.json). Every probe JSON (incl. llama-bench) is a row of
[`receipts/probes.jsonl`](receipts/probes.jsonl) keyed by window (`w-20261122a`…`h`); window logs, argv and
server-log tails are in [`receipts/windows.txt`](receipts/windows.txt). The harness (`window.sh`, `tt.py`, chains)
is [`receipts/harness.txt`](receipts/harness.txt); the local engine patches are in
[`receipts/offload-wiring.patch`](receipts/offload-wiring.patch).

Checkpoints are pinned by sha256, verified on the workstation and again on Aegis ([`receipts/sha256.txt`](receipts/sha256.txt); lease and serving-unit restore in [`receipts/lease.txt`](receipts/lease.txt); engine root-cause lines in [`receipts/engine-root-causes.txt`](receipts/engine-root-causes.txt)):
Inkling-2-Mini@b85c3cfa `65e60f51…`, TML-d24-varloop-research@657b53ba `79e60b65…`, and Unsloth
UD-Q3_K_XL in its `Mainline_Support` layout `e2d03e8c…` (root layout `f72f396d…`).

## Verdict

On this 16 GB consumer Blackwell card, the fastest arm measured is **llama.cpp with a 3-bit GGUF
held entirely on the GPU, at 80.4 tok/s decode**. Serving the HF NVFP4 checkpoint directly needs
CPU expert offload. Of the engines tried, that worked only in **vLLM with a one-line local wiring
patch**, at **14.1 tok/s**. SGLang reached decode only in a later window, with a
local OffloaderV2 patch. It ran at **3.57 tok/s** with a KV pool below 32k tokens
([SGLang V3 addendum](#addendum--sglang-v3-offload-decode-2026-11-23)).
fak native refuses the family with a typed error (fak `d835bf8f35`). The native forward is ticket
`inkling2-loop-native-forward-port-20261122`.

| arm | engine @ rev | weights | ctx | placement | decode tok/s (n=3) | TTFT s | state |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | llama.cpp `ad50351f5` (has #32704) | UD-Q3_K_XL mainline GGUF | 32768 | all on GPU, `-b 512 -ub 128` | **80.37** (80.34–80.38) | 0.058 | MEASURED |
| L1b | llama-bench, same build | same | — | all on GPU | tg128 **80.99 ±0.14**, pp512 **2362 ±13** | — | MEASURED |
| V1 | vLLM `e72fc9ae5` + wiring patch | HF NVFP4/MXFP8 | 32768 | 10 GB experts in UVA host memory, `--max-num-batched-tokens 1024` | **14.12** (14.115–14.116) | 0.84 | MEASURED |
| V0 | same | same | 4096 | same, default batch | 14.11 | 0.84 | MEASURED |
| VL1 | same | varloop, 1 pass | 32768 | same | **31.16** | 0.39 | MEASURED |
| VL3 | same | varloop, 3 passes | 32768 | same | **14.12** | 0.84 | MEASURED |
| VL5 | same | varloop, 5 passes | 16384 | same | **9.13** | 1.29 | MEASURED (32k at default batch needed 4.5 GiB KV, 4.26 available) |
| S | SGLang PR #51926 `9c64454b` | HF | 16–32k | V1 / V2 offloader | — | — | ABSTAIN (see 2); superseded by S3 |
| S3 | same + V3 patch | HF NVFP4/MXFP8 | 32768 configured, **19968-token KV pool** | V2 `cpu` mode: layer 0–5 and 18–23 expert w13/w2 in pinned host, prefetch 1, eager | **3.57** (3.566–3.567) | 0.97 | MEASURED (2026-11-23, pool-capped; a ≥32k pool is a typed ABSTAIN, see addendum) |
| L0 | llama.cpp `29b52d39` | Unsloth root GGUF | — | — | — | — | ABSTAIN: Unsloth-native looped layout, `blk.24.attn_q` not found |

All arms are C1, greedy, 57-token prompt, 256 completion tokens counted from `usage`, after one
discarded warmup. Every arm produced coherent reasoning text. **Not measured here:** quality or eval
accuracy. Q3_K_XL is not quality-equivalent to the native NVFP4/MXFP8 weights. The
[addendum](#addendum--prefill-and-3-bit-quality-2026-11-22) measures prefill and the logit-level quality gap.

## What was learned

1. **vLLM and SGLang never wire CPU offload into the Inkling2 decoder** (source-read, not run unpatched
   here). Both build the decoder as a plain `nn.ModuleList` and never call
   `get_offloader().wrap_modules`, so `--cpu-offload-gb` cannot move Inkling2 layer weights. Every vLLM
   and SGLang run in this receipt used the local wiring in `receipts/offload-wiring.patch`. For vLLM
   the working form is `--cpu-offload-params experts --offload-backend uva`.
2. **SGLang never reached decode.**
   (a) With the V1 wiring at `--mem-fraction-static 0.88`, prefill CUDA-graph capture OOMed in a
   CuTe kernel (w-20261122a).
   (b) With graphs disabled at 0.70, the first request failed: `Pointer argument cannot be accessed
   from Triton (cpu tensor?)`. Source-read cause: `prepare_loop()` stacks the loop-controller weights
   into a plain attribute that V1's `functional_call` never swaps (w-20261122d). Also source-read:
   NVFP4 block scales are swizzled only `if scale.is_cuda`, so CPU-offloaded scales would reach the
   expert GEMM in checkpoint layout. That would mean wrong output or a layout error; it was not
   witnessed, because no V1 run decoded. Any future SGLang-V1 number needs that path fixed first.
   (c) The V2 attempt (prelude/coda experts only) OOMed at model construction (`Tried to allocate 648 MiB`,
   w-20261122f). This was the local patch's fault: it built all 24 layers on the GPU before calling
   `wrap_modules`, while V2 itself offloads at wrap time. A lazy generator was not tried, so this is a
   patch-order failure, not an SGLang V2 defect. It was fixed and measured on 2026-11-23
   (arm S3, [SGLang V3 addendum](#addendum--sglang-v3-offload-decode-2026-11-23)).
   Separately, the first-request FlashInfer sm_120 JIT compiled about a dozen kernels in parallel at
   about 1–3 GB host RAM each. That host-OOM-killed the scheduler (w-20261122c) until it was capped
   with `MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1`.
3. **Offloaded-expert decode is far slower than the resident GGUF.** vLLM, with 10 GB of experts read
   zero-copy from UVA host memory, decodes at 14.1 tok/s against 80.4 tok/s for llama.cpp with a 3-bit
   GGUF fully on the GPU (5.7x). That comparison changes engine, quantization and placement together.
   Host-transfer boundedness is the likely cause but was not profiled.
4. **Cost per layer visit is roughly constant, rising about 12% from 1 to 5 passes.** tok/s x visits is
   about 748 at 1 pass (24 visits), 678 at 3 (48) and 657 at 5 (72, at 16k context). Varloop at 1 pass
   is 2.2x faster than at 3, so pass count is a direct speed dial. Its quality cost was not measured.
5. **On this card, vLLM's KV admission is set mainly by batched-token sizing, not context.** At the same
   offload and `--max-num-seqs 1`, 8k with the default batch size needed 4.06 GiB of KV against
   3.69–3.89 GiB available (w-20261122c/d), while 32k with `--max-num-batched-tokens 1024` started and
   measured (w-20261122f). For comparison, 32k with the default batch size needed 4.81 GiB
   (w-20261122a). The figures are in `receipts/engine-root-causes.txt`. Source-read attribution: the
   hybrid planner charges the 48 short-conv cache groups (window 4) for in-flight tokens
   (`max_concurrent_batches x max_num_batched_tokens`).
6. **There are two incompatible GGUF layouts.** Mainline llama.cpp `29b52d39` cannot load Unsloth's
   root files (`blk.24.attn_q.weight not found`). `Mainline_Support/*.gguf_file` loads and serves on
   mainline `ad50351f5`, which contains #32704 (`690b87d881`). The pre-#32704 gibberish/PPL figures
   (about 130 vs 7.65) are cited from Unsloth's Mainline_Support README, not measured here.

## Next

- fak native: ticket `inkling2-loop-native-forward-port-20261122`. The decode target to beat on this
  card is 80.4 tok/s (on-GPU llama.cpp). Getting there needs resident low-bit experts, not host offload.
- Upstream candidates, not filed: the one-line offload wiring for vLLM `vllm/models/inkling2/nvidia/model.py`,
  plus the SGLang controller-stack and offloaded-scale-swizzle issues in item 2.

## Addendum — prefill and 3-bit quality (2026-11-22)

**[HW-WITNESSED]** Same card and same engine builds. Six windows, `w-20261122p` to `u` (19:16Z–22:05Z),
all run under three exclusive measure leases, all released. The holder for all three was
`claude-inkling-prefillq-20261122`:

| lease | windows |
| --- | --- |
| `hwl-969db3285cc71091` | p |
| `hwl-df1dbc9cb71e5b1c` | q–t |
| `hwl-8961bd50ee2d2523` | u |

The grant and release events are in the windows receipt. `fak-llama-qwen27b.service` was stopped
for each window and confirmed active afterwards. Ticket: `inkling2-prefill-quality-aegis-20261122`.

Receipts:
- Every prefill, llama-bench and analysis JSON is a row of [`receipts/addendum-probes.jsonl`](receipts/addendum-probes.jsonl).
- [`receipts/addendum-windows.txt`](receipts/addendum-windows.txt) holds:
  - window logs and server argv;
  - per-window `STAGES`, which carry the `llama-perplexity` flags for each arm;
  - selected `llama-perplexity` lines and vLLM KV/error lines;
  - GGUF download sha256 and a GGUF header dump;
  - lease events and the analysis summaries. The file is [`receipts/addendum-windows.txt`](receipts/addendum-windows.txt).
- All scripts are in [`receipts/addendum-harness.txt`](receipts/addendum-harness.txt).
- Machine-readable: the `addendum` key of [`perf-receipt-v2.json`](perf-receipt-v2.json).

### Prefill

Each prompt is token ids sliced from wikitext-2 at a distinct, non-wrapping offset, run with
`max_tokens 1` and streamed. TTFT runs from request start to the first streamed token; prefill tok/s
is prompt tokens / TTFT, median of 3 after a 512-token warmup. The llama.cpp server runs with
`--cache-ram 0` and vLLM with `--no-enable-prefix-caching`. Server-internal timings
(llama.cpp `timings`, vLLM `request_prefill_time_seconds`) agree with the client figures within 0.3%.

| prompt tokens | llama-bench ub512 | llama-bench ub128 | llama.cpp server (L1) tok/s · TTFT | vLLM (V1) tok/s · TTFT | llama.cpp / vLLM |
| --- | --- | --- | --- | --- | --- |
| 512 | 2296 ±51 | 1601 ±6 | 1318 · 0.39 s | 229 · 2.24 s | 5.8x |
| 2048 | 1954 ±4 | 1488 ±2 | 1331 · 1.54 s | 810 · 2.53 s | 1.6x |
| 8192 | 1621 ±0 | 1326 ±1 | 1244 · 6.59 s | 806 · 10.16 s | 1.5x |
| 32768 / 32000 | OOM (3356 MiB compute buffer) | 959 ±0 | 930 · 34.4 s | 784 · 40.8 s | 1.2x |

Arm configurations:
- **llama-bench**: UD-Q3_K_XL all on GPU, `-fa 1 -b 2048 -n 0 -r 3`, `-p 32768` for the last row.
- **llama.cpp server**: the L1 serving config (`-c 32768 -b 512 -ub 128 -fa on`), 32000-token prompt for
  the last row, peak VRAM 15212 MiB.
- **vLLM**: as specified, `--cpu-offload-gb 10 --cpu-offload-params experts --offload-backend uva
  --max-num-batched-tokens 1024 --max-num-seqs 1 --max-model-len 32768`, peak VRAM 13872 MiB.
  **Chunked prefill is forced**: any prompt longer than 1024 tokens runs as ceil(N/1024) scheduler
  steps. The first 32000-token rep was 707 tok/s (45.3 s); the warmup at 512 tokens took 4.46 s.

1. **vLLM offload prefill is flat at about 800 tok/s from 2k to 32k.** A full 1024-token step costs
   about 1.26 s whatever the context. That fits a fixed per-step cost of reading the UVA-resident experts,
   but it was not profiled. llama.cpp loses 30% from 2k to 32k as attention cost grows, so its lead
   shrinks from 1.6x to 1.2x at 32k.
2. **vLLM steps of 512 and 768 tokens take 1.8–2.2x longer than a full 1024-token step**
   (steps of 256 tokens or fewer do not).
   Extra sizes measured in `w-20261122q`:

   | prompt tokens | TTFT s | tok/s |
   | --- | --- | --- |
   | 128 | 0.98 | 131 |
   | 256 | 1.07 | 239 |
   | 512 | 2.23 | 230 |
   | 768 | 2.81 | 273 |
   | 1024 | 1.26 | 816 |
   | 1536 | 3.51 | 438 |

   A 512-token prompt takes 1.8x longer than a 1024-token one, and 1536 = 1024 + 512 costs
   1.26 + 2.25 s. The proposed cause is **source-read only**: the sm_120 MoE path
   (`trtllm_bmm.py`, `moe_sm120.py` tile selection) appears to pick a 64-token tile with 32-row blocks
   for 257–1023-token steps, which reads each expert's weights twice; at 1024 it picks a 128/128 tile
   and reads once. The confirming experiment (forcing the tile at 512) was not run. Practical effect:
   on this configuration, padding a 512–768-token prompt up to 1024 tokens lowers TTFT. Prompts of
   256 tokens or fewer are already faster than 1024.
3. **Larger vLLM batch sizes do not fit next to the offload.** These were refused at engine start by
   KV admission and recorded as ABSTAIN (`w-20261122p`/`q`), so 1024 is the largest tried that serves:

   | `--max-num-batched-tokens` | max-model-len | KV needed | KV available |
   | --- | --- | --- | --- |
   | 4096 | 32768 | 8.44 GiB | 5.34 GiB |
   | 4096 | 8192 | 7.52 GiB | 5.34 GiB |
   | 8192 | 8192 | 7.52 GiB | 4.5 GiB |

   This matches baseline item 5: batched-token sizing, not context, sets the KV charge.
   llama-bench at ub512 likewise OOMs at 32k, so on 16 GB both engines trade prefill batch size for
   context.

### How close is 3-bit to full?

**Method and why.** We scored next-token distributions on one shared token stream:
wikitext-2-raw test, 80 chunks of 512 tokens, scoring positions 256–510 of each chunk, which gives
**20,400 scored positions** per arm. All 95% CIs are bootstrap over chunks with 1000 resamples.
This was the cheapest honest method:
- It needs one forward pass per arm. A task eval needs reasoning-length decode, which runs at
  14 tok/s on the native engine, so gsm8k-class evals were too slow for these windows.
- KLD and top-1 agreement measure distribution shift at 20,400 positions in one forward pass. A
  1k-question eval cannot resolve gaps this small.

Sources and references:
- **GGUF arms** use `llama-perplexity --kl-divergence-base`, which saves full-vocabulary log-probs.
- **Native NVFP4** is the HF checkpoint served by vLLM `e72fc9ae5`. Its layer 0–19 experts are true
  W4A4 NVFP4 (`tl.dot_scaled` e2m1×e2m1 on sm_120) and layers 20–23 are MXFP8 W8A8. That kernel
  path is source-read from vLLM `e72fc9ae5`, not profiled. vLLM is fed
  the identical token ids through `prompt_logprobs=20`, so only the native top-20 is available.
  A 10-chunk rerun was bit-identical.
- **KLD vs native** is therefore KL over the native top-20 plus one tail bucket, which is a
  **lower bound**. Against the full-vocabulary BF16 reference the same bound captures **75–78% of
  the full KL** on every arm (q3 0.0140/0.0183, q8 0.00236/0.00305), so we scale bound/0.76 to
  estimate full KL.
- **Harness check:** Q8_0 vs BF16 measures **KL 0.00305, top-1 97.65%**, against ggml-org's published
  0.0030 / 97.6% (40 chunks; model card of
  [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) at the downloaded
  revision `e8a2cecf`). Our PPL decode matches `llama-perplexity`'s own PPL within 0.07%.

GGUF arms. Expert tensor types come from the GGUF header dump in the windows receipt, which also
holds the download sha256 values. All four files have `add_bos_token = false`, so llama.cpp and vLLM
see the same first token.
- `q3`: Unsloth UD-Q3_K_XL (Mainline_Support), `--fit on` = all on GPU, the L1 serving placement.
  - ffn_down: NVFP4 in layers 0–19, IQ4_NL in 20–23.
  - ffn_gate_up: NVFP4 in 12 of the 20 layers 0–19. The other 8 are IQ2_S/IQ2_XS/IQ3_XXS/Q3_K, and
    layers 20–23 are IQ3_S/IQ3_XXS.
  - Non-expert tensors are a Q5_K/Q6_K/Q4_K/Q8_0 mix.
- `q3cpu`: the same file with every expert tensor in host RAM.
- `q4km`: Unsloth UD-Q4_K_M.
  - All 40 expert tensors of layers 0–19 are stored as NVFP4, the checkpoint's own format. Byte
    identity with the checkpoint was not checked.
  - Q8_0/Q5_K in 20–23; non-expert tensors Q8_0.
  - Experts of layers 0–3 are on the CPU.
- `q4cpu`: the same file with all experts on the CPU.
- `q8`: ggml-org Q8_0 with all experts on the CPU.
- `bf16`: ggml-org BF16, with all experts on the CPU. Its README describes the NVFP4/MXFP8 experts as
  upcast to BF16.

All CPU-expert arms run `--fit off -ngl 99 --no-op-offload --n-cpu-moe N` (see finding 3 below).

| arm | PPL | ΔPPL vs native (ln ratio, 95% CI) | top-1 vs native | KLD vs native (top-20 bound) | full KL vs BF16 | top-1 vs BF16 |
| --- | --- | --- | --- | --- | --- | --- |
| native NVFP4 (vLLM) | 7.194 | — | — | — | n/a (top-20 only) | 95.64% |
| bf16 | 7.181 | −0.19% [−0.41, +0.05] | 95.64% [95.36, 95.93] | 0.0084 [0.0074, 0.0098] | — | — |
| q8 | 7.179 | −0.22% [−0.45, +0.02] | 95.42% | 0.0088 | 0.0031 [0.0028, 0.0033] | 97.65% |
| q4cpu | 7.186 | −0.12% [−0.35, +0.13] | 95.58% | 0.0087 | 0.0033 | 97.55% |
| q4km | 7.207 | +0.17% [−0.07, +0.40] | 94.72% | 0.0108 | 0.0094 | 95.65% |
| q3cpu | 7.237 | +0.59% [+0.25, +0.92] | 94.37% | 0.0145 | 0.0137 | 95.26% |
| **q3 (served L1)** | **7.246** | **+0.71% [+0.37, +1.06]** | **93.75% [93.39, 94.12]** | **0.0168 [0.0155, 0.0181]** | **0.0183 [0.0170, 0.0195]** | **94.31% [93.98, 94.65]** |

**Answer.**
- **Against BF16** (the checkpoint's own weights at full activation precision), the served 3-bit GGUF
  has full KL **0.018**, **94.3%** top-1 agreement and **+0.9% PPL**.
- **Against native NVFP4**, the bound is **0.0168** (about 0.022 full KL estimated), top-1 is
  **93.7%** and PPL is **+0.71%** (CI excludes zero).
- In the same metric, Q3 sits about **2x as far from native as BF16 does**.
- Q4_K_M with CPU experts is indistinguishable from Q8_0/BF16 against native (bound 0.0087 vs
  0.0088/0.0084).

**Is BF16/Q8 an acceptable proxy for native NVFP4? For ranking quants, yes. For absolute distance,
no.**
- BF16 and native disagree on top-1 at **4.4%** of positions, with KL(native‖BF16) ≥ 0.0084 (about
  0.011 full estimated). That is as large as Q4_K_M-on-GPU's distance from BF16.
- Every GGUF arm measured at least that far from native (bound 0.0084–0.0168).
- The extra distance is not additive. For q3, the BF16-referenced full KL (0.018) is about 0.004
  below the native estimate (about 0.022). For q8 and q4cpu the shortfall is about 0.008
  (0.003 vs about 0.0115).
- The PPL difference between BF16 and native is not significant (−0.19%, CI spans zero), so the
  native vLLM path does not measurably cost wikitext PPL. It changes *which* distribution is served,
  not how good it is on this corpus.
- The arm order is the same against both references, except that q8 and q4cpu swap places. They
  are within CI of each other on both.
- The BF16↔native gap mixes NVFP4 activation quantization with every other engine-numerics
  difference (vLLM vs llama.cpp kernels, attention, softcap). This receipt does not separate them.

### Further findings

1. **Placement alone moves llama.cpp numerics by as much as a quant step.** The same file, with
   experts mostly on GPU vs all on CPU:
   - Q4_K_M (experts of layers 4–23 on GPU vs all on CPU): full KL vs BF16 **0.0094 vs 0.0033**,
     top-1 95.65% vs 97.55%.
   - UD-Q3_K_XL (all on GPU vs experts on CPU): 0.0183 vs 0.0137.

   The GPU expert path is also *farther* from native (q4km 0.0108 vs q4cpu 0.0087), so it does not
   bring llama.cpp closer to the native output. The mechanism (CUDA MMQ activation quantization vs
   CPU dot products) is not attributed here. Consequence: a quant's quality figure is only valid for the placement it was
   measured with. The served L1 arm is `q3`.
2. **The GGUFs encode the same experts in three different ways.**
   - UD-Q4_K_M stores every layer 0–19 expert as NVFP4.
   - UD-Q3_K_XL stores layer 0–19 ffn_down and 12 of the 20 ffn_gate_up tensors as NVFP4, and
     requantizes the rest to 2–4 bits.
   - ggml-org Q8_0/BF16 re-encode or upcast all experts.

   The requantized ffn_gate_up layers are the likely source of Q3's extra KLD over Q4_K_M at the same
   placement: 0.006 on the native bound, and 0.009–0.010 full KL vs BF16. This was not isolated,
   because the non-expert tensor types also differ.
3. **llama.cpp `ad50351f5` op-offload bug.** With Inkling2 experts in host memory and op offload
   active (batch ≥ 32), logits can be **NaN**. Configurations witnessed as NaN from chunk 1:
   - `--n-cpu-moe 24` on Q4_K_M and on Q8_0, `--n-cpu-moe 22` on Q4_K_M, and `-ngl 0` on Q8_0
     (`w-20261122r`/`s`).
   - `--fit on` auto-spill on Q8_0 (`w-20261122q`).

   Other configurations:
   - `--n-cpu-moe 4` with op offload is finite but perturbed: chunk-1 PPL 5.298, against 5.213 for
     the same placement with `--no-op-offload` (`w-20261122t` q4km).
   - At `--n-cpu-moe 24`, `--no-op-offload` (5.228) and `-ub 16` (5.214) were both finite over
     2 chunks.
   - `--fit on` auto-spill on Q4_K_M was finite: KL 0.0102 vs Q8_0, against 0.0095 for the
     `--n-cpu-moe 4 --no-op-offload` arm. Placement also differs there and the CIs overlap. This is
     the superseded arm `q4km-fit-opoffload`, `w-20261122p` `lppl-q4km`.

   Upstream candidate, not filed. CPU-expert serving of this model on `ad50351f5` should use
   `--no-op-offload` (or `-ub` below 32).

### Scope

This addendum covers wikitext-2 only, at context 512 with 20,400 positions. Not covered:
- Long-context quality, chat or reasoning-mode quality, and task accuracy.
- Full native KL: native is limited to top-20, so its KLD is a bound, scaled by a measured factor of
  about 0.76.
- An independent reference implementation: "native" means vLLM `e72fc9ae5` numerics, which are
  deterministic but not checked against one.
- A loop-count sweep: the varloop quality cost stays unmeasured.
- Prefill profiling: findings 1 and 2 under Prefill are timing observations plus a source-read
  explanation.
- Batch > 1.

## Addendum — SGLang V3 offload decode (2026-11-23)

**[HW-WITNESSED]** This addendum uses the same card, driver and SGLang build (PR #51926 `9c64454b`). It covers one window, `w-20261122v`, 08:06Z–08:30Z, run under exclusive measure lease `hwl-7a63ec8e07aa6301` (holder `claude-inkling-sglang-v3-20261122`), which has been released. `fak-llama-qwen27b.service` was active at the start and was stopped for the window. systemd then refused the harness's restarts between arms (`start-limit-hit`), so the unit was reset and started after the last arm, and was confirmed active with `/health` reporting ok. Ticket: `inkling2-sglang-v3-offload-decode-20261122`.

Receipts:
- Probe JSON for every arm: [`receipts/sglang-v3-probes.jsonl`](receipts/sglang-v3-probes.jsonl).
- Window logs, argv, and the relevant server-log lines (offloader placement, weight memory, KV pool, error): [`receipts/sglang-v3-windows.txt`](receipts/sglang-v3-windows.txt).
- The engine patch (a diff against the PR's `inkling2.py`) and the chain scripts: [`receipts/sglang-v3-harness.txt`](receipts/sglang-v3-harness.txt).
- Machine-readable: arm `S3`, abstain `S3-32k`, and the `sglang_v3` key of [`perf-receipt-v2.json`](perf-receipt-v2.json).

### The fix

`OffloaderV2.wrap_modules` consumes its generator lazily. For each layer it yields, `_CpuParamOffloader` copies the whitelisted parameters into pinned host memory at once. The V2 attempt (baseline item 2c) built a list of all 24 decoder layers on the GPU before calling `wrap_modules`, so it OOMed at construction.

The V3 patch builds each layer inside the generator instead. It yields only the 12 layers visited once per forward (prelude 0–5, coda 18–23) with whitelist `w13_weight`/`w2_weight`. Loop-core layers 6–17 are built in the same pass and stay resident, because they run 3 times per forward, which a one-way V2 prefetch ring cannot serve. Expert scales, the loop-controller stacks and every non-expert weight stay on the GPU. This also avoids V1's two failure modes:
- the Triton CPU-pointer crash on the controller stacks;
- skipped scale swizzles, because `_prepare_sm120_scales` still sees CUDA scales.

The server log witnesses the placement. `memory_allocated` grows about 118 MB per prelude layer, against roughly 486 MiB of NVFP4 experts per layer had they stayed resident. Weight load ends at 10.79 GB on the GPU.

### Result

| arm | config | KV pool tokens | decode tok/s (n=3) | TTFT s | state |
| --- | --- | --- | --- | --- | --- |
| **S3** `sglang-v3-f78-eager` | `--mem-fraction-static 0.78 --chunked-prefill-size 1024`, CUDA graphs off, `--context-length 32768` | 19968 | **3.57** (3.566–3.567) | 0.97 | MEASURED |
| `f78-decodegraph` | same + decode CUDA graph | — | — | — | ABSTAIN: graph capture OOM, 648 MiB |
| `f78-fp8kv` | same + `--kv-cache-dtype fp8_e4m3` | 43264 | — | — | ABSTAIN: `q.dtype bf16 != k.dtype float8_e4m3fn` (attention backend) |
| `f79`/`f80` | 0.79 (chunk 256, mamba 6) / 0.80 (chunk 1024) | 26752 / 25984 | — | — | ABSTAIN: first prefill OOM, 324 MiB |
| `f805`…`f83` | 0.805–0.83, chunk 256, `expandable_segments`, mamba 6 where set | 31744–36736 | — | — | ABSTAIN: first prefill OOM, 324 MiB |

The workload and reader are the same as the baseline: C1, greedy, the 57-token prompt, 256 completion tokens from `usage`, and one discarded warmup. The output is coherent reasoning text (`text_head` in the probe JSON). Every arm in this table ran eager except `f78-decodegraph`.

1. **SGLang decodes, but at 3.57 tok/s**: 4.0x slower than vLLM's UVA offload (14.1) and 22.5x slower than on-GPU llama.cpp (80.4).
   - The cause is structural. V2 `cpu` mode copies a layer's **whole** expert tensors to the GPU on every forward: about 7.6 GiB per token (8 NVFP4 layers at 486 MiB, 4 MXFP8 layers at 972 MiB).
   - At 3.57 tok/s that is about 27 GiB/s of host-to-device traffic, which is consistent with PCIe-bound copy. This was not profiled.
   - vLLM UVA reads expert bytes in place, so at decode it should touch only the routed experts' pages. That is source-read and was not profiled.
   - V2 cannot be made selective without engine changes, so this placement will not get close to vLLM.
2. **32k context is a narrower typed ABSTAIN, `KV_POOL_BELOW_CONTEXT`.**
   - SGLang accepts `--context-length 32768`, but on 16 GB a bf16 KV pool of 32768 tokens or more does not fit beside the V2 prefetch ring. The ring holds the current and next layer's expert copy, up to about 1.9 GiB for the MXFP8 coda layers.
   - Every fraction from 0.79 up, which gives pools from 25984 to 36736 tokens, OOMs at the first prefill while allocating a 324 MiB expert copy. Smaller prefill chunks, `expandable_segments` and a smaller mamba-state cache did not change that.
   - The only serving point is 0.78, with a 19968-token pool. S3's decode figure is therefore for a short prompt with 32k configured, not for a 32k-resident request.
   - Two routes out are closed in this build: fp8 KV fails in the attention backend, and `--max-mamba-cache-size 4` is refused because mamba_ratio 5 gives max_num_reqs 0.
3. **The decode CUDA graph does not combine with V2 `cpu` offload here.** Capture materializes each offloaded layer's device copy inside the graph pool and OOMs at the first MXFP8 `w13` (648 MiB). That leaves S3 eager.

Not covered: prefill throughput on S3, a long-prompt decode, and a profile of the host-to-device copy. Upstream candidates, not filed:
- the lazy-construction wiring;
- a routed-expert-only (UVA-style) V2 mode;
- fp8 KV in the Inkling2 attention backend.
