# NVIDIA CUDA Perf-Gap Attribution — fak vs llama.cpp b11870

Campaign: `GOAL-nvidia-cuda-perf-gap` (2026-10-31). Method: six read-only wave-1 cohorts
(kernel inventory, VRAM byte-budget, prefill decomposition, decode roofline, config-delta
census, MTL/Halo transfer audit) over committed receipts + public source at pinned commits;
synthesis below. Cohort reports: `_scratch/goals/nvidia-gap/cohort-{A..F}-*.md`.
Evidence classes: **HW-WITNESSED** (committed node receipts), **SW-VERIFIED** (source at the
measured commit), **CODE-TRACED** (source at HEAD), **ESTIMATED** (labeled bands, wide under
the 95 W cap). Estimates stay ESTIMATED until a profile/ABBA receipt lands.

## 0. Headline numbers

| Corpus | Node / envelope | Model | fak decode | fak prefill | ratio decode / prefill |
|---|---|---|---|---|---|
| raider-5090 h2h-head-20261029 | native Linux, RTX 5090 Laptop @ 95 W cap, ctx 2048 | Qwen3.8-27B-Q4_K_M (48 GDN + 16 full-attn hybrid) | 13.31 vs 24.41 tok/s | 171.5 vs 566.6 tok/s (TTFT 1219 vs 364 ms) | **0.545x / 0.303x** [HW-WITNESSED] |
| anthony-current-20261025 internal4070 | Windows 11; fak in WSL2, llama native; 4070 Laptop 8 GB, ctx 8192 | Qwen2.5-Coder-3B-Instruct-Q4_K_M (dense) | 46.10 vs 95.91 | 38.67 vs 4237.3 (cold-512, llama cache 0.0) | **0.48x / ~0.009x** [HW-WITNESSED] |
| anthony-current-20261025 egpu | eGPU RTX 5070 Ti over TB4 | same 3B dense | ~69–73 vs ~230 | ~48–49 vs ~6630–6660 | **~0.32x / ~0.007x** [HW-WITNESSED] |
| VRAM, raider same model+ctx | — | — | fak 23.8–24.1 GB resident vs llama ~16.2 GB; fak h2h cold-512 OOMs (`cudaMalloc(5255168)`, `h2h-head-20261029/raw/fak-serve-excerpt.txt:6`), llama completes; the typed 170 MiB ffn-gate scratch failure (`cudaMalloc(178257920)`) was witnessed in the Oct 28 service runs (`hwbench-20261028/fak-arm-journal-excerpt.txt`; `qwen38-150w-20261028/raw/p150b-native-loadgen.journal-tail.txt:15`) | | **+6.6 GB gap** [HW-WITNESSED] |

Cross-corpus ratios are NOT comparable (different models/architectures); only within-corpus
arm ratios are meaningful (cohort E §corpora). The 95 W cap explains cross-node absolute
deltas, **not** the same-node raider ratio.

## 1. Structural facts (shared denominator)

- Model geometry [CODE-TRACED, pinned GGUF fixture `internal/ggufload/testdata/qwen38_q4km_header.gguf.gz`]:
  64 active layers = 48 GDN linear-attention (nK=16, nV=48, kHd=vHd=128) + 16 full-attention
  (24 Q / 4 KV heads, head dim 256, partial RoPE 64) + 1 MTP block (dropped unless
  `FAK_SPECULATIVE=mtp`). hidden 5120, FFN 17408, vocab 248,320. Quant census: Q4_K 11.370 GB,
  Q6_K 4.527 GB, Q5_K 1.038 GB, F32 0.105 GB, Q8_0 0.052 GB (payload 17.09 GB).
- Per-token weight traffic ≈ **16.09 GB for BOTH engines** (same GGUF, single read; cohort D §3).
  Bytes moved are NOT the gap — kernel/orchestration efficiency under the 95 W cap is.
- Implied effective bandwidth [cohort D §4]: fak@h2h ≥219 GB/s, fak@HEAD-default 297 GB/s,
  llama 400 GB/s vs 896 GB/s spec; plausible capped sustained BW ESTIMATED 550–700 GB/s →
  perfect-engine ceiling ≈ 30–37 tok/s. L3 bar (≥0.8x llama ≈ 19.5 tok/s) needs ≥~330 GB/s
  effective — reachable without touching the power envelope.
- Raider h2h fak serve flags [HW-WITNESSED, `raw/argv.txt`]: `--engine inkernel --backend cuda
  --kv-precision fp32 --native-context-tokens 2048`; env `FAK_GPU_BUDGET_MB=auto FAK_STREAM_Q4K=1
  FAK_INKERNEL_RADIX=off FAK_CUDA_Q4K_PREFILL=mmq` recorded in `RESULTS.md:68`. forward_path
  `cuda/qwen35-gdn-ssm-decode-v1` (sequence spine, not scalar loop).

## 2. Prefill attribution (canonical raider cell: 209-tok prompt, 1142 ms server-side p50)

Server-side prefill p50 = 1142 ms (TTFT 1219 − 77 ms gateway/client, both HW-WITNESSED).
Bottom-up sum of medians ≈ 380–420 ms → **~600–800 ms residual unlocated until profiled**.
Llama floor for the same work at the same cap: ~175–180 ms (pp512/pp2048 oss-baselines).

| ID | Cause | Est. impact (raider) | Evidence | Fix candidate | Status |
|---|---|---|---|---|---|
| A1 | **Dense Q4_K never reaches panel prefill**: `promptPanelFor` requires `PromptPanelRowExact(dt)` true only for Q8_0 (`internal/compute/cuda_prompt_panel.go:95` @2cc7d389; `:110` at HEAD, behavior identical) → dense Q4_K models prefill one P=1 forward per token, re-streaming ~16 GB weights each. Anthony's ~100x prefill gap is dominated by this route choice (~55–85%), not WSL2 | ~55–85% of Anthony ~0.009x; ~0% for raider (hybrid has the sequence spine) | SW-VERIFIED @2cc7d389 + HW-WITNESSED timing signature (15–26 ms/tok growing with prompt length; `gemm=none` every row) | Admit Q4_K to the panel path (mmq row-exactness) or ship `-lean`-style Q8 for serve; PHL-153/154 prove the panel mechanism hits 185–722 tok/s on the same GPU | OPEN — largest architectural cost, largest absolute win |
| A2 | **Q8 dequant-compensated GEMMs for all GDN projections + full-attn q/k** (272 calls/prefill × dequant→f32→fp16 split + 3×cuBLAS; `internal/compute/cuda/quant_matmul.cuh:331-333`, P≥128 arm `:666-680`): ~67 GB extra traffic/prefill + 816 cuBLAS calls + 272 scratch-alloc cycles. Survives `FAK_CUDA_Q4K_PREFILL=mmq` (that arm only covers Q4_K tensors) | ESTIMATED 80–150 ms (12–23% of TTFT) | CODE-TRACED + HW-WITNESSED route (serve log `q4k prefill arm=mmq`, Q8 path unchanged) | Serve these projections as Q4_K via mmq, or route P≥128 Q8 through the rowtile panel GEMM (`fcuda_q8_matmul_rows_f32`) instead of dequant+cuBLAS | OPEN — largest structural delta vs llama, which never dequants these |
| A3 | **SIMT scalar O(P²) prompt attention is the default**; tensor-core hd256 port exists but dormant: `QWEN38_PREFILL_ATTN_MMA_DEFAULT 0` + unset `FAK_CUDA_QWEN38_PREFILL_ATTN` (`internal/compute/cuda/attention.cuh:301-308`); kernel `fattn_mma_hd256.cuh` present | ESTIMATED 10–50 ms (2–7%) | CODE-TRACED; MTL precedent: same-class default flip worth 1.4–2.5x (`receipts/mac-m3pro-qwen35/fak-vs-llama-grid-20261017.md`) | P-05 hardware A/B (ABBA, greedy parity, P∈{128,512,2048}) then flip the macro | OPEN — "landed-dormant" failure mode, exactly what MTL fixed |
| A4 | **Launch storm**: ~2,000–2,400 eager cgo/kernel round-trips per prefill, no CUDA graph (graph capture is decode-only today, opt-in `FAK_CUDA_GRAPH`) | ESTIMATED 20–80 ms (1–3%+) | CODE-TRACED launch census (cohort C §1.1 table); no nsys witness yet | CUDA-graph the spine or batch small-op launches | OPEN |
| A5 | **Sequential GDN recurrence** across the whole prompt per layer (one thread/value-col, block-wide sync per token step; chunked scan exists behind `FAK_CUDA_GDN_RECURRENCE=chunk16`, `qwen35_gdn.cuh:1409-1416`) | ESTIMATED 5–15 ms (0.4–1.6%) | CODE-TRACED | Qualify chunk16 on the gap node | OPEN |
| A6 | **Unlocated residual ~600–800 ms (50–65% of TTFT)**: bottom-up estimate vs measured 1142 ms; either effective compute/BW under the cap far below assumed rates for these exact shapes, or allocator churn on per-call compensated scratch | 600–800 ms until located | gap arithmetic [HW-WITNESSED anchors]; rows above ESTIMATED | Prerequisite: rerun h2h-fak-b with `FAK_CUDA_TIMESTAMP_PROFILE=1` (committed harness, `kernel_profile.cuh`; one env var, no code) | OPEN — must precede any kernel work |
| A7 | Gateway + tokenizer + client delta (TTFT − server prefill) | 77 ms (6.3%) | HW-WITNESSED delta | Separate workstream, not engine | OUT OF SCOPE here |

## 3. Decode attribution (raider, single-stream, ctx 2048)

| ID | Cause | Est. impact | Evidence | Fix candidate | Status |
|---|---|---|---|---|---|
| D1 | **Baseline is stale vs trunk**: h2h ran `daa1178b`, pre-mmvq-default. The f32-coeff GEMV (2 GiB coeff ring vs ~5 GB working set → redundant mint+read passes) was replaced by the llama.cpp-style DP4A MMVQ default (`internal/compute/cuda/q4k_mmvq.cuh`; commits `99995e395b` opt-in, `b8116b0304` default — both ON origin/main, verified 2026-10-31) | **WITNESSED +53.7%**: 11.78 → 18.10 tok/s same lane/envelope (cross-session A/B per that receipt, not same-session ABBA); takes ratio 0.545x → ~0.74x | HW-WITNESSED (`raider-5090/cuda-decode-defaults-20261029/RESULTS.md`) + git ancestry check | Re-measure h2h at HEAD (no code needed) | FIXED on trunk, re-measure pending |
| D2 | **No CUDA graph; ~570 kernel launches + 48 `cudaStreamSynchronize` fences per token** (one per GDN layer, `qwen35_gdn.cuh:936`), all under global `cudaMu` (`cuda_specialized.go:266`) → zero inter-layer pipelining. Graph replay is documented "the only way past the proven ~12 tok/s op-per-call floor" but Instantiate OOMs at this fit (graph-capacity KV prealloc `internal/compute/graph_cuda.go:24-33` not charged to the fit plan) | ESTIMATED 3–6 ms/tok ≈ 5–10% of token time | CODE-TRACED launch count + HW-WITNESSED OOM record; no nsys µs witness yet | Fix graph fit accounting + reclaim VRAM headroom (V2–V4), then ABBA graph on/off | OPEN — gated by VRAM |
| D3 | **KV cache FP32 3-row (pre-RoPE Kraw + K + V) vs llama f16 2-row**: 2x read traffic at depth (268 vs 134 MB/tok @2048), 3x resident pressure → OOM rows; f16/bf16 tiers exist and are realized on CUDA (`internal/compute/cuda/kv_precision.cuh:12-40`, typed store/load/attention wired at `cuda.go:1386-1399`), default stays f32 (`cmd/fak/serve_native_config.go:47`) | <1% at ctx≤512; ~1% @2048; large for capacity (cold-512 OOM) | CODE-TRACED + HW-WITNESSED OOM rows; R2 long-ctx drop validates KV-byte sensitivity | Flip serve default to f16 (or fit-based selection) with a parity witness at depth | OPEN — one-flag change, parity witness owed |
| D4 | **GDN recurrent state FP32, 302 MB/tok r+w constant** (allocated via `uploadHostF32`, `qwen35_hal.go:515,534,543`) | Δ151 MB/tok vs an f16-state engine ≈ 0.5% (IF llama uses f16 state — UNVERIFIED) | CODE-TRACED; low confidence | Verify llama state dtype; consider f16 state tier | OPEN, low confidence |
| D5 | **Shared ceiling: lm_head Q6_K 1.043 GB read every token** (vocab 248,320 logits materialized for both engines) | ≈2.6–3.5 ms/tok ≈ 4–7% — caps the ceiling, not a differentiator | CODE-TRACED census | Top-k logit / pruned-vocab paths would help both | OPEN (shared) |
| D6 | **Per-token host embedding H2D upload** (no `EmbeddingRow` on cudaBackend; `internal/model/hal.go:966-984`, counted `hostFallbackEmbedRowUpload`) | small (2,880 B + launch) | CODE-TRACED | Device-resident embedding row or pinned staging | OPEN, minor |

MMVQ vectorization verdict: P=1 grid (out blocks × 128 threads, 4 warps, vdr=2) matches
upstream llama.cpp `MMVQ_PARAMETERS_GENERIC` — no identifiable GEMV vectorization gap vs
llama; the residual lives in orchestration (D2), KV/state dtypes (D3/D4), shared lm_head (D5).

## 4. VRAM footprint attribution (fak 22,814 MiB vs llama 16,172 MiB, same model+ctx; fak figure = mem-breakdown ledger after prefill + 128 decode tokens, f32 arm)

Gap = **6,642 MiB** (f32 arm; 4,594 MiB under mmvq). Component ledger
(`raider-5090/mem-breakdown-20261029/RESULTS.md` + code trace, cohort B):

| ID | Component | Size | Share | Code site | Fix direction |
|---|---|---:|---:|---|---|
| V1 | **Q8_0 re-encode of every `.linear_attn.*` + self_attn q/k/qkv projection** (9 bpw vs ~6.1 native; 6.65e9 elems → 7,135 MiB vs ~4,826) **+ device-resident token embedding** (+682 MiB vs llama CPU_Mapped) | **+3,001 MiB** | 45% | gate `internal/model/quant_q4k.go:880` `ResidentQ4KEligible`; `quant.go:407`; `hal.go:414`; `cuda_upload.go:327` | Extend row-normalized raw-K-quant residency to GDN layouts; keep embedding host-resident — largest single shrink |
| V2 | **Fixed 2 GiB `q4_coeff` ring arena** — allocated ONLY by the f32 P<4 decode arm (`q4_coeffs_for` sole caller `quant_matmul.cuh:691`); **dead under the mmvq trunk default** | **+2,048 MiB** (at daa1178b) | 31% | `quant_matmul.cuh:527` (`kQ4CoeffArenaBytes`), lazy alloc `:575` | Vanishes automatically at HEAD default — confirm by re-measure; do not re-introduce |
| V3 | **Uncapped idle-pool retention** of per-request KV + dequant/split scratch (freed buffers retag into exact-size pool, returned only on refusal-triggered pressure-trim) | **+1,619 MiB gross** | 17–24% | `runtime_memory.cuh:115` (`device-pool`), trim `cuda.go:297` | Watermark the pool; proactive `TrimLarge(watermark)` between requests |
| V4 | **Admission under-prices real prefill scratch**: plan prices ~23 MiB transients (`capacity.go:206,226`), the P≥128 dequant path needs up to 340+170 MiB per large GEMM (`cuda_prompt_panel.go:56` `cudaDequantScratch`; the witnessed typed failure = 170 MiB FFN gate [17,408×5,120] fp16 split, `cudaMalloc(178257920)`, Oct 28 service runs: `hwbench-20261028/fak-arm-journal-excerpt.txt` + `qwen38-150w-20261028/raw/p150b-native-loadgen.journal-tail.txt:15`) → requests pass the precheck then OOM mid-prefill, poisoning the context | not in static gap; root of the red cold-512 row | — | `internal/compute/capacity.go` vs `internal/compute/cuda_prompt_panel.go:56` | Price `cudaDequantScratch` into `requestMemoryPlanWithExtras` (or route prefill through arms that skip dequant) — resolves ticket `raider-fak-arm-oom-20261028` |
| V5 | **Precheck reads live `cudaMemGetInfo` free × 0.85, not a configured KV size** — so it sees ~18–34 MiB once V1–V3 consume the card; logged `pressure-trim ... want=450217984 budget=33813299 margin=-416404685` (`raider-5090/qwen38-150w-20261028/raw/p150b-native-loadgen.journal-tail.txt:15`, Oct 28 12:19Z loadgen service run) | mechanism behind the ticket symptom | — | free read `inkernel_planner_memory.go:215` (`requestDevicePressureFit` → `compute.DeviceMemoryInfo`); headroom const `inkernel_planner.go:2776` (0.15); verdict `capacity.go:1059` `RefuseMemoryPlanIfTooBig` | Honest after V1–V4 land; keep the fail-open policy, fix the inputs |
| V6 | CUDA context/module/driver overhead delta | ~+372 MiB | 6% | M−L residual | structural; observe only |

Ranks V1–V3 sum to 6,668 MiB ≈ the entire witnessed gap: fixing them lands fak at ~16.1 GB,
matching llama.cpp and restoring ~7.9 GB of KV/scratch headroom. V4 is what keeps the
precheck honest once headroom returns.

## 5. Config / environment deltas (from cohort E; classified, not engine-intrinsic)

| Delta | Class | Effect on observed ratios |
|---|---|---|
| WSL2 guest (Anthony only; no native-Linux fak arm on that box) | ENV | est. ×1.5–3 multiplier on Anthony numbers; NOT isolable from committed data; zero effect on raider (both arms native) |
| `lm_head=cpu-q6k` + host sampling (Anthony; triggered by WSL2 fit decision; no CUDA device Q6_K head route) | EI (placement) | est. ×1.5–2.5 share of Anthony decode gap; absent on raider (27B fits fully on device) |
| Prefix-cache asymmetry (fak zero reuse for dense+CUDA: `InKernelBackendPrefixReuseSupported()` false @2cc7d389; llama slot cache active) | EI (feature) | **~0% of the prefill RATE ratio** (client measures uncached tokens; internal cold_512 had llama cache=0.0 yet still 0.009x); up to ×1.9 p50 inflation in warm-ish eGPU cells (true-cold ≈0.014x, still ~70x) |
| 95 W SW cap (raider, both arms) | ENV | explains cross-node absolute deltas, explicitly NOT the same-node ratio |
| Context capacity / concurrency / model-file identity / driver | matched within every paired run | no within-ratio effect (the "2048 vs 8192" appears only across non-paired runs) |
| Binary provenance (fak daa1178b vs 2cc7d389 across corpora; llama b11870 two builds) | ENV | the Q4_K panel gate (A1) is present at both fak commits → does not explain either ratio |

## 6. Why MTL/Halo did not hit this wall (transfer audit, cohort F)

MTL reached near-parity by flipping exactly the class of defects found here — "landed-dormant"
optimized kernels activated with a receipt-pinned default (Q4_K wide-tile 1.49x @P=64;
attention-kernel choice alone 2.2x deep decode; f16 KV mirror halving 537→268 MB/token).
CUDA status of each MTL/Halo success factor:

| Factor | CUDA status | Note |
|---|---|---|
| Tensor-core prompt-panel attention | PARTIAL — dormant (A3) | `fattn_mma_hd256.cuh` present; macro default 0 |
| Q4_K prefill GEMM arm activation | PARTIAL — opt-in (A2 adjacent) | mmq exists; default arm = compensated dequant+cuBLAS; raider receipt already set `=mmq` |
| 16-bit KV planes | PARTIAL — shipped, off (D3) | typed kernels wired into decode + panel; f32 default |
| Whole-token captured graph decode | PARTIAL — off + OOM-broken at large fits (D2) | `graph_cuda.go:24-33` fit-accounting gap |
| Chunked prefill walk, device-resident state | PRESENT | sequence spine default; no per-panel host round-trip observed |
| Radix-shared prefix cache | PRESENT (but raider receipt ran `FAK_INKERNEL_RADIX=off`) | verify campaign argv before attributing residuals |
| DP4A quantized decode GEMV default | PRESENT (D1, +53.7% witnessed) | done |
| Device argmax / per-head GDN norm / 2–8-lane decode batching / async+GPUDirect load | PRESENT | transferred already |

Conclusion: the NVIDIA lag is NOT a missing-kernel problem — nearly every MTL/Halo winning
mechanism already exists on CUDA. It is a **default-selection + memory-footprint** problem:
dormant optimized paths (A1–A3), a stale-as-measured baseline (D1), and a 6.6 GB footprint
excess that OOMs prefill and blocks graph capture (V1–V5).

## 7. What changes at trunk today (re-measure expectations, no code)

Verified against origin/main (2026-10-31): `b8116b0304` (mmvq default) and `99995e395b` are
on trunk; f16/bf16 KV tiers realized (`serve_native_config.go:47`); q4_coeff arena unreachable
under the default arm. Therefore, re-running the h2h acceptance workload at HEAD should show,
before any new code: decode ≈ 18.1 tok/s (~0.74x, D1), VRAM −2 GiB (V2), prefill unchanged
(A2–A5 all still open). This re-measure is the cheapest possible move and the new baseline.

## 8. Ranked wave-2 candidates (gain per cost)

1. **Re-measure h2h at HEAD** (0 code): converts the stale 0.545x baseline to its current
   value; confirms V2 arena absence. Next checkable step for the whole campaign.
2. **Profile the prefill** with `FAK_CUDA_TIMESTAMP_PROFILE=1` (0 code): locates the 600–800 ms
   residual (A6) and sizes A2/A4 before kernel work.
3. **Price dequant scratch into admission (V4)**: bounded change in `internal/compute/capacity.go`
   + planner extras; makes the precheck honest and stops the witnessed cold-512 OOM that
   poisons the context — turns the red acceptance row green and resolves
   `docs/tickets/raider-fak-arm-oom-20261028`.
4. **Retire the Q8 dequant-compensated path for GDN/attn-qk projections (A2)**: ~67 GB
   traffic/prefill removed; largest structural prefill delta vs llama.
5. **MMA prompt-attention A/B + default flip (A3)**: MTL precedent 1.4–2.5x; receipt-pinned.
6. **f16 KV default flip (D3)**: one flag; halves bytes/token at depth; restores capacity.
7. **Graph fit-accounting fix + idle-pool watermark (V3/D2)**: unblocks CUDA-graph decode.
8. **Admit dense Q4_K to panel prefill (A1)**: largest absolute win (Anthony ~100x → 100s of
   tok/s, PHL-153/154 proof-of-mechanism), largest architectural cost — schedule after 1–5.

Stop rule: marginal gain no longer pays against remaining goal budget (max_iters 40); stop or
reassess when a candidate's ESTIMATED band no longer clears its measured cost.

## 9. Assumptions and weak points (carried from cohorts, labeled)

- llama b11870 default KV dtype = f16 (no `-ctk/-ctv` in committed argv; auto-resolution) — medium
  confidence, not witnessed in h2h stderr.
- llama GDN state dtype unknown — assumed f32-equivalent for byte parity (D4 low confidence).
- 95 W-capped sustained bandwidth 550–700 GB/s — ESTIMATED, anchored by SM-clock telemetry
  (915–1297 MHz of 3090) + cross-node eGPU effective-BW back-out; no committed BW measurement
  on this node.
- Host launch overhead 3–6 ms/tok and all A2–A5 ms bands — ESTIMATED until the §8.2 profile lands.
- V1 per-tensor Q8_0 vs raw-K-quant residency is arithmetic-derived from the gate logic, not yet
  tensor-witnessed (cohort B open item: `/debug/vars` site breakdown by tensor family).
- Anthony internal-4070 block carries the co-residency caveat (2,935 MiB unowned WSL residency
  during the llama block; declared `observed_negative_co_resident_reference_not_accepted`).
- llama "-b 512 -ub 128" from earlier briefs does NOT appear in any committed argv — treated
  as unverified (cohort E row 9).

## 10. Wave-2 re-measure status (2026-10-31, HW-WITNESSED)

Re-measured at public trunk `c7caeb22d` on raider-fak under measure lease
`hwl-0900772181a96b06`; full receipt in [RESULTS.md](RESULTS.md) + `raw/`. §7 predictions vs outcome:

| Prediction (§7) | Outcome |
| --- | --- |
| decode ≈ 0.74x via mmvq trunk default | **CONFIRMED**: 13.31 → 18.19 tok/s = 0.745x llama (+36.7%) |
| VRAM −~2 GiB, cold-512 OOM cleared | **CONFIRMED**: peak 21,186 MiB; cold-512 completes 3/3 (was OOM at daa1178b) |
| prefill roughly unchanged until profiled/fixed | **CONFIRMED**: 160.9 tok/s (flat within spread), 0.284x llama |

Two new findings supersede parts of this doc's ranking:

1. **Trunk regression (witnessed, isolated, not yet fixed)**: HEAD unknobbed cannot serve the
   baseline config — bisect first-bad `217d2d0f1` (i-quant dtype advertisement flips the
   `qwen35DeviceQKNativeRow` gate; GDN warmup fails closed `tensor-preflight`). All timed arms
   used the documented revert knobs (`FAK_QWEN35_NATIVE_{QK,GDN_INPROJ}_ROWS=0`,
   `FAK_QWEN35_NATIVE_SSM_OUT=0`), which restore the historical route byte-for-byte.
   Ticket owed against `217d2d0f1`.
2. **Prefill hot spot now measured, not estimated**: `FAK_CUDA_TIMESTAMP_PROFILE=1` ts-stage
   profile puts GDN projections at **34.1% of prefill stage time** (`k_qwen35_gdn_fused_in_proj`
   24.6% + `k_qwen35_gdn_out_proj` 9.5%), making the §4 GDN-layout lever the confirmed next hill;
   the A2–A5 ms bands in §9 are replaced by these measurements for the GDN share.
