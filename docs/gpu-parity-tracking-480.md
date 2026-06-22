---
title: "fak GPU parity tracker: Go-CUDA tok/s vs llama.cpp"
description: "The apples-to-apples batch-1 protocol, lever status, and residual for measuring fak's Go-CUDA throughput against the llama.cpp baseline on Qwen2.5-7B."
---

# GPU throughput-parity tracking — Go-CUDA tok/s vs `llama.cpp` (#480)

> **Umbrella tracker for [#480](https://github.com/anthony-chaudhary/fak/issues/480)** —
> *"a measured Go-CUDA tok/s number (prefill+decode, same protocol) on Qwen2.5-7B-Q4_K_M
> next to the llama.cpp baseline — apples-to-apples (same GPU, same batch-1 regime, same
> token ids)."*
>
> **House rule for this repo: every number comes from a real run.** This doc is written on
> a host with **no GPU and no CUDA toolkit** (win32 dev box; the GCP GPU quota is walled —
> see §4). So the real `tok/s`/ratio cells below are deliberately **`pending GPU run`**, each
> annotated with the *exact* command + host needed to fill it. **No number in this doc is
> invented.** The two things that *are* measured already — the `llama.cpp` GPU baseline and
> the SmolLM2-135M head-to-head fak *can* run today — live in [`GPU.md`](../GPU.md) §3/§3b
> and are cited here, not restated as new.
>
> This is the protocol + lever-status + residual; it is **not** a parity claim. fak has not
> yet measured Go-CUDA tok/s on the 7B target, and this doc says so plainly.
>
> **Refreshed 2026-06-22.** The device levers this tracker once listed as *not started* — #484
> (fp16 HGEMM), #485 (Q8_0/Q4_K GEMM), #486 (flash attention) — **landed after the first write**
> and are now **built + GPU-gated** (§3.1). The open work is no longer *writing kernels*; it is a
> GPU run to record their numbers and an end-to-end real-7B decode. Every numeric parity cell
> stays `pending GPU run`.

---

## 1. The apples-to-apples protocol (the peer contract)

The peer protocol is **defined in code** by
[`internal/model/bench_llamacpp.py`](../internal/model/bench_llamacpp.py) and the matching
`cmd/modelbench` flags. It is a **batch-1, single-stream, autoregressive** forward-pass
benchmark — fak's actual claim regime, *not* GPU continuous-batching serving (vLLM/SGLang
are a different regime, deliberately excluded; `bench_llamacpp.py:1-14`).

| Parameter | Value | Source |
|---|---|---|
| Prefill sizes P | **{16, 64, 256}** tokens | `bench_llamacpp.py:23` (`PREFILL_SIZES`) |
| Decode prompt | 16 tokens | `bench_llamacpp.py:24` (`DECODE_PROMPT`) |
| Decode steps D | **32** incremental steps | `bench_llamacpp.py:25` (`DECODE_STEPS`) |
| Token ids | **deterministic LCG**, *identical sequence fed to every engine* | `bench_llamacpp.py:28-33` (`lcg_ids`, seed `2463534242`) |
| Timing | **median wall-clock** of `reps` (default 3) | `bench_llamacpp.py:36-59` (`statistics.median`) |
| What is timed | the **forward pass only** — token ids fed via the low-level eval loop; no sampling / no detokenization | `bench_llamacpp.py:11-14, 47-58` |
| Precisions | **F16, Q8_0, Q4_K_M** (precision axis explicit, not hidden) | `bench_llamacpp.py:7-9` |
| Metric | prefill `tok/s = P / median_s`; decode `tok/s = 1 / per_token_median_s` | `bench_llamacpp.py:73-79` |
| Regime | batch-1 autoregressive decode, single stream | `bench_llamacpp.py:6-8` |

**Apples-to-apples invariants** — all three must hold for a row to be comparable:

1. **Same GPU host** — both engines run on the *same* card (4070 Laptop / GPU server / a GCP
   GPU VM), same driver, same VRAM budget.
2. **Same model + same quantization** — the **Qwen2.5-7B-Instruct-Q4_K_M** GGUF (the GPU
   parity target chosen in `GPU-MODEL-PICK.md`; weights ~4.68 GB, fits 8 GB — see `GPU.md`
   §3).
3. **Same token ids** — the deterministic LCG sequence above, so timing differences are the
   forward pass, not the inputs.

**The wired cross-engine harness.** [`tools/gcp_bench.py`](../tools/gcp_bench.py) runs *both*
peers on one provisioned GPU against the same model and folds them into one `result.json`
(schema `fak.gcp-vm-bench.v2`, an `engines` map). It harmonizes the shape across engines so
the rows are directly comparable (`gcp_bench.py:311-339`):

- **`llama` (baseline):** `llama-bench -m <MODEL> -ngl 99 -p 512 -n 128 -o json`
  (`gcp_bench.py:312`) — full GPU offload.
- **`fak-cuda` (the deliverable):** `modelbench -backend cuda -require-non-reference
  -prefill-sizes 512 -prefill-reps 5 -decode-steps 128 -decode-reps 5 -decode-prompt 16`
  (`gcp_bench.py:327-338`). `-require-non-reference` **fails loudly** if the run silently fell
  back to CPU, so a green number can never be a masked non-GPU result.

> The `gcp_bench` cross-engine shape is P=512 / D=128 (matching `llama-bench`'s `pp512/tg128`);
> the `bench_llamacpp.py` canonical shape is P∈{16,64,256} / D=32. Either is valid for #480
> **as long as both engines run with identical flags on the same host/model/ids** — the
> table in §3.2 records which shape each pending cell will use.

---

## 2. What is already measured (cited, not restated)

Two real-run results already exist in [`GPU.md`](../GPU.md). They are the *bar* and the
*fitting-model head-to-head*; #480's open target is the 7B number that neither yet provides.

| Already measured | Where | What it is |
|---|---|---|
| `llama.cpp` CUDA baseline on Qwen2.5-7B-Q4_K_M, RTX 4070 | `GPU.md` §3 (`llama-bench -ngl 99`, median of 5) | the **bar to beat**: prefill `pp512` 2256 ± 45 t/s, decode `tg128` 48.0 ± 0.4 t/s; weights 4.36 GiB resident |
| fak-CUDA **vs** `llama.cpp` on SmolLM2-135M (a model that *fully fits* the GPU) | `GPU.md` §3b (`FAK_CUDA_GRAPH=1`, decode median over 128 steps) | fak-CUDA reusable-graph path **≈120 tok/s decode — even with `llama.cpp` Q8_0 (120)**, at f32; output argmax-exact |

**The gap #480 still owns:** fak-CUDA has **not** *measured* tok/s on **Qwen2.5-7B-Q4_K_M**.
The two device levers that gated the 7B — native quantized GEMM (**#485**) and the K-quant GGUF
load (**#489**) — **both landed after this doc's first write** (§3.1), so the 7B cell is no longer
*lever-gated at the op level*: the Q4_K weight stays ≈4.7 GB resident (the #485 VRAM witness keeps
it int4-narrow), well inside any GPU, and on a datacenter GPU even an f32 7B (~28 GB) fits — the
WSL ~15 GB ceiling no longer bounds the run. **What remains is to *run* it:** execute the §4.2 /
§4.3 commands on a GPU host (to record the realized cosines + tok/s), and wire the real Qwen2.5
config through the model engine for an end-to-end decode (the op-level kernels are already proven
by the synthetic-Llama forward gate `TestCUDAForwardMatchesRef`; a real-7B end-to-end serve is the
open follow-up — the model loop still fetches f32 weights, so quant serving needs the loop to
consume the resident Q4_K/F16 buffers). It is now **GPU-gated**, not lever-gated.

---

## 3. Lever status — every issue this umbrella covers

### 3.1 Status table (landed / built+gated / not-started + deciding evidence)

Verified against `git log` and the working tree on **2026-06-21**, re-verified **2026-06-22**
(the #484/#485/#486 device-lever rows below, which landed after the first write). "Landed this
session" means a commit referencing the issue is in `main`'s recent history.

| # | Lever | Status | Deciding commit / `file:line` |
|---|---|---|---|
| **#482** | async backend — ops enqueue, fence **only** at Read/Argmax | ✅ **LANDED this session** | commit `1b9f68a`; `Caps{Async: true, …}` at `internal/compute/cuda.go:181`; fence-generation machinery `cuda.go:74-115` (`fenceGen`, `Ready()`); device-side Argmax + sync-vs-async tok/s delta in `tools/run_482_acceptance_on_gpu.sh` |
| **#479** | device-resident KV Evict / Clone — **no host round-trip**, quarantine witness preserved | ✅ **LANDED this session** | commit `30438ad`; on-GPU compaction + single-rotation re-RoPE at `internal/compute/cuda.go:522-593`; `Host()` stays `(nil,false)` (resident); witness `tools/run_479_acceptance_on_gpu.sh` |
| **#489** | GGUF loader + dequant-on-load (7B *loads*) | 🟢 **K-quant LANDED — Q4_K/Q6_K load + resident path** | commit `13ec795` (legacy Q4_0/Q4_1) **plus the K-quant path**: `internal/ggufload/gguf.go` decodes `Q4_K`(type 12)/`Q6_K`(14) (`dequantF32` at `gguf.go:1695`, sizes `:1653-1665`); `quant_q4k_loader.go` keeps the raw super-block bytes **resident** (`AddResidentQ4K`/`ResidentQ4KEligible`, no f32 materialization); `ComputeSource.Weight(name, want)` (`compute_source.go:70`) uploads at the requested resident dtype. The Q4_K geometry (`getScaleMinK4`) is the one #485's device tile reproduces. |
| **#483** | CUDA Graphs / capture for the batch-1 decode step | 🟢 **BUILT + GATED — issue open, live worker extending** | reusable graph **ships** gated `FAK_CUDA_GRAPH=1`: `cuda.go:44,62,149,479`; `cudaGraphExecUpdate` instantiate-once at `internal/compute/cuda_kernels.cu:277-300`. Drove the SmolLM2 7.5→120 tok/s win (`GPU.md` §3b). Open residual: fixed-capacity KV (1024) → dynamic/ring. *Disjoint live-worker lane — not touched by this doc.* |
| **#485** | native device matmul for **Q8_0 / Q4_K** (no dequant-to-f32) | 🟢 **BUILT + GATED** (landed after this doc's first write) | commits `e1513da` (native Q8_0/Q4_K device GEMM) + `a8cb3fc` (max-norm dominant-row fix, **caught on the 4070 GPU witness**). `cuda.go:291` now advertises `Caps{… UploadDtype:true …}`; `Upload(t, Q8_0/Q4_K)` narrows at H2D, dequant **fused into the GEMM tile** (`k_q8_gemm`/`k_q4k_gemm`), weight stays int8/int4-resident (VRAM witness). Gates `cudaQ8CosineMin=0.999` / `cudaQ4KCosineMin=0.995` (`cuda.go:103-104`); witness `cuda_quant_test.go`; acceptance `tools/run_485_acceptance_on_gpu.sh`. **The 7B device-GEMM gate is cleared in code; realized cosines/tok/s are the GPU residual.** |
| **#484** | fp16 compute path (cuBLAS HGEMM / tensor cores) | 🟢 **BUILT + GATED** (landed after this doc's first write) | commit `f6eb11d`; `Upload(t, F16)` narrows weights to `__half` at H2D (+ ColMajor transpose-repack), GEMM via `cublasGemmEx` tensor-core HGEMM, F32 accumulate (`fcuda_matmul_f16`, `cuda_backend.h:62-66`). Gate `cudaFP16CosineMin=0.997` (`cuda.go:75`); witness `cuda_fp16_test.go`; acceptance `tools/run_484_acceptance_on_gpu.sh`. |
| **#486** | flash / paged-attention CUDA kernel for the Attention op | 🟢 **BUILT + GATED** (landed after this doc's first write) | commit `49d445b`; `cuda.go:291` advertises `Caps{… FusedAttn:true}`; fused online-softmax `k_flash_attention` over the KV window (no `scores[nPos]` row materialized), across MHA/GQA/MQA. Gate `cudaFlashAttnCosineMin=0.999` (`cuda.go:124`); it is also the per-layer attention in the end-to-end forward gate `TestCUDAForwardMatchesRef` (`cuda_test.go:205`); acceptance `tools/run_486_acceptance_on_gpu.sh`. |
| **#481** | native-Windows signed `-tags cuda` build (off the WSL workaround) | 🔴 **NOT STARTED** | only the WSL toolchain exists (`internal/compute/setup_cuda_wsl.sh`, `build_cuda.sh`); `GPU.md` §2 "A native Windows `-tags cuda` build (signed) is the portability follow-up" |

**Issue-checkbox crosswalk** (the umbrella's living checklist in #480):

- [x] flash/paged attention kernel → **#486, built + GPU-gated `49d445b`** (`Caps.FusedAttn`, `cuda_flash_test.go`)
- [x] native Q8_0/Q4_K device matmul → **#485, built + GPU-gated `e1513da`/`a8cb3fc`** (`Caps.UploadDtype`, `cuda_quant_test.go`)
- [x] fp16 HGEMM / tensor cores → **#484, built + GPU-gated `f6eb11d`** (`cublasGemmEx`, `cuda_fp16_test.go`)
- [x] CUDA Graphs decode capture → **#483, shipped gated `FAK_CUDA_GRAPH=1`** (issue open: dynamic KV residual)
- [x] async enqueue (fence only at Read/Argmax) → **#482, landed `1b9f68a`**
- [x] enablers: device-resident Evict → **#479, landed `30438ad`**; quant-on-load + GGUF K-quant → **#489, Q4_K/Q6_K load + resident path `13ec795`+**; native build → **#481, not started**

### 3.2 The parity result matrix — the cells a real GPU run fills

**Every cell below is `pending GPU run`.** None is estimated. The command that produces each
is in §4.

**Target model: Qwen2.5-7B-Instruct-Q4_K_M** (the #480 target). *Op-level levers cleared: #485
(Q4_K device matmul) and #489 (K-quant load + resident path) landed (§3.1). Remaining: a GPU run
+ wiring the real Qwen2.5 config end-to-end through the model engine (the loop still fetches f32).*

| Engine | Precision | Prefill t/s (P=512) | Decode t/s (D=128) | Ratio vs llama | Status |
|---|---|---:|---:|---:|---|
| `llama.cpp` (CUDA, `-ngl 99`) | Q4_K_M | 2256 ± 45 ✓ | 48.0 ± 0.4 ✓ | 1.00 (baseline) | **measured** — `GPU.md` §3 |
| **fak-CUDA** (`-backend cuda`) | f32 | `pending GPU run` | `pending GPU run` | `pending` | **GPU-gated** (kernels landed; needs a GPU host) |
| fak-CUDA (Q4_K, #485 landed) | Q4_K | `pending GPU run` | `pending GPU run` | `pending` | **GPU-gated**: #485 built (`e1513da`), run `tools/run_485_acceptance_on_gpu.sh` |
| fak-CUDA (f16, #484 landed) | f16 | `pending GPU run` | `pending GPU run` | `pending` | **GPU-gated**: #484 built (`f6eb11d`), run `tools/run_484_acceptance_on_gpu.sh` |

> *Producing cmd (once a 7B-capable fak-CUDA path + a GPU host both exist):*
> `python tools/gcp_bench.py --engine all` on the GPU host (§4), or the per-host
> `modelbench` + `llama-bench` pair in §4.2.

**Fitting model: SmolLM2-135M** (the head-to-head fak *can* run today — canonical
`bench_llamacpp.py` shape P∈{16,64,256} / D=32, *and* the `GPU.md` §3b graph result).

| Engine | Precision | Decode t/s | Status |
|---|---|---:|---|
| `llama.cpp` | Q8_0 | 120 ± 15 ✓ | **measured** — `GPU.md` §3b |
| `llama.cpp` | F16 | 261 ± 10 ✓ | **measured** — `GPU.md` §3b |
| **fak-CUDA** (reusable graph, `FAK_CUDA_GRAPH=1`) | f32 | ≈119–120 ✓ | **measured** — `GPU.md` §3b (parity with Q8_0) |
| fak-CUDA on a **datacenter GPU** (L4/B200, no WSL per-call tax) | f32 | `pending GPU run` | **blocked**: GPU quota (§4) |

The §3b SmolLM2 row is the **proof the protocol + backend produce a real, comparable
number** the moment a GPU is reachable; the datacenter-GPU row would re-measure it off the
WSL per-call-overhead floor.

---

## 4. The residual — exact command(s) + named host to fill every `pending` cell

The numbers are unobtainable on this host **by design**: win32, no CUDA toolkit, GCP GPU
quota walled. Here is precisely what produces them and where.

### 4.1 Cross-engine, one command (preferred) — `tools/gcp_bench.py`

Provisions a GCP GPU VM, ships the *local working tree* (exact code under test, incl.
uncommitted), builds + runs **both** `llama` and `fak-cuda` against the same model on the
same GPU, folds into one `result.json`, and **always tears the VM down** (`gcp_bench.py:1-60`):

```bash
# Offline plan review — works on THIS host, touches nothing, provisions nothing:
python tools/gcp_bench.py --dry-run

# The real cross-engine head-to-head — REQUIRES GPU quota (see blocker below):
python tools/gcp_bench.py --tier g2-l4 --engine all     # L4: llama + fak-cpu + fak-cuda
python tools/gcp_bench.py --tier a4-b200 --blackwell    # the flagship Blackwell run
```

**Blocker (today): GPU quota is walled.** `gcp_bench.py:542-543` stops with:

> `STOP: no tier in the ladder has live GPU quota. Request quota (IAM > Quotas) or a
> reservation; or use --proof to try L4.`

→ **Residual action:** raise the GCP GPU quota (or attach a reservation) for the project,
then re-run the command above. Until then only `--dry-run` runs here.

### 4.2 Per-host, on an *already-provisioned* GPU (4070 Laptop / GPU server)

On a host that already has a reachable NVIDIA GPU + CUDA toolkit, run the two peers directly
with identical model/ids, then divide:

```bash
# fak-CUDA side (decode; reusable-graph path) — from GPU.md §3b:
FAK_CUDA_GRAPH=1 go run -tags cuda ./cmd/modelbench \
    -gguf <Qwen2.5-7B-Instruct-Q4_K_M.gguf> -backend cuda -require-non-reference \
    -prefill-sizes 512 -prefill-reps 5 -decode-steps 128 -decode-reps 5 -decode-prompt 16
#   (7B path additionally needs #485 Q4_K matmul + #489 K-quant load; until then this runs
#    only on a fitting model, e.g. internal/model/.cache/smollm2-135m)

# llama.cpp side (the bar) — from GPU.md §3:
llama-bench -m <Qwen2.5-7B-Instruct-Q4_K_M.gguf> -ngl 99 -p 512 -n 128 -r 5
```

### 4.3 Per-issue device acceptance gates (correctness witnesses + #482 tok/s delta)

The levers that **landed this session** ship their own on-GPU gates. They exit non-zero on a
SKIP (a skip is not a pass), so a green run is real device evidence:

```bash
bash tools/run_479_acceptance_on_gpu.sh   # on-GPU Evict == never-saw, no host round-trip
bash tools/run_482_acceptance_on_gpu.sh   # async==sync argmax parity + sync-vs-async tok/s delta
bash tools/run_484_acceptance_on_gpu.sh   # fp16 HGEMM == cpuref within cudaFP16CosineMin (0.997)
bash tools/run_485_acceptance_on_gpu.sh   # Q8_0/Q4_K device GEMM cosines + VRAM witness + quant-vs-f32 tok/s
bash tools/run_486_acceptance_on_gpu.sh   # fused flash attention == cpuref/naive within cudaFlashAttnCosineMin
```

Each resolves the CUDA toolchain (`~/cudaenv` else PATH `nvcc`), builds `libfakcuda.a`, and runs
the `-tags cuda` witness; `run_482` additionally prints `sync decode … tok/s` /
`async decode … tok/s` / `delta (async − sync)` — the **#482 tok/s evidence**, also
`pending GPU run` here.

### 4.4 Named hosts

| Host | Reaches it | Use for |
|---|---|---|
| RTX 4070 Laptop (Anthony's box, WSL2+CUDA 12.6, sm_89) | `.\tools\fak_laptop_test.ps1 accept` then §4.2/§4.3 | the existing baseline + correctness gates (carries WSL per-call tax) |
| GPU server | direct SSH; §4.2/§4.3 with `FAK_CUDA_ARCH=sm_90` | native-Linux numbers off the WSL floor |
| GCP GPU VM (L4 sm_89 / B200 sm_100) | `tools/gcp_bench.py` once quota is raised (§4.1) | the one-command cross-engine `result.json` |

---

## 5. Why this doc carries no fak-CUDA 7B tok/s (the honest block)

1. **No GPU on this host.** win32 dev box, no CUDA toolkit (`nvcc` absent); the `-tags cuda`
   backend type-checks and `go build ./...` is green, but device execution needs a GPU. The
   acceptance scripts (§4.3) detect this and exit `4` (INCONCLUSIVE — "a skip is not a pass").
2. **GCP GPU quota walled.** `gcp_bench.py` STOPs at the quota gate (§4.1); only `--dry-run`
   runs here.
3. **The 7B target's op-level levers have landed; the *run* hasn't.** The device GEMM (#485) and
   K-quant load + resident path (#489) that gated Qwen2.5-7B-Q4_K_M **shipped after this doc's
   first write** (§3.1): the int4 device GEMM exists (`e1513da`), the Q4_K weight loads + stays
   resident (`internal/ggufload`, `AddResidentQ4K`), and the #485 VRAM witness keeps it int4-narrow
   — so the ~28 GB-f32 / WSL-RAM argument no longer bounds it (it was the **dev-laptop** ceiling,
   `GPU.md` §3; a datacenter GPU sidesteps it). What is still `pending GPU run` is (a) the realized
   device cosines / tok/s and (b) wiring the real Qwen2.5 config end-to-end through the model engine
   (the loop still fetches f32) — both need a GPU host, not another lever.

A doc that says *"these numbers require executing X on host Y"* is the correct deliverable
here; the one unacceptable outcome — an invented `tok/s` — is avoided. The moment a GPU host
clears (quota raised, or the 4070/DGX runs §4.2–4.3), every `pending GPU run` cell is filled
by the named command and folded back into the §3.2 matrix and `GPU.md`'s house style.

---

## 6. Provenance

- **Protocol:** `internal/model/bench_llamacpp.py:1-79`; cross-engine harness
  `tools/gcp_bench.py:1-60, 311-339`.
- **Measured baselines (cited, not restated):** `GPU.md` §3 (llama.cpp 7B baseline), §3b
  (fak-CUDA vs llama.cpp SmolLM2-135M); house style `docs/benchmarks/MODEL-BASELINE-RESULTS.md`.
- **Lever evidence:** commits `1b9f68a` (#482), `30438ad` (#479), `13ec795` (#489), and —
  **landed after the 2026-06-21 write** — `f6eb11d` (#484), `e1513da`/`a8cb3fc` (#485),
  `49d445b` (#486); `internal/compute/cuda.go`, `internal/compute/cuda_kernels.cu`,
  `internal/compute/cuda_{fp16,quant,flash}_test.go`, `internal/ggufload/`,
  `cmd/modelbench/main.go`; acceptance gates `tools/run_{479,482,484,485,486}_acceptance_on_gpu.sh`.
- **Residual:** `tools/gcp_bench.py` (quota STOP at `:542-543`); per-host commands §4.2.
- Written 2026-06-21 on a host with no GPU; **lever status refreshed 2026-06-22** (the #484/#485/#486
  device levers landed and are now built + GPU-gated). Every numeric parity cell is `pending GPU run`
  by design.
