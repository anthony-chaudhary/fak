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

1. **Same GPU host** — both engines run on the *same* card (4070 Laptop / lab DGX / a GCP
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

**The gap #480 still owns:** fak-CUDA has **not** measured tok/s on **Qwen2.5-7B-Q4_K_M**.
`GPU.md` §3 states why "by construction": the 7B head-to-head is gated on the loaders (GGUF +
quant-on-load, #489) **plus** native quantized device GEMM (#485) — a 7B f32 blob (~28 GB)
will not fit WSL's ~15 GB RAM, and there is no Q4_K device matmul yet. So the 7B cell is
**both** GPU-gated (no GPU here) **and** lever-gated (#485 not started). Both must clear.

---

## 3. Lever status — every issue this umbrella covers

### 3.1 Status table (landed / built+gated / not-started + deciding evidence)

Verified against `git log` and the working tree on **2026-06-21**. "Landed this session"
means a commit referencing the issue is in `main`'s recent history.

| # | Lever | Status | Deciding commit / `file:line` |
|---|---|---|---|
| **#482** | async backend — ops enqueue, fence **only** at Read/Argmax | ✅ **LANDED this session** | commit `1b9f68a`; `Caps{Async: true, …}` at `internal/compute/cuda.go:181`; fence-generation machinery `cuda.go:74-115` (`fenceGen`, `Ready()`); device-side Argmax + sync-vs-async tok/s delta in `tools/run_482_acceptance_on_gpu.sh` |
| **#479** | device-resident KV Evict / Clone — **no host round-trip**, quarantine witness preserved | ✅ **LANDED this session** | commit `30438ad`; on-GPU compaction + single-rotation re-RoPE at `internal/compute/cuda.go:522-593`; `Host()` stays `(nil,false)` (resident); witness `tools/run_479_acceptance_on_gpu.sh` |
| **#489** | GGUF loader + dequant-on-load (7B *loads*) | 🟡 **PARTIAL — landed this session** | commit `13ec795` (legacy Q4_0/Q4_1 32-elem dequant); leaf `internal/ggufload/` (`gguf.go`, `quant_q4k_loader.go`, `dequant_q40_test.go`). K-quant Q4_K device path is the remaining gate for 7B (joins #485). |
| **#483** | CUDA Graphs / capture for the batch-1 decode step | 🟢 **BUILT + GATED — issue open, live worker extending** | reusable graph **ships** gated `FAK_CUDA_GRAPH=1`: `cuda.go:44,62,149,479`; `cudaGraphExecUpdate` instantiate-once at `internal/compute/cuda_kernels.cu:277-300`. Drove the SmolLM2 7.5→120 tok/s win (`GPU.md` §3b). Open residual: fixed-capacity KV (1024) → dynamic/ring. *Disjoint live-worker lane — not touched by this doc.* |
| **#485** | native device matmul for **Q8_0 / Q4_K** (no dequant-to-f32) | 🔴 **NOT STARTED** | `cuda.go:181` advertises **no `UploadDtype`** (so `modelbench` rejects `-quant` on the cuda backend, `cmd/modelbench/main.go:352-357`); `GPU.md` §4 "No quantized device GEMM". **This is the gate for a 7B-on-fak-CUDA run.** |
| **#484** | fp16 compute path (cuBLAS HGEMM / tensor cores) | 🔴 **NOT STARTED** | `GPU.md` §4 "F32 compute only (no fp16 / tensor cores)"; §3b names this the lever from the Q8_0 number to the F16 number (4070 tensor cores idle, ~2–4× on the table) |
| **#486** | flash / paged-attention CUDA kernel for the Attention op | 🔴 **NOT STARTED** | `GPU.md` §4 "Naive decode attention (per-call scratch, one block/head) — no flash/paged attention" |
| **#481** | native-Windows signed `-tags cuda` build (off the WSL workaround) | 🔴 **NOT STARTED** | only the WSL toolchain exists (`internal/compute/setup_cuda_wsl.sh`, `build_cuda.sh`); `GPU.md` §2 "A native Windows `-tags cuda` build (signed) is the portability follow-up" |

**Issue-checkbox crosswalk** (the umbrella's living checklist in #480):

- [ ] flash/paged attention kernel → **#486, not started**
- [ ] native Q8_0/Q4_K device matmul → **#485, not started**
- [ ] fp16 HGEMM / tensor cores → **#484, not started**
- [x] CUDA Graphs decode capture → **#483, shipped gated `FAK_CUDA_GRAPH=1`** (issue open: dynamic KV residual)
- [x] async enqueue (fence only at Read/Argmax) → **#482, landed `1b9f68a`**
- [x] enablers: device-resident Evict → **#479, landed `30438ad`**; quant-on-load + GGUF → **#489, partial `13ec795`**; native build → **#481, not started**

### 3.2 The parity result matrix — the cells a real GPU run fills

**Every cell below is `pending GPU run`.** None is estimated. The command that produces each
is in §4.

**Target model: Qwen2.5-7B-Instruct-Q4_K_M** (the #480 target). *Additionally lever-gated:
needs #485 (Q4_K device matmul) + #489 K-quant load before fak-CUDA can run it end-to-end.*

| Engine | Precision | Prefill t/s (P=512) | Decode t/s (D=128) | Ratio vs llama | Status |
|---|---|---:|---:|---:|---|
| `llama.cpp` (CUDA, `-ngl 99`) | Q4_K_M | 2256 ± 45 ✓ | 48.0 ± 0.4 ✓ | 1.00 (baseline) | **measured** — `GPU.md` §3 |
| **fak-CUDA** (`-backend cuda`) | f32 | `pending GPU run` | `pending GPU run` | `pending` | **blocked**: GPU + #485/#489 |
| fak-CUDA (after #485) | Q4_K | `pending GPU run` | `pending GPU run` | `pending` | **blocked**: #485 not started |
| fak-CUDA (after #484) | f16 | `pending GPU run` | `pending GPU run` | `pending` | **blocked**: #484 not started |

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

### 4.2 Per-host, on an *already-provisioned* GPU (4070 Laptop / lab DGX)

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
```

Both resolve the CUDA toolchain (`~/cudaenv` else PATH `nvcc`), build `libfakcuda.a`, and run
the `-tags cuda` witness; `run_482` additionally prints `sync decode … tok/s` /
`async decode … tok/s` / `delta (async − sync)` — the **#482 tok/s evidence**, also
`pending GPU run` here.

### 4.4 Named hosts

| Host | Reaches it | Use for |
|---|---|---|
| RTX 4070 Laptop (Anthony's box, WSL2+CUDA 12.6, sm_89) | `.\tools\fak_laptop_test.ps1 accept` then §4.2/§4.3 | the existing baseline + correctness gates (carries WSL per-call tax) |
| Lab DGX | direct SSH; §4.2/§4.3 with `FAK_CUDA_ARCH=sm_90` | native-Linux numbers off the WSL floor |
| GCP GPU VM (L4 sm_89 / B200 sm_100) | `tools/gcp_bench.py` once quota is raised (§4.1) | the one-command cross-engine `result.json` |

---

## 5. Why this doc carries no fak-CUDA 7B tok/s (the honest block)

1. **No GPU on this host.** win32 dev box, no CUDA toolkit (`nvcc` absent); the `-tags cuda`
   backend type-checks and `go build ./...` is green, but device execution needs a GPU. The
   acceptance scripts (§4.3) detect this and exit `4` (INCONCLUSIVE — "a skip is not a pass").
2. **GCP GPU quota walled.** `gcp_bench.py` STOPs at the quota gate (§4.1); only `--dry-run`
   runs here.
3. **The 7B target is additionally lever-gated.** Even *with* a GPU, fak-CUDA cannot run
   Qwen2.5-7B-Q4_K_M end-to-end until **#485** (native Q4_K device matmul) and the **#489**
   K-quant load path land — both **not started** / **partial** (§3.1). A 7B f32 blob (~28 GB)
   does not fit, and there is no int4 device GEMM yet (`GPU.md` §3, §4).

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
- **Lever evidence:** commits `1b9f68a` (#482), `30438ad` (#479), `13ec795` (#489);
  `internal/compute/cuda.go`, `internal/compute/cuda_kernels.cu`, `internal/ggufload/`,
  `cmd/modelbench/main.go`; acceptance gates `tools/run_479_acceptance_on_gpu.sh`,
  `tools/run_482_acceptance_on_gpu.sh`.
- **Residual:** `tools/gcp_bench.py` (quota STOP at `:542-543`); per-host commands §4.2.
- Written 2026-06-21 on a host with no GPU; every numeric parity cell is `pending GPU run`
  by design.
