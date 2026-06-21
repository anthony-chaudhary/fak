# M3-LLAMACPP-RESULTS — fak on Apple Silicon: the NEON Q8 lane, the best model it runs, and the measured gap to llama.cpp

> **Honest verdict up front.** Before this work fak had **no SIMD Q8 kernel on arm64** — the
> int8 dot was amd64-only (AVX2/AVX-512), so on Apple Silicon it fell back to scalar Go and
> Q8_0 quantization was a *net loss*: int8 decode was **1.24× slower than f32** (17.7 vs 14.0
> ms/tok) despite streaming 3.5× fewer weight bytes. This session added the missing NEON lane
> (SDOT decode kernel + NEON prefill GEMM) and the pure-Go HuggingFace loaders, which together
> (a) flip int8 to **1.9× faster than f32** and prefill **5–6× faster**, and (b) let fak run a
> real model — **Qwen2.5-1.5B-Instruct**, 13× the params of the old SmolLM2-135M — on this box.
> The result is that fak now lands in the **same order of magnitude** as llama.cpp's CPU path
> (decode ~2.2× behind, prefill ~6.5× behind) instead of not being in the race. It is **not at
> parity**: llama.cpp's hand-tuned ggml kernels, Accelerate BLAS, and Metal backend remain
> ahead, exactly as `LLAMACPP-HEADTOHEAD-RESULTS.md` found on Zen5 — the residual is a
> hand-tuned-assembly / GEMM-tiling / GPU boundary, not an architecture one.
>
> Box: Apple **M3 Pro** (6P+6E, 36 GB unified, ~150 GB/s). fak = pure-Go in-kernel forward
> pass + the new arm64 NEON SDOT kernel (`internal/model/quant_arm64.{go,s}`), no cgo, no GPU.
> llama.cpp = Homebrew build **8200** (`541bf3762`), the same machine. Apples-to-apples Q8_0
> on both sides. Native `go test`/`go run` (this is macOS, not the WDAC-blocked Windows host).

## 1. The kernel flip (what the NEON lane bought) — SmolLM2-135M, the bit-identity fixture

The arm64 NEON kernel is **bit-identical** to the scalar reference (`TestQdot8NEONMatchesScalar`;
the per-block int32 SDOT sum is order-independent, the float combine matches gc's arm64 FMA
fusion via a single `FMADDS`). So this is pure speed, same numbers — argmax-exact vs the HF
oracle stays 25/25.

| axis (SmolLM2-135M, Q8_0) | before (scalar arm64) | after (NEON arm64) | delta |
|---|---:|---:|---:|
| int8 **decode** | 17.68 ms/tok | **7.30 ms/tok** | **2.42× faster** |
| int8 decode vs **f32** decode (14.0 ms/tok) | 1.24× *slower* | **1.91× faster** | the flip |
| int8 **prefill** (P=256) | 126 tok/s | **631 tok/s** | **5.0× faster** |

Reproduce: `go run ./cmd/q8bench` (prints the argmax-exact gate + decode/prefill A/B).

## 2. The best model fak runs here — Qwen2.5-1.5B-Instruct, loaded in pure Go

fak could previously only run the one checkpoint `export_oracle.py` (torch) had baked into its
custom format. Two pure-Go loaders (no torch) removed that ceiling — the forward pass is already
generic Llama/Qwen2 (GQA, RoPE θ, SwiGLU, tied embeddings, Qwen2 qkv-bias):

- `modelbench -hf <snapshot>` — load a HuggingFace `config.json` + `model.safetensors` directly
  (bf16→f32 in Go).
- `modelbench -lean` / `model.LoadSafetensorsQuant` — **quantize the big matmul weights at load
  and drop their f32**, so resident cost falls from f32's ~5.1 B/param to ~1.1. This is the
  lever for the *best possible* model on 36 GB: f32-resident load tops out ~3B; the lean load
  fits ~7B. Pinned bit-identical to a regular load+quantize (`TestLoadSafetensorsQuantMatchesRegular`).
  This is the same model→engine portability the HAL seam predicts
  (`../docs/explainers/hardware-portability.md`): the arm64 NEON Q8 kernel is the seam's
  assumption-#3 (x86-only dispatch) closed on a real second ISA, and the lean loader is its
  residency lever — both arrive as added backends/loaders, not a fork of the proven f32/Q8 loops.

| Qwen2.5-1.5B load (Apple M3 Pro) | peak RSS | load time | decode |
|---|---:|---:|---:|
| regular `-hf` (f32-resident) | 18.8 GB | 14.3 s | 28.9 tok/s |
| `-lean` (quantize-at-load, f32 dropped) | **11.1 GB** | **4.9 s** | 28.9 tok/s (same Q8 weights) |

## 3. Head-to-head — Qwen2.5-1.5B-Instruct, Q8_0, same machine, 6 threads

| axis | **fak** (pure-Go CPU, NEON Q8) | llama.cpp **CPU** (`-ngl 0`) | llama.cpp **Metal** (`-ngl 99`) |
|---|---:|---:|---:|
| **decode** (tg64) | 28.9 tok/s (34.6 ms/tok) | 63.2 ± 0.5 tok/s | 67.5 ± 0.9 tok/s |
| **prefill** (pp256) | 55.5 tok/s | 363.2 ± 4.0 tok/s | 1746.8 ± 2.0 tok/s |
| fak ÷ llama.cpp CPU | — | decode **0.46×** · prefill **0.15×** | — |

Two facts the numbers make plain:

- **Decode is memory-bandwidth-bound on this SoC** — llama.cpp's own Metal decode (67.5) is barely
  above its CPU decode (63.2), because GPU and CPU share the same ~150 GB/s unified memory. So
  the decode gap is not "fak lacks a GPU"; it is that llama.cpp's ggml Q8_0 NEON kernel extracts
  ~2× more effective bandwidth per core than fak's (fak ≈ 48 GB/s, llama.cpp ≈ 90 GB/s on the
  same 1.1 B/weight Q8_0 byte stream). Same bytes, tighter kernel.
- **Prefill is compute-bound** — Metal's 1747 tok/s (≈31× fak) is the GPU's FLOP advantage, and
  even llama.cpp's *CPU* prefill (363) beats fak's per-cell NEON GEMM (55) because it uses a
  register-blocked GEMM / Accelerate BLAS with weight reuse, where fak still issues one NEON
  GEMV per output cell.

## 4. Why the residual gap (and what would close it)

- **Decode (~2.2×):** the hand-tuned-kernel boundary. fak's `qdot8asm` does, per 32-wide block,
  2× SDOT + a `VADDV`/`VMOV`/`SCVTF`/`FMADD` reduce; ggml amortizes the float reduction across
  blocks and uses the i8mm `SMMLA` path on i8mm-capable parts (M3 has it). Matching that forfeits
  the current bit-identity-to-scalar property, so it was left as a deliberate, documented choice
  (fak's trust posture values the bit-exact kernel). Activation-quant is NEON-able too but is a
  smaller term than the dot.
- **Prefill (~6.5× vs CPU, ~31× vs Metal):** fak has **no register-blocked Q8 GEMM tile on
  arm64** (the `qgemm8tile` tile kernel is amd64 asm). The per-cell NEON dot already bought 5–6×;
  a real NEON tile (weight-block × token-block register blocking, the arm64 twin of the AVX-512
  tile) is the next prefill lever, and it is the bigger one. Metal-class prefill is out of reach
  for a pure-CPU runner by construction.

## 5. What shipped this session (all on `darwin/arm64`, full `go test ./...` green)

| commit | change |
|---|---|
| `feat(model): arm64 NEON Q8_0 kernel` | `qdot8asm` (SDOT, FEAT_DotProd; auxv detect on linux), bit-identical to scalar; decode 2.4× |
| `feat(model): pure-Go HF loaders` | `-hf` safetensors load + `LoadSafetensorsQuant` lean load (fits a 7B on 36 GB) |
| `feat(model): arm64 NEON Q8_0 prefill GEMM` | route the batched GEMM through `qdot8asm`; prefill 5–6× |

## 6. Reproduce

```sh
# kernel flip + argmax-exact gate (SmolLM2 fixture)
go run ./cmd/q8bench

# the best model, lean-loaded, NEON Q8 (Qwen2.5-1.5B-Instruct HF snapshot)
FAK_WORKERS=6 go run ./cmd/modelbench -hf <snapshot> -lean -decode-reps 6 -prefill-reps 3

# llama.cpp side, same GGUF, same machine
llama-bench -m qwen2.5-1.5b-instruct-q8_0.gguf -ngl 0  -t 6 -p 256 -n 64   # CPU
llama-bench -m qwen2.5-1.5b-instruct-q8_0.gguf -ngl 99 -t 6 -p 256 -n 64   # Metal

# the bit-identity / lean-equivalence gates
go test ./internal/model/ -run 'TestQdot8NEONMatchesScalar|TestLoadSafetensorsQuantMatchesRegular' -v
```

## Bottom line

The "needed updates to fak" for this M3 were the missing **arm64 NEON Q8 lane** (decode + prefill)
and the **pure-Go HF loaders**. With them, fak goes from *quantization-is-a-loss* to running
**Qwen2.5-1.5B at ~29 tok/s decode / 55 tok/s prefill in pure Go on CPU**, within ~2.2× of
llama.cpp's CPU decode on the identical Q8_0 weights. Full parity remains a register-blocked-GEMM
+ hand-tuned-kernel + Metal effort — the same honest boundary the Zen5 head-to-head reached — but
the architecture gap (no NEON at all) is closed. For where this sits in the product story —
local model + kernel matching the frontier on safety + cost while capability climbs the on-box
size ladder — see `../docs/explainers/local-vs-frontier-parity.md`: the honest read there is that
fak is the in-kernel *reference* runner and `llama-server` remains the speed-tuned serving engine
for the 7-9B ramp.
