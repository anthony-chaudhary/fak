# The int8 precision rung — independent verification (beat HF on int8, argmax-exact)

> **Goal:** descend one rung of the precision ladder (f32 → 8-bit) and **beat HuggingFace on
> the int8 rung** — measured *same-precision* (HF int8, not just HF f32), and **correct**
> (argmax-exact vs the HF oracle), not merely fast.
>
> **Result (native, this 32-core box, no GPU): MET.** fak int8 decodes at **7.68 ms/tok**,
> beating the same-rung **HF int8 peer (22.78 ms/tok) by 2.97×** and HF f32 (28.90) by 3.76×,
> while staying **argmax-EXACT vs the HF-authored oracle (25/25 positions)**. It is now
> near-parity with llama.cpp Q8_0 (6.9 ms/tok) — the int8 rung closed the decode gap to the
> SOTA CPU int8 engine from f32's ~2.9× down to **~1.1×**, in one pure-Go binary.
>
> This document is the **independent verification lane**: the int8 forward kernels
> (`internal/model/quant.go`, `quant_forward.go`, `quant_amd64.{go,s}`, `quant_noasm.go`) and
> their bit-identity tests are authored by the implementation lane; this report (a) supplies
> the **same-rung HF int8 peer** the comparison needs and (b) re-checks the win with a
> **witness that did not author the kernels** — the fleet's distrust discipline applied to
> the speedup claim. Tools: `internal/model/bench_hf_quant.py` (HF int8 peer),
> `cmd/q8bench` (correctness + speed verifier, public-API-only), `cmd/q8kernel` (standalone
> kernel microbench). Data: `hf-int8.json`, `fak-int8.json` (this dir).

## The int8 rung, head-to-head (decode = the agent-loop regime)

Decode = batch-1 autoregressive. Prefill-256 = full 256-token ingestion, last-token logits
only on every engine (apples-to-apples). All rows are the same SmolLM2-135M, same
deterministic LCG token-ids, same box.

| engine | precision | threads | decode ms/tok | prefill-256 ms | correctness |
|---|---|---:|---:|---:|---|
| **fak int8 (Q8_0, SIMD)** | int8 | all | **7.68** | **146** | **argmax-exact (25/25)** |
| fak f32 (parity lane) | f32 | all | 18.06 | 677 | argmax-exact |
| **HF transformers (dynamic int8)** | int8 | 1 | **22.78** | 153 | — |
| HF transformers (dynamic int8) | int8 | 32 | 30.6 | 80 | — |
| HF transformers (eager) | f32 | 1 | 28.90 | 515 | reference oracle |
| llama.cpp | Q8_0 | 1 | 6.91 | 75 | — |
| llama.cpp | Q8_0 | 32 | 6.05 | 83 | — |

(fak/HF-int8 decode = interleaved-min on a box shared with other fleet sessions; see
Methodology. fak f32 here = 18.1 ms/tok, consistent with the parity lane. The fak int8
prefill **146 ms** is the Act-4 **register-blocked tile GEMM**, re-measured 2026-06-17 on an
idle box and folded into `comparison.json`; the original as-found legacy Q8 prefill measured
269 ms in this same table under fleet contention — decode is unchanged by Act 4.)

- **Decode — beat HF on the int8 rung, decisively.** fak int8 **7.68 ms/tok < HF int8
  22.78** (2.97× faster) and < HF f32 28.90 (3.76×). So when *both* engines quantize to
  int8, fak still wins decode by ~3× — and it wins even though HF's int8 uses torch's
  fbgemm/oneDNN int8 GEMM. fak int8 is **near-parity with llama.cpp Q8_0** (7.68 vs 6.91 →
  1.11×), up from f32's 2.87× behind: descending the precision rung closed ~80% of the
  remaining decode gap to the SOTA CPU int8 engine, in pure Go (Go assembly, one static
  binary — no cgo, no FFI, the in-kernel thesis intact).
- **Prefill — int8 + the register-blocked tile GEMM (Act 4) cut fak's prefill 4.6× (677 →
  146 ms; the as-found legacy Q8 prefill measured 269 ms here under fleet contention)** but it
  still trails HF int8 (80–153) and llama.cpp (75): pure-Go batched GEMM vs MKL/GGML AVX-512 —
  the same labeled boundary the f32 lane named, now ~1.8× behind llama.cpp Q8 (not ~3.6×).
  Decode is the regime an agent loop actually runs in, and that is the rung fak wins.

## Correctness — the gate, not an afterthought

Quantization is lossy by construction, so the int8 path is **not** bit-identical to f32. It
is held to the **same bar the f32 oracle test enforces**: for every HF-authored oracle
prompt, the int8 KV-session's **per-position argmax must equal the oracle's `argmax_per_pos`**.

```
[int8 correctness] prompt 0: argmax 5/5    greedy 4/12   last|Δ|=1.21 argmaxOK=true
[int8 correctness] prompt 1: argmax 11/11  greedy 12/12  last|Δ|=0.68 argmaxOK=true
[int8 correctness] prompt 2: argmax 9/9    greedy 4/12   last|Δ|=0.81 argmaxOK=true
[int8 correctness] TOTAL argmax 25/25 -> ARGMAX-EXACT vs HF oracle
```

25/25 argmax-exact; last-position logit max|Δ| vs HF ≈ 0.7–1.2 (small). Greedy continuation
agreement is strong (prompt 1: 12/12) and partial on the others — expected, since greedy is
chaotic and a tiny logit perturbation eventually flips a downstream token; the per-position
argmax gate is the deterministic, localized correctness witness, and it is exact. The
implementation lane's own `go test ./internal/model/` (incl. the scalar/AVX2/AVX-512
bit-identity test) passes green on an independent run (27.2 s).

## Why the win needed SIMD (the kernel microbench)

A standalone kernel microbench (`cmd/q8kernel`) on the memory-bound head matmul [49152×576],
pure-Go, all cores, isolated the kernel question:

```
f32                      2.04 ms   (55.6 weight-GB/s)   1.00×
int8×int8  (scalar Go)   2.00 ms   (15.9 weight-GB/s)   0.98×   ← no win
```

The scalar int8 dot is **no faster than f32**: Go emits per-byte sign-extend + scalar `IMUL`,
so the 4×-fewer-weight-bytes advantage is spent on slower per-element integer arithmetic —
decode stays compute-bound, not bandwidth-bound. The byte savings only become speed once the
dot itself goes vectorized: the implementation lane's `qdot8asm`/`qdot8asm512`
(`quant_amd64.s`, AVX2/AVX-512, dispatched by CPUID) is what turns the format win into the
2.4× end-to-end decode speedup measured above. This is the same lever that makes llama.cpp's
Q8_0 fast — and it is why fak now lands next to it.

## Methodology / honesty

- **Same inputs, same box.** SmolLM2-135M; deterministic LCG token-ids reproduced bit-for-bit
  across `cmd/modelbench`, `bench_hf.py`, and `bench_hf_quant.py`; token values don't affect
  matmul/attention cost, so synthetic ids measure the identical work.
- **HF int8 peer = `torch.ao.quantization.quantize_dynamic(Linear → qint8)`** — weight-only
  int8 Linears, activations dynamically quantized, run on CPU via fbgemm/oneDNN. The
  embedding lookup stays f32 (nn.Embedding is not a Linear), exactly as fak keeps the
  embedding f32 and quantizes only its use as the head. This is the closest HF analogue to
  fak's Q8_0 and to llama.cpp's Q8_0, making "beat HF on the int8 rung" a *same-precision*
  claim, not f32-vs-int8.
- **Contention-robust timing.** This box runs several fleet sessions concurrently, so absolute
  latencies are load-sensitive (the f32 lane noted the same). `cmd/q8bench` measures f32 and
  int8 **interleaved per rep** (both sample the same load) and reports the **min over 25 reps**
  (least-contended estimate); min and median agree tightly (int8 7.68 / 8.98). The 2.97× win
  over HF int8 is far outside any plausible noise.
- **Native vs WSL.** fak numbers here are native Windows (WDAC permitting); a WSL cross-check
  read ~12% slower for fak (f32 22.3 WSL vs 18.1 native) — so a WSL-measured fak int8 (11.75)
  would *understate* the win. Either way fak int8 beats HF int8 by ~2–3×.

## Coordination note (multi-session)

This was produced concurrently with the implementation lane (a peer session authored the int8
kernels + their tests). To avoid colliding on actively-built code, this lane is **strictly
additive**: it adds only `bench_hf_quant.py`, `cmd/q8bench`, `cmd/q8kernel`, the two JSON data
files, and this doc; it edits **no** existing file and **no** `.go` in `internal/model`.
`fak-int8.json` + `hf-int8.json` are ready to fold into `compare.py`/`comparison.json`
(owned by the implementation lane). Nothing here is committed — the `fak/` tree has live
in-flight work from several sessions; the implementation lane should commit the coherent
increment.
