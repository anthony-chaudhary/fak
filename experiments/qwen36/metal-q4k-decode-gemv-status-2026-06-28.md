# Metal q4_k decode GEMV — closure status (#68)

**Status: SHIPPED + on-device-parity-witnessed (per the #68 thread); throughput tracked
under #67 — closeable as implemented.** This note binds
[#68](https://github.com/anthony-chaudhary/fak/issues/68) (*per-token decode matmul on the
GPU (q4_k); a resident-weight Metal GEMV plus the command-buffer fix should approach the
llama.cpp Metal decode*) to the tree as it actually stands. The decode-GEMV deliverable the
issue asks for **is implemented and wired on `main`** — the per-token q4_k GEMV kernel, its
Go dispatch, and the grouped/fused decode seams all landed under the v0.30.0 release
baseline, the resident-decode-forward epic [#67](https://github.com/anthony-chaudhary/fak/issues/67),
and the grouped/fused follow-up `62462799`, and were never cited back to #68, so the issue
sat open with its work already done. **The issue owner has since confirmed it on-device** in
the #68 thread (M3 Pro, `-tags fakmetal`): `TestMetalQ4KGemvMatchesCPU` cosine **1.000000**
(maxRel ≤ 1.2e-6 at [256,256] / [512,1024] / [5120,5120]) and `TestMetalQ4KDecodeMatchesCPU`
(GPU greedy decode == CPU `[433 92 166 106]`) — and ruled that **a per-token GEMV in
isolation does not move decode throughput** (each GEMV is one command buffer at ~360 µs fixed
overhead, so the win is the one-command-buffer resident forward), so the throughput residual
is **#67's**, not #68's. This note was authored on a `windows/amd64`, `CGO_ENABLED=0` box
where the `-tags fakmetal` kernel cannot compile or be measured; the on-device numbers above
are the **owner's reported witness from the #68 thread**, recorded here with attribution —
not re-run on this host (§5). The host-independent CPU-parity gate that pins the dispatch the
GPU kernel mirrors **was** re-run here, green (§3).

Sibling evidence in this cluster:
[`metal-q4k-device-gemm-status-2026-06-28.md`](metal-q4k-device-gemm-status-2026-06-28.md)
(#70, the **prefill GEMM** twin — the batched matmul, *not* the per-token decode GEMV this
issue is about),
[`metal-resident-weights-status-2026-06-28.md`](metal-resident-weights-status-2026-06-28.md)
(#69, the resident weights the decode GEMV reads), and
[`QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
(the `7.29 / 6.12 tok/s` decode llama.cpp-Metal bar this lane closes against).

---

## 1 — The issue's premise vs the tree today

The issue's "Step" is one sentence: *"Per-token decode matmul on the GPU (q4_k)."* Decode is
bandwidth-bound (~16.5 GB/token q4_k_m at ~100 GB/s CPU ≈ ~6 tok/s ceiling; the fak CPU
decode is ~0.9 tok/s — far below even the CPU ceiling — and llama.cpp-Metal hits 7.29 / 6.12).
That per-token GPU matmul **is shipped**: the `q4k_gemv` Metal kernel
(`internal/metalgemm/q4k.m:94`) is one thread per output row, each thread reconstructing its
weight row's f32 values from the resident q4_k super-blocks and dotting against the f32
activation (one `simd_sum` reduction across the 32-lane threadgroup). It is the on-device twin
of the CPU `q4kMatRowsRange` inner loop.

The issue conflates two things that have *different* statuses on `main`, and naming the
difference is what keeps the #68 scope honest:

1. **"A resident-weight Metal GEMV."** — **DONE.** The per-token decode GEMV runs on the GPU
   against resident raw q4_k super-blocks (§2). The weights are uploaded once and cached per
   `*Model` (the #69 residency deliverable), so per-token the only host↔device traffic is the
   activation `x` and the result `y` — never the weights.
2. **"…plus the command-buffer fix should approach the llama.cpp Metal decode."** —
   **#67's job, not #68's.** The *kernel* throughput is there (the fused decode MLP sustains
   ~76.9 GB/s on the M3 Pro — bandwidth-bound, exactly as the issue predicted), but a
   per-token decode GEMV in isolation is *command-buffer-overhead-bound*: the owner measured
   `BenchmarkMetalQ4KGemv` at 457 µs/op = 32 GB/s (~21% of peak BW), each GEMV a separate
   command buffer at ~360 µs fixed overhead, so a clean 27B decode is only ~1.2 tok/s. The
   end-to-end throughput win comes from the one-command-buffer-per-token resident forward
   (#67), which removes the CPU↔GPU serialization each per-matmul `waitUntilCompleted`
   incurs — the owner's #68-thread ruling is explicit: *"close this as implemented, and let
   #67 carry the speed."* So the throughput is the residual §5 reassigns to #67, **not** a
   #68 deliverable.

So #68 is **not "build a q4_k decode GEMV"** (done, and on-device-parity-confirmed by the
owner in-thread) — and its throughput residual is #67's, not a #68 gate. #68 is **closeable
as implemented**.

---

## 2 — What is implemented (the decode-GEMV closure)

**The per-token decode GEMV kernel.** `q4k_gemv` (`q4k.m:94`) is one thread per output row:
thread `o` walks its row's `nblk` super-blocks (one `q4k_block_dot` per 256-weight block,
striped across the 32 SIMD lanes), reduces with `simd_sum`, and writes `Y[o]`. This is the
decode GEMV — **distinct from `q4k_gemm`** (`q4k.m:145`, the register-blocked tiled *batched
prefill* GEMM that is #70's deliverable). The two share the dequant primitive and the
resident weight table, not the grid: decode is `[out]` threads over one activation; prefill
is `[out,P]` threads over a panel.

**The Go dispatch.** `Q4KWeight.GEMV` (`internal/metalgemm/q4k.go:53`) is the decode GEMV —
its doc comment reads *"computes y[Out] = W · x for one f32 activation row x … This is the
decode GEMV"* — and calls `C.mg_q4k_gemv`. The routing twin of the prefill dispatch is
`(*Session).q4kMatRowsDispatch` (`internal/model/metal_q4k_on.go:51`): under `s.MetalQ4K`
with a device present it uploads the weight once via `metalQ4KWeight` → `UploadQ4K` and runs
`w.GEMV(xf, y)`; otherwise it falls back to the proven CPU `q4kMatRows`. Routing **both**
decode and prefill q4_k matmuls to the GPU is what lets `metalQ4KWeight` free the CPU copy
(single residency) — the comment at `metal_q4k_on.go:47-50` says so explicitly.

**The amortized decode seams (commit `62462799`).** Per-token decode fires many GEMVs that
share one activation (q/k/v, gate/up, the GDN in-proj quad, the MLP down). The follow-up
commit collapsed those into fewer command buffers, each still a q4_k decode GEMV at heart:

- `metalgemm.GEMVGroup` (`q4k.go:76`) / `mg_q4k_gemv_group` (`q4k.m:442`) — N decode GEMVs
  sharing one activation in **one** command buffer (pipelined dispatches).
- `metalgemm.FusedMLP` (`q4k.go:111`) / `mg_q4k_mlp` (`q4k.m:281`) + the `q4k_swiglu` MSL
  kernel (`q4k.m:214`) — a whole dense SwiGLU MLP for one decode token in **one** command
  buffer, the intermediate-wide buffer resident on-GPU.
- `(*Session).q4kGroupDispatch` (`metal_q4k_on.go:71`) and `q4kFusedMLP`
  (`metal_q4k_on.go:120`) wire them into the decode forward; the Q8/Q6_K minority falls back
  per-call. Results are bit-identical to the per-matmul path up to GPU float-order.

**Resident across steps.** The #67 epic moved the whole decode loop onto resident state, so
decode pays no per-step upload of weights **or** KV — the GEMV reads a handle uploaded once
and never re-touched (full commit list in the #69 closure note §2).

Landing commits (none bound to #68 — the gap this note closes):

- `1029e37c` `feat: fak — the Fused Agent Kernel (public release v0.30.0)` — the per-token
  `mg_q4k_gemv`, `Q4KWeight.GEMV`, and `q4kMatRowsDispatch` are all present in this baseline,
  so the core decode GEMV predates the granular public history.
- `62462799` `perf(metal): fuse the q4_k decode MLP + add the grouped-GEMV decode seam`
- the `#67` resident-decode-forward epic (GPU-resident Q8 decode, resident decode KV, flash
  decode attention, on-GPU final norm + LM head, …).

---

## 3 — Correctness witness (what I ran here)

The host-independent half is the CPU-side dispatch contract — the same resident super-block
bytes the GPU kernel consumes, dequanted the same way — and it is pinned by pure-Go parity
tests that **do** run on this box (the `-tags fakmetal` GPU tests are excluded from a
`windows/amd64` build, so only the CPU half compiles here):

```
go test ./internal/model  -run Q4K -count=1   →  ok  (1.076s)
go test ./internal/metalgemm        -count=1   →  ok  (0.169s)
```

Covering:

- `TestQ4KMatRowsMatchesF32` — the resident-Q4_K f32 GEMV (`q4kMatRowsRange`, the scalar
  reference the GPU kernel mirrors) is deterministic and matches the reference reduction.
- `TestQ4KGemmMatchesMatRows` / `TestQ4KGemmInt8MatchesMatRowsInt8` — the batched GEMM is
  **bit-identical** per `(output row, token)` to the per-token decode GEMV, on both the f32
  and int8-SDOT reductions. A dispatch/packing bug blows these up by orders of magnitude.
- the cpu-ref HAL parity (`q4k_hal_test.go`) — the backend's Q4_K MatMul reproduces
  `q4kMatRowsRange` bit-for-bit, the floor the GPU kernel sits above.

The **on-device** GPU half is pinned by the `-tags fakmetal` tests
(`TestMetalQ4KGemvMatchesCPU` holds the kernel to cosine ≥ 0.9999 / maxRel ≤ 5e-3 vs the CPU
f32 reference; `TestMetalQ4KDecodeMatchesCPU` runs a greedy decode with `MetalQ4K` and
asserts the **same tokens** as the CPU path, including after the single-residency CPU-copy
free). This host cannot run them, but **the owner has now reported them green on an M3 Pro
in the #68 thread**: `TestMetalQ4KGemvMatchesCPU` cosine **1.000000** (maxRel ≤ 1.2e-6 at
[256,256] / [512,1024] / [5120,5120]) — *tighter* than the test's own 5e-3 bound — and
`TestMetalQ4KDecodeMatchesCPU` greedy decode == CPU `[433 92 166 106]`. Recorded with
attribution (owner, in-thread); not re-run on this `windows/amd64` host.

---

## 4 — Why this is #68's deliverable

The issue's "Step" — *per-token decode matmul on the GPU (q4_k)* — is exactly satisfied:
`q4k_gemv` is the per-token decode GEMV, dispatched by `q4kMatRowsDispatch`, reading resident
raw q4_k weights. The bandwidth-bound prediction is **confirmed**: the fused decode MLP
sustains ~76.9 GB/s on the M3 Pro (`BenchmarkMetalQ4KFusedMLP`, cited in `62462799`), i.e.
the kernel is at the bandwidth wall the issue's ceiling math assumed, not compute-bound like
the CPU int8-SDOT path (~23 GB/s, ~1.4 tok/s decode ceiling). The one thing this kernel
**cannot** do alone is move *end-to-end* decode tok/s — the gap between "the GEMV is
bandwidth-bound" and "the whole decode forward approaches llama.cpp" is CPU↔GPU
serialization, which the owner reassigned to #67 (§5), not a #68 residual.

---

## 5 — Closure state (what is #68's, and what is #67's)

The two things this note previously named as "remaining" have resolved into one that is
**done for #68** and one that is **not #68's gate**:

1. **On-device parity — DONE (owner, in the #68 thread).** The gate
   `go test ./internal/model -tags fakmetal -run 'MetalQ4K(Gemv|Decode)' -count=1` was run by
   the issue owner on an M3 Pro: `TestMetalQ4KGemvMatchesCPU` cosine **1.000000**,
   `TestMetalQ4KDecodeMatchesCPU` greedy decode == CPU `[433 92 166 106]`. This is the
   decode-GEMV correctness witness #68 asks for, and it is green. (Recorded with attribution;
   not re-run on this `windows/amd64` host, which cannot build `-tags fakmetal`. A Mac worker
   can re-confirm with the one-liner above, but the witness already exists in-thread.)
2. **End-to-end decode tok/s — #67's deliverable, not #68's.** A per-token decode GEMV in
   isolation cannot move throughput: the owner measured it command-buffer-overhead-bound
   (`BenchmarkMetalQ4KGemv` 457 µs/op = 32 GB/s, ~21% BW; ~360 µs fixed per-buffer overhead;
   clean 27B decode ~1.2 tok/s). The wall is CPU↔GPU serialization — each per-matmul
   `waitUntilCompleted` idles the GPU while the CPU runs the inter-matmul norms / attention /
   GDN recurrent — and the fix is #67's one-command-buffer-per-token resident forward, of
   which this GEMV is the kernel. Capturing the `7.29 / 6.12 tok/s` llama.cpp-Metal decode
   bars (`FAK_QPROFILE=1`, Qwen3.6-27B q4_k_m, no co-resident llama-server) is therefore the
   **#67** gate, tracked there — full data in #59.

**Verdict:** #68 is **closeable as implemented** — the per-token GPU matmul on resident q4_k
weights is shipped, CPU-parity-pinned here (§3), and on-device-parity-witnessed in-thread by
the owner — exactly the owner's #68-thread ruling: *"close this as implemented, and let #67
carry the speed."* The *approach-llama.cpp decode throughput* is **not** a #68 claim; it is
#67's, and stays `not yet` there until #67's resident forward records it.

---

## 6 — Out of scope for #68

The **batched prefill GEMM** is #70 (a different kernel grid, `q4k_gemm`). **Resident weights
/ the literal zero-copy upload** is #69. **Lifting `requirePreNorm` for the hybrid arch** is
#71. The **CPU int8-SDOT decode GEMV** (`e0823ab1`, the amd64 AVX2 reducer) is the CPU-side
lever under #209, not the Metal one. And the MPS `-tags metal` registry backend's quantized
device GEMV is still genuinely open (`internal/compute/metal.go` panics on non-F32) — a
different lane from the `-tags fakmetal` `internal/metalgemm` lane closed here. None of those
are part of #68.

---

## 7 — Why this is the right increment

This is the host-independent slice: it binds #68 to the per-token decode GEMV that already
shipped under the v0.30.0 baseline + the #67 resident-decode-forward epic + the `62462799`
grouped/fused seams, corrects the issue body's conflation (the GEMV is done; the throughput
residual is the end-to-end forward, which is **#67's**, not the kernel), folds in the owner's
on-device parity witness from the #68 thread (recorded with attribution, not re-run here), and
runs the one gate this box can run (the pure-Go CPU-parity tests that pin the dispatch the GPU
kernel mirrors — re-confirmed green: `TestQ4KMatRowsMatchesF32`, `TestQ4KGemmMatchesMatRows`,
`TestQ4KGemmInt8MatchesMatRowsInt8`, `TestSessionQ4KKernelMixedDispatch`, `internal/metalgemm`).
It is **not** a fresh on-device measurement on this host — the M3 Pro numbers are the owner's,
attributed — and it does **not** claim the *approach-llama.cpp decode throughput*, which is
#67's gate and stays `not yet` there. No fabricated pass: the gates run for *this* change are
the pure-Go parity tests above (this host) plus the doc/commit gate on the `experiments` lane;
the on-device parity is the owner's in-thread witness.
