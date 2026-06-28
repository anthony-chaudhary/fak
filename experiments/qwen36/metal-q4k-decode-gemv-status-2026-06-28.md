# Metal q4_k decode GEMV — closure status (#68)

**Status: SHIPPED (code) / on-device measurement gated.** This note binds
[#68](https://github.com/anthony-chaudhary/fak/issues/68) (*per-token decode matmul on the
GPU (q4_k); a resident-weight Metal GEMV plus the command-buffer fix should approach the
llama.cpp Metal decode*) to the tree as it actually stands. The decode-GEMV deliverable the
issue asks for **is implemented and wired on `main`** — the per-token q4_k GEMV kernel, its
Go dispatch, and the grouped/fused decode seams all landed under the v0.30.0 release
baseline, the resident-decode-forward epic [#67](https://github.com/anthony-chaudhary/fak/issues/67),
and the grouped/fused follow-up `62462799`, and were never cited back to #68, so the issue
sat open with its work already done. It was authored on a `windows/amd64`,
`CGO_ENABLED=0` box where the `-tags fakmetal` kernel cannot compile, run, or be measured,
so the one genuinely-gated step — the on-device end-to-end decode tok/s on an M3 Pro — is
named as the remaining work in §5, not faked green.

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
   **PARTIALLY.** The *kernel* throughput is there (the fused decode MLP sustains ~76.9 GB/s
   on the M3 Pro — bandwidth-bound, exactly as the issue predicted), but the end-to-end
   decode tok/s is still gated on the one-command-buffer-per-token resident forward (#67),
   which removes the CPU↔GPU serialization each per-matmul `waitUntilCompleted` incurs. That
   is the residual §5 names, not the GEMV itself.

So #68 is **not "build a q4_k decode GEMV"** (done) — the remaining work is the on-device
end-to-end measurement of the path the GEMV is already the kernel of, on a Mac.

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
free) — but those are the witnesses a Mac worker re-confirms, **not** ones this host can run.

---

## 4 — Why this is #68's deliverable

The issue's "Step" — *per-token decode matmul on the GPU (q4_k)* — is exactly satisfied:
`q4k_gemv` is the per-token decode GEMV, dispatched by `q4kMatRowsDispatch`, reading resident
raw q4_k weights. The bandwidth-bound prediction is **confirmed**: the fused decode MLP
sustains ~76.9 GB/s on the M3 Pro (`BenchmarkMetalQ4KFusedMLP`, cited in `62462799`), i.e.
the kernel is at the bandwidth wall the issue's ceiling math assumed, not compute-bound like
the CPU int8-SDOT path (~23 GB/s, ~1.4 tok/s decode ceiling). What is not yet closed is the
*end-to-end* decode tok/s — the gap between "the GEMV is bandwidth-bound" and "the whole
decode forward approaches llama.cpp" — which §5 scopes.

---

## 5 — The remaining gate (the honest "not yet")

Two things remain, both requiring an Apple-Silicon M3 Pro with the macOS Metal toolchain —
the capability this `windows/amd64` host does not have:

1. **End-to-end decode tok/s on the full resident forward.** The `62462799` diagnosis is
   explicit: the decode wall is **not** kernel throughput (the big MLP matmuls already
   sustain ~73–77 GB/s) but CPU↔GPU serialization — each per-matmul `waitUntilCompleted`
   leaves the GPU idle while the CPU runs the inter-matmul norms / attention / GDN recurrent.
   The fix is #67's one-command-buffer-per-token resident forward (keeps the GPU busy across
   the whole token), which the per-token GEMV is the kernel of. Once that is wired, capture
   with `FAK_QPROFILE=1` the decode of Qwen3.6-27B q4_k_m, against the `7.29 / 6.12 tok/s`
   llama.cpp-Metal bars — **without a co-resident llama-server** (the 36 GiB
   swap-contamination rule from the parity-status §3). The box's ~30% run-to-run variance
   means the noise-free microbench is the honest signal until a stable end-to-end number is
   recorded.
2. **Re-confirm the on-device parity gates** on that Mac node:
   `go test ./internal/model -tags fakmetal -run 'MetalQ4K(Gemv|Decode)' -count=1` (confirms
   `TestMetalQ4KGemvMatchesCPU` / `TestMetalQ4KDecodeMatchesCPU` green on the real device).

**Next checkable step:** run gate (2) on a Mac node, then capture (1). Until the on-device
end-to-end decode tok/s is recorded, the *approach-llama.cpp* claim is `not yet`; the **decode
GEMV** deliverable (per-token GPU matmul on resident q4_k weights, bandwidth-bound as
predicted) is shipped and CPU-parity-pinned.

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
grouped/fused seams, corrects the issue body's conflation (the GEMV is done; the residual is
the end-to-end forward, not the kernel), and runs the one gate this box can run (the pure-Go
CPU-parity tests that pin the dispatch the GPU kernel mirrors). It is **not** a claim that the
on-device decode tok/s is measured — that is the gated step in §5, and the *approach-llama.cpp*
speedup stays `not yet` until a Mac worker records it. No fabricated pass: the only gates run
for *this* change are the pure-Go parity tests above and the doc/commit gate on the
`experiments` lane.
