# Metal q4_k device GEMM for the full-attention prefill twin — implementation spec (#70)

**Status: SPEC / host-independent slice.** This document is the code-level recipe a
Mac-equipped worker executes to land [#70](https://github.com/anthony-chaudhary/fak/issues/70)
(*upload the q4_k-majority weights as raw blocks and run the device GEMM directly, instead of
`dequantQ8 → f32 → metalgemm.Upload` which doubles the upload bytes and discards the q4_k
compactness*). It was authored on a `windows/amd64`, `CGO_ENABLED=0` box, where the
`metal_prefill.go` kernel (`//go:build darwin && cgo && fakmetal`) **cannot be compiled, run,
or measured** — so the two genuinely-gated steps (the `fakmetal` build and the on-device M3 Pro
measurement) are named here as the remaining work in §6, not faked green. #70 stays **OPEN**
until a Mac worker lands and witnesses the kernel using this spec.

Sibling evidence in this cluster:
[`QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
(the `51.55 tok/s pp22` llama.cpp-Metal bar and the `FAK_QPROFILE` per-phase breakdown) and
[`metal-hybrid-prefill-spec-20260627.md`](metal-hybrid-prefill-spec-20260627.md) (the same
host-independent-slice pattern for the adjacent #71).

---

## 1 — The fence today (precise mechanism)

The issue body says "`metal_prefill.go` currently `dequantQ8 → f32 → metalgemm.Upload`,
doubling the upload bytes." Reading the tree, that is exactly true of **one** of the model's
two Metal q4_k surfaces, and getting the boundary right is what makes this spec executable:

- **The full-attention Metal prefill twin `prefillBatchedMetal`** (`internal/model/metal_prefill.go:180`,
  reached from `kv.go:534-537` when `s.Metal` is set, i.e. `FAK_METAL`) is the path the issue
  names. Its weight table `metalWeights()` (`metal_prefill.go:68`) reads the Q8_0 store
  `m.q8(name)`, calls `dequantQ8(qt)` (`metal_prefill.go:48`) to reconstruct the full `[out,in]`
  **f32** matrix, then `metalgemm.Upload(...)` (`internal/metalgemm/metalgemm.go:64`) stores an
  **f16** copy on-device; the `mm` closure (`metal_prefill.go:202-212`) runs the projection via
  `gw[name].MatMul`. So a q4_k weight is dequantized to f32, then re-quantized to f16 — the
  device holds **f16 (2 bytes/elt)** where the GGUF carried **q4_k (~0.5 byte/elt)**. On the
  27B that is the difference between a ~16 GB resident set and a ~54 GB one (`q4k.go:4-6`):
  f16 of a 27B does not fit the 36 GB unified pool at all, which is why the f16 twin is only
  ever exercised on small models and the 27B real-weight path can never use it.

- **The resident-Q4_K *hybrid* lane is already correct.** `prefillBatchedQ4K`
  (`prefill_q4k.go:51`, reached when `s.Q4K`/`s.MetalQ4K`) already dispatches each projection
  to **EITHER `q4kw` (raw Q4_K majority) or `q8w` (Q8 minority)**, and under `MetalQ4K` its
  q4_k matmuls run on the GPU via `q4kGemmDispatch` (`metal_q4k_on.go:34`) →
  `metalQ4KWeight` (`metal_q4k_on.go:146`) → `metalgemm.UploadQ4K` (raw GGUF super-blocks, no
  f32 round-trip) → `Q4KWeight.GEMM` (`metalgemm/q4k.go:127`). That lane is pinned green on a
  Mac by `TestMetalQ4KPrefillMatchesCPU` (`metal_q4k_test.go:170`).

**So the device q4_k GEMM and its model-layer wiring already exist and ship** (the kernel under
#1085 — `q4k.m` `mg_q4k_gemm` + `UploadQ4K`/`Q4KWeight.GEMM`; the dispatch under #67 —
`metal_q4k_on.go`). #70 is therefore **not "write a quantized device GEMM"** (done) — it is the
small wiring change of routing the **full-attention** twin `prefillBatchedMetal` through that
same already-proven `metalQ4KWeight`/`Q4KWeight.GEMM` path it already uses for the hybrid,
deleting the `dequantQ8 → f16` doubling.

---

## 2 — The change (`internal/model/metal_prefill.go`, `-tags fakmetal`)

One substitution, applied at the two seams the f16 path owns: the weight table and the `mm`
closure. The model must be loaded q4_k (`s.Q4K`, `m.q4kw` populated) so the projections resolve
to raw q4_k blocks; the q4_k-majority projections upload raw and GEMM on the GPU, the Q8
minority keeps the existing Q8→f16 path. This is the identical per-projection
`q4kw`-else-`q8` dispatch `prefillBatchedQ4K` already makes (`prefill_q4k.go:51`).

### 2a — Weight table: prefer raw q4_k, fall back to f16

Replace the body of `metalWeights()` (`metal_prefill.go:68-88`) so that, per projection name,
it uploads the **raw q4_k blocks** when the weight is q4_k-resident and only dequant→f16 for the
Q8 minority. The q4_k upload reuses the proven `metalQ4KWeight` (`metal_q4k_on.go:146`), which
already caches one handle per `(model, name)` and (under `FAK_Q4K_FREE_CPU`) frees the CPU copy:

```go
// metalWeights builds this model's GPU projection table for the full-attention Metal twin.
// A q4_k-resident projection is uploaded as RAW q4_k super-blocks (UploadQ4K, ~0.5 B/elt) and
// run with Q4KWeight.GEMM; only a Q8-minority projection (no q4kw entry) falls back to the
// dequant→f16 Upload+MatMul. This removes the f32 round-trip the issue names (#70) and is the
// same per-projection q4kw-else-q8 dispatch prefillBatchedQ4K uses (prefill_q4k.go:51).
func (m *Model) metalWeights() map[string]*metalProj { /* per name: see mm in 2b */ }
```

In practice the simplest landing keeps the existing f16 table for the Q8 minority and adds a
parallel q4_k table, then makes `mm` (2b) pick. Either factoring is fine; the invariant is *no
q4_k weight is dequantized to f32*.

### 2b — `mm`: dispatch raw-q4_k GEMM vs f16 MatMul per projection

Replace the `mm` closure (`metal_prefill.go:202-212`) so each projection routes by store. This
is byte-for-byte the dispatch `q4kGemmDispatch` already performs — reuse it directly so the
kernel and its fallback are the single proven implementation, not a copy:

```go
mm := func(name string, X []float32, out, in int) []float32 {
	if qt := m.q4kw[name]; qt != nil {
		// q4_k-majority: raw blocks resident on the GPU, device dequant-GEMM. No f32/f16 copy.
		return s.q4kGemmDispatch(name, qt, X, P) // metal_q4k_on.go:34 (GPU GEMM, CPU fallback)
	}
	Y := make([]float32, P*out) // Q8 minority: existing dequant→f16 device GEMM
	gw[name].MatMul(X, P, Y)
	return Y
}
```

Because `q4kGemmDispatch` already falls back to the CPU `q4kGemm` when `!s.MetalQ4K` or the
device is absent, the twin should set `s.MetalQ4K` (or call `metalQ4KWeight` + `GEMM` directly)
so the q4_k projections stay on the GPU — inside the Metal twin a device is present by
construction. The seven `metalProjNames` (`metal_prefill.go:34`) and the resident-forward
handles in `metalResidentConfig` (`metal_prefill.go:98`) use the same `gw[...].ID()` handles, so
the resident path (`prefillMetalResident`, `metal_prefill.go:134`) must consume the q4_k handle
IDs too (or stay on its f16 IDs as a documented fallback) — see §5.

### 2c — `dequantQ8` becomes Q8-minority-only

`dequantQ8` (`metal_prefill.go:48`) is no longer called for q4_k weights. Keep it for the Q8
minority; it is unreachable for a pure-q4_k load and that is correct.

---

## 3 — Why this is the entire lever (and the memory win)

- **Upload bytes halve-and-then-some.** f16 is 2 B/elt; q4_k is ~0.5 B/elt (144 B per 256-elt
  super-block, `q4k.go:27`). The H2D copy and the resident footprint both drop ~4× for the
  q4_k-majority projections — the "doubling the upload bytes" the issue calls out, removed.
- **It is the only route that fits 27B on 36 GB.** f16 of a 27B ≈ 54 GB does not fit the
  unified pool; q4_k_m ≈ 16 GB does (`q4k.go:4-6`). So this is not just faster upload — it is
  what makes the full-attention Metal twin *runnable at all* on the real Qwen3.6-27B weights,
  which is the precondition for the on-device pp22 measurement in §6.
- **Single residency caveat (carried from #1067).** `metalQ4KWeight` frees the CPU q4_k copy
  after upload *only* under `FAK_Q4K_FREE_CPU=1` (`metal_q4k_on.go:31,159-167`); default OFF
  keeps both copies (~2× q4 bytes) because the CPU fallbacks `q4kGemm`/`q4kMatRows` read
  `qt.raw` and panic on nil if some tensor is not GPU-routed. The twin inherits this exactly:
  free the CPU copy only once *every* projection of every layer is provably GPU-routed. The
  honest default for #70 is double-residency (correct, ~30 GB) with `FAK_Q4K_FREE_CPU` as the
  opt-in single-residency (~16 GB) once the full-attention twin is proven fully GPU-routed.

The GPU kernel cost is already proven competitive: `mg_q4k_gemm` is register-blocked + uses a
SIMD-group dot reduction (#1085, commits `e7cee5ec`/`816f3319`) — ~6.5× at realistic P over the
naive kernel — and `TestMetalQ4KGemmMatchesCPU` (`metal_q4k_test.go:259`) already pins it on the
real `[17408,5120]` Qwen3.6-27B gate/up panel.

---

## 4 — What stays unchanged

Everything outside the projection upload/GEMM is identical f32 Go and is copied verbatim — the
twin's structure does not change, only the bytes the GPU holds:

- the per-layer RMSNorm, RoPE, causal GQA over the f32 KV cache, SwiGLU, residuals
  (`metal_prefill.go:226-303`) — pure CPU f32, already the proven reference;
- the f32 KV-cache appends (`Kraw` pre-RoPE, `K` post-RoPE, `V`, `pos`), so decode / Evict /
  Clone stay valid — the same contract `prefillBatchedQ4K` and `prefillMetalResident` honor;
- the `FAK_QPROFILE` timing seams (`metal_prefill.go:308-314`).

This is the same compute-bound split the twin already makes — only the matmuls on the GPU, the
cheap elementwise on the CPU — now fed compact q4_k weights instead of doubled f16.

---

## 5 — The resident forward + default-build stub

- **Resident forward.** `prefillMetalResident` (`metal_prefill.go:134`) and
  `metalResidentConfig` (`metal_prefill.go:98`) register the f16 projection handle IDs with
  `forward.m`. If the resident forward is to consume q4_k weights it needs q4_k handle IDs +
  a q4_k device forward; that is a larger change. The honest minimal #70 lands the **hybrid /
  per-matmul** path (§2) and lets the resident forward **decline** (return nil) for a q4_k
  load so `prefillBatchedMetal` falls back to the per-matmul q4_k path (`metal_prefill.go:181-186`
  already has exactly this decline-and-fall-back seam). A q4_k-resident on-GPU forward is then
  a separately-measured follow-up, not bundled into #70.
- **Default-build stub.** `prefillBatchedMetal` has a non-`fakmetal` stub
  (`metal_prefill_stub.go:9`); no new stub is needed because the change stays inside the
  already-tagged twin body. `q4kGemmDispatch`/`metalQ4KWeight` already have their `metal_q4k_off.go`
  stubs, so the default build stays one cgo-free artifact.

---

## 6 — The remaining gate (the honest "not yet")

Two steps remain, both requiring an Apple-Silicon M3 Pro with the macOS Metal toolchain — the
capability this `windows/amd64` host does not have:

1. **`fakmetal` build + parity witness.** On the Mac, after applying §2:
   `go test ./internal/model -tags fakmetal -run MetalQ4K -count=1`. The kernel itself is
   already pinned by `TestMetalQ4KGemmMatchesCPU` (`metal_q4k_test.go:259`). Add a wiring gate
   that mirrors `TestMetalQ4KPrefillMatchesCPU` (`metal_q4k_test.go:170`) but drives the
   **full-attention** twin: build a non-hybrid PreNorm synthetic model loaded q4_k, run
   `prefillBatchedMetal` with the new raw-q4_k upload and assert the logits match the CPU
   `prefillBatchedQ4K` path within the documented GPU float-accumulation band
   (`argmax` equal, `cosine ≥ 0.999`). **This is the green gate that closes the correctness
   half of #70** — it cannot be run here (no darwin/cgo).
2. **On-device tok/s measure.** `FAK_QPROFILE=1` prefill of Qwen3.6-27B q4_k_m at pp22 with
   the full-attention Metal twin on the raw-q4_k upload, against the `51.55 tok/s` llama.cpp-Metal
   bar — captured **without a co-resident llama-server** (the status-doc §3 swap-contamination
   rule: on a 36 GiB box two 27B copies page to swap; trust the clean isolated number). Record
   the before (f16, if it even fits) / after (raw q4_k) upload bytes and pp22 tok/s in this
   cluster. This is the number that proves the "closes most of the prefill compute gap" claim.

**Next checkable step:** apply §2 on a Mac node, add the §6.1 wiring test, run
`go test ./internal/model -tags fakmetal -run MetalQ4K`. Until that test is green on a witnessed
commit, #70 is `not yet` — the wiring is authored-as-spec, the kernel it reuses is shipped.

---

## 7 — Why this is the right increment (and what it is not)

This is the host-independent slice: it turns the issue's prose ask into a precise wiring change
bound to the **already-shipped** q4_k device GEMM (#1085) and its **already-proven** model-layer
dispatch (`metalQ4KWeight`/`q4kGemmDispatch`, #67) — so the Mac worker reuses a pinned kernel
rather than writing one — and it fixes the one imprecision worth fixing: the q4_k device GEMM is
not missing; it already runs for the *resident-Q4K hybrid* lane, and #70 is the small job of
extending that same upload+GEMM to the *full-attention* twin `prefillBatchedMetal`. It is **not**
a claim that the Metal kernel is built, compiled, or measured for the full-attention twin — those
are the gated steps in §6, and #70 stays open until a Mac worker witnesses them. No fabricated
pass: the only gate run for *this* change is the doc/commit gate on the `experiments` lane.
