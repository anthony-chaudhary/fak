# Metal resident weights in unified memory — closure status (#69)

**Status: SHIPPED (residency) / literal-zero-copy + on-device measurement gated.** This
note binds [#69](https://github.com/anthony-chaudhary/fak/issues/69) (*keep q4_k/f16 weights
resident in GPU/unified memory across calls instead of re-uploading per forward;
`m.metalResidentConfig()` is the seam — extend it to a zero-copy unified-memory residency,
the Apple analog of the Vulkan `FAK_GPU_BUDGET_MB` Stage-1 offload*) to the tree as it
actually stands. The core deliverable the issue asks for — **no per-token / per-call weight
upload on the decode + prefill hot path** — **is implemented and shipping on `main`**. It
landed under the sibling Metal-resident-forward epic #67 (the resident decode forward) plus
the existing per-`*Model` upload caches, and was never cited back to #69, so the issue sat
open with its work already done. This note was authored on a `windows/amd64`,
`CGO_ENABLED=0` box where the `-tags fakmetal` / `-tags metal` Metal lanes cannot compile,
run, or be measured, so the two genuinely-gated steps — the literal `newBufferWithBytesNoCopy`
upgrade and the on-device M3 Pro residency-win measurement — are named in §5 as remaining
work, **not faked green**.

Sibling evidence in this cluster:
[`metal-q4k-device-gemm-status-2026-06-28.md`](metal-q4k-device-gemm-status-2026-06-28.md)
(#70, the q4_k device GEMM that consumes these resident weights),
[`metal-hybrid-prefill-spec-20260627.md`](metal-hybrid-prefill-spec-20260627.md) (#71), and
[`QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
(the M3 Pro box, the 36 GiB unified pool, and the `51.55 tok/s pp22` llama.cpp-Metal bar this
lane closes against).

---

## 1 — The issue's premise vs the tree today

The issue's literal ask is two things, and they have *different* statuses on `main`:

1. **"Keep weights resident across calls instead of re-uploading per forward."** — **DONE.**
   Both Metal lanes allocate + fill each weight buffer **once** and reuse it for every
   subsequent matmul; the per-`*Model` caches guarantee the upload runs once per model, not
   per token. Per-forward, the *only* host↔device traffic is the activation X and the result
   Y — never the weights (§2).
2. **"Extend it to a *zero-copy* unified-memory residency."** — **PARTIALLY.** The weights
   are resident in `MTLResourceStorageModeShared` buffers (true Apple unified memory: one
   physical allocation the CPU and GPU both address; the per-call activation round-trip is a
   coherency flush, **not** a copy — `metal.m:8-9`). But the **one-time upload itself still
   copies** into that shared buffer (`newBufferWithLength` + `memcpy`/`f32→f16`), where a
   literal `newBufferWithBytesNoCopy:` over the page-aligned host GGUF mapping would wrap the
   bytes with **zero** copy and collapse the q4_k double-residency. That is the genuine open
   residual (§5).

So #69 is **not "make weights resident"** (done) — it is the narrower, Apple-hardware-gated
job of turning the one-time *resident upload* into a true *zero-copy* wrap, plus measuring the
residency win on device.

---

## 2 — What is implemented (the residency closure)

**The seam #69 names is wired.** `m.metalResidentConfig()`
(`internal/model/metal_prefill.go:98`) registers a model with the GPU-resident forward
**once** (`metalResidentReady[m]`, line 102/125): it reuses the already-uploaded projection
handles via `gw[...].ID()` (lines 119-122) and uploads the per-layer RMSNorm + q/k/v bias
vectors once. `prefillMetalResident` (`metal_prefill.go:134`) then runs the whole fresh
prefill on those resident handles in **one command buffer, one sync** (line 153) — no
per-matmul, and certainly no per-weight, round-trip.

**f16 / Q8 lane — resident, reused, never re-uploaded.**
`mg_upload` (`internal/metalgemm/metal.m:67`) allocates the f16 weight in a
`MTLResourceStorageModeShared` buffer (lines 82-83) and **retains it across calls** with
`CFBridgingRetain` into the `gW[]` table (line 91). `mg_matmul` (line 99) reads `W.buf`
(line 104) for every projection and only converts the per-call activation/result into the
*reused* `gXBuf`/`gYBuf` scratch (lines 107-120, 146) — the weight buffer is never touched
again. The model side caches the whole table per `*Model`: `metalWeights()`
(`metal_prefill.go:68`) returns early on `metalWt[m]` (lines 71-73), so the upload loop runs
exactly once per model.

**q4_k lane — resident raw blocks, cached per (model, name).**
`metalQ4KWeight` (`internal/model/metal_q4k_on.go:146`) uploads each weight's raw GGUF
super-blocks **once** via `UploadQ4K` (`internal/metalgemm/q4k.go:37`) and caches the handle
in `metalQ4KW[m][name]` — **caching `nil` too** "so a failed upload doesn't retry every
token" (`metal_q4k_on.go:158`, 24-25). The resident bytes are the compact form — 144 B per
256-weight super-block, ≈ 0.5 B/elt (`q4k.go:27`), the only residency that fits a 27B on the
36 GB pool (`q4k.go:4-6`). Both the decode GEMV (`q4kMatRowsDispatch`, line 51) and the
prefill GEMM (`q4kGemmDispatch`, line 34) read that one resident handle.

**Resident decode forward — weights *and* KV stay on the GPU across steps.** The #67 epic
moved the whole decode loop onto resident state, so decode pays no per-step upload of weights
**or** KV:

- `5a11a319` `feat(metal): add the GPU-resident Q8 decode forward (#67)`
- `6806ecd2` `perf(metal): keep the decode KV resident on the GPU across steps (#67)`
- `69276457` `perf(metal): drop the per-layer KV-append blits from the resident decode forward (#67)`
- `f8f331fb` `perf(metal): use f16 block scales + fuse projection bias in the resident decode GEMV (#67)`
- `4bf74979` `perf(metal): run the final norm + LM head on the GPU in the resident decode forward (#67)`
- `e116128e` `perf(metal): parallelize the resident-decode RMSNorm (#67)`
- `d6188182` `perf(metal): add key-parallel flash-decode attention to the resident decode forward (#67)`
- `62462799` `perf(metal): fuse the q4_k decode MLP + add the grouped-GEMV decode seam`

**Anti-leak discipline.** `mg_reset` / `mg_q4k_reset` (`metal.m:169`, `q4k.go:139`) tear down
the resident table only on model reload, so residency does not *accumulate* across reloads —
the per-load leak that helped exhaust unified memory on 2026-06-18 (`metal.m:161-168`). The
model side drops `metalWt` / `metalResidentReady` in the same critical section
(`metalgemm.go:127-136`).

---

## 3 — The Apple analog of `FAK_GPU_BUDGET_MB`, drawn precisely

The issue frames the target as "the Apple analog of the Vulkan `FAK_GPU_BUDGET_MB` Stage-1
offload." The two are analogs of *intent* (keep the hot weights device-resident), but the
mechanism differs because the memory model differs — and naming that difference is what keeps
the #69 scope honest:

| | Vulkan / CUDA (`FAK_GPU_BUDGET_MB`) | Apple unified memory (this lane) |
|---|---|---|
| memory model | discrete VRAM, separate from host RAM | one physical pool, CPU + GPU coherent |
| the lever | the **Spills** lever — cap device-local residency; weights past the cap spill *in upload order* to host-visible / managed memory (`internal/compute/vulkan.go:57`) | residency is the **default** when the set fits (q4_k_m 27B ≈ 16 GB ≤ 36 GB); nothing to budget or spill |
| per-call weight copy | a real H2D copy across PCIe unless resident | **none** — the shared buffer is the same physical bytes the GPU reads (`metal.m:8-9`) |
| the residual | overflow handling on a too-small card | the **one-time upload copy** → `newBufferWithBytesNoCopy` (§5) |

So on Apple there is no per-call copy to eliminate (it is already gone) and no spill budget to
honor (the q4_k set fits). The only "copy" left to remove is the *one-time* fill of the
resident buffer — the literal "zero-copy" word in the issue title — which §5 scopes.

---

## 4 — Correctness witness (what I ran here)

The host-independent half is the CPU-side dispatch that every resident GPU path mirrors, pinned
by pure-Go parity tests that **do** run on this box (via an isolated `git archive HEAD` build,
since native `go test` is App-Control-blocked in the repo dir):

```
go test ./internal/model     -run Q4K -count=1   →  ok
go test ./internal/metalgemm          -count=1   →  ok
```

These pin the dispatch contract the resident GPU handles consume — `q4kGemm`/`q4kMatRows`
bit-identical per `(row, token)`, and the batched prefill consistent with the per-token path.
The **on-device** residency is pinned by the `-tags fakmetal` GPU tests
(`TestMetalQ4KGemmMatchesCPU` records cosine 1.0 vs the CPU f32 reference; the resident decode
forward has its own `fakmetal` parity gates) — but those are the witnesses a Mac worker
re-confirms, **not** ones this `windows/amd64` host can run.

---

## 5 — The remaining gate (the honest "not yet")

Two steps remain, both requiring an Apple-Silicon M3 Pro with the macOS Metal toolchain — the
capability this `windows/amd64` host does not have:

1. **Literal zero-copy upload.** Today `mg_q4k_upload` (`q4k.m:359-364`) does
   `newBufferWithLength:...StorageModeShared` + `memcpy(b.contents, raw, bytes)` — the raw
   GGUF bytes are *copied* into a fresh resident buffer. Replace that with
   `newBufferWithBytesNoCopy:length:options:deallocator:` over a **page-aligned**
   (`getpagesize()`) host mapping of the q4_k tensor, so the device addresses the GGUF bytes
   in place: **zero upload copy** *and* automatic single-residency (no separate CPU+GPU
   copy → retires the `FAK_Q4K_FREE_CPU` double-residency caveat at `metal_q4k_on.go:8-13`,
   159-167). Preconditions: the GGUF must be mmap'd page-aligned and kept alive for the
   buffer's lifetime, and `in*…*144` must be a multiple of the page size or padded. The f16
   lane's upload copy is **not** removable this way — it is a genuine `f32→f16` precision
   conversion (`metal.m:89`) — so this lever is q4_k-only. Gate: build under `-tags fakmetal`
   on a Mac and re-run the q4_k parity tests to confirm the no-copy buffer reads identically.
2. **On-device residency-win measure.** With `FAK_QPROFILE=1`, capture decode + pp22 prefill
   of Qwen3.6-27B q4_k_m on the resident path vs a per-call-reupload baseline, against the
   `51.55 tok/s` pp22 / `7.29` decode llama.cpp-Metal bars — **without a co-resident
   llama-server** (the 36 GiB swap-contamination rule from the parity-status §3). Record the
   per-phase GEMM-vs-attention-vs-rest breakdown and the upload-bytes delta in this cluster.
   **Precondition — the reupload baseline does not exist in the tree yet.** Residency is
   unconditionally on: `metalWeights` short-circuits on `metalWt[m]` (`metal_prefill.go:71`)
   and `metalQ4KWeight` on `metalQ4KW[m][name]` with no flag to bypass either cache, and a
   tree-wide grep of `internal/model` + `internal/metalgemm` at `HEAD` finds no `reupload` /
   `NORESIDENT` / `NOCOPY` toggle (the only residency-adjacent env knobs that exist are
   `FAK_Q4K_FREE_CPU`, `FAK_QPROFILE`, `FAK_METAL_RESIDENT`, `FAK_QKERNEL`). So there is no A/B
   switch to force the per-call-upload arm and the residency-*win* number cannot be obtained by
   env alone. Step (2) therefore depends on a small **`(fak model)`-lane** code add first — a
   `FAK_METAL_REUPLOAD=1` baseline that re-runs the upload each forward (skip/clear the cache
   guard) so the resident path can be diffed against it. That makes the measurement itself
   cross-lane-gated, not a pure Mac-side capture.

**Next checkable step:** on a Mac node, first add the `FAK_METAL_REUPLOAD` baseline toggle
(a `(fak model)`-lane change, since this `experiments` lane cannot host it), apply (1), run
`go test ./internal/model -tags fakmetal -run MetalQ4K -count=1`, then capture (2) as the
resident-vs-reupload diff. Until the on-device residency-win number is recorded, the *zero-copy
speedup* claim is `not yet`; the **residency** deliverable (no per-forward weight upload, weights
resident in unified memory) is shipped and CPU-parity-pinned.

---

## 6 — Out of scope for #69

The MPS `-tags metal` registry backend's quantized device GEMM is still genuinely open
(`internal/compute/metal.go` panics on non-F32) — a different lane from the `-tags fakmetal`
`internal/metalgemm` lane closed here. Lifting `requirePreNorm` for the hybrid arch is #71;
routing the full-attention Q8 twin off its `dequantQ8 → f32 → Upload` form is #70. Those are
separate tickets in this same cluster, not part of #69.

---

## 7 — Why this is the right increment

This is the host-independent slice: it binds #69 to the resident-weights work that already
shipped under #67 (and the per-`*Model` upload caches), corrects the issue body's implicit
assumption (residency is not missing — it shipped; only the *literal* zero-copy upload and its
on-device measurement remain), draws the Apple/Vulkan analog precisely so the `FAK_GPU_BUDGET_MB`
comparison is not over-read, and runs the one gate this box can run (the pure-Go CPU-parity
tests that pin the dispatch the resident GPU handles mirror). It is **not** a claim that the
`newBufferWithBytesNoCopy` upgrade is built or that the residency-win tok/s is measured — those
are the gated steps in §5, and the zero-copy speedup stays `not yet` until a Mac worker records
it. No fabricated pass: the only gates run for *this* change are the pure-Go parity tests above
and the doc/commit gate on the `experiments` lane.
