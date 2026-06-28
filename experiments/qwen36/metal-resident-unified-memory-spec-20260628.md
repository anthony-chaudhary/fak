# Metal resident weights in zero-copy unified memory (no per-call upload) — implementation spec (#69)

**Status: SPEC / host-independent slice.** This document is the code-level recipe a
Mac-equipped worker executes to land [#69](https://github.com/anthony-chaudhary/fak/issues/69)
(*keep q4_k/f16 weights resident in GPU/unified memory across calls instead of re-uploading per
forward; `m.metalResidentConfig()` is the seam; extend it to a zero-copy unified-memory residency
— the Apple analog of the Vulkan `FAK_GPU_BUDGET_MB` Stage-1 offload*). It was authored on a
`windows/amd64`, `CGO_ENABLED=0` box, where the Metal kernel files (`//go:build darwin && cgo &&
fakmetal`) **cannot be compiled, run, or measured** — so the two genuinely-gated steps (the loader
page-alignment wiring + `fakmetal` build, and the on-device M3 Pro resident-set / tok-s
measurement) are named here as the remaining work in §6, not faked green. #69 stays **OPEN** until
a Mac worker lands and witnesses the change using this spec.

Sibling evidence in this cluster (the same host-independent-slice pattern for the adjacent Metal
steps): [`metal-q4k-device-gemm-spec-20260628.md`](metal-q4k-device-gemm-spec-20260628.md) (#70,
the q4_k device GEMM for the full-attention twin) and
[`metal-hybrid-prefill-spec-20260627.md`](metal-hybrid-prefill-spec-20260627.md) (#71, the hybrid
prefill). The measurement bar this step is judged against lives in
[`QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
(the `51.55 tok/s pp22` llama.cpp-Metal prefill bar and the swap-contamination rule for a clean
isolated number).

---

## 1 — What "resident" already is today (don't re-ship it)

Reading the tree, **the first half of the issue's prose — "across calls instead of re-uploading
per forward" — already ships for the f16 lane.** Getting this boundary right is what keeps #69
from being a no-op re-claim:

- **f16 weights are uploaded once and cached per `*Model`.** `metalWeights()`
  (`internal/model/metal_prefill.go:68`) builds the seven-projection × `NumLayers` f16 table once,
  memoized in the package-level `metalWt map[*Model]map[string]*metalgemm.Weight`
  (`metal_prefill.go:42`); every later `prefillBatchedMetal` (`metal_prefill.go:198`) and the
  GPU-resident forward (`prefillMetalResident`, `metal_prefill.go:134`) reuse the cached handles —
  the `mm` closure calls `gw[name].MatMul` (`metal_prefill.go:206`), it does **not** re-upload.
- **The named seam already registers a resident topology.** `metalResidentConfig()`
  (`metal_prefill.go:98` — the function #69 names) already reuses those f16 handles, uploads the
  per-layer RMSNorm/bias vectors, and records the geometry with `forward.m` so the whole prefill
  runs in one command buffer with no per-matmul host round-trip.
- **q4_k weights are also cached resident per `(model, name)`.** `metalQ4KWeight`
  (`metal_q4k_on.go:146`) uploads each raw q4_k tensor once into `metalQ4KW` and reuses the handle.

So "upload once, reuse the handle across calls" is **done**. The part of #69 that is *not* done —
and the only part worth a commit — is the word the title leads with: **zero-copy**.

---

## 2 — The actual fence: residency today still costs a SECOND copy

Both upload paths allocate a **fresh** device `MTLBuffer` and copy the weight into it, so at the
moment of upload the weight exists **twice** in the unified pool — the CPU-side source AND the
device buffer:

- **q4_k (the lane that matters for 27B).** `mg_q4k_upload` (`internal/metalgemm/q4k.m:348`)
  does `id<MTLBuffer> b = [gDev newBufferWithLength:bytes options:MTLResourceStorageModeShared]`
  then `memcpy(b.contents, raw, bytes)` (`q4k.m:359-364`) — a verbatim copy of the GGUF
  super-blocks. The source `raw` is the Go slice `qt.raw` (`quant_q4k.go:59`), which the loader
  populated *verbatim* from the GGUF payload (`quantizeQ4KFromRaw`, `quant_q4k.go:342-355`,
  `raw: raw`). So immediately after upload there are two ~16 GB copies of a 27B q4_k_m: `qt.raw`
  and the device buffer.
- **f16 (small-model / Q8-minority lane).** `mg_upload` (`internal/metalgemm/metal.m:67`) allocs
  a shared buffer and converts f32→f16 into it (`metal.m:82-89`); the Go doc is explicit — "cgo
  copies it into the device buffer" (`metalgemm.go:63`).

The existing mitigation is a *free*, not a *no-copy*: `metalQ4KWeight` drops the CPU copy
(`qt.raw = nil`, `metal_q4k_on.go:166`) **only** under `FAK_Q4K_FREE_CPU=1`
(`freeCPUCopyAfterUpload`, `metal_q4k_on.go:31`). That is gated OFF by default because it is
**unsafe** (#1067): the batched-prefill CPU fallback `q4kGemm` and the decode `q4kMatRows` still
read `qt.raw`, so a not-fully-GPU-routed weight hits the freed `nil` slice — the `requireRawCPU`
guardrail (`quant_q4k.go:74-80`) now turns that into a legible panic instead of a cryptic
slice-bounds crash, but it is still a refusal, not residency. So today the honest options are
**double residency (~30 GB, swaps the 36 GB box)** or **a fragile single residency that fails
closed unless every matmul is provably GPU-routed.** Neither is the zero-copy the issue asks for.

---

## 3 — The change: alias the CPU bytes, don't copy them

On Apple Silicon the CPU and GPU share one physical pool, so a weight that is already in CPU
memory does not need a second device copy at all — the GPU can read the *same* pages. Metal
exposes exactly this:

```objc
// mg_q4k_upload_nocopy — zero-copy residency: wrap the caller's page-aligned q4_k region in a
// device buffer that ALIASES it (no memcpy, no second allocation). The bytes are the GGUF bytes
// and stay owned by the caller; deallocator:nil means Metal will not free them.
int mg_q4k_upload_nocopy(void *raw, int out, int in) {
    if (gDev == nil || !q4k_init() || in % 256 != 0) return -1;
    if (gNQ4 >= MG_MAX_Q4) return -1;
    int  nblk  = in / 256;
    long bytes = (long)out * nblk * 144;
    // raw MUST be page-aligned (getpagesize(), 16 KiB on Apple Silicon); newBufferWithBytesNoCopy
    // returns nil otherwise — the caller guarantees this (see §4) and we surface the nil.
    id<MTLBuffer> b = [gDev newBufferWithBytesNoCopy:raw
                                             length:(NSUInteger)bytes
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
    if (b == nil) { NSLog(@"mg_q4k_upload_nocopy: not page-aligned or alloc failed (%.1f MB)",
                          (double)bytes / 1e6); return -1; }
    int id = gNQ4++;
    gQ4[id].buf = CFBridgingRetain(b);   // retain the wrapper, NOT a copy of the bytes
    gQ4[id].out = out; gQ4[id].in = in; gQ4[id].nblk = nblk;
    return id;
}
```

Everything downstream is **unchanged**: `gQ4[id].buf` is consumed exactly as before by the
register-blocked `mg_q4k_gemm` (`q4k.m:481`), the SIMD-group `mg_q4k_gemv` (`q4k.m:374`), the
group/fused-MLP paths, and `mg_q4k_reset` (`q4k.m:527`) — the q4_k kernels read `device const
uchar* W`, and a no-copy shared buffer presents the identical bytes. The **only** thing that
changes is that there is now one physical copy of the weight, simultaneously CPU-readable (so the
`q4kGemm`/`q4kMatRows` fallbacks stay valid — no `requireRawCPU` panic, no `FAK_Q4K_FREE_CPU`
gamble) and GPU-readable. This is why zero-copy is strictly better than the free: it gives single
residency **and** keeps the CPU fallback live, dissolving the #1067 tension instead of guarding it.

This is the Apple analog of the Vulkan `FAK_GPU_BUDGET_MB` Stage-1 offload (`vulkan.go:59`,
`vulkan_plan.go:367` — on a *discrete* GPU the budget decides which cold weights spill
host-visible, because device VRAM and host RAM are separate pools). On unified memory there is no
separate device pool to budget against, so the budget concept **collapses to zero-copy
residency**: every weight is host-visible and device-visible at once, at no copy. The honest
exposure is therefore not a byte budget but a boolean lever — call it `FAK_METAL_NOCOPY` — that
selects the no-copy upload over the copying one (default-on once §6 is witnessed; off falls back
to today's copying `mg_q4k_upload`, which stays as the portable/un-aligned path).

---

## 4 — The real cost: the loader must hand over a page-aligned, stably-owned region

`newBufferWithBytesNoCopy` has two hard preconditions that today's `qt.raw` does not meet, and
satisfying them is where the genuine work is:

1. **Page alignment.** The base pointer must be aligned to `getpagesize()` (16 KiB on Apple
   Silicon) or the call returns `nil`. A `qt.raw` from `make([]byte, …)` (`quant_q4k.go`) has no
   such guarantee.
2. **Stable, non-Go-owned lifetime.** Metal retains the buffer (and thus the backing memory)
   across many calls. cgo forbids C from retaining a Go pointer past the call that passed it, and
   the Go GC may move/collect a `[]byte`. So the aliased region must be C-owned (or OS-mapped),
   not a plain Go slice.

Both are solved the way llama.cpp's Metal backend solves them — **memory-map the GGUF and alias
each tensor's region** — and the tree already has the darwin seam:

- `internal/model/mmap_unix.go:11` (`mmapOpen` via `syscall.Mmap`) returns a **page-aligned**,
  read-only mapping of a file (mmap base addresses are always page-aligned), with a `Close` that
  unmaps; `mmap_other.go:15` is the Windows/`js`/`plan9` stub (`errMmapUnsupported`). The q4_k GGUF
  loader would map the file once and set each `q4kTensor.raw` to the **sub-slice of the mapping**
  at that tensor's payload offset (the bytes are already verbatim GGUF — `quant_q4k.go:55`), and
  pass `&raw[0]` to `mg_q4k_upload_nocopy`. The mapping outlives every buffer; the GPU pages the
  weights in on demand; the CPU fallback reads the same pages. Zero H2D copy, single residency.
  - Caveat the loader must honor: a *tensor's* payload offset inside the GGUF is only
    page-aligned if the file lays it out that way. GGUF does not guarantee 16 KiB tensor
    alignment, so the mapping base is aligned but `&raw[tensorOffset]` may not be. Two honest
    landings: (a) require/encode a re-aligned GGUF (a one-time `fak`-side repack that pads tensor
    payloads to page boundaries), or (b) for any unaligned tensor fall back to today's copying
    `mg_q4k_upload`. Mixed is fine — the win is dominated by the few huge MLP/attention tensors,
    which are easy to pad.
- The non-mmap alternative (no file map, e.g. a quantize-in-place path) is a C `posix_memalign`
  region the loader fills and owns; same alignment, same lifetime, but it reintroduces one copy at
  load, so mmap is the genuinely-zero-copy route and the one to land.

**f16 lane:** zero-copy does *not* apply to the Q8→f16 path, because that path's bytes are
synthesized (dequant Q8 → cast f16, `metal.m:89`) — there is no pre-existing f16 region to alias.
That lane's residency is already the "upload once, cache the handle" of §1; #69's zero-copy win is
specifically the **raw-block (q4_k) lane**, which is also the only lane that fits 27B in 36 GB
(f16 of 27B ≈ 54 GB does not; q4_k_m ≈ 16 GB does — `q4k.go:4-6`). So #69 and #70 stack: #70 routes
the full-attention twin through the q4_k device GEMM; #69 makes that q4_k weight resident at zero
copy.

---

## 5 — The Go-side wiring (`internal/metalgemm` + `internal/model`, `-tags fakmetal`)

- **`internal/metalgemm/q4k.go`:** add `UploadQ4KNoCopy(raw []byte, out, in int) *Q4KWeight`
  mirroring `UploadQ4K` but calling `mg_q4k_upload_nocopy((unsafe.Pointer)(&raw[0]), …)`. Because
  Metal retains the bytes, the Go side must keep the backing mapping alive for the handle's
  lifetime — store the mmap closer on the `*Model` (it already memoizes the handle table), and
  drop it only in the `Reset`/teardown path alongside `mg_q4k_reset`.
- **`internal/model/metal_q4k_on.go`:** in `metalQ4KWeight` (`:146`), when `FAK_METAL_NOCOPY` is
  set and `qt.raw` is page-aligned (mmap-backed), call `UploadQ4KNoCopy` and **do not** free
  `qt.raw` (it IS the GPU buffer now); otherwise keep today's `UploadQ4K` + `FAK_Q4K_FREE_CPU`
  behavior verbatim. This makes `requireRawCPU` (`quant_q4k.go:74`) unreachable on the no-copy
  path — the CPU bytes are never freed — which is the correctness upgrade over the free.
- **`internal/model/metal_prefill.go`:** `metalResidentConfig` (`:98`) is unchanged for the f16
  projections; it already records resident handles. The only seam touch is documenting that under
  `FAK_METAL_NOCOPY` the q4_k projections consumed by the §70 full-attention twin come from
  no-copy handles — the handle id contract (`Weight.ID()`/`Q4KWeight` ids) is identical, so the
  resident forward needs no change.
- **Stub parity:** `metal_q4k_off.go` already stubs the q4_k entry points for the default
  (non-`fakmetal`) build; add the `UploadQ4KNoCopy` stub there so `go build ./...` stays one
  cgo-free artifact. No new platform stub is needed in `metalgemm_stub.go` beyond mirroring the
  existing `Upload*` stubs.

---

## 6 — The remaining gate (the honest "not yet")

Three steps remain, all requiring an Apple-Silicon M3 Pro with the macOS Metal toolchain — the
capability this `windows/amd64`, `CGO_ENABLED=0` host does not have:

1. **Loader page-alignment wiring + `fakmetal` build.** Map the q4_k GGUF via the `mmap_unix.go`
   seam, point `q4kTensor.raw` at the aligned tensor region (pad/fallback per §4), add
   `mg_q4k_upload_nocopy` + `UploadQ4KNoCopy`, then
   `go build ./... -tags fakmetal && go vet ./... -tags fakmetal` on the Mac. **Cannot be run
   here** (no darwin/cgo).
2. **Parity witness.** Add a test mirroring `TestMetalQ4KPrefillMatchesCPU`
   (`internal/model/metal_q4k_test.go:170`) but with `FAK_METAL_NOCOPY=1`: load a synthetic q4_k
   model through the mmap path, run `prefillBatchedQ4K` (and, with #70, `prefillBatchedMetal`) and
   assert the logits match the copying-upload path bit-for-bit — the no-copy buffer presents the
   *same* bytes, so this must be **exactly equal**, not just `cosine ≥ 0.999`. Also assert the CPU
   fallback (`q4kGemm` with `s.MetalQ4K=false`) still reads `qt.raw` without panic — the #1067
   regression guard for the no-copy path. `go test ./internal/model -tags fakmetal -run MetalQ4K
   -count=1`. **This is the green gate that closes the correctness half of #69.**
3. **On-device residency + tok/s measurement.** With Qwen3.6-27B q4_k_m on the 36 GB box, record:
   peak unified-memory resident set (copying `mg_q4k_upload` ≈ 2× q4 bytes at upload vs no-copy ≈
   1×), whether the 27B now loads without `FAK_Q4K_FREE_CPU`, and `FAK_QPROFILE=1` pp22 prefill
   tok/s against the `51.55 tok/s` llama.cpp-Metal bar — captured **without a co-resident
   llama-server** (the status-doc §3 swap-contamination rule). Record before/after in this
   cluster. This is the number that proves "removes per-call upload from the hot path."

**Next checkable step:** map the q4_k GGUF page-aligned on a Mac node, add `mg_q4k_upload_nocopy` +
the §6.2 parity test, run `go test ./internal/model -tags fakmetal -run MetalQ4K`. Until that test
is green on a witnessed commit, #69 is `not yet` — the wiring is authored-as-spec; the residency
cache and the q4_k device kernels it reuses are already shipped (§1).

---

## 7 — Why this is the right increment (and what it is not)

This is the host-independent slice: it turns the issue's prose into a precise change bound to the
**already-shipped** residency cache (`metalWt`/`metalQ4KW`) and the **already-shipped** q4_k device
kernels (`mg_q4k_gemm`/`mg_q4k_gemv`, #1085/#67), and it fixes the one imprecision worth fixing —
"resident across calls" is *already true*; what is missing is the **zero-copy** that turns today's
two-copies-or-a-fragile-free into one physical copy that is both CPU- and GPU-readable. It is
**not** a claim that the no-copy upload is compiled, page-aligned in the loader, or measured —
those are the gated steps in §6, and #69 stays open until a Mac worker witnesses them. No
fabricated pass: the only gate run for *this* change is the doc/commit gate on the `experiments`
lane.
