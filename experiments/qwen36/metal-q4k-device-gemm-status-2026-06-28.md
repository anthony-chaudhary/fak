# Metal q4_k device GEMM — closure status (#70)

**Status: SHIPPED (code) / first on-device number measured (~11× under SOTA) — the
matmul-only per-phase split and the kernel work to close the gap stay Mac-gated (§5.1).**
This note binds
[#70](https://github.com/anthony-chaudhary/fak/issues/70) (*upload q4_k as f16/int8
and run the device GEMM directly, not `dequantQ8 → f32 → Upload`*) to the tree as it
actually stands. The upload-side deliverable the issue asks for **is implemented and
wired on `main`** — it landed under the sibling tickets #1071 (model dispatch) and
#1085 (the register-blocked GEMM kernel), and was never cited back to #70, so the issue
sat open with its work already done. It was authored on a `windows/amd64`,
`CGO_ENABLED=0` box where the `-tags fakmetal` kernel cannot compile, run, or be
measured, so the one genuinely-gated step — the on-device matmul-only re-measure on an
M3 Pro — is named as the remaining work in §5, not faked green.

Sibling evidence in this cluster:
[`metal-hybrid-prefill-spec-20260627.md`](metal-hybrid-prefill-spec-20260627.md) (#71,
the hybrid-arch Metal prefill twin) and
[`QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
(the `51.55 tok/s pp22` llama.cpp-Metal bar this lane closes against).

---

## 1 — The issue's premise vs the tree today

The issue body reads against `internal/model/metal_prefill.go`: *"`metal_prefill.go`
currently `dequantQ8 → f32 → metalgemm.Upload`, doubling the upload bytes and losing the
q4_k compactness."* That sentence is still literally true of `prefillBatchedMetal`
(`metal_prefill.go:79` still calls `metalgemm.Upload(dequantQ8(qt), …)`) — **but that is
the Q8 lane (`s.Metal`), not the q4_k lane.** The q4_k device GEMM did not land by
editing `prefillBatchedMetal`; it landed as a **separate resident-Q4_K prefill path**,
exactly as the CPU side split `prefillBatchedQ` (Q8) from `prefillBatchedQ4K` (Q4_K).
So "fix `metal_prefill.go`" was satisfied by a sibling file, and the issue's named file
is a now-superseded lane for the q4_k case.

Routing today (`internal/model/kv.go:502-538`):

| selector | prefill path | weight upload |
|---|---|---|
| `s.Q4K` + `s.MetalQ4K` | `prefillBatchedQ4K` → `q4kGemmDispatch` | **`UploadQ4K` — raw q4_k super-blocks resident, device GEMM directly** |
| `s.Metal` (Q8) | `prefillBatchedMetal` | `Upload(dequantQ8 → f32)` — the issue's old lane |

A q4_k_m artifact loads through the `s.Q4K` selector, so it takes the **first** row —
the device GEMM #70 asked for — never the `dequantQ8 → f32` row.

---

## 2 — What is implemented (the upload-side closure)

**Raw upload, no dequant.** `metalgemm.UploadQ4K`
(`internal/metalgemm/q4k.go:37`) copies the *verbatim GGUF super-block bytes* resident
onto the GPU — length `out*(in/256)*144`, i.e. **144 bytes per 256-weight super-block**
(4-bit codes + the packed 6-bit sub-scales/mins + the f16 super-scale). Nothing is
dequantized host-side; each GPU thread dequants its weight row on the fly inside the
kernel. Contrast the byte cost the issue flagged:

| on-device weight form | bytes / 256 weights | vs raw q4_k |
|---|---|---|
| **raw q4_k (this path)** | **144** | 1.0× |
| f16 (`metalgemm.Upload` half) | 512 | 3.6× |
| f32 (`dequantQ8 → f32`, the old Q8 lane) | 1024 | 7.1× |

**Device GEMM directly.** `Q4KWeight.GEMM` (`q4k.go:127`) dispatches `q4k_gemm`
(`internal/metalgemm/q4k.m:145`), the **register-blocked tiled prefill GEMM** from #1085:
each threadgroup computes a 64×64 output block, walks the K axis one q4_k sub-block at a
time, and reuses each staged value through a 4×4 register micro-tile. Measured on an M3
Pro at the real `[17408,5120]` gate/up shape: **~1375 GFLOP/s, flat across P=64..2048**
(~6.5–7.2× the prior one-threadgroup-per-row kernel, which went DRAM-bound on redundant
activation re-reads), ~20% of the f32 FLOP ceiling.

**Wiring.** `prefillBatchedQ4K` (`internal/model/prefill_q4k.go:53`) routes each q4_k
projection through `q4kGemmDispatch`; under `-tags fakmetal` with `s.MetalQ4K` set,
`q4kGemmDispatch` (`internal/model/metal_q4k_on.go:34`) uploads the raw weight once via
`metalQ4KWeight` → `UploadQ4K` and runs `w.GEMM`. The pure-Go build keeps the same seam
on the CPU `q4kGemm` (`metal_q4k_off.go`), so the shipped binary stays one artifact and
`MetalQ4K` is opt-in.

Landing commits (none bound to #70 — the gap this note closes):

- `e7cee5ec` `perf(metalgemm): implement a register-blocked q4k_gemm prefill GEMM — ~6.5x at realistic P (#1085)`
- `816f3319` `perf(metalgemm): q4k_gemm prefill GEMM uses a SIMD-group dot reduction for small-P occupancy (#1085)`
- `ba5af664` `fix(model): fix resident-Q4_K prefill to dispatch its GEMM to the Metal GPU (#1071)`
- `62462799` `perf(metal): fuse the q4_k decode MLP + add the grouped-GEMV decode seam`

(`UploadQ4K` / `mg_q4k_upload` itself predates the granular public history — it is in the
v0.30.0 release baseline `1029e37c`, so the raw-upload primitive has been resident the
whole time; what #1071/#1085 added is the **prefill GEMM** that consumes it.)

---

## 3 — Correctness witness (what I ran here)

The host-independent half is the CPU-side dispatch contract — the same byte stream the
GPU kernel consumes — and it is pinned by pure-Go parity tests that **do** run on this
box (via an isolated `git archive HEAD` build, since native `go test` is App-Control-
blocked in the repo dir):

```
go test ./internal/model    -run Q4K -count=1   →  ok  (1.270s)
go test ./internal/metalgemm        -count=1   →  ok  (0.255s)
```

Covering:

- `TestQ4KGemmMatchesMatRows` / `TestQ4KGemmInt8MatchesMatRowsInt8` — the batched
  `q4kGemm` is **bit-identical** per `(output row, token)` to the proven per-token decode
  GEMV, on both the f32 and int8-SDOT reductions. So the q4_k majority contributes **zero**
  drift; a dispatch/packing bug blows these up by orders of magnitude.
- `TestPrefillBatchedQ4KMatchesSerial` / `…Deterministic` — the whole resident-Q4_K
  batched prefill builds a KV cache consistent with the per-token path, within the same
  attention reduction-order band `prefillBatchedQ` already shares.

The **on-device** GPU half is pinned by `TestMetalQ4KGemmMatchesCPU` (the `q4k.m` header
records **cosine 1.0** vs the CPU f32 reference) — but that test is `-tags fakmetal` and
is the witness a Mac worker re-confirms, not one this host can run.

A first M3 Pro run has since landed (under #64,
[`metal-fak-q4k-27b-20260626.json`](metal-fak-q4k-27b-20260626.json)): the on-device
**decode** greedy parity holds (`TestMetalQ4KDecodeMatchesCPU` PASS — the Metal lane
emits the same token sequence as the CPU reference), and the artifact carries the first
on-device prefill/decode tok/s. That witnesses the decode lane and the end-to-end path;
the prefill **GEMM** parity (`TestMetalQ4KGemmMatchesCPU` / `…PrefillMatchesCPU`) is the
half still awaiting a clean Mac re-confirm. The tok/s themselves are read in §5.

---

## 4 — Why this is #70's deliverable

The issue's "Step" is exactly satisfied: the q4_k-majority weights are uploaded **raw**
(`UploadQ4K`, 144 B/super-block — the compact form, not `dequantQ8 → f32`), and the
device GEMM (`q4k_gemm`) runs **directly** on them. "Closes most of the prefill compute
gap" is the GPU-vs-CPU (~14×) factor the issue decomposed; the kernel realizes ~6.5–7.2×
of it as a GEMM-only number today, with the full path-level number still gated on §5.

---

## 5 — The remaining gate (the honest "not yet")

The issue's own comment names it: *"per the method, re-run the matmul-only split after to
verify the q4_k GEMM stays bandwidth-bound."* That measurement requires an Apple-Silicon
M3 Pro with the macOS Metal toolchain — the capability this `windows/amd64` host does not
have. On a Mac node:

1. **Build + on-device parity:** `go test ./internal/model -tags fakmetal -run MetalQ4K -count=1`
   (confirms `TestMetalQ4KGemmMatchesCPU` / `…PrefillMatchesCPU` green on the real device).
2. **Matmul-only split:** `FAK_QPROFILE=1` prefill of Qwen3.6-27B q4_k_m at pp22 with
   `s.Q4K + s.MetalQ4K`, captured **without a co-resident llama-server** (the 36 GiB
   swap-contamination rule), against the `51.55 tok/s` llama.cpp-Metal bar. Record the
   per-phase GEMM-vs-attention-vs-rest breakdown in this cluster, and confirm the q4_k
   GEMM is bandwidth-bound (not occupancy-bound) at realistic P.

**Next checkable step:** run gate (1) on a Mac node, then capture (2). Until the
matmul-only split is recorded, the *per-phase* speedup attribution is `not yet`;
the **code** deliverable (raw upload + device GEMM) is shipped and CPU-parity-pinned.

### 5.1 — Update: a first whole-path on-device number now exists (still `not yet`)

The end-to-end on-device rate is **no longer fully unrecorded**. A clean M3 Pro run is
committed (under #64,
[`metal-fak-q4k-27b-20260626.json`](metal-fak-q4k-27b-20260626.json) — node-macos-a,
18-GPU-core M3 Pro / 36 GB, `-tags fakmetal`, `FAK_Q4K=1 FAK_METAL=1`, llama booted out
per the 36 GiB swap rule):

| phase | on-device (fak Q4_K + Metal) | llama.cpp-Metal bar | gap |
|---|---|---|---|
| prefill (P=421, agentic) | **4.5 tok/s** | 51.55 tok/s (pp22) | **~11× under** |
| prefill (P=29, tiny) | 0.6 tok/s | — | weight-read-dominated, provenance only |
| decode | 1.2 tok/s | 7.29 / 6.12 tok/s | ~5–6× under |

So the issue's headline — *"closes most of the prefill compute gap"* — is now a
**measured** claim, not an unmeasured one: the q4_k device GEMM runs on the GPU and the
agentic-length prefill is up off the ~0.8 tok/s CPU floor, but it still sits **~11× under
the 51.55 tok/s llama.cpp-Metal SOTA** at ~20 % of the f32 FLOP ceiling. The 4.5 tok/s is
a **whole-path** number (GEMM + attention + norm + sampling), so it does **not** by itself
prove the q4_k GEMM is the bound; that attribution is exactly the §5 step-(2) matmul-only
`FAK_QPROFILE=1` per-phase split, which remains the open Mac-gated gate. #70 therefore
stays a genuine `not yet`: **measured-but-gap-remains**, with the per-phase split (and the
kernel work to close the ~11×) the remaining step — both Apple-Silicon-only.

---

## 6 — Out of scope for #70

The Q8 lane (`prefillBatchedMetal`, `s.Metal`) still does `dequantQ8 → f32 → Upload`
(`metal_prefill.go:79`). That is the **full-attention Q8** path, a different selector
from the q4_k lane this issue is about; reworking its upload form (or retiring it now
that the q4_k lane exists) is a separate concern, not part of #70. Likewise the MPS
backend's quantized device GEMM is still genuinely open
(`internal/compute/metal.go:276` panics on non-F32) — that is the `-tags metal` registry
lane, distinct from the `-tags fakmetal` `internal/metalgemm` lane closed here.

---

## 7 — Why this is the right increment

This is the host-independent slice: it binds #70 to the work that already shipped under
#1071/#1085, corrects the issue body's implicit assumption (the fix is a separate q4_k
lane, not an edit to `prefillBatchedMetal`), and runs the one gate this box can run (the
pure-Go CPU-parity tests that pin the dispatch the GPU kernel mirrors). It is **not** a
claim that the on-device tok/s number is measured — that is the gated step in §5, and the
end-to-end speedup stays `not yet` until a Mac worker records it. No fabricated pass: the
only gates run for *this* change are the pure-Go parity tests above and the doc/commit
gate on the `experiments` lane.
