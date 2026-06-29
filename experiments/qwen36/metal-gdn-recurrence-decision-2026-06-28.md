# Metal Gated-DeltaNet recurrence — GPU kernel vs CPU-hybrid decision (#65)

**Status: DECIDED + WITNESSED (CPU-hybrid, both phases).** This note answers the "Step" of
[#65](https://github.com/anthony-chaudhary/fak/issues/65) (*the 48 linear_attn
Gated-DeltaNet layers' recurrent scan — GPU scan kernel vs CPU-hybrid; decide + benchmark
both*). The issue carries the measured finding itself — the GDN recurrence is **≈0.5% of
prefill** on this M3 Pro, the projections are the wall — and asks for the decision and a
two-arm benchmark, not for a new kernel. The original host-runnable decision was authored
on a `windows/amd64`, `CGO_ENABLED=0` box where the `-tags fakmetal` model lane cannot run;
the remaining on-device M3 Pro capture has now been run and recorded as
[`metal-gdn-recurrence-m3pro-20260629.json`](metal-gdn-recurrence-m3pro-20260629.json). The
decision is grounded in measurement that is on disk and in code that exists — the honesty
rule of [`../../docs/proofs/00-METHOD.md`](../../docs/proofs/00-METHOD.md) applies (a
counter-witness would be recorded with its counterexample, not rounded away).

Sibling evidence in this cluster:
[`QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
(§3 — the `FAK_QPROFILE` per-phase breakdown this decision rests on: `mlp_gate_up ≈ 43%`,
`mlp_down ≈ 20%`, **GDN recurrence ≈ 0.5%**; and the `51.55 tok/s pp22` llama.cpp-Metal
bar) and [`metal-hybrid-prefill-spec-20260627.md`](metal-hybrid-prefill-spec-20260627.md)
(#71 — the hybrid Metal prefill twin that **realizes** this decision for prefill). #92 is
the separate device-GDN ticket this decision routes the kernel arm to, gated behind the §5
trigger.

---

## 1 — The decision, in one line

**Keep the Gated-DeltaNet recurrence on the CPU (the CPU-hybrid arm) for both prefill and
decode; do _not_ write a Metal GDN-scan kernel speculatively.** The recurrence is ≈0.5% of
the prefill wall — a GPU scan kernel would chase 0.5% while the projection GEMMs (≈63%) are
the entire lever. A device GDN kernel (#92) becomes warranted **only if** an on-device
profile, captured _after_ the projections move to the GPU, shows the recurrence (or its
CPU↔GPU round-trip) has become the wall. That is a measured trigger (§5), not an assumption
baked in now — the #977 playbook forbids guessing the bottleneck.

---

## 2 — The two arms, in code terms

| arm | what it is | exists today? |
|---|---|---|
| **A — CPU-hybrid** | projection / MLP GEMMs on the GPU; the conv1d+SiLU mixer, q/k L2-norm, the per-head **delta-rule recurrent scan**, and the gated-RMSNorm readout stay in the f32 CPU recurrence | **yes** — `prefillBatchedMetalQwen35Hybrid` (`internal/model/metal_prefill_hybrid.go`) injects a GPU GEMM into `prefillQwen35HybridViaMM` / `prefillQwen35LinearLayerMM` (`internal/model/metal_prefill_hybrid_core.go`); the resident-Q4_K lane uses `prefillQwen35HybridQ4KHidden` / `prefillQwen35LinearLayerQ4K` (`internal/model/qwen35_prefill_q4k.go`); decode's `linearAttnStep` (`qwen35.go:399`) is the same CPU recurrence |
| **B — GPU scan kernel** | the delta-rule scan itself written as a Metal kernel (per-head rank-1 state update across the sequence), so the recurrence never leaves the GPU | **no** — would be new `-tags fakmetal` work; scoped to **#92** (chunked/Q8/device GDN), explicitly deferred here |

Arm A is the proven path: its recurrence math is **byte-for-byte** the scalar CPU reference
(`linearAttnSeq` → `linearAttnSeqBatched`, pinned by
`TestQwen35LinearAttnBatchedMatchesScalar`), and the `linearAttnCache` it advances is the
same object decode / `Evict` / `Clone` consume. Arm B would have to re-establish that parity
from scratch on a Metal kernel — cost the 0.5% measurement says is not yet worth paying.

---

## 3 — Prefill decision: CPU-hybrid (measured, not assumed)

The split is the one `prefillQwen35HybridViaMM` already makes: `isLinearAttnLayer(l)` routes
to `prefillQwen35LinearLayerMM` (five GDN projections via the injected GEMM, recurrence on
CPU); the full-attention minority routes to `prefillQwen35FullAttnLayerMM`. The
resident-Q4_K lane mirrors that same split in `prefillQwen35HybridQ4KHidden`. The
justification is on-disk measurement:

- `FAK_QPROFILE` per-phase breakdown on this M3 Pro (status doc §3): **`mlp_gate_up ≈ 43%`,
  `mlp_down ≈ 20%`** of the prefill wall, while the **GDN recurrence ≈ 0.5%**. #65's own body
  carries the same finding.
- Moving the ~63% that is MLP GEMMs (plus q/k/v/o and the GDN in/out projections) to the GPU's
  FLOP-rich path is the whole prefill lever. A CPU recurrence at 0.5% is sufficient; it cannot
  be the thing that closes the ~14× GPU-vs-CPU factor #65 decomposes.

The #71 twin encodes exactly this rationale in `internal/model/metal_prefill_hybrid.go` and
`internal/model/metal_prefill_hybrid_core.go`, citing `#65, #977`; this doc is the
canonical decision record those comments point back to.

---

## 4 — Decode decision: still CPU recurrence, but the cost is _serialization_, not FLOPs

Decode changes the **denominator**, so it is decided separately rather than by the prefill
0.5%. Per #65's first comment (the corrected #67 diagnosis): the GDN recurrence is **~6–8% of
decode** by compute, but the operative cost is that it forces a **CPU↔GPU round-trip every GDN
layer** — the GPU idles while the CPU runs the inter-matmul scan, because each `mg_q4k_*`
blocks on `waitUntilCompleted`. On a 3:1-interleave 27B that is 48 such stalls per token.

The naive reading ("the round-trip is expensive ⇒ write a GPU scan kernel") is the wrong
conclusion, and naming why is the substance of this decision:

- The stall is not the recurrence's **arithmetic**; it is the **blocking boundary** between a
  GPU GEMM and a CPU step. A GPU GDN-scan kernel removes one boundary but the fix that removes
  _all_ of them — for the recurrence, the norms, the gating, the full-attn softmax — is a
  **single resident forward** that keeps one command buffer in flight across the whole token
  (#61 shared resident activation buffer, #67 resident decode). That lever subsumes the GDN
  case without a bespoke scan kernel.
- A CPU recurrence **inside** a resident forward (results staged through the shared device
  buffer, no per-layer `waitUntilCompleted`) pays the 0.5%/6–8% compute but **not** the stall.
  So the decode decision is: **keep the recurrence on the CPU; spend the effort on #61/#67's
  resident buffer, not on a Metal scan kernel.**

A GPU scan kernel only earns its place if, after #61/#67 land and the projections are GPU-
resident, the recurrence-plus-its-staging is _measured_ to be the remaining decode wall. That
is the §5 trigger into #92.

---

## 5 — Benchmark both (the protocol, and the trigger for arm B)

"Benchmark both" does not mean build a kernel we have already decided not to build. It means
**measure arm A on the device, and define the number that would flip the decision to arm B.**

**Arm A is already instrumented.** `prefillQwen35HybridViaMM` and the resident-Q4_K
`prefillQwen35HybridQ4KHidden` path print the hybrid split directly under `FAK_QPROFILE`:

```
[metalprof-hybrid P=22] total=<t>  gemm+roundtrip=<g>  rest(recurrence/attn/norm)=<r> ms
```

The `rest(...)` term **is** the CPU-hybrid recurrence upper bound (plus the cheap norms/attn)
— the arm-A measurement #65 asks for, already in the binary. No new harness is needed; the
Mac run is recorded below (§6).

**A device-independent arm-A witness is now on disk and host-runnable** (no Mac, no GPU):
[`gdn-recurrence-bench/`](gdn-recurrence-bench/) computes the recurrence-vs-projection
**compute ratio** at the real Qwen3.6-27B GDN shapes — exact FLOP arithmetic over the layer
dims, reproducible on any box. Measured here: the recurrent scan is **2.725%** of the
linear_attn projections (and ~0.5% of whole prefill once the MLP GEMMs enter the
denominator — agreeing with the on-device profile), with a native-CPU wall-time
corroboration of 1.99%. This bounds arm B: a perfect zero-cost GPU scan kernel could save
**at most** that few percent, while the projections (already GPU-routed by the CPU-hybrid)
are the ~97% lever. It does **not** replace the §6 Mac capture — that measures the
*serialization* fraction after the projections move to the GPU, a different number.

**The on-device arm-A capture is now on disk.** On `node-macos-a` (Mac15,7 / Apple M3 Pro
18-core GPU / 36 GB unified / macOS 26.5 / Metal 4), the resident-Q4_K Metal path for the
real `Qwen3.6-27B.q4_k_m.gguf` was run at pp22 with `FAK_QPROFILE=1`, `FAK_Q4K=1`,
`FAK_METAL=1`, `CGO_ENABLED=1`, and `-tags fakmetal`. Witness:
[`metal-gdn-recurrence-m3pro-20260629.json`](metal-gdn-recurrence-m3pro-20260629.json).

```
[metalprof-hybrid P=22] total=6720.7  gemm+roundtrip=6051.6  rest(recurrence/attn/norm)=669.1 ms  path=q4k
```

That puts projection GEMM/round-trip at **90.0%** of the measured hybrid split, with the
entire non-projection bucket at **10.0%**. The `rest(...)` bucket is an **upper bound** for
GDN recurrence because it also contains full-attention, norms, q8 panel quantization, conv,
gating, residuals, cache bookkeeping, and the linear-attention recurrence. Combined with the
device-independent recurrence compute bound above, the live Mac result keeps the same
decision: the projections remain the wall; a bespoke GPU scan kernel still chases the wrong
part of the profile.

**Arm B (the device GDN scan) is benchmarked counterfactually, then conditionally.** The
on-disk 0.5% prefill figure already adjudicates the prefill comparison: even a perfect,
zero-cost GPU scan kernel can return at most that 0.5% of prefill — below the noise of the
projection GEMMs — so arm B cannot beat arm A on prefill, and building it to measure it would
spend M3-Pro kernel-author effort to confirm a bound the profile already proves. The honest
benchmark is therefore the **trigger**, not a premature kernel:

> **Build + benchmark arm B (#92) only when** an on-device `[metalprof-hybrid]` (prefill) or
> the resident-decode profile (post #61/#67) shows `rest(recurrence/…)` rising to a non-trivial
> fraction of `total` (rule of thumb: recurrence-attributable time ≳ the largest single
> projection GEMM). Until that number appears, arm B stays deferred and #92 stays gated.

This keeps the comparison evidence-driven: arm A has a real measurement path; arm B has a
real, falsifiable condition under which it would be built and raced against arm A.

---

## 6 — The on-device gate (now reached)

The former "not yet" gate was the Apple-Silicon capture of arm A after the projections move
to Metal. It is now reached:

| field | value |
|---|---|
| witness | [`metal-gdn-recurrence-m3pro-20260629.json`](metal-gdn-recurrence-m3pro-20260629.json) |
| host | `node-macos-a` — Mac15,7 / Apple M3 Pro 18-core GPU / 36 GB unified / macOS 26.5 / Metal 4 |
| command shape | `FAK_QPROFILE=1 FAK_METAL=1 FAK_Q4K=1 CGO_ENABLED=1 go run -tags fakmetal ./cmd/modelbench ... -q4k -metal -prefill-sizes 22 -prefill-reps 1` |
| profile line | `[metalprof-hybrid P=22] total=6720.7  gemm+roundtrip=6051.6  rest(recurrence/attn/norm)=669.1 ms  path=q4k` |
| interpretation | projection GEMM/round-trip is **90.0%**; the whole non-projection bucket is **10.0%** and is an upper bound on recurrence |

Building arm B remains **#92's** scope and is gated behind the §5 trigger — not part of #65.
The trigger did not fire here: this capture does not isolate recurrence as the wall; it shows
the projection/round-trip bucket is still dominant.

---

## 7 — Why this is the right increment (and what it is not)

This is the smallest close for #65: it turns "decide + benchmark both" into (a) a **made
decision** — CPU-hybrid for both phases, with the prefill call grounded in the 0.5% prior
profile, the host-runnable recurrence/projection compute ratio, and the fresh M3 Pro
resident-Q4_K `[metalprof-hybrid]` capture; and (b) a **two-arm benchmark protocol** — arm A
measured, arm B reduced to a falsifiable trigger into #92 rather than a speculative kernel.
It corrects the tempting-but-wrong inference (that the decode round-trip justifies a GPU scan
kernel) and binds the decision to the code that realizes it (#71). It is **not** a claim that
a Metal GDN-scan kernel was built or beaten — #65 decides not to build it unless the #92
trigger fires. No fabricated pass: the on-device gate is now recorded, and the rest bucket is
kept as an upper bound rather than mislabeled as recurrence-only time.
