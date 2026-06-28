# Metal Gated-DeltaNet recurrence — GPU kernel vs CPU-hybrid decision (#65)

**Status: DECIDED (CPU-hybrid, both phases) / one benchmark arm instrumented, the
on-device capture gated.** This note answers the "Step" of
[#65](https://github.com/anthony-chaudhary/fak/issues/65) (*the 48 linear_attn
Gated-DeltaNet layers' recurrent scan — GPU scan kernel vs CPU-hybrid; decide + benchmark
both*). The issue carries the measured finding itself — the GDN recurrence is **≈0.5% of
prefill** on this M3 Pro, the projections are the wall — and asks for the decision and a
two-arm benchmark, not for a new kernel. It was authored on a `windows/amd64`,
`CGO_ENABLED=0` box where the `-tags fakmetal` model lane **cannot compile, run, or be
measured**, so the one genuinely-gated step (the on-device `FAK_QPROFILE` capture of the
recurrence split on the M3 Pro) is named in §6 as the remaining work, not faked green. The
decision is grounded in measurement that is already on disk and in code that already
exists — the honesty rule of [`../../docs/proofs/00-METHOD.md`](../../docs/proofs/00-METHOD.md)
applies (a counter-witness would be recorded with its counterexample, not rounded away).

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
| **A — CPU-hybrid** | projection / MLP GEMMs on the GPU; the conv1d+SiLU mixer, q/k L2-norm, the per-head **delta-rule recurrent scan**, and the gated-RMSNorm readout stay in the f32 CPU recurrence | **yes** — `prefillQwen35LinearLayerMetal` (`internal/model/metal_prefill.go:494`), driven by `prefillBatchedMetalQwen35Hybrid` (`metal_prefill.go:384`); decode's `linearAttnStep` (`qwen35.go:399`) is the same CPU recurrence |
| **B — GPU scan kernel** | the delta-rule scan itself written as a Metal kernel (per-head rank-1 state update across the sequence), so the recurrence never leaves the GPU | **no** — would be new `-tags fakmetal` work; scoped to **#92** (chunked/Q8/device GDN), explicitly deferred here |

Arm A is the proven path: its recurrence math is **byte-for-byte** the scalar CPU reference
(`linearAttnSeq` → `linearAttnSeqBatched`, pinned by
`TestQwen35LinearAttnBatchedMatchesScalar`), and the `linearAttnCache` it advances is the
same object decode / `Evict` / `Clone` consume. Arm B would have to re-establish that parity
from scratch on a Metal kernel — cost the 0.5% measurement says is not yet worth paying.

---

## 3 — Prefill decision: CPU-hybrid (measured, not assumed)

The split is the one `prefillBatchedMetalQwen35Hybrid` already makes (`metal_prefill.go:432`):
`isLinearAttnLayer(l)` routes to `prefillQwen35LinearLayerMetal` (five GDN projections to the
GPU via `mm`, recurrence on CPU); the full-attention minority routes to
`prefillQwen35FullAttnLayerMetal`. The justification is on-disk measurement:

- `FAK_QPROFILE` per-phase breakdown on this M3 Pro (status doc §3): **`mlp_gate_up ≈ 43%`,
  `mlp_down ≈ 20%`** of the prefill wall, while the **GDN recurrence ≈ 0.5%**. #65's own body
  carries the same finding.
- Moving the ~63% that is MLP GEMMs (plus q/k/v/o and the GDN in/out projections) to the GPU's
  FLOP-rich path is the whole prefill lever. A CPU recurrence at 0.5% is sufficient; it cannot
  be the thing that closes the ~14× GPU-vs-CPU factor #65 decomposes.

The in-flight #71 twin encodes exactly this rationale at its head comment
(`metal_prefill.go:379-381`, citing `#65, #977`) — this doc is the canonical decision record
that comment points back to.

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

**Arm A is already instrumented.** `prefillBatchedMetalQwen35Hybrid` under `FAK_QPROFILE`
(`metalProf`, `metal_prefill.go:31`) prints the split directly (`metal_prefill.go:483`):

```
[metalprof-hybrid P=22] total=<t>  gemm+roundtrip=<g>  rest(recurrence/attn/norm)=<r> ms
```

The `rest(...)` term **is** the CPU-hybrid recurrence cost (plus the cheap norms/attn) — the
arm-A measurement #65 asks for, already in the binary. No new harness is needed; it needs a
Mac run (§6).

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

## 6 — The remaining gate (the honest "not yet")

One step remains, requiring an Apple-Silicon M3 Pro with the macOS Metal toolchain — the
capability this `windows/amd64`, `CGO_ENABLED=0` host does not have:

1. **On-device arm-A capture.** On the M3 Pro Mac verify node over Tailscale (the only place
   `-tags fakmetal` builds), once #71's twin is built:
   `FAK_QPROFILE=1` prefill of `Qwen3.6-27B.q4_k_m.gguf` at pp22, captured **without a
   co-resident llama-server** (the status doc §3 36-GiB swap-contamination rule). Record the
   `[metalprof-hybrid]` line in this cluster. That is the arm-A number; the §5 trigger then
   reads directly off `rest(...)` vs `gemm+roundtrip`.

Building arm B is **#92's** scope and is gated behind the §5 trigger — not part of #65.

**Next checkable step:** run gate (1) on the Mac node and append the `[metalprof-hybrid]`
result here. Until that line is on disk, the *on-device recurrence fraction* stays `not yet`
(the design decision, however, is **made** — from the 0.5% prefill profile already on disk and
the serialization analysis in §4, both independent of that capture).

---

## 7 — Why this is the right increment (and what it is not)

This is the host-independent slice #65 actually asks for: it turns "decide + benchmark both"
into (a) a **made decision** — CPU-hybrid for both phases, with the prefill call grounded in
the on-disk 0.5% profile and the decode call grounded in the round-trip/serialization analysis
that a GPU scan kernel does **not** address (#61/#67 do); and (b) a **two-arm benchmark
protocol** — arm A already instrumented (`[metalprof-hybrid]`), arm B reduced to a falsifiable
trigger into #92 rather than a speculative kernel. It corrects the tempting-but-wrong inference
(that the decode round-trip justifies a GPU scan kernel) and binds the decision to the code
that realizes it (#71). It is **not** a claim that the on-device recurrence fraction has been
re-measured here, nor that a Metal GDN-scan kernel was built or beaten — those are the gated /
deferred steps in §6 and §5. No fabricated pass: the only gate run for *this* change is the
doc/commit gate on the `experiments` lane.
