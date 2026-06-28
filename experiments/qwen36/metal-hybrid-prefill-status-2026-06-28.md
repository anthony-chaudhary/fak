# Metal hybrid (Qwen3.6 Gated-DeltaNet) prefill twin — closure status (#71)

**Status: CODE-COMPLETE on this host (host-independent slice done) / Mac `fakmetal` parity
witness pending — #71 stays `not yet`.** This is the *status* half of the spec→status pair for
[#71](https://github.com/anthony-chaudhary/fak/issues/71) (*lift `requirePreNorm` so the hybrid
Qwen3.6 Gated-DeltaNet arch can use the Metal prefill*). Its *spec* half —
[`metal-hybrid-prefill-spec-20260627.md`](metal-hybrid-prefill-spec-20260627.md) — laid out the
code-level recipe (§2–§5); this note records that **the recipe is now realized as code** and
pins the single gate that still cannot be reached on this `windows/amd64`, `CGO_ENABLED=0` box.
It does **not** claim the Metal kernel was compiled, run, or measured here — those need an
Apple-Silicon M3 Pro with the macOS Metal toolchain, named in §3 as the honest remaining work,
not faked green (the [`../../docs/proofs/00-METHOD.md`](../../docs/proofs/00-METHOD.md) honesty
rule applies: a counter-witness would be recorded, not rounded away).

Sibling evidence in this cluster:
[`metal-hybrid-prefill-spec-20260627.md`](metal-hybrid-prefill-spec-20260627.md) (the recipe
this realizes), [`metal-gdn-recurrence-decision-2026-06-28.md`](metal-gdn-recurrence-decision-2026-06-28.md)
(#65 — why the GDN recurrence stays on the CPU, the decision this prefill twin encodes), and
[`QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
(the `FAK_QPROFILE` per-phase split and the `51.55 tok/s pp22` llama.cpp-Metal bar).

---

## 1 — What the spec asked for, and what now exists in code

The spec's §2–§5 are realized in the `internal/model` package — the hybrid-aware Metal prefill
twin plus the gate/stub that route the Qwen3.6 Gated-DeltaNet arch to it instead of falling back
to the CPU. Each spec step maps to a concrete symbol now present in the tree:

| spec step | symbol realized | build tag |
|---|---|---|
| §2 — the gate (pure `Config` logic, no cgo) | `metalQwen35HybridPrefillOK` (delegates to `q8Qwen35HybridPrefillOK`); routed from `Session.Prefill` and `Session.PrefillNoLogits` behind `s.Metal && …OK && s.Cache.Len()==0` | **default** (`kv.go`) |
| §3 — the twin (structural copy of the CPU `prefillQwen35Hybrid…` template) | `prefillBatchedMetalQwen35Hybrid` — per-layer `isLinearAttnLayer(l)` split into `prefillQwen35LinearLayerMetal` (5 GDN projections to GPU, conv1d+SiLU / q·k L2-norm / delta-rule scan / gated-RMSNorm readout on CPU) and `prefillQwen35FullAttnLayerMetal` (q⧺gate/k/v/o to GPU, RoPE/GQA/output-gate/QK-norm on CPU); shared `mlp.{gate,up,down}` GEMMs to GPU | `fakmetal` |
| §3 (upload) — the hybrid-aware weight table | `metalWeightsQwen35Hybrid` — per layer uploads the **layer-kind-correct** projection set (`linear_attn.{in_proj_qkv,z,b,a,out_proj}` vs `self_attn.{q,k,v,o}_proj`) plus `mlp.*`, freeing the f32 staging buffer after each `metalgemm.Upload` | `fakmetal` |
| §5 — the default-build stub | `prefillBatchedMetalQwen35Hybrid` one-line panic twin of the `prefillBatchedMetal` stub, so `kv.go`'s dispatch compiles cgo-free | **default** (`metal_prefill_stub.go`) |

The twin keeps the GDN recurrence on the CPU and batches the projection/MLP GEMMs on the GPU —
exactly the split the spec's §4 decision and the #65 decision doc justify from the on-disk
profile (projection GEMMs ≈ 63 % of the prefill wall; the GDN scan ≈ 0.5 %). The f32
`Kraw/K/V/pos` + `linearAttnCache` it builds is the same object decode / `Evict` / `Clone`
already consume, so the path is drop-in with the proven CPU template.

---

## 2 — The gate run here (host-independent witness)

The only gate runnable on this `windows/amd64`, `CGO_ENABLED=0` host is the **default cgo-free
build** of the touched package — the half of §2/§5 that is not tag-gated (the gate predicate in
`kv.go` and the panic stub). It is green:

```
$ CGO_ENABLED=0 go vet ./internal/model/
$ echo $?
0
```

This proves the dispatch wiring and the stub compile and type-check in the default build with the
twin present — i.e. the un-tagged surface of #71 is sound. It does **not** exercise the
`-tags fakmetal` kernel body (that build constraint excludes it on every non-darwin host), so it
is a *compile* witness of the wiring, not a *correctness* witness of the GPU math. The
correctness witness is the Mac gate in §3.

---

## 3 — The remaining gate (the honest "not yet")

Two steps remain, both requiring an Apple-Silicon M3 Pro with the macOS Metal toolchain — the
capability this host does not have — and one repo step that belongs to the `model` lane:

1. **Model-lane commit of the twin.** The §3 kernel body + `metalWeightsQwen35Hybrid` live in
   `internal/model/metal_prefill.go` (the `model` lane, `-tags fakmetal`), with the gate/stub in
   `kv.go` / `metal_prefill_stub.go`. Those are a `model`-lane ship, by the file-tree rule — this
   `experiments`-lane note records their state, it does not commit them. Until that commit lands
   on `main`, a Mac worker cannot `git pull` the twin.
2. **`fakmetal` build + parity witness (closes the correctness half).** On the Mac, with the twin
   committed: `go test ./internal/model -tags fakmetal -run Qwen35Hybrid -count=1`. Add a parity
   test mirroring `TestPrefillQwen35HybridQ4KMatchesTokenLoop` — run the twin and the proven CPU
   template on the same prompt, assert the hidden states match within the documented Q8
   deferred-reduction / FMA-rounding drift band. **This is the green gate that closes #71's
   correctness half; it cannot be run here (no darwin/cgo/Metal).**
3. **On-device tok/s measure.** `FAK_QPROFILE=1` prefill of `Qwen3.6-27B.q4_k_m.gguf` at pp22,
   against the `51.55 tok/s` llama.cpp-Metal bar, captured **without a co-resident llama-server**
   (the swap-contamination rule: on a 36 GiB box two 27B copies page to swap — trust the clean
   isolated number). Record the `[metalprof-hybrid …]` line in this cluster; §5 of the #65
   decision doc then reads the arm-A recurrence fraction directly off it.

**Next checkable step:** land step (1) on the `model` lane, then run step (2) on the Mac verify
node and append the parity verdict + the `[metalprof-hybrid]` line here. Until that test is green
on a witnessed commit, the *on-device kernel correctness* of #71 stays `not yet`; the
host-independent slice (the recipe + its cgo-free wiring) is **done**.

---

## 4 — Why this is the right increment (and what it is not)

This note is the `experiments`-lane closure-status record #71 needs to turn it from "needs
implementation" into "implemented — needs the Mac witness," so a Mac worker reads the one gate
left rather than re-deriving §2–§5. It is grounded in the committed spec and in the proven CPU
template, witnesses the only gate this host can run (the default cgo-free build), and names the
exact host capability it cannot reach (the M3 Pro `fakmetal` toolchain) plus the `model`-lane
commit dependency. It is **not** a claim that the Metal kernel was compiled, run, or measured
here, nor that #71 is closed — #71 stays open until the Mac parity test is green on a witnessed
commit. No fabricated pass: the only gate run for *this* change is the default-build vet above
and the doc/commit gate on the `experiments` lane.
