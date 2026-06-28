# Metal hybrid (Qwen3.6 Gated-DeltaNet) prefill twin — closure status (#71)

**Status: the host-independent slice is now FULLY COMMITTED on `main` — the backend-agnostic core
(`c80d64fa`) plus the Metal twin, the `requirePreNorm`-lifting gate, the default-build stub, and the
`kv.go` dispatch (`5c065118`, `dos commit-audit` `diff-witnessed`) — and its entire CPU-side logic is
host-independently WITNESSED green on this box; the GPU f16 GEMM numerics + on-device tok/s are the
only Mac-bound residual — #71 stays `not yet`.** This is the *status* half
of the spec→status pair for
[#71](https://github.com/anthony-chaudhary/fak/issues/71) (*lift `requirePreNorm` so the hybrid
Qwen3.6 Gated-DeltaNet arch can use the Metal prefill*). Its *spec* half —
[`metal-hybrid-prefill-spec-20260627.md`](metal-hybrid-prefill-spec-20260627.md) — laid out the
code-level recipe (§2–§5); this note records that **the recipe is now realized as committed
code**: the prefill body is the backend-agnostic core `prefillQwen35HybridViaMM`
(`internal/model/metal_prefill_hybrid_core.go`, default build, no cgo, committed `c80d64fa`), and
its whole CPU-side orchestration is proven equal to the shipped CPU template by a test that runs
**green on this `windows/amd64`, `CGO_ENABLED=0` box** (§2) — which narrows the gate that still
cannot be reached here to just the GPU f16 GEMM numerics and the on-device throughput.
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

| spec step | symbol realized | build tag | commit state |
|---|---|---|---|
| §3 — the prefill **body** (structural copy of the CPU `prefillQwen35HybridQHidden` template) | **backend-agnostic core** `prefillQwen35HybridViaMM` (`metal_prefill_hybrid_core.go`): per-layer `isLinearAttnLayer(l)` split into `prefillQwen35LinearLayerMM` (5 GDN projections via the injected `mm`; conv1d+SiLU / q·k L2-norm / delta-rule scan / gated-RMSNorm readout on CPU) and `prefillQwen35FullAttnLayerMM` (q⧺gate/k/v/o via `mm`; RoPE/GQA/output-gate/QK-norm on CPU); shared `mlp.{gate,up,down}` GEMMs via `mm`. Only the projection GEMMs are abstracted behind `hybridGemmFn`; every elementwise op is the identical f32 CPU math. | **default** (no cgo) | **committed `c80d64fa`** |
| §3 — the Metal **twin** (the one substitution the core abstracts) | `prefillBatchedMetalQwen35Hybrid` + `metalWeightsQwen35Hybrid` (`metal_prefill_hybrid.go`): a GPU f16 GEMM per projection (dequant-once-to-f16, cached per `*Model`, **layer-kind-correct** upload set — `linear_attn.{in_proj_qkv,z,b,a,out_proj}` vs `self_attn.{q,k,v,o}_proj` plus `mlp.*`), fed into the same core | `fakmetal` | committed `5c065118` |
| §2 — the gate (pure `Config` logic, no cgo) | `metalQwen35HybridPrefillOK` (delegates to `q8Qwen35HybridPrefillOK`); routed from `Session.Prefill` / `Session.PrefillNoLogits` behind `s.Metal && …OK && s.Cache.Len()==0`, **lifting `requirePreNorm("Metal prefill")`** for the hybrid | **default** (`metal_prefill_hybrid_gate.go` / `kv.go`) | committed `5c065118` |
| §5 — the default-build stub | `prefillBatchedMetalQwen35Hybrid` one-line panic twin (`metal_prefill_hybrid_stub.go`), so `kv.go`'s dispatch compiles cgo-free | **default** | committed `5c065118` |

The twin keeps the GDN recurrence on the CPU and batches the projection/MLP GEMMs on the GPU —
exactly the split the spec's §4 decision and the #65 decision doc justify from the on-disk
profile (projection GEMMs ≈ 63 % of the prefill wall; the GDN scan ≈ 0.5 %). The f32
`Kraw/K/V/pos` + `linearAttnCache` it builds is the same object decode / `Evict` / `Clone`
already consume, so the path is drop-in with the proven CPU template.

---

## 2 — The gate run here (host-independent **correctness** witness)

The strongest gate runnable on this `windows/amd64`, `CGO_ENABLED=0` host is now a *correctness*
witness, not just a compile one — it exists because `c80d64fa` factored the prefill body into the
backend-agnostic core, so the twin's **entire CPU-side logic** is exercised without a Mac or
`-tags fakmetal`:

```
$ go test ./internal/model -run TestQwen35HybridViaMMMatchesCPUTemplate -count=1
ok  	github.com/anthony-chaudhary/fak/internal/model	0.334s
```

`TestQwen35HybridViaMMMatchesCPUTemplate` (committed with the core, `c80d64fa`) drives
`prefillQwen35HybridViaMM` with a CPU Q8 `mm` that reproduces the proven `prefillQwen35HybridQHidden`
path per projection, and asserts the **whole** prefill — logits, the KV cache, and the linear-attn
cache — matches that proven path within the documented grouped-vs-ungrouped Q8 float-order drift
(~1e-6). That witnesses every CPU-side op the Metal twin runs: the conv1d+SiLU mixer, the q/k
L2-norm, the per-head delta-rule recurrent scan, the gated RMSNorm readout, the full-attention
RoPE/GQA/output-gate/QK-norm, both RMSNorms, and every residual. It catches the exact bug class an
off-device Metal lane is otherwise blind to — a transcription error when the body was hand-copied
from the CPU template, which diverges O(1) per layer and blows past the close tolerances.

The witness above was captured against the **committed** tree (`git archive HEAD` into a scratch
build root); pin it to the committed core `c80d64fa`, because a peer's in-flight `forward.go`
refactor (a `normWeights` change, orthogonal to #71) transiently reddens the *live working-tree*
build of this package. The weaker, secondary gate is the default cgo-free build of the dispatch
wiring + panic stub (`go vet ./internal/model/`), now that the model-lane twin/gate/stub have
landed (`5c065118`). What
**neither** gate exercises is the `-tags fakmetal` body's two irreducibly device-bound residuals —
the GPU f16 GEMM numerics and the on-device throughput — which is the Mac gate in §3.

---

## 3 — The remaining gate (the honest "not yet")

The model-lane code has fully landed: the backend-agnostic **core** (`c80d64fa`) and the thin GPU
**twin** + the `requirePreNorm`-lifting **gate** + the default-build **stub** + the `kv.go`
dispatch (`5c065118`). Both commits are `dos commit-audit` `diff-witnessed` (claim_kind
`code_effect`), and §2's correctness witness is green against this committed tree. What remains is
**only** the two device-bound steps that need an Apple-Silicon M3 Pro with the macOS Metal
toolchain — the capability this host does not have:

1. **Model-lane code — LANDED (`c80d64fa` + `5c065118`).** All on `main`: the backend-agnostic core
   (`metal_prefill_hybrid_core.go`), the `-tags fakmetal` GPU wrapper
   `prefillBatchedMetalQwen35Hybrid` + `metalWeightsQwen35Hybrid` (`metal_prefill_hybrid.go`), the
   un-tagged gate `metalQwen35HybridPrefillOK` (`metal_prefill_hybrid_gate.go`) that lifts
   `requirePreNorm("Metal prefill")`, the default-build stub (`metal_prefill_hybrid_stub.go`), and
   the `kv.go` dispatch. A Mac worker can now `git pull` the wrapper directly — no model-lane commit
   is outstanding.
2. **`fakmetal` build + GPU-numerics parity witness (closes the last correctness residual).** On
   the Mac, with the wrapper committed:
   `go test ./internal/model -tags fakmetal -run Qwen35Hybrid -count=1`. Because §2 already proves
   the CPU-side orchestration host-independently, this Mac run adds the **one** residual it cannot
   reach: the GPU f16 GEMM numerics. Mirror `TestPrefillQwen35HybridQ4KMatchesTokenLoop` — run the
   twin and the proven CPU template on the same prompt, assert the hidden states match within the
   f16-GEMM + Q8 deferred-reduction / FMA-rounding drift band. **This is the green gate that closes
   #71's last correctness residual; it cannot be run here (no darwin/cgo/Metal).**
3. **On-device tok/s measure.** `FAK_QPROFILE=1` prefill of `Qwen3.6-27B.q4_k_m.gguf` at pp22,
   against the `51.55 tok/s` llama.cpp-Metal bar, captured **without a co-resident llama-server**
   (the swap-contamination rule: on a 36 GiB box two 27B copies page to swap — trust the clean
   isolated number). Record the `[metalprof-hybrid …]` line in this cluster; §5 of the #65
   decision doc then reads the arm-A recurrence fraction directly off it.

**Next checkable step:** step (1) has landed (`5c065118`); run step (2) on the Mac verify node and
append the GPU-numerics parity verdict + the `[metalprof-hybrid]` line here. Until that test is
green on a witnessed commit, the *on-device GPU correctness + throughput* of #71 stays `not yet`;
the host-independent slice — the recipe, its committed backend-agnostic core, the committed Metal
twin/gate/stub/`kv.go` wiring, and that core's green CPU-side correctness witness — is **done**.

---

## 4 — Why this is the right increment (and what it is not)

This note is the `experiments`-lane closure-status record #71 needs to turn it from "needs
implementation" into "implemented — needs the Mac witness," so a Mac worker reads the one gate
left rather than re-deriving §2–§5. It is grounded in the committed spec and in the proven CPU
template, witnesses the strongest gate this host can run (§2's host-independent correctness test,
green), and names the exact host capability it cannot reach (the M3 Pro `fakmetal` toolchain) plus
the `model`-lane commit dependency. It is **not** a claim that the Metal kernel was compiled, run,
or measured here, nor that #71 is closed — #71 stays open until the Mac GPU-numerics parity test is
green on a witnessed commit. No fabricated pass: the only gate run for *this* change is §2's
host-independent correctness test above (green at the committed core) and the doc/commit gate on
the `experiments` lane.
