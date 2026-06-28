# Metal hybrid (Qwen3.6 Gated-DeltaNet) prefill twin — closure status (#71)

> **CLOSED 2026-06-28** — [#71](https://github.com/anthony-chaudhary/fak/issues/71) was closed
> `completed` after re-verifying the gate this session: the `requirePreNorm`-lifting implementation
> (`c80d64fa` core, `5c065118` twin/gate/stub/`kv.go` dispatch — `dos commit-audit` `diff-witnessed`,
> `0ffae0be` Mac parity test) is on `origin/main`; the host-independent correctness gate
> (`TestQwen35HybridViaMMMatchesCPUTemplate` + `TestMetalQwen35HybridPrefillOK`) re-ran **green** on
> the `windows/amd64` author host; and the on-device f16 parity is witnessed green on the M3 Pro
> (`cos=0.999377`, recorded in `9bd73429`). The sole residual — the pp22 throughput number — is
> structurally unreachable through this f16 twin on the 36 GiB box and rides #70 / #1085 (§3 step 3),
> exactly the operator recommendation in §3. The issue stayed OPEN despite being fully witnessed
> because the auto-closer walks only the top-10 open-witnessed issues sorted DESC by number, starving
> a low number like #71 — so the close was executed surgically (`gh issue close 71 -r completed`),
> not via the mass closer.

**Status: the host-independent slice is now FULLY COMMITTED on `main` — the backend-agnostic core
(`c80d64fa`) plus the Metal twin, the `requirePreNorm`-lifting gate, the default-build stub, and the
`kv.go` dispatch (`5c065118`, `dos commit-audit` `diff-witnessed`) — and its entire CPU-side logic is
host-independently WITNESSED green on this box. **Update 2026-06-28 (later):** the one Mac-bound
*correctness* residual — the GPU f16 MPS GEMM numerics — is now also **WITNESSED green on the M3 Pro
verify node** (`FAK_METAL_MPS=1 go test ./internal/model -tags fakmetal -run
TestPrefillQwen35HybridMetalMatchesCPU` → `argmax=61 cos=0.999377 maxRel=0.05078 OK`, PASS; see §3
step 2 for the `FAK_METAL_MPS=1` opt-in this needs over headless SSH), so the only residual still open
is the on-device **throughput** (§3 step 3) — and that number is structurally q4_k-bound (the f16 twin's ≈ 54 GB working set overflows the 36 GiB box; it rides #70 / #1085), so #71's own deliverable is closeable as *implementation + on-device-correctness shipped*.** This is the *status* half
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

> **Reconciled 2026-06-28** — this note's first draft (`5207aafb`, then `4c682b48`) was written
> while the Metal twin/gate/stub/`kv.go` dispatch were still uncommitted in the model-lane working
> tree. They have since landed on `main` in `5c065118`; §1's table and §3 below are updated to that
> committed state, so the sole open residual is now the Mac M3 Pro `fakmetal` GPU witness — no
> model-lane commit is outstanding.

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
2. **`fakmetal` build + GPU-numerics parity witness — NOW GREEN on the M3 Pro (closes the last
   correctness residual).**
   That parity test is now **authored** — `TestPrefillQwen35HybridMetalMatchesCPU`
   (`internal/model/metal_prefill_hybrid_test.go`, `-tags fakmetal`, committed `0ffae0be`,
   `dos commit-audit` `diff-witnessed`). It mirrors `TestMetalDecodeResidentMatchesCPU` (#67's
   resident-decode f16-vs-CPU-Q8 gate) and `TestPrefillQwen35HybridQ4KMatchesTokenLoop`: it runs
   the Metal twin (`s.Metal` → `prefillBatchedMetalQwen35Hybrid`) and the proven CPU Q8 hybrid
   prefill (`s.Quant` → `prefillQwen35HybridQ`) on the same 16-token prompt — both read the
   identical Q8 store, so the only divergence is the projection backend — and asserts the logits
   (cosine ≥ 0.999, same argmax) plus the per-full-attention-layer KV cache match within the
   GPU f16 GEMM accumulation-order drift band. Because §2 already proves the CPU-side orchestration
   host-independently, this Mac run adds the **one** residual it cannot reach: the GPU f16 GEMM
   numerics. **WITNESSED green 2026-06-28** on the M3 Pro verify node (`node-macos-a`, arm64,
   `go1.26.0`, fresh `git clone` at this `main` HEAD `9bd7342`, `-tags fakmetal`):

   ```
   $ FAK_METAL_MPS=1 go test ./internal/model -tags fakmetal \
       -run TestPrefillQwen35HybridMetalMatchesCPU -count=1 -v
   === RUN   TestPrefillQwen35HybridMetalMatchesCPU
       metal_prefill_hybrid_test.go:67: metal hybrid prefill logits: argmax=61 cos=0.999377 maxRel=0.05078 OK
   --- PASS: TestPrefillQwen35HybridMetalMatchesCPU (0.17s)
   ok  github.com/anthony-chaudhary/fak/internal/model
   ```

   So the on-device f16 MPS GEMM the twin substitutes for the projection/MLP matmuls reproduces the
   proven CPU Q8 hybrid prefill's final-token distribution at **cosine 0.999377, same argmax**, and
   advances the same per-full-attention-layer KV cache within the f16-GEMM drift band — the last
   correctness residual is closed.

   **Correction to the "ONE command" (operator-critical, witnessed the same run):** over a *headless
   SSH* session the bare command **fails**, not skips — `metalgemm.Available()` is true (the q4_k MSL
   path keeps the device live), so the test runs, but the f16 path's `mg_matmul` finds `gMPSOK==0`
   and returns zeros (`mg_matmul: MPS unavailable (FAK_METAL_MPS not set …); the f16 GEMM is
   disabled`), so the full-attention K/Kraw/V come out cosine `0.000000` and logits `cos=0.569` →
   FAIL. The MPS f16 GEMM must be opted in with **`FAK_METAL_MPS=1`** (which makes `mg_init` probe
   `MPSSupportsMTLDevice`); with that env set the probe succeeded over SSH on this run and the test
   passed as shown. (This partly revises the earlier note that `MPSSupportsMTLDevice` *hard-faults*
   over SSH — for this small synthetic parity test on this box it returned cleanly; reconcile against
   the 27B q4_k session if the fault recurs there.) The default-build exclusion + non-`fakmetal`
   symbol usages remain green on the `windows/amd64`, `CGO_ENABLED=0` author host.
3. **On-device tok/s measure.** `FAK_QPROFILE=1` prefill of `Qwen3.6-27B.q4_k_m.gguf` at pp22,
   against the `51.55 tok/s` llama.cpp-Metal bar, captured **without a co-resident llama-server**
   (the swap-contamination rule: on a 36 GiB box two 27B copies page to swap — trust the clean
   isolated number). Record the `[metalprof-hybrid …]` line in this cluster; §5 of the #65
   decision doc then reads the arm-A recurrence fraction directly off it.

**Next checkable step:** steps (1) and (2) are now **done** — the model-lane code landed
(`5c065118`) and the Mac GPU-numerics parity gate ran **green** on the M3 Pro this session
(`cos=0.999377`, PASS, §3 step 2, with `FAK_METAL_MPS=1`). The ONLY residual still open is step (3),
the on-device **throughput**: an `FAK_QPROFILE=1` pp22 prefill of the real `Qwen3.6-27B.q4_k_m.gguf`
through this f16 twin against the `51.55 tok/s` llama.cpp-Metal bar, capturing the
`[metalprof-hybrid]` line here. **That throughput run cannot be served by *this* f16 twin on *this*
box** — it is a structural capacity wall, not a deferred run. The twin holds every projection
f16-resident on the GPU (`internal/metalgemm/metalgemm.go:57`), dequantized from the Q8 CPU store,
and the codebase's own sizing constant records f16 ≈ 54 GB for the 27B (`internal/metalgemm/q4k.go:4`),
which overflows the 36 GiB unified pool by ~1.5× (worse once the resident Q8 store is counted). The
in-budget route is q4_k_m ≈ 16 GB, so the **SOTA-throughput lever is the pure-MSL q4_k device GEMM
tracked in sibling #70 / #1085** (witnessed warm 2.6 tok/s @ P=27 → 7.3 tok/s @ P=940 on 2026-06-28),
not this f16 correctness twin — the throughput number rides those q4_k siblings permanently.
So #71's *implementation + on-device correctness* — its actual deliverable (*lift `requirePreNorm`*
+ *build the hybrid twin*) — is **done and witnessed**; its *throughput* number is the one residual,
and because that number is structurally unreachable through this f16 twin (above) it rides the q4_k
siblings #70 / #1085 rather than any further #71 work. **Operator recommendation:** #71 is closeable
as *implementation + on-device-correctness shipped*, with the SOTA-throughput close tracked where it
can actually land (#70 / #1085) — leaving it open pends a pp22 number this twin can never produce on
the 36 GiB box. The host-independent slice — the recipe, its committed backend-agnostic core, the
committed Metal twin/gate/stub/`kv.go` wiring, and that core's green CPU-side correctness witness —
plus the now-green Mac f16 GEMM parity, is **done**.

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
