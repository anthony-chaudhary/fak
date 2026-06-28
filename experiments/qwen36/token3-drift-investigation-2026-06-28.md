# Qwen3.6-27B token-3 correctness drift — investigation + per-layer divergence-probe design (2026-06-28)

**Status: DESIGN / spec doc — host-independent.** This note is the *correctness*
companion to the speed work. The speed gap (decode 0.1→0.9→1.2 vs the 7.29 tok/s
llama.cpp-Metal bar) is kernel optimization; the **token-3 drift is the real
"true parity" blocker** — fak and llama.cpp agree for two tokens then disagree on
the third, so fak is computing a *different model*, not a slower one. This doc
(1) restates the phenomenon precisely, (2) ranks the candidate compounding-float
sources against the *actual* `internal/model/qwen35.go` ops, (3) designs a
**per-layer divergence probe** that turns "drifts at token 3" into "first
diverges at layer L, op O", (4) weighs a scaled-fixture reproduction, and (5)
lists next checkable steps with the host each needs.

Honesty rider ([`../../docs/proofs/00-METHOD.md`](../../docs/proofs/00-METHOD.md)):
the orchestrator host is `win32`, **no Apple Silicon, no NVIDIA GPU, no 27B
artifact**. No 27B or GPU run is claimed here. The probe *logic* is unit-testable
on this host against the tiny fixture; the 27B capture is Mac/artifact-gated and
named as such. Where a llama.cpp flag is uncertain it is marked
"uncertain — next checkable step is X" rather than asserted.

Sibling evidence in this cluster:
[`../../docs/benchmarks/QWEN36-PARITY-RESULTS.md`](../../docs/benchmarks/QWEN36-PARITY-RESULTS.md)
(the witnessed token ids + the "Token-3 drift RE-DIAGNOSED" section this builds on)
and [`metal-gdn-recurrence-decision-2026-06-28.md`](metal-gdn-recurrence-decision-2026-06-28.md)
(why the GDN recurrence stays on the CPU — the same recurrence this doc suspects).

---

## 1 — The phenomenon, precisely

On the fixed 22-token ChatML prompt (`#93` real-artifact oracle), greedy decode:

| engine / path | tokens | text |
|---|---|---|
| **llama.cpp b9707** (Metal, q4_k_m) | `[248068, 198, 90700]` | `<think>\nThinking` |
| **fak GGUF→Q8** (CPU) | `[248068, 198, 8160]` | `<think>\nHere's` (drifts at tok 3) |
| **fak resident-q4k** (CPU, no Q8 round-trip) | tok 3 = `Here's a thinking` | drifts at tok 3 |

Both fak paths **match for tokens 1–2 (`248068`, `198`) then diverge on token 3.**
(`QWEN36-PARITY-RESULTS.md`: "fak's current GGUF->Q8 path returns `[248068, 198,
8160]`" and "the resident-q4k path … **also drifts at token 3** … 2-token match
then divergence … artifact `experiments/model-ladder/qwen36-resident-q4k-parity-20260619.json`".)

Two facts bound the cause:

- **The arch math is correct.** The tiny text-only `qwen3_5` fixture
  (`internal/model/make_qwen35_tiny.py`: 3 Gated-DeltaNet + 1 gated full-attn
  layer, hidden 32, head_dim 8) is **bit-exact vs HF transformers** —
  `TestOptionalQwen35HybridOracleForwardMatchesHF` (`oracle_test.go:1401`)
  asserts per-layer hidden cosine ≥ 0.9999 (it reports 1.000000, max|Δ|~4e-9) and
  argmax parity at every position. So `linearAttnSeq` / the gated full-attn path
  reproduce the reference *recurrence, output gate, qk-norm, partial RoPE, (1+w)
  RMSNorm* exactly at fixture scale.
- **It is not a Q8 round-trip quant artifact.** The drift survives the move to
  native q4_k resident weights (`QWEN36-PARITY-RESULTS.md`, RE-DIAGNOSED
  section); and it is not a reference-path bug (the tiny fixture is bit-exact).

So the residual is a **kernel-numerics divergence at 27B scale on the hybrid GDN
path** — small float differences vs llama.cpp's kernels that are negligible for
two tokens and tip an argmax on the third. The "2 tokens then diverge" signature
is exactly what a *recurrent* error source produces: the GDN state carries error
forward, and at decode-token-3 enough has accumulated (across 48 GDN layers ×
prior positions) to flip a near-tie logit. **Important caveat:** llama.cpp is not
a bit-exact oracle the way HF-eager is — it uses its own quant dequant order, FMA
fusion, and SIMD reductions, so a per-op cosine of ~1−1e-4 between fak and
llama.cpp is *expected and acceptable*. The probe's job (§3) is to find the layer
where the divergence is **anomalously large** (a real algorithmic/ordering
mismatch), not merely nonzero.

---

## 2 — Hypothesis ranking (tied to real `qwen35.go` symbols)

Ranked by plausibility *as the first anomalous (algorithmic, not rounding)
divergence*. Each names the function and the lines, why it would match for 2
tokens then diverge, and what confirms/refutes it. Line numbers are at the commit
this doc is written against.

### H1 — GDN delta-rule recurrent scan: accumulation order + state carry (HIGH)
`linearAttnStep` (`qwen35.go:399`), inner `headStep` (`qwen35.go:481-517`); prefill
twins `linearAttnSeq` (`qwen35.go:332-383`) and `prefillQwen35LinearLayerMM`
(`metal_prefill_hybrid_core.go:202-246`).

The delta rule is a chain of rank-1 updates: `st[i,d] *= g`; `kvmem[d] = Σ_i
st[i,d]·k[i]`; `delta[d] = (v[d]−kvmem[d])·β`; then `st[i,d] += k[i]·delta[d]` and
`out[d] += st[i,d]·q[i]`, accumulated in a **fixed i-then-d scalar order**
(`qwen35.go:497-516`). This is the single most plausible *first-anomalous* op
because:
- it is the **only stateful op** — `st` (`lst.recurrent[h]`, `[kHd·vHd]` f32)
  persists across tokens, so a per-step reduction-order difference vs llama.cpp's
  `ggml_metal_kargs_gated_delta_net` kernel **compounds**, matching the "2 tokens
  then diverge" signature exactly.
- the **reduction order is fak-specific**: fak does serial `+=` over `i∈[0,kHd)`
  in f32; llama.cpp's kernel almost certainly uses a different lane/threadgroup
  reduction order (and possibly f32 accumulation of f16 products). Same math,
  different rounding — which is invisible for 2 tokens and decisive on the 3rd.
- the decay `g = exp(−exp(A_log)·softplus(a+dt_bias))` (`qwen35.go:488-490`) is a
  near-1 multiplier; tiny errors in `g` re-multiply the *entire* state every step,
  so they don't cancel — they integrate.

*Confirms it:* the probe (§3) shows the first large per-layer cosine drop on a
`linear_attention` layer, and a per-op tap shows the divergence enters at the
recurrent-scan output (`core[t]`), not at the conv or the projections.
*Refutes it:* the first drop is on a `full_attention` layer, or on a linear layer
but already present at `convOut` before the scan.

### H2 — per-head q/k L2-norm + 1/√kHd query scale ordering (HIGH–MED)
`l2normInto` (`qwen35.go:108-117`) + the scale loop (`qwen35.go:465-471`,
`metal_prefill_hybrid_core.go:193-199`).

fak L2-normalizes each `kHd`-wide head with `inv = 1/√(Σx²+1e-6)` (sum, not mean;
eps **inside** the sqrt), then multiplies **only q** by `scale = 1/√kHd`. Two
divergence surfaces: (a) the `+1e-6` eps placement and the f32 `Σx²` reduction
order vs llama.cpp's `l2norm` (FLA uses `rsqrt`; rounding of `rsqrt` vs
`1/sqrt` differs); (b) whether the query scale is folded **before or after** the
L2-norm. fak applies `scale` *after* normalizing q (`qwen35.go:468-470`). A
near-tie at token 3 is sensitive to this because q/k feed directly into the
recurrent `kvmem`/`out` products (H1) — so H2 errors are *amplified by* H1.
Matches "2 then diverge" only weakly on its own (it is not stateful), but it
**feeds** the stateful op, so a small steady q/k bias accumulates through `st`.

*Confirms/refutes:* per-op tap on `qNorm`/`kNorm` (pre-scan) vs llama.cpp's
normalized q/k — if those already diverge before the scan, H2 is upstream of H1.

### H3 — gated RMSNorm of the readout `(1+w)` vs plain-weight confusion (MED)
`rmsNormGatedInPlace` (`qwen35.go:122-131`) vs the ordinary `(1+w)` norm
`applyRMSNormInPlaceCfg` (`arch.go:238-251`).

The GDN readout uses `x = weight·(x·rsqrt(mean(x²)+eps))·silu(gate)` with the
**plain** ones-init `norm.weight` (NOT the `(1+w)` form), while every *other*
norm in the qwen35 stack (`input/post/q/k/final`) uses `(1+w)`
(`cfg.NormGain1p`, `arch.go:246`). The fixture is specifically built to catch a
`(1+w)`-vs-plain mix-up (`make_qwen35_tiny.py:79-82` perturbs the norm weights so
the distinction is non-trivial), and it passes — so a *gross* mix-up is already
refuted. The residual risk is **mean-vs-sum** or eps placement in the gated norm
at 27B `vHd=128` width vs llama.cpp. Lower than H1/H2 because it is per-token
(not stateful) and fixture-guarded.

*Confirms/refutes:* per-op tap on `core[t]` before vs after `rmsNormGatedInPlace`.

### H4 — depthwise causal conv1d + SiLU window (MED–LOW)
`linearAttnStep` conv loop (`qwen35.go:435-452`), prefill
(`metal_prefill_hybrid_core.go:145-169`).

K=4 causal depthwise conv over `concat(q,k,v)`, left-padded, then SiLU. The decode
path reads the persistent `lst.conv` window (`qwen35.go:443-447`) and prefill
seeds it. A boundary/index bug here would mis-feed the scan — but a *bug* would
likely break token 2 too, and it does not. The live risk is only the SiLU/`acc`
f32 rounding vs llama.cpp's `ssm_conv` kernel, which is small and per-token. Ranked
below H1–H3.

*Confirms/refutes:* per-op tap on `convOut` vs llama.cpp's post-conv activations.

### H5 — partial-RoPE 0.25 rotation (LOW, full-attn layers only)
`applyRopeRow` (`rope.go:191-206`), `rotaryDim()` (`rope.go:49-64`),
`invFreqDenom()` (`rope.go:39-47`).

Only the leading `int(head_dim·0.25)=64` of 256 lanes rotate; `invFreqDenom()`
returns `rotaryDim()` for the qwen35 family. `applyRopeRow` **deliberately pins
each product to f32 (`float32(a*cos)−float32(b*sin)`) to block FMA fusion**
(`rope.go:197-205`) — this was the fix for an Apple-Silicon reposition drift, so
RoPE is already hardened for cross-arch determinism. Only the 16 full-attn layers
use it. Lower than the GDN hypotheses because it is non-stateful, fixture-tested,
and FMA-hardened.

### H6 — mRoPE section split `[11,11,10]` (REFUTED for the text path)
The task brief lists mRoPE interleave as a candidate, but the code shows the
**text forward collapses mRoPE to plain partial-RoPE**: `invFreqDenom()` returns
`rotaryDim()` with no section interleave, and `TestOptionalOrnithOracleForwardMatchesHF`
(`qwen35_ornith_oracle_test.go:100-113`) explicitly guards that
`rope_dim_per_layer` is **uniform** ("mrope section interleave leaked into the
text path" is the asserted *failure*). So the `[11,11,10]` multimodal section
split is **not on the text decode path** and is not a candidate for this drift.
Recorded here so the probe does not chase it.

### H7 — grouped-vs-ungrouped reduction order in the projection GEMMs (LOW)
The `metal_prefill_hybrid_core.go` header (lines 7-10) documents a known
"grouped-vs-ungrouped float-order drift" between the Q8 CPU GEMM and the Metal
GEMM. This is a *per-element* rounding difference in the projections, ~1 ULP, and
is the expected llama.cpp↔fak baseline noise — not an anomalous algorithmic
divergence. Ranked lowest; it is the *floor* the probe's threshold must sit above.

**Ranking summary:** H1 (recurrent scan / state carry) ≫ H2 (q/k L2-norm feeding
the scan) > H3 (gated RMSNorm) > H4 (conv) > H5 (partial-RoPE) ≫ H6 (refuted),
H7 (baseline noise). The two stateful/state-feeding ops (H1, H2) are the only ones
whose error model *integrates* across tokens, which is what "match 2, diverge on 3"
demands.

---

## 3 — The per-layer divergence probe (the high-value design)

**Goal:** turn "drifts at token 3" into "first diverges at layer L, op O."
**Method:** dump per-layer (and within the suspect layer, per-op) hidden states
from *both* engines on the *same fixed prompt at the same decode step*, compare
with cosine + max|Δ|, and report the first layer whose cosine drops below a
noise-floor threshold — then within that layer the first op that does.

### 3a — What it dumps from llama.cpp (the reference)
The 27B (b9707, q4_k_m), the fixed 22-token ChatML prompt, **decode step that
produces token 3** (i.e. prompt + the 2 agreed tokens `248068, 198` already in
context, predicting the 3rd). Per `linear_attention` / `full_attention` decoder
layer: the **residual-stream hidden state after that layer** (`[hidden=5120]`).

Mechanism — **uncertain, next checkable step named.** llama.cpp exposes layer
activations through (a) `--verbose` / `LLAMA_LOG`-level tensor printing, (b) the
`ggml` eval-callback (`ggml_backend_sched_set_eval_callback` /
`llama_set_eval_callback`-style hook used by `llama-eval-callback`), or (c) a
`GGML_GRAPH_DUMP` of the compute graph. The eval-callback path is the most likely
to give clean per-layer `l_out`/`ffn_out` tensors by name. **The exact b9707 flag
+ tensor-name filter is uncertain from this host.** *Next checkable step:* on the
Mac artifact node, run `llama-eval-callback --help` (or grep `b9707` source for
`eval_callback`) and confirm which hook yields `blk.<l>.<...>` residual tensors;
record the chosen mechanism in this cluster before the capture.

Output: `llamacpp_layers.f32` = `[n_layers, hidden]` for the token-3 step + a
sidecar `llamacpp_meta.json` (`{commit, prompt_ids, decode_step, layer_names}`).

### 3b — What it dumps from fak (minimal instrumentation — a named follow-on)
There is **no per-layer hidden tap in `qwen35.go` today** (grep over
`internal/model` for `FAK_*` env hooks finds quant/metal/profiling gates but no
hidden-state dump). Propose an **env-gated tap** — `FAK_HIDDEN_TAP=<dir>` — added
in the decode forward at the residual-stream boundary, i.e. right after each
layer's `X[i] += o[i]` / `X[i] += Down[i]` in the per-token step path (the decode
twin of `prefillQwen35HybridViaMM`'s loop, `metal_prefill_hybrid_core.go:48-106`).
When set, write `X` (`[hidden]`) for every layer at the configured decode step to
`<dir>/fak_layer_<l>.f32`. **Within the first-diverging layer**, a finer tap
writes the GDN intermediates already named by the existing `phaseStart/phaseEnd`
spans (`qwen35.go:417-541`): `convOut` (after `qwen35_linear_step_conv`),
`qNorm/kNorm` (after `qwen35_linear_step_qk_norm`), `core` pre- and
post-`rmsNormGatedInPlace` (`qwen35_linear_step_recurrent` / `_gated_norm`), and
`out` (after `_out_proj`). These span names already exist — the tap reuses them as
op labels, so the per-op localization needs no new structure, only a write.

**This doc does not add the tap** (DESIGN only; the trunk forward stays
unchanged). The tap is the named follow-on `#<TBD> qwen35 FAK_HIDDEN_TAP per-layer
hidden dump`.

### 3c — The comparison + first-divergence finder (unit-testable HERE)
A small comparator (`cmd/` tool or a `_test.go`) loads both layer dumps and computes:
- `cos_l = cosine(fak_layer_l, llama_layer_l)` and `maxabs_l = max|Δ|` for each l;
- `firstDivergeLayer = min{ l : cos_l < THRESHOLD }`, where THRESHOLD is set above
  the H7 baseline noise (start at 0.9999, the same floor the in-tree oracle uses,
  `oracle_test.go:1547`; widen if the llama.cpp↔fak quant-dequant noise floor is
  higher — record the measured baseline cosine of the *agreeing* early layers and
  set THRESHOLD just below it);
- within `firstDivergeLayer`, the same finder over the per-op taps →
  `firstDivergeOp`.

The comparison **primitives already exist in-tree**: `cosine` (`oracle_test.go:185`)
and `argmax` (`oracle_test.go:195`). The finder is pure arithmetic over two float
slices — **host-independent and unit-testable on this win32 box**: a test can
synthesize two `[n_layers, hidden]` arrays, inject a known divergence at layer 7
op "recurrent", and assert the finder returns `(7, "recurrent")`. That test pins
the probe logic green here, *before* any Mac/27B run, exactly the
fixture-first discipline `export_oracle.py` established.

### 3d — The gradeable witness (JSON schema)
The probe emits `experiments/qwen36/token3-divergence-<commit>.json`:

```json
{
  "schema": "qwen36-token3-divergence/v1",
  "commit": "<git sha of the fak build>",
  "llamacpp_build": "b9707",
  "quant": "q4_k_m",
  "prompt_ids": [/* the fixed 22-token ChatML prompt */],
  "decode_step": 3,
  "threshold": 0.9999,
  "baseline_cosine_floor": 0.99996,
  "per_layer": [
    {"layer": 0, "kind": "linear_attention", "cosine": 1.000000, "max_abs": 3.1e-6},
    {"layer": 7, "kind": "linear_attention", "cosine": 0.9831,   "max_abs": 4.2e-1}
  ],
  "first_divergence_layer": 7,
  "first_divergence_kind": "linear_attention",
  "per_op_in_first_layer": [
    {"op": "qk_norm",   "cosine": 0.99998},
    {"op": "recurrent", "cosine": 0.9831}
  ],
  "first_divergence_op": "recurrent"
}
```

A grader reads `first_divergence_layer` + `first_divergence_op` and checks the
`per_layer` cosine array is monotone-credible (≈1 until the named layer). That
single artifact converts the prose verdict into a checkable witness and, when the
op is fixed, the *same* probe re-run shows `first_divergence_layer = null`
(parity) — a falsifiable closure condition.

---

## 4 — Scaled-fixture reproduction WITHOUT the 27B artifact?

**Verdict: uncertain, leaning feasible-as-a-stressor / not-guaranteed-as-a-repro.**

The idea: build a *medium* `qwen3_5` fixture — more layers (e.g. 16–32, keeping
`full_attention_interval=4`) and wider GDN heads/state than the tiny 4-layer/32-hidden
one — still CPU-runnable, to see whether the compounding error appears without the
27B weights. Honest assessment:

- **What a scaled fixture CAN do:** it can exercise *deeper recurrent compounding*.
  The tiny fixture has only 3 GDN layers and `vHd=kHd=8`; the 27B has 48 GDN layers
  and `vHd=128`. The error model in H1 *integrates with depth and width*, so a
  16–32-layer, `vHd=64`-ish fixture is the right knob to **stress** the recurrent
  accumulation and is trivially buildable by editing `make_qwen35_tiny.py`'s
  `Qwen3_5TextConfig` (`num_hidden_layers`, `linear_*_head_dim`, `hidden_size`).
  It stays CPU-runnable and HF-comparable through the *existing* `export_oracle.py`
  path — so it costs almost nothing to try.
- **Why it likely will NOT reproduce the *llama.cpp* drift:** the tiny-fixture
  oracle is **HF transformers (eager)**, against which fak is **bit-exact**. A
  scaled fixture compared to HF would *also* be ~bit-exact (same kernels, same
  reductions), so it would **not** show a divergence — because the drift is not
  fak-vs-HF, it is **fak-vs-llama.cpp** (two different kernel implementations).
  To reproduce *the actual phenomenon* on a fixture you would need to run that
  fixture through **llama.cpp**, which requires (a) exporting the random fixture to
  GGUF and (b) llama.cpp loading a `qwen35` GGUF — runnable in principle on the Mac
  node, but that re-introduces the llama.cpp dependency the fixture was meant to
  avoid. On a plain CPU box with only HF, a scaled fixture **cannot** surface a
  fak↔llama.cpp divergence.
- **The genuinely useful host-independent variant:** build the scaled fixture and
  run **two fak code paths** against it (e.g. the scalar `linearAttnStep` vs a
  *deliberately reordered* recurrent reduction, or f32 vs a simulated f16-accumulate
  scan) to measure *how fast* a known reduction-order perturbation compounds with
  depth. That **quantifies H1's sensitivity** — "a 1-ULP per-step state error
  reaches argmax-flipping magnitude by layer N / token M" — entirely on this win32
  host, and predicts whether the 27B's 48-layer depth is enough to explain the
  observed token-3 flip. That is feasible **yes**, and is the highest-value
  host-independent experiment short of the Mac capture.

So: a scaled fixture is **feasible and cheap to build**, **useful as an H1
sensitivity stressor** (host-independent), but **not a faithful reproduction of
the fak↔llama.cpp drift** without putting llama.cpp in the loop (Mac-gated).

---

## 5 — Next checkable steps (each with the host capability it needs)

1. **(this host, win32)** Implement the §3c first-divergence finder + its
   synthetic unit test (inject a known divergence, assert `(layer, op)`), reusing
   `cosine`/`argmax`. Pins the probe logic green before any 27B run. *Needs: nothing
   beyond this box.*
2. **(this host, win32)** Build the medium scaled fixture (§4) by editing
   `make_qwen35_tiny.py`, and run the **H1 sensitivity stressor** (two fak
   reduction orders / f32-vs-simulated-f16 scan) to measure how a per-step state
   error compounds with depth × tokens. Reports whether 48 layers can explain a
   token-3 flip. *Needs: CPU + transformers (the existing fixture toolchain).*
3. **(this host, win32 — follow-on commit)** Add the `FAK_HIDDEN_TAP` env-gated
   per-layer + per-op tap to the decode forward (§3b). Code-only; witnessable by a
   unit test that asserts the tap writes one file per layer at the configured step.
   *Needs: nothing beyond this box; it is a `qwen35.go` edit, out of scope for THIS
   design doc.*
4. **(Mac artifact node)** Resolve the llama.cpp b9707 per-layer dump mechanism
   (§3a) — `llama-eval-callback`/eval-callback vs graph dump — and record the
   chosen flag in this cluster. *Needs: the M3 Pro Mac with b9707 + the 27B GGUF.*
5. **(Mac artifact node)** Run both dumps on the fixed 22-token prompt at the
   token-3 decode step, feed them to the §3c finder, and emit the §3d witness
   JSON. This is the step that names `(L, O)`. *Needs: the M3 Pro Mac + the 27B
   q4_k_m artifact + steps 1,3,4 landed.*
6. **(this host, after step 5)** Given the named `(L, O)`, write the targeted fix
   (most likely an H1 reduction-order / accumulation-dtype change in
   `headStep`/`linearAttnSeq`), re-run the in-tree tiny-fixture oracle to confirm
   no regression, and re-run the probe (Mac) to confirm `first_divergence_layer =
   null`. *Fix + tiny-fixture check: this host; final 27B confirmation: Mac.*

---

## 6 — What this is, and is not

This is the host-independent slice of the correctness blocker: a precise restatement,
a hypothesis ranking **bound to real `qwen35.go` symbols** (with mRoPE explicitly
*refuted* for the text path rather than hand-waved in), a probe design whose
comparison logic is **unit-testable on this win32 box today**, a gradeable JSON
witness schema, and an honest read on the scaled-fixture idea. It is **not** a
claim that any 27B or GPU run was performed here, that the first-diverging layer is
already known, or that the tap/probe is implemented — those are the gated/named
follow-ons in §5. The only gate run for *this* change is the doc/commit gate on the
`experiments` lane.
