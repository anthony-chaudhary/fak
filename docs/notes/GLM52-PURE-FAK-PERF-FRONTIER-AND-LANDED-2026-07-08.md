---
title: "GLM-5.2 pure-fak performance: the true frontier (witness, not wiring) + what landed 2026-07-08"
description: "An evidence-grounded audit of the GLM-5.2 pure-fak performance levers — which are shipped, which are witnessed, which are genuinely missing — plus the code that landed this session to make the pure-fak decode number DISCOVERABLE (glmdsatput -> benchcli ledger; recorder auto-land), the file-anchored plans for the three hard levers (MTP self-speculation, constrained tool-JSON decode, continuous-batching onto the resident chat serve) with their minimal GPU-free landable slices, and the exact box commands owed. Names no served tok/s — the sole WITNESSED anchor stays 23.2 tok/s single-stream."
---

# GLM-5.2 pure-fak performance: the true frontier, and what landed

_2026-07-08. Companion to the [costed L1–L10 lever map](GLM52-L1-L10-COSTED-LEVER-MAP-2026-07-06.md),
the [pure-fak benchmark gap](GLM52-PURE-FAK-BENCHMARK-GAP-2026-07-06.md), and the
[native throughput plan](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md). This note
does not produce a tok/s — the sole WITNESSED anchor stays **23.2 tok/s single-stream**
(llama.cpp 8-GPU resident). It records the audited state and the code that made the pure-fak
number landable._

## TL;DR — the finding that reframes the work

The question "what known concepts are we not fully using for GLM-5.2 pure-fak?" has a
surprising answer: **almost all of them are already wired. The bottleneck is not wiring — it
is witness.** The L1 row-split knob, the L4 flash-attn/CUDA-graph knobs, the L8 bench harness,
and a native continuous-batching scheduler are all shipped and tested. But:

- **Zero measured lever tok/s exist.** `experiments/benchmark/runs/by-machine/` has **no
  `gpu-server-3` directory** — not one row-vs-layer, FA-on-vs-off, or batched-throughput number.
  The only measured decode is the **23.498 tok/s layer baseline** from before the row-split
  toggle existed (`experiments/glm-gpu-witness/mgpu-glm52-fullgpu-serve-witness-2026-06-29.json`).
- **L1 (#3075) was closed COMPLETED on the *wiring* landing**, while its acceptance bar (a
  recorded A/B artifact) was never met — a false-close the L4 triage explicitly warns about.
- **Three measured pure-fak numbers are stranded** outside the discoverable ledger (below).

So the highest-value work is not "invent a new optimization" — it is **run the levers that are
already built and land the witnesses**, and **close the last mile that keeps measured numbers
out of the ledger**. The second half is GPU-free; that is what this session landed.

## What landed this session (GPU-free, build+test verified)

### 1. `glmdsatput -out` → the pure-fak decode number is now discoverable (closes both notes' P1)

`cmd/glmdsatput` measures fak's own `glm_moe_dsa` decode on real kernels (bit-exact, cosine
1.000000), but it only printed a bespoke `glm-throughput/1` line that **no index recognizes** —
so the pure-fak path was measured yet undiscoverable (zero artifacts under the runs tree). The
4-config synthetic sweep (`26.53/15.80/13.44/8.49` decode tok/s) has been sitting on private
box scratch as `glmw116b7ed250b7.result` since 2026-06-25, named "a P1" in two notes.

Added a `-out <dir>` flag (`cmd/glmdsatput/main.go`) that writes, per run:
- `manifest.json` — the `glm-throughput/1` body wrapped **verbatim** (the load-bearing `scope`
  caveat survives) in a `benchcli` lineage + `benchmark_artifact` envelope via
  `benchcli.WriteReport`, so it folds into `benchcli.BuildLineageIndex` and is bindable by
  `dos verify`;
- `result.json` (raw record) + `RESULTS.md` (scope-forward human page).

Witness (GPU-free): new `TestWriteLedgerArtifactIsDiscoverable` asserts `benchcli.DecodeArtifact`
recognizes the manifest, `BuildLineageIndex` folds **exactly one** artifact (result.json is not
double-counted), and `scope` survives. `go test ./cmd/glmdsatput/` green; `gofmt`/`go vet` clean;
a real host-backend run emits a discoverable manifest. Adversarially reviewed → verdict
`correct`, seam right, no regressions.

### 2. Recorder auto-lands the sweep (closes the structural half of the stranding)

`tools/dgx_glm_throughput_run.sh` drove the synthetic sweep with `-json` only, folding to a
`/tmp` fetch record — so **every on-box sweep kept stranding** even after glmdsatput learned to
land. Wired `-out` into the sweep (one discoverable subdir per config) plus lineage exports
(`FAK_BENCH_COMMIT`/`FAK_BENCH_NODE`), default-on under
`experiments/benchmark/runs/by-machine/<node>/<UTC>-glm52-native-decode/`, opt-out with
`GLM_LAND_DIR=""`. The existing `/tmp` fetch pipeline is untouched (the `-json` line the
recorder folds is unchanged). `bash -n` clean.

### 3. `internal/guideddecode` (slice-1 of constrained tool-JSON decode, #26)

_Status recorded separately at the bottom once its build/test is confirmed this session._

## The costed lever program — audited status (2026-07-08)

Every multiplier is COMPUTED/ESTIMATED off the single WITNESSED 23.2 tok/s anchor; none is
served. "shipped-script" = the wiring/driver exists; "witnessed" = a recorded
`experiments/benchmark/runs` result exists.

| Lever | What | Wiring | Witnessed | The real gap |
|---|---|---|:--:|---|
| **LF** #3074 | active-set from GGUF header | triage-only | ✗ | box-agnostic (server 2); re-derives every ceiling; cheapest |
| **L8** #3082 | serve+bench harness | **shipped** | n/a | force-multiplier (not a tok/s lever) — `glm52_bench_lever.sh` |
| **L1** #3075 | 8-GPU tensor/row split | **shipped** | ✗ | **run the A/B** — dominant ~3–6× (real ~3× on MoE); base topology L2/L3/L4/L5/L7 stack on |
| **L2** #3079 | continuous batching + KV | scheduler shipped, not on serve | ✗ | 10–40× aggregate; scheduler exists but not on the chat serve (see plan) |
| **L4** #3076 | flash-attn + CUDA graphs | **shipped** | ✗ | run the A/B; graphs may read inert until L1 removes the layer-split bubble |
| **L3** #3078 | speculative decoding | triage-only | ✗ | n-gram arm unblocked; no draft checkpoint; MTP head dropped (see plan) |
| **L9** #3085/6 | real prefill path (chunked+FA) | needs harness | ✗ | prefill axis; no prefill-sweep script yet |
| **L5** #3077 | decode quant sweep | triage-only | ✗ | 1.1–1.5× |
| **L6** #3087 | INT8 tensor-core expert GEMM | kernel unwritten | ✗ | up to ~2× agg; blocked-by L2; fak int8 GEMM accumulates on scalar SIMT not tensor cores |
| **L10** #3089/#1482 | native fak resident-EP | script exists | ✗ | separate engine-honest row vs 23.2 |
| **L7** #3088 | true DSA sparsity | hardware-floored | ✗ | **sm_90 kernel floor** — un-winnable on sm_80; only the full-MLA ctx curve is measurable now |

**Hardware ceiling (name it, don't backlog it):** both lab boxes are **sm_80**. FP8 experts and
GLM-5.2's native DSA sparse-attention kernel both need **sm_90 (Hopper)**; on sm_80 GLM-5.2 is
served as full-MLA and L6 falls back to int8. Those two levers are hardware-gated, not
wiring-gated.

## The three stranded pure-fak numbers (land them)

None is discoverable by `benchcli.BuildLineageIndex`:

1. **Headline (both notes' P1):** 4-config synthetic `glmdsatput` sweep — decode
   `26.53/15.80/13.44/8.49` tok/s (HEAD `b68a182`), on private box scratch
   `glmw116b7ed250b7.result`.
2. 3-config small-context bisection — decode `49.78/56.61/40.38` tok/s.
3. **Real-weight (most consequential):** the 466 GB in-kernel serve witness — load 150 s / 136 s
   (3.04/3.75 GB/s), decode ~0.2 tok/s (HEAD `6d727be7`) — the only non-synthetic pure-fak
   decode number, live only in
   [GLM52-FAK-NATIVE-SERVE-LOAD-SPEED](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md).

With this session's wiring, (1) lands GPU-free-mechanically by re-running the sweep with `-out`
on the sm_80 CUDA box (the two largest configs still hit the P0 DSA illegal-memory-access), or
by scrubbing + wrapping the stranded `.result` through `benchcli.MarshalReport` (needs off-box
access to the private scratch file).

## The three hard levers — file-anchored plans + minimal GPU-free slices

### C2 — GLM MTP self-speculation (missing by construction)
GLM-5.2's own MTP head (`.nextn.*` in GGUF; `mtp.*` in safetensors) is **dropped at load in all
four loader paths** (`internal/model/safetensors.go:356`, `safetensors_quant.go:375`,
`internal/ggufload/gguf_glm_tensors.go:132` via `gguf_weightsource.go:180,331` +
`quant_q4k_loader.go:188`) and byte-accounting (`estimate.go:308`), referenced by **zero**
forward code, and `verify.go:69` excludes `glm_moe_dsa` from the batched verify (sequential
fallback only). The generic accept core exists (`internal/polymodel` `AcceptGreedy:410`/
`AcceptTree:473`, `model.VerifyForward`) but drafts from a **separate** co-resident model and is
`FAK_POLYMODEL`-gated off.
- **Minimal GPU-free slice:** retain `.nextn`/`mtp.` tensors behind a `Config` flag (touch the
  four skip callsites + estimator), add the `nextn` config fields, and a unit test asserting
  retention-on / byte-unchanged-drop-off (inverse of `glm_test.go:96`). Plus a GLM-facing
  wrapper test over the shipped `AcceptGreedy` on synthetic logits. **Caveat:** the MTP-head
  sub-tensor spellings are PROVISIONAL — real acceptance is checkpoint/box-gated. This slice is
  scaffolding, not a speedup.

### C4 — constrained tool-JSON decode (seam shipped, compiler missing; #26/#929)
The decode-boundary seam ships (`internal/model/constraint.go`: `LogitMask`, `StepMask`,
`GenerateConstrained`, `FAK_NATIVE_GUIDED_DECODE` default off; the logit_bias half is wired into
the live loop at `internal/agent/inkernel_planner.go:1182`). Ride-mode passthrough to vLLM/SGLang
works. **Missing:** the tokenizer-aware **schema→per-step-token-mask compiler** (#26 umbrella;
#929 shipped only the seam). `internal/grammar` is post-hoc arg-shape repair, not a decode-time
compiler.
- **Minimal GPU-free slice (landing this session):** `internal/guideddecode` — a sound
  byte-level `AllowedNextBytes(prefix, tools)` FSM over the canonical
  `{"name":"<enum>","arguments":…}` envelope, table-driven tested. Next slices: a `model.LogitMask`
  adapter needing a new `tokenizer.TokenBytes(id)` accessor, then wiring the mask into the agent
  sampler (must apply inside `sampleLogitsWithPenalty` to preserve stochastic sampling, not route
  through the greedy `GenerateConstrained`).

### C5 — continuous batching onto the resident chat serve (#401, L2)
A real native continuous-batching scheduler is **shipped and correct GPU-free**
(`internal/modelengine/nativesched.go` — one `BatchSession.StepBatch` per loop iteration, FIFO
admit/retire), reachable from the gateway **only via `/v1/fak/syscall`** with a fixed 16-token
decode. The **resident GLM chat serve** (`/v1/chat/completions`, `/v1/messages`) decodes
**serially**, one `Session` per request, through `agent.InKernelPlanner`
(`inkernel_planner.go:1205`), holding `devMu` across the whole forward. Worse, even via the
scheduler GLM gets **no weight-stream amortization**: `batchPreNormFastPathOK`
(`internal/model/batch.go:109`) excludes `IsMoE()` **and** `isGLMMoeDsa()`, so GLM falls to the
per-user serial fallback (bit-exact but no speedup) — the fused MoE+DSA batched GEMM is the named
open speed lever.
- **Minimal GPU-free slice:** extract the planner decode into a lane + single-step, add an opt-in
  (`FAK_INKERNEL_BATCH`) multi-lane loop over `BatchSession.StepBatchActive`, and a synthetic
  equivalence test (each concurrent lane's tokens bit-identical to serial). Lands the *wiring* on
  the chat serve; the batched `glm_moe_dsa` kernel (also CPU-testable via `batch_glm_test.go`)
  supplies the actual throughput. Aggregate curve is box-gated.

## Ready-to-dispatch box commands (the witnesses owed)

On the resident 8-GPU sm_80 host (GPU server 3), step-0 is the endpoint gate (the driver self-runs it):

```bash
# L1 — the dominant lever, unmeasured. Produces l1-ab-verdict.json (layer vs row + gpus-busy):
DEVICES=0,1,2,3,4,5,6,7 bash tools/glm52_l1_rowsplit_ab.sh
# then commit the verdict + bench legs under experiments/benchmark/runs/by-machine/gpu-server-3/…

# Land the headline pure-fak sweep discoverably (now that -out is wired), per config:
go run -tags cuda ./cmd/glmdsatput -backend cuda \
  -layers 8 -hidden 2048 -heads 16 -inter 8192 -index-topk 256 \
  -decode-prompt 512 -decode-steps 64 -decode-reps 5 -json \
  -out experiments/benchmark/runs/by-machine/<node>/$(date -u +%Y%m%dT%H%M%SZ)-glm52-native-decode
# or drive the whole sweep (auto-lands): bash tools/dgx_glm_throughput_run.sh <nonce> auto
```

## Honesty fences

- No served tok/s is claimed here. Every lever multiplier is COMPUTED/ESTIMATED off 23.2.
- The landed code is WITNESSED at the **wiring** level (build+unit-test+e2e), not at the
  throughput level — it makes numbers *landable/discoverable*, it does not make them faster.
- "L1/L4 shipped" means the *knob/driver* shipped, not that the lever is proven; #3075's close is
  not backed by an artifact.
- The MTP and cont-batch minimal slices are correctness scaffolding; their speedups are
  box-gated and must not be quoted before a recorded run.
