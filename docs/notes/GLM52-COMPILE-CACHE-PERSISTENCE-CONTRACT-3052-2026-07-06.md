# GLM-5.2 compile-cache persistence contract (#3052), 2026-07-06

**Status:** specification + generation classification. The runtime implementation
is **not yet shipped and not yet witnessed** — this note pins the exact contract a
follow-on serve/infra pass implements in one shot, and classifies #3052's horizon
so it can be dispatched. It does not itself make a second boot fast; it names the
witness that would.

**Scope of this pass:** triage-only. #3052 routed to the `docs` lane and carried no
generation label and no milestone (intake drift). The deliverable here is the
classification + the implementable contract; the code lands under the serve/infra
lanes named in [Decomposition](#decomposition).

## The problem, in one line

The ~500s GLM-5.2 first-request tax is one-time backend **warmup** (axis A in
[`GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md`](GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md)),
not a KV-prefix cache miss (axis B). Today it recurs on **every** boot / VM
re-create because the JIT-compiled kernels are recomputed each process start
instead of being persisted and reused. The warmup work is visible in the bring-up
log ([`GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md`](GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md),
steps 14–16): `Entering DeepGEMM JIT Pre-Compile session`, `Try DeepGEMM JIT
Compiling for <GEMM_NT_F8F8BF16>`, `Capture target verify CUDA graph begin`.

## What actually persists across a restart (and what does not)

This is the load-bearing engineering distinction, because the issue title bundles
three things that have **different** persistence stories. Getting it wrong is how a
"warm cache" boot still pays most of the tax.

| Warmup component | Disk-cacheable across a process restart? | Pin it with |
|---|---|---|
| TorchInductor codegen (`torch.compile` FX → Triton) | **yes** — reused if the cache dir survives | `TORCHINDUCTOR_CACHE_DIR` |
| Triton kernel compile (PTX/cubin) | **yes** | `TRITON_CACHE_DIR` |
| DeepGEMM JIT pre-compile (`GEMM_NT_F8F8BF16`, …) | **yes** — DeepGEMM keys a JIT cache dir | DeepGEMM JIT cache dir (`DG_JIT_CACHE_DIR`, older builds `DG_CACHE_DIR`; **verify against the pinned DeepGEMM version** — this is the one env name that has drifted) |
| vLLM `torch.compile` artifact cache | **yes** | `VLLM_CACHE_ROOT` (default `~/.cache/vllm`) — also honors `TORCHINDUCTOR_CACHE_DIR` |
| **CUDA graph capture** | **no** — captured into device memory at warmup, not serialized to disk by SGLang/vLLM | (re-done every boot; see below) |
| Model weight load into VRAM | **no** (VRAM is not persistent) — but weight *bytes* can avoid re-download via `HF_HOME` / `HF_HUB_CACHE` on a persistent disk | `HF_HOME` / `HF_HUB_CACHE` |

**The honest ceiling on the win.** Persisting the compile caches removes the
**JIT-compile** term, which the bring-up log shows dominates the tax. CUDA-graph
*capture* and weight-load-into-VRAM are re-paid every boot regardless — but capture
is cheap once the kernels it dispatches are already compiled and warm, and weight
load is I/O-bound, not compute-bound. So the reachable outcome is **"a fraction of
the first boot"** (acceptance bullet 1), **not zero**. Anyone who expects a warm
boot to be instant has mis-scoped the CUDA-graph line — call it out in the witness.

## The persistence contract

### 1. Pin the JIT/compile cache dirs to a persistent, tuple-keyed location

Set, in the serve wrapper's environment before launching the engine:

```
PERSIST_ROOT   = <persistent disk mount, e.g. /mnt/compile-cache>   # survives boot/VM re-create
CACHE_KEY      = ${MODEL_SLUG}/${QUANT}/sm${ARCH}/tp${TP}/ctx${CTX}/${ENGINE}-${ENGINE_VER}-torch${TORCH_VER}
CACHE_DIR      = ${PERSIST_ROOT}/${CACHE_KEY}

TORCHINDUCTOR_CACHE_DIR = ${CACHE_DIR}/inductor
TRITON_CACHE_DIR        = ${CACHE_DIR}/triton
DG_JIT_CACHE_DIR        = ${CACHE_DIR}/deepgemm      # name is version-dependent — verify
VLLM_CACHE_ROOT         = ${CACHE_DIR}/vllm
HF_HUB_CACHE            = ${PERSIST_ROOT}/hf         # weights: keyed by repo, not by the tuple
```

### 2. Invalidation is by construction, not by a check

The cache key is a **path prefix built from the full tuple** (arch, model, quant,
tp, ctx, engine, engine version, torch version). Any tuple field change ⇒ a
**different directory** ⇒ a guaranteed rebuild. A stale cache can never silently
mis-serve, because a mismatched tuple never resolves to a populated dir. This
satisfies acceptance bullet 2 (key includes GPU arch **and** backend version:
`sm${ARCH}` distinguishes GPU server sm_80 from H100 Mega sm_90; `${ENGINE_VER}` /
`${TORCH_VER}` distinguish toolchains) and proposed-fix bullet 3 (invalidate on any
tuple change).

`ARCH` comes from the preflight (`tools/glm52_serve_preflight.py` already computes
`arch=Hopper (sm_90)`); `ENGINE_VER` / `TORCH_VER` from the engine's version string;
`MODEL_SLUG` is the checkpoint repo with `/` → `-`.

### 3. A preflight cache hit/rebuild readout

Before launch, the wrapper/preflight emits one structured line so an operator knows
whether this boot will be slow:

```
COMPILE_CACHE hit    dir=${CACHE_DIR} tuple=${CACHE_KEY}   # dir exists and is non-empty → warm, fast boot
COMPILE_CACHE rebuild dir=${CACHE_DIR} tuple=${CACHE_KEY}  # dir absent/empty → this boot pays the JIT tax
```

This is the same state `internal/vllmcompile.Block` already models
(`CompileCacheEnabled`, `CompileCacheKey`) for the benchmark-honesty gate — the
preflight readout should populate those fields so the bench path and the operator
path read one source of truth, and a cold boot is never quoted as a tuned baseline.

## Decomposition

The runtime work is one issue but several leaves, in the lanes that own the files:

| Leaf | Lane | File(s) | What it does |
|---|---|---|---|
| L1 pin cache dirs + emit readout | serve/tools | `tools/glm52_sglang_vllm_serve.sh`, `tools/glm52_serve_preflight.py` | export the tuple-keyed dirs; print `COMPILE_CACHE hit\|rebuild` (the largest slice of the fix) |
| L2 persist the store | infra/gcp | `scripts/gcp-glm-serve.sh` | mount a persistent disk at `PERSIST_ROOT`, or bake a pre-warmed cache into the serve image |
| L3 surface the gate | cmd/serve | `internal/vllmcompile`, preflight | feed `CompileCacheKey` / `CompileCacheEnabled` from the live tuple so bench + operator agree |
| L4 witness | infra | `experiments/agent-live/` | two-boot dogfood on a fixed tuple: capture ready-time(boot1) ≫ ready-time(boot2) — the acceptance proof and the gen/next→gen/now promotion evidence |

## Generation classification

**`generation` + `gen/next`; milestone `Generation G1 - Next Gen` (#13).**

Applied from [`docs/generation.md`](../generation.md) and the binding
[`GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md`](GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md).

- **Why `gen/next`, not `gen/now`.** The program map's `gen/next` row is "near-term
  cache/context integration… runnable soon after **gates, handoffs, and visibility**
  exist… first evidence: **dogfood runs on real serve sessions**." #3052 needs
  exactly that trio: a preflight cache-hit **gate**, a persistent-disk **handoff**,
  and its acceptance **witness is a real GPU two-boot dogfood** — not a cheap
  already-available test. `gen/now`'s bar ("focused tests / captured CLI output /
  operator readout showing the current path is safer *now*") is not met: no runtime
  code exists to witness yet.
- **Why not `gen/second-next`.** It needs no cross-engine compatibility contract or
  new serving architecture; it runs on the existing SGLang/vLLM path. That keeps it
  out of the architecture-option horizon.
- **Promotion evidence (→ `gen/now`).** A two-boot dogfood witness showing boot2 ≪
  boot1 on a fixed tuple; the preflight cache-hit gate landing default-on; a
  tuple-change forcing an observed rebuild (invalidation proven).
- **Demotion / retirement evidence.** Demote/park if backend/toolchain version churn
  thrashes the tuple key so most boots still rebuild (the key change makes the cache
  net-negative), or if a warm cache cannot prove cold-path correctness across arch.
  Retire if upstream SGLang/vLLM ships its own persistent compile cache keyed on the
  same tuple, making this wrapper-level pinning redundant.
- **Invalidating assumption (must be measured before promotion).** This whole issue
  assumes the ~500s is **dominated by disk-cacheable JIT** (TorchInductor / Triton /
  DeepGEMM), not by CUDA-graph *capture* + weight-load, which are **not** disk-
  persisted across a process restart. If per-phase timing shows capture/weight-load
  dominate, pinning the compile dirs buys only a small win and #3052 demotes. So the
  **first evidence to collect** is a per-phase breakdown of the warmup, not just the
  aggregate ready time.

## Refs

- Axis A/B ablation: [`GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md`](GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md)
- Bring-up trail: [`GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md`](GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md)
- Benchmark-honesty gate this contract feeds: `internal/vllmcompile`
- Companion tickets: #3051 (readiness-gate the first real turn) · #3053 (de-conflate the warmup tax from aggregate `cache_bit`)
