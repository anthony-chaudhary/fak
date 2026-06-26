---
title: "GLM-5.2 on the pure fak kernel: witnessed results (2026-06-21)"
description: "Witnessed go-test results showing the GLM-5.2 architecture runs in the pure fak kernel, why serving the real 753B checkpoint is hardware-gated, plus turn demos."
---

# GLM-5.2 on the pure fak kernel + the large-scale agent/turn demos — witnessed results (2026-06-21)

> **Goal:** *get GLM-5.2 self-hosted on our pure kernel up and running, and prove the
> large-scale agent/turn demos.*
>
> This doc is closed by witnesses the author did not write — `go test` exit codes, a
> serving-preflight verdict, and benchmark output fields — not by self-report. Every
> command below was run on-box at `HEAD` on 2026-06-21 (`go build ./...` green;
> `go version` = go1.26.3 windows/amd64; GLM `go test` run under WSL go1.26.0
> linux/amd64 because Windows app-control blocks freshly-compiled native test binaries).

> **Current direction (see the [staged plan](native-753b-track-staged-plan.md), #917).**
> This is a 2026-06-21 point-in-time witness. The second arm of its scope split — "serving
> the real 753B checkpoint is hardware-gated; this dev box has no NVIDIA GPU" (§2) — has
> since been superseded by progress on the datacenter GPU node: `--cpu-offload-experts`
> shipped and fak's **own** engine loads the full 466 GB `glm_moe_dsa` model natively and
> binds `/v1/*` ([2026-06-25 native-serve note](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md);
> the staged plan is the single current direction). The architecture-runs-green witnesses
> (§1) and the model-agnostic turn demos (§3) stand unchanged; only the serving gate moved.

## The honest scope split (read this first)

"Self-host GLM-5.2 on our kernel" is two different things, and conflating them is the
trap:

1. **The GLM-5.2 architecture running inside the pure fak kernel** — DSA (DeepSeek-Sparse-
   Attention) + the learned indexer + GLM MoE group-routing + shared experts, executed by
   the in-kernel pure-Go fusion, proven correct on synthetic / tiny `glm_moe_dsa` fixtures
   with **no HF download and no 753B checkpoint**. **This is achievable on-box and is
   witnessed green below (§1).**

2. **Serving the real 753B `zai-org/GLM-5.2` checkpoint** — hardware-gated. Stock
   SGLang/vLLM DSA kernels need sm_90 (Hopper) / sm_100 (Blackwell); even INT4 weights are
   ~376 GB. This dev box has **no NVIDIA GPU**, so it fails the gate — *correctly and by
   design*. The preflight tool (§2) turns that into a reproducible go/no-go and names the
   hardware that would serve it.

The **large-scale agent/turn demos (§3)** are model-agnostic kernel demos — they measure
cross-agent cache reuse, turn-tax elimination, and fan-out, independent of which model
sits behind the kernel.

---

## §1 — GLM-5.2 pure-kernel: up and running, witnessed green

`go test` under WSL, all packages **PASS** (`-count=1`):

```
ok  github.com/anthony-chaudhary/fak/internal/model      (GLM/DSA/MoE witnesses)
ok  github.com/anthony-chaudhary/fak/internal/agent      (GLM message↔segment coherence)
ok  github.com/anthony-chaudhary/fak/internal/gateway    (GLM serving conformance)
ok  github.com/anthony-chaudhary/fak/internal/cachemeta  (GLM turn-segment shaping)
```

The 23 named `internal/model` witnesses that prove the `glm_moe_dsa` arch executes in the
kernel (all `--- PASS`):

| Area | Witness | What it proves |
|---|---|---|
| Family / loader | `TestGLMFamilyDerivationFromConfig` | `model_type:"glm_moe_dsa"` → `isGLM` + `isGLMMoeDsa`; dense `glm4` and `llama` correctly excluded |
| Loader | `TestGLMDropsMtpAndVisualTensorsAtLoad` | the MTP head + vision tower are dropped at load (family-gated, Llama-invariant) |
| MoE FFN | `TestGLMMoEForwardRunsThroughNativeKernel` | router→top-k→per-expert SwiGLU→weighted-sum runs finite through native Prefill + decode |
| MoE | `TestGLMMoeDsaQuantizeBuildsSharedExperts` | GLM shared-expert FFN quantizes + runs |
| MoE batch | `TestGLMMoeBatchedDecodeMatchesSerial` | multi-user batched GLM-MoE decode == serial |
| **DSA attention** | `TestDSASparseAttentionMatchesDenseMaskedReference` | **sparse DSA attention is bit-correct vs a dense masked reference** |
| DSA attention | `TestDSASparseAttentionRejectsInvalidSelections` | invalid index selections are rejected, not silently wrong |
| DSA indexer | `TestDSAIndexScoresUsesProjectedWeightsAndRelu` | the learned per-key indexer scoring path |
| DSA indexer | `TestDSATopKIndicesAreCausalAndPrefixReusable` | top-k selections stay causal + prefix-reusable |
| DSA IndexShare | `TestDSAIndexShareReusesPreviousFullIndexer` / `...RejectsMissingFullIndexer` | IndexShare reuse + its guard |
| DSA↔cache | `TestDSAIndexDecisionFeedsAttentionIndexCachemeta` | the DSA index decision feeds the attention-index cache metadata |
| DSA shared layers | `TestGLMMoeDsaQuantizeAllowsSharedLayersWithoutIndexerTensors` | shared (indexer-less) layers load + run |
| Quant residency | `TestGLMMoeDsaQuantLoadRunsResidentDSAProjections` | q_a/q_b/kv_a/kv_b + indexer projections are q8-resident (no f32 round-trip), Prefill/Step match the cacheless Q8 head |
| Quant residency | `TestGLMMoeDsaRegularQuantizeBuildsResidentDSAProjections` | `.Quantize()` builds the same resident q8 tensors |
| Quant head | `TestGLMMoeDsaQuantSessionUsesUntiedQ8Head` | untied lm_head q8 residency |
| Session | `TestGLMMoeDsaQuantSessionCoversNoLogitsPrefixAndGenerate` | PrefillNoLogits / SessionFromPrefix / Generate parity vs cacheless forward |
| Quant load | `TestGLMMoeDsaQuantLoadBF16MatchesDecodedQ8` | BF16-on-disk → decoded-Q8 equality |
| Sharded load | `TestGLMMoeDsaQuantDirLoadsShardedBF16Weights` | sharded (`-of-`) BF16 GLM-DSA dir loads q8-lean |
| **Bit-exact evict** | `TestGLMMoeDsaQuantEvictMatchesNeverSawAndReropes` | **mid-run span evict on GLM-DSA == a run that never saw the span, with re-rope — the poison-quarantine proof on the GLM path** |
| Batch parity | `TestGLMDsaBatchedDecodeMatchesSerial` (dense+moe), `TestGLMDsaGenerateBatchMatchesSerial` (dense+moe) | batched GLM-DSA decode/generate == serial |
| Accel no-op | `TestDecodeBandGLMDsaMonolithicNoOp` | the GLM-DSA decode-band path is a proven no-op vs the scalar reference |

**Honest boundary (skipped, not failed):** `TestOptionalGLMMoeDsaOracle*` (6 tests:
export-metadata, DSA-boundary, DSA-attention-trace, dense-prefix-layer, cacheless-forward,
session-cache) `t.Skip` because no real re-exported HF `glm_moe_dsa` oracle is on disk.
Numeric parity against the *real* DSA math is gated on exporting a tiny GLM-MoE-DSA oracle
(epic #474 / #413). The synthetic tier above proves loader + family + MoE + DSA *wiring*
and bit-exact KV behavior; it does **not** claim HF numeric parity for DSA.

**Reproduce:**
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/work/fak && \
  go test ./internal/model/  -run "GLM|Dsa|DSA|MoE" -v -count=1 && \
  go test ./internal/agent/ ./internal/gateway/ ./internal/cachemeta/ -run "GLM|Glm|Coherence" -count=1'
```

---

## §2 — Serving the real 753B GLM-5.2: the honest hardware gate

`tools/glm52_serve_preflight.py` is a fail-closed go/no-go that reads the DSA kernel arch
floor (sm_90) and the per-quant weight footprint.

**On this box (no GPU):**
```
node_verdict     : BLOCKED_ARCH
any_engine_ready : false
arch             : unknown   compute_cap: null   total_vram_gb: 0.0
```
Correct: stock SGLang/vLLM cannot serve GLM-5.2 here. (The kernel's own *architecture*
still runs CPU-resident — §1 — it is the 753B serving stack that is gated.)

**As a planner for an 8×H200 node** (`--gpu-name "NVIDIA H200" --gpu-count 8 --gpu-memory-total-gb 1128 --no-probe-engines`):
```
node_verdict     : READY_PENDING_INSTALL
ready_engines    : [sglang, vllm]
recommended_quant: fp8        arch: Hopper (sm_90)   compute_cap: 9.0   total_vram_gb: 1128.0
```
That is the precise answer to "what does self-hosting the real 753B need": a Hopper /
Blackwell node (H100/H200/B200/B300/GB200/GB300), fp8 by default, with per-GPU TP-shard
fit checked — not this consumer box.

---

## §3 — The large-scale agent/turn demos, proven on-box

All model-agnostic; run native (`go run`) on Windows, no weights required.

### 3.1 Fan-out to N=1024 sub-agents (`cmd/fanbench`)
`go run ./cmd/fanbench -agent-max 1024 -grid log` — the N-ladder corner:

| N | calls | shared | isolated (warm) | **cross** | tax_clawed_back | parallel_speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1028 | 785 | 536 | 255 | 61.7% | 57.7× |
| 512 | 2052 | 1569 | 1069 | 483 | 61.7% | 66.9× |
| **1024** | **4100** | **3155** | **2152** | **1005** | **61.7%** | **72.8×** |

At N=1024: **1005 sibling-only tool-result saves** over isolated worlds, **61.7%** of the
multi-agent token tax clawed back, 72.8× critical-path speedup. (Matches STATUS §0.)

### 3.2 Fleet sweep 50×50 corner (`cmd/fleetbench`)
`go run ./cmd/fleetbench -agents 50 -turns 50 -trials 24 -profile read-heavy -granularity resource`:

```
T=50 A=50  calls=2500  shared=2344  isolated=1974(warm)  cross=370
tokens_saved_shared=3,094,080   $12.66 saved (shared)
```
The read-fleet corner **deletes 2,344 / 2,500 calls** with **+370 cross-agent turns** over
isolated (warm per-agent KV) worlds. (Matches STATUS §0.) The full 50×50 surface is the
same run without `-agents/-turns` (a heatmap over all 2,500 cells).

### 3.3 Turn-tax A/B through the real kernel (`cmd/fak turntax`)
| Suite | turns saved | breakdown | vDSO ON / OFF | safety floor (separate axis) |
|---|---:|---|---|---|
| `turntax-airline` | **9** | forced 5 (grammar+dedup) + elision 4 (pure+static) | 9 / 2 → vDSO = 7 turns | injections admitted base 1 → fak **0**; destructive executed base 1 → fak **0** |
| `turntax-happy` | **0** | — (the anti-inflation control: a clean path inflates nothing) | 0 / 0 | base 0 / fak 0 |

The safety floor is reported on a **deliberately separate axis** and never folded into the
turn count.

### 3.4 RadixAttention prefix reuse + cache-aware scheduling (`cmd/radixbench -scale 1`)
| Workload | reqs | cache hit | cross-subtree reuse | bounded sched (FCFS → cache-aware) |
|---|---:|---:|---:|---|
| few-shot | 16 | 88.2% | 1.00× | 88.2% → 88.2% (100% of optimal) |
| multi-turn-chat | 8 | 79.5% | 2.50× | 79.5% → 79.5% |
| tree-of-thought | 27 | 77.2% | 1.40× | 77.2% → 77.2% |
| **agents (5×6)** | 30 | 86.7% | 1.48× | **62.1% → 86.7%** (cache-aware lift) |

Plus a policy-eviction witness: a verdict freed exactly 8 tokens and **kept the benign
sibling** warm.

### 3.5 Context-changing fleet token accounting (`cmd/ctxdemo -print`)
Exact, timing-free prefill-token work per scenario (decode excluded):

```
scenario       C   T    P   no-cache    warmKV     fak    fak-win  (ref×)  maxCtx
fleet-5x50     5  50 1024  1,259,857    39,591   35,495    1.1×    35.5×    9569
deep-research  4   5 1536     40,188     9,358    4,750    2.0×     8.5×    2642
```
The 5-agent × 50-turn fleet re-reads **1.26M tokens cold**; fak does **35,495** — 35.5× vs
cold, 1.1× on top of an already-warm per-agent KV cache (the honest serving baseline).

---

## What is proven vs. what is not

**Proven on-box today:** the GLM-5.2 `glm_moe_dsa` architecture (DSA sparse attention, the
learned indexer + IndexShare, GLM MoE group-routing + shared experts, quant residency,
sharded load, batched-decode parity, and **bit-exact mid-run eviction**) runs inside the
pure fak kernel, green under `go test`; and the large-scale agent/turn demos reproduce
their headline numbers live (fan-out N=1024, fleet 50×50, turn-tax, radix, fleet token
accounting).

**Not proven here (labeled, not hidden):** (a) HF *numeric* parity for DSA — gated on a
real exported `glm_moe_dsa` oracle (`TestOptionalGLMMoeDsaOracle*` skip; #474/#413); (b)
serving the real 753B checkpoint — hardware-gated, needs an sm_90+ node (§2); (c) the
accelerated `compute.Backend` GLM-DSA decode path still panics honestly rather than diverge
(`requireGLMDsaSession`, #86). None of these is on the critical path for "the architecture
runs in our kernel" — they are the next rungs for "serve the flagship at scale."

---

_Artifacts (regenerable; written to a scratch dir, not committed): `fanout.{json,csv}`,
`fleet-corner.{json,csv}`, `fleet-sweep.{json,csv}`, `turntax-airline.json`,
`turntax-happy.json`, `radix.json`. Reproduce with the commands inline above._
