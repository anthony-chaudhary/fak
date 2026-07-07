---
title: "GLM-5.2 GPU-server drive (#3073): the costed L1–L10 lever map — value/cost priority + per-lever exact witness command, 2026-07-06"
description: "The single consolidated costing of every lever in epic #3073's GLM-5.2 throughput drive on the two sm_80 lab nodes: each of L1–L10 (+ Lane F) with its expected multiplier (labelled COMPUTED/ESTIMATED off the one WITNESSED anchor, 23.2 tok/s single-stream on GPU server 3, 2026-07-01), engineering cost S/M/L, node + lane, dependency order, readiness, a value/cost priority rank, and the EXACT recorded-artifact witness command the operator runs. Reconciles the epic's original 'two boxes in parallel' plan with the 2026-07-06 hardware correction (only GPU server 3 holds the 433.82 GiB model resident, so the resident levers serialize and the L8 fast-iteration harness becomes the force-multiplier). Advances every lever to ready-to-dispatch without fabricating a single tok/s."
---

# GLM-5.2 GPU-server drive: the costed L1–L10 lever map (epic #3073)

This note is the **costed** form of the lever table in
[`GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md`](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) §5.
The ceiling doc names the levers and their expected multipliers; this doc adds the three columns an
operator needs to *dispatch* them — **engineering cost, dependency order, and the exact recorded-artifact
witness command** — and ranks them by value/cost. It pairs with the
[per-box baseline](GLM52-DGX-PER-BOX-BASELINE-AND-NEXT-STEPS-2026-07-06.md) (the hardware split) and the
per-lane triage notes linked per row below.

> **The one WITNESSED anchor.** The only measured GLM-5.2 decode number is **23.2 tok/s single-stream**
> (llama.cpp 8-GPU resident, GPU server 3, 2026-07-01). Every multiplier in this map is **COMPUTED** from
> the roofline or **ESTIMATED** — none is a served result. Acceptance for *every* lever is a recorded
> `experiments/benchmark/runs` artifact, never a self-report. This map is planning + costing; it produces
> no tok/s.

## 1. The two nodes, and the step-0 gate

| Fact | Consequence |
|---|---|
| GLM-5.2 UD-Q4_K_M is **433.82 GiB** | fits resident only on the 640 GiB box |
| **GPU server 3** = 8×80 GiB → 640 GiB | the **resident node** — carries **all of L1–L10** |
| **GPU server 2** = 8×40 GiB → 320 GiB | **cannot** hold the model resident; carries only box-agnostic Lane F and its own-law lanes (L11/L12, out of L1–L10 scope) |
| GPU server 3 holds ~536 GiB weights resident but its **:8000 endpoint was NOT answering** at probe time (epic comment 2026-07-06T23:41Z) | **Step 0 = verify-or-restart the resident serve** gates every resident lever below |

> **Correction to the epic's parallel plan.** The epic body schedules "#3075 (L1) ∥ #3079 (L2) — the two
> biggest single levers — on separate boxes in parallel." The 2026-07-06 hardware correction removes that:
> both are resident-model levers, so **both run on GPU server 3 and serialize** (one resident serve,
> relaunched per lever). GPU server 2's 7 free 40 GiB cards cannot hold the model, so the only true
> cross-box parallelism is **Lane F on server 2 ∥ the resident levers on server 3**. This makes **L8 (the
> fast-iteration serve+bench harness) the force-multiplier** — it turns the now-serialized resident lane
> from a ~500 s-cold-tax-per-lever crawl into a <10 min/lever cycle.

## 2. Costing legend

- **Engineering cost** — **S** = config/flag change, first A/B in <1h · **M** = script + sweep, hours,
  no kernel · **L** = new kernel/engine code, days.
- **Provenance** — **WITNESSED** = measured on-box · **COMPUTED** = derived from the roofline ·
  **ESTIMATED** = order-of-magnitude only. The 23.2 tok/s baseline is the sole WITNESSED datum.
- **Readiness** — **ready** = triage + witness command exist, dispatch as-is · **needs-wiring** = a flag,
  script, or kernel must be added first (part of the cost) · every row's step 0 is the endpoint gate.
- **Priority** = value/cost, honouring dependencies. Lower = do sooner.

## 3. The costed lever map

| # | Lever | Tickets | Node | Bound | Expected (prov) | Eng | Readiness | Pri | Triage of record |
|---|---|---|---|---|---|:--:|---|:--:|---|
| **LF** | active-set from GGUF header | #3074 | server 2 (box-agnostic) | ceiling-truth | de-risks every ceiling; K=4 vs 8 moves target ~2× (COMPUTED) | **S** | ready | **1** | [Lane F](GLM52-LANE-F-ACTIVE-SET-FROM-GGUF-HEADER-2026-07-06.md) |
| **L8** | warm-start + compile caches + bench harness | #3082/#3083/#3084 | both | iteration (kills ~500 s cold tax) | makes every lever <10 min/node (COMPUTED) | **M** | needs-wiring | **1** | [#3051](GLM52-READINESS-WARMUP-GATE-CONTRACT-3051-2026-07-06.md) · [#3052](GLM52-COMPILE-CACHE-PERSISTENCE-CONTRACT-3052-2026-07-06.md) · [cold-start](GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md) |
| **L1** | 8-GPU tensor/row split (kill `-sm layer`) | #3075 | server 3 | A single-stream | **~3–6×** (23.2→~70–140) (COMPUTED) | **S**→M | needs-wiring | **2** | this doc §5 |
| **L2** | continuous batching + KV budget | #3079/#3080 | server 3 | B aggregate | **10–40× aggregate** (→~9–11k @80%) (COMPUTED) | **M** | needs-wiring | **3** | [cont-batch](GLM52-DGX2-LANEB-L2-CONTBATCH-TRIAGE-2026-07-06.md) · [KV budget](GLM52-DGX2-LANEB-KV-BUDGET-TRIAGE-2026-07-06.md) |
| **L4** | flash-attention + CUDA graphs | #3076 | server 3 | A single-stream | 1.2–1.8× (COMPUTED) | **S–M** | ready | **4** | [Lane A L4](GLM52-DGX3-LANEA-L4-FA-CUDAGRAPH-TRIAGE-2026-07-06.md) |
| **L3** | speculative decoding (n-gram / draft) | #3078 | server 3 | A single-stream | 1.5–2.5× (acceptance-driven) (COMPUTED) | **M** | needs-wiring | **5** | this doc §5 |
| **L9** | real prefill path (chunked + FA) | #3085/#3086 | server 3 | C prefill | stands up Ceiling C (~11–14k target) (COMPUTED) | **M** | needs-wiring | **6** | this doc §5 |
| **L5** | decode quant sweep (Q4_K fast path) | #3077 | server 3 | A single-stream | 1.1–1.5× (COMPUTED) | **M** | ready | **7** | [L5 quant](GLM52-L5-QUANT-SWEEP-TRIAGE-2026-07-06.md) |
| **L6** | INT8 tensor-core expert GEMM | #3087 | server 3 | B aggregate | up to ~2× aggregate (ESTIMATED) | **L** | needs-wiring | **8** | this doc §5 |
| **L10** | native fak resident-EP device-NCCL | #3089 / #1482 | server 3 | ceiling-truth | pure-fak kernel witness vs 23.2 baseline (ESTIMATED) | **M–L** | needs-wiring | **9** | [pure-fak gap](GLM52-PURE-FAK-BENCHMARK-GAP-2026-07-06.md) |
| **L7** | true DSA sparsity (not full-MLA) | #3088 | server 3 | A/C long-ctx | ctx-dependent; ~1.0–1.1× @8k, material only 32k+ (ESTIMATED) | **M**(curve)/**L**(kernel) | needs-wiring | **10** | [Lane E L7](GLM52-DGX3-LANEE-L7-DSA-TRUE-SPARSE-TRIAGE-2026-07-06.md) · this doc §5 |

## 4. Prioritized dispatch order

Two things run truly in parallel; the rest serialize on the one resident box.

1. **Now, before anything measures (parallel, no resident serve needed):**
   - **LF (#3074)** on GPU server 2 — pin active-params/bytes from the GGUF header; **re-derives every
     ceiling** in this map (it scales `1/active`). Cheapest, unblocks nothing downstream but corrects the
     target all other levers are measured against.
   - **L8 harness (#3082)** — build the one-command serve+bench emitter first; every row below records its
     acceptance artifact through it. Post-correction this is the highest-leverage engineering item.
2. **Step 0 (gates all resident levers): verify-or-restart the GPU server 3 :8000 endpoint.**
3. **L1 (#3075)** — the dominant ~3–6× lever and the base topology L4/L5/L3 stack on. One-flag A/B.
4. **L2 (#3079/#3080)** — stands up aggregate; **blocked-by L1** (measure batching on the 8-GPU-per-token
   serve, not the layer-split-throttled one).
5. **Lane A stack on top of L1, one resident serve at a time:** L4 (#3076) → L3 (#3078) → L5 (#3077).
   **L9 (#3085/#3086)** prefill can interleave (compute-bound, independent of the L1 decode fix).
6. **Architecture/kernel, later:** L6 (#3087, **blocked-by L2**) · L10 (#3089/#1482, separate engine, runs
   concurrent) · L7 (#3088, measure the full-MLA ctx curve to *size the prize* before any sparse kernel).

## 5. Per-lever detail (the un-documented levers)

The six levers with an existing triage doc are linked in §3; their exact witness commands live there. The
five below had **no prior triage** — this section is their triage of record.

### L1 — 8-GPU tensor/row split (kill `-sm layer`) · #3075 · server 3 · Lane A
**Mechanism.** Single-stream decode is memory-bandwidth-bound (`tok/s = eff_BW ÷ active_bytes/token`).
llama.cpp `-sm layer` pipelines the layers across the 8 GPUs, so each token streams from **one GPU, seven
idle** — ~1/8 of the box's ~16 TB/s aggregate HBM. That structural 1-of-8 *is* the 23.2 tok/s wall.
`-sm row` shards every weight tensor across all 8 cards and all-reduces over NVLink, so **every token draws
all 8 GPUs' bandwidth**. Dominant lever: it attacks the parallelism denominator, not bytes or launch cost.
**Cost S→M:** `--split-mode row` is a one-flag add to `glm52_mgpu_serve.sh`; escalates to M only if row
underdelivers and RPC/SGLang-TP is needed. **Provenance COMPUTED** (~3–6×, 23.2→~70–140); realism caveat:
llama.cpp `-sm row` MoE gains are empirically below the pure-bandwidth ratio, so the A/B may land ~3×.
**Witness:**
```bash
# step 0: verify-or-restart the resident serve (endpoint down at probe time)
curl -sf -m5 http://127.0.0.1:8002/health || \
  DEVICES=0,1,2,3,4,5,6,7 bash tools/glm52_mgpu_serve.sh           # EXISTING; default -sm layer = arm A
# arm B (lever): relaunch SAME weights with row split (PROPOSED one-flag add), port 8003
SHARD1=$(ls /mnt/sglang_dv3/glm52-q4/GLM-5.2-UD-Q4_K_M-*-00001-of-*.gguf | head -1)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid llama-server \
  --model "$SHARD1" --alias glm-5.2 --jinja --n-gpu-layers 999 \
  --split-mode row --tensor-split 1,1,1,1,1,1,1,1 \
  --host 127.0.0.1 --port 8003 --ctx-size 8192 &
# decode 256 tok on both ports + prove all 8 GPUs busy under row:
nvidia-smi dmon -s u -d 1 -c 30 > util.dmon &
for PORT in 8002 8003; do curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Count to 200."}],"max_tokens":256}' \
  | grep -o '"tokens_per_second":[0-9.]*'; done
# land as experiments/benchmark/runs/by-machine/gpu-server-3/<UTC>-glm52-L1-smrow-vs-smlayer/ + util.dmon
```
**Blocker:** :8000/:8002 endpoint not answering (step 0); `--split-mode row` not yet wired into the serve
script.

### L3 — speculative decoding (n-gram / draft) · #3078 · server 3 · Lane A
**Mechanism.** Draft K tokens cheaply, **verify all K in one target forward pass** — the expensive weight
read amortizes across accepted tokens, so net speedup ≈ mean-accepted-length minus overhead. The multiplier
is a direct function of **token-acceptance rate**; agentic traffic (repetitive, tool-heavy, quoting prior
output) is the high-acceptance regime a zero-cost n-gram drafter exploits.
**Feasibility on the sm_80 llama.cpp engine:** prompt-lookup/n-gram (no draft checkpoint, no extra VRAM) is
the unblocked first move; draft-model (`-md`) needs a **vocab-compatible GLM checkpoint** — none staged;
EAGLE/Medusa are heads on vLLM/SGLang, **not the llama.cpp resident engine** → out of scope here.
**Cost M** (wiring + acceptance sweep); EAGLE would be L and is not this ticket. **COMPUTED 1.5–2.5×.**
Measure **on top of L1** so the multiplier composes with per-token bandwidth.
**Witness:**
```bash
DEVICES=0,1,2,3,4,5,6,7 PORT=8000 bash tools/glm52_mgpu_serve.sh    # EXISTING; spec-OFF baseline
# SPEC arm (prompt-lookup, zero-checkpoint) — PROPOSED flags, port 8003:
llama-server --model "$SHARD1" --alias glm-5.2 --jinja --n-gpu-layers 999 \
  --host 127.0.0.1 --port 8003 --ctx-size 8192 \
  --draft-max 6 --draft-min 1 --draft-p-min 0.6          # sweep --draft-max {4,6,8}
python3 tools/glm52_serving_witness.py --base-url http://127.0.0.1:8003/v1 --model glm-5.2 \
  --context-length 8192 --out experiments/benchmark/runs/<host>-glm52-specdecode-<UTC>.json
# acceptance artifact = token-acceptance rate + net decode tok/s (spec on vs off) on a fixed agentic set
```
**Blocker:** step-0 endpoint; no vocab-compatible draft checkpoint staged (n-gram arm is the unblocked one).

### L9 — real prefill path (chunked + FA) · #3085 (baseline) + #3086 (chunked+FA) · server 3 · Lane D
**Mechanism.** Prefill is compute-bound (~64 GFLOP/token), practical ceiling ~11–14k tok/s (§3.3). Today it
is **not a number we hold** — the only prompt-eval figure (46 tok/s, 07-01) is an **11-token prompt**,
launch-overhead-dominated, not a prefill-regime measurement. L9 first *stands up the curve*
(prompt_len→prefill tok/s), then A/Bs **chunked prefill** (`-ub`, overlaps prefill with concurrent decode)
and **flash-attention** (`-fa`, cuts the O(L²) attention traffic). **Not a decode multiplier** — a separate
axis. **Cost M** (no prefill-sweep script exists; TTFT-off-SSE loop over 5 lengths + recorded artifact).
Independent of L1 (compute-bound), so it can interleave once the endpoint is up; **#3085 before #3086.**
**Witness:**
```bash
curl -sf -m8 http://127.0.0.1:8000/v1/models | grep -q glm-5.2 \
  || CTX=8192 bash tools/glm52_stage_serve_dgx3.sh                  # EXISTING; waits GLM52_SERVE_READY
# #3085 baseline sweep — PROPOSED (new harness): TTFT = wall to first SSE token
OUT=experiments/benchmark/runs/by-machine/gpu-server-3/glm52-prefill-sweep-$(date -u +%Y%m%dT%H%M%SZ)
for N in 128 512 2048 4096 8192; do
  P=$(python3 -c "print('word '*$N)"); T0=$(date +%s.%N)
  curl -sN -m120 http://127.0.0.1:8000/v1/chat/completions \
    -d "{\"model\":\"glm-5.2\",\"stream\":true,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}]}" \
    | grep -m1 -q 'data: '; T1=$(date +%s.%N)
  echo "{\"prompt_len\":$N,\"ttft_s\":$(echo "$T1-$T0"|bc -l)}" | tee -a "$OUT/prefill.jsonl"
done
# #3086 A/B — PROPOSED: relaunch with  -fa  -ub 512  (sweep 256/512/1024) + a concurrent-decode stream
```
**Blocker:** step-0 endpoint; `glm52_stage_serve_dgx3.sh` does not yet pass `-fa`/`-ub`.

### L6 — INT8 tensor-core expert GEMM (sm_80) · #3087 · server 3 · Lane B/E
**Mechanism.** At concurrency the expert GEMMs become **compute-bound** (Ceiling B). sm_80 has no FP8; its
top math rate is the **int8 tensor core (624 TOP/s = 2× BF16's 312)**. W8A8 experts on the int8 path double
the compute ceiling (~78k vs ~39k tok/s raw aggregate). Does **nothing at batch 1** — throughput-only.
**fak vs llama.cpp:** fak *already* ships an int8 GEMM (`fcuda_q8_matmul_f32`/`k_q8_gemm`) but it accumulates
with **scalar int32 SIMT**, not the tensor cores — L6 = replace that inner accumulate with int8-tensor MMA
**and** route the Q4_K routed-experts into the int8 path (cosine ≥ 0.999). llama.cpp's `MMQ` already uses
int8 `dp4a`/`mma`, so on that baseline L6 is a confirm-and-measure (M); the **fak-kernel** witness the
ticket asks for is **L (kernel, days)**. **ESTIMATED up to ~2×.** **Blocked-by L2** (no compute-bound region
to help in until batching stands up).
**Witness:**
```bash
DEVICES=0,1,2,3,4,5,6,7 PORT=8000 bash tools/glm52_mgpu_serve.sh    # EXISTING resident bring-up
bash tools/glm52_e2e_after_serve_dgx3.sh                            # EXISTING e2e witness (#413)
# PROPOSED fak-kernel A/B: FAK_INT8_EXPERTS off then on, fixed concurrency C, via the native EP path
for INT8 in 0 1; do
  FAK_INT8_EXPERTS=$INT8 FAK_Q4K=1 RANKS=8 FIRST_GPU=0 PORT=8071 \
    setsid bash tools/glm52_ep_witness.sh &     # then drive CONCURRENCY=64 aggregate bench
done
# accept: runs artifact with int8-expert cosine >= 0.999 vs f32 ref + aggregate tok/s int8 vs bf16
```
**Blocker:** step-0 endpoint; blocked-by L2 (needs the aggregate harness); fak int8-tensor kernel unwritten.

### L7 — true DSA sparsity (not full-MLA) · #3088 · server 3 · Lane E
**Mechanism.** GLM-5.2 is `glm_moe_dsa` (DeepSeek-Sparse-Attention): a top-k indexer makes attention grow
sub-linearly with context. But llama.cpp serves it as **full MLA**, and the fused sparse-DSA kernel has an
**sm_90 floor** (the same floor that makes stock SGLang/vLLM `BLOCKED_ARCH` on these boxes). **On sm_80 there
is no resident sparse kernel to turn on** — so a true-sparse tok/s win is **un-witnessable on this hardware
today**, and L7 is *not* a config flip. The honestly measurable artifact now is the **full-MLA ctx-scaling
curve** (decode+prefill tok/s vs ctx {2k,8k,32k}) — it **sizes the prize**: flat out to 32k ⇒ deprioritize;
a cliff at depth ⇒ quantifies the DSA headroom that would justify the sm_90-kernel work. **Cost M** for the
curve, **L** for any real sparse path. **ESTIMATED ~1.0–1.1× @8k**, material only 32k+. Lowest priority of
the resident levers. Measure the curve **downstream of L1** at a fixed FA setting.
**Witness:**
```bash
for CTX in 2048 8192 32768; do
  CTX=$CTX PORT=8002 DEVICES=1,2,3,4,5,6,7 setsid bash tools/glm52_mgpu_serve.sh &   # EXISTING serve, CTX flag
  until grep -q GLM52_MGPU_READY /tmp/glm52_mgpu/PHASE; do sleep 10; done
  python3 tools/glm52_serving_witness.py --base-url http://127.0.0.1:8002/v1 --model glm-5.2 \
    --context-length $CTX --out experiments/benchmark/runs/glm52-l7-mlactx-$CTX.json   # EXISTING witness
  pkill -f "llama-server.*port 8002" || true
done
# headline = the ctx -> decode/prefill curve on FULL-MLA (the prize L7 would capture), labelled WITNESSED
```
**Blocker:** step-0 endpoint; **sm_90 DSA kernel floor** — ship the full-MLA baseline curve now, not a
sparse-vs-dense speedup.

## 6. What this map does NOT claim

- **No served tok/s.** Every multiplier is COMPUTED/ESTIMATED off the single WITNESSED 23.2 baseline. This
  is a costing, not a benchmark.
- **No lever is "done."** Each row's acceptance is a recorded `experiments/benchmark/runs` artifact that only
  GPU server 3 (resident) or GPU server 2 (Lane F only) can produce — none of which exists yet, and the
  resident endpoint was down at probe time. The map hands the operator each lever *ready to dispatch* with
  the exact command; it does not run them.
- **The pure-fak kernel (L10) stays fenced** from the llama.cpp baseline — its bar is a *first* live
  resident-EP tok/s held against 23.2, on a separate engine-honest row.

*Companions:* [theoretical ceiling + lever table](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) ·
[per-box baseline](GLM52-DGX-PER-BOX-BASELINE-AND-NEXT-STEPS-2026-07-06.md) ·
[8-GPU full-resident serve (23.2 WITNESSED)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md) ·
per-lane triage notes linked per row in §3.
