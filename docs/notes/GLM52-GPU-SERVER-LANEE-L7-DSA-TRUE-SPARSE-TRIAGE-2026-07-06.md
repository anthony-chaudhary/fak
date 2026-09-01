---
title: "GLM-5.2 L7 true DSA sparsity (#3088): triage â€” the sparse arm is un-witnessable on sm_80; the honest witness is the full-MLA ctx curve (2026-07-06)"
description: "Triage for issue #3088 (epic #3073, GPU server 3 / Lane E / L7): refutes the issue's core assumption (a portable DSA sparse-path exists to enable) against the sm_90 kernel floor on the lab's sm_80 boxes, corrects the node designation (title says GPU server 2; the resident lever runs on GPU server 3), classifies the lever gen/second-next with named re-promotion triggers, and hands a server-3 operator the turnkey full-MLA ctx-scaling-curve witness {2k,8k,32k} that sizes the DSA prize. NO number here is a served measurement; the only WITNESSED figure is the 23.2 tok/s baseline carried from the 07-01 resident-serve note."
---

# GLM-5.2 L7 true DSA sparsity (#3088): triage + the curve that sizes the prize

> **What this is.** A *triage / classification* record for issue
> [#3088](https://github.com/anthony-chaudhary/fak/issues/3088) â€” the **Lane E / L7**
> lever under epic [#3073](https://github.com/anthony-chaudhary/fak/issues/3073). It
> (1) records why the issue's Do/Accept as written cannot execute on the lab's hardware,
> (2) classifies the generation horizon, and (3) hands a GPU server-3 operator the exact
> full-MLA ctx-scaling-curve protocol that produces the honest WITNESSED artifact â€” the
> one that decides whether L7 is worth kernel work at all.
>
> **What this is NOT.** Not the benchmark, and **not a sparse-vs-dense A/B** â€” no sparse
> arm can run on this iron today (Â§2). The only served measurement referenced is the
> **23.2 tok/s WITNESSED** single-stream baseline from
> [GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md).
> Every curve cell below is **PENDING** until an operator runs it; nothing here is fabricated.

## 1. The ask (verbatim intent) + node correction

Lever **L7** (ctx-dependent, the long-context attention lever in the
[ceiling map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) Â§5). The issue: llama.cpp runs
GLM DSA as full MLA (the sparse indexer is WIP upstream), so long-context attention pays
the dense cost. **Do:** enable/port the DSA indexer sparse path in the serve; A/B decode +
prefill tok/s vs full-MLA at ctx {2k, 8k, 32k}; verify output parity within the DSA
tolerance. **Accept:** a ctxâ†’{sparse vs full-MLA tok/s} table with the parity check;
headline = the long-ctx crossover.

**Node correction.** The issue title says **GPU server 2**, but GLM-5.2 UD-Q4_K_M (433.82 GiB) is
resident **only on GPU server 3** (640 GiB aggregate); GPU server 2's 320 GiB cannot hold it
(ceiling doc, 2026-07-06 correction â€” "the two nodes are NOT identical"). L7 is a
**GPU server 3 Â· Lane E** lever, per both the corrected ceiling Â§6 and the
[costed lever map](GLM52-L1-L10-COSTED-LEVER-MAP-2026-07-06.md) Â§3. Any witness run lands
on server 3.

## 2. Why the sparse arm is un-witnessable on this hardware today

The issue's Assumption 1 â€” *"a DSA indexer sparse-path implementation (even WIP upstream)
exists to port or enable"* â€” is **false for sm_80**, on three grounds:

1. **llama.cpp serves `glm_moe_dsa` as full MLA today.** Its sparse indexer is WIP
   upstream â€” and *implementing* it is explicitly excluded by the issue's own Out-of-scope
   clause ("not upstream llama.cpp's own DSA indexer implementation"). So there is nothing
   in-scope to "enable."
2. **The existing fused sparse-DSA kernels (DeepSeek lineage) have an sm_90 floor** â€” the
   same floor that makes stock SGLang/vLLM refuse GLM-5.2 on these boxes (`BLOCKED_ARCH`,
   vLLM #35021). Both lab nodes are **sm_80** (compute 8.0). There is no resident sparse
   kernel that can run there; porting one is **Cost L** kernel work, not a serve-config
   change (lever map Â§5 L7).
3. **The pure-fak DSA path is out-of-lane for this issue.** fak's own DSA index-selection /
   sparse-attention kernels exist
   ([index selection](GLM52-DSA-INDEX-SELECTION-ON-PURE-KERNEL-2026-06-23.md) Â·
   [sparse attention](GLM52-DSA-SPARSE-ATTENTION-ON-PURE-KERNEL-2026-06-23.md) Â·
   [full forward](GLM52-DSA-FULL-FORWARD-ON-PURE-KERNEL-GPU-SERVER-2026-06-23.md)), but the
   epic rule is explicit: llama.cpp is the engine-honest baseline; pure-fak numbers stay in
   a separate artifact (the #1482 / L10 lane, #3089).

Consequence: **the "sparse" column of the issue's Accept table cannot be produced on the
lab's iron today**, by anyone, regardless of operator access. The issue's parity-tolerance
question ("what is the DSA output-parity tolerance?") stays **unknown â€” needs operator
input**, and is moot until a sparse arm exists.

## 3. Generation classification â€” `gen/second-next`

Classified per [docs/generation.md](../generation.md); label `gen/second-next` + milestone
*Generation G2 â€” Second Next Gen* applied on the issue (this record is the basis).

- **Why not `gen/now`:** the `gen/now` test requires "no dependency on a future
  architecture bet." The sparse path depends on kernels that do not exist for this arch
  (Â§2) â€” marking it now-horizon would be the *current-work laundering* anti-pattern.
- **Why not `gen/next`:** `gen/next` work "should be runnable by agents soon" pending a
  gate/dogfood/schema. No gate unblocks L7; it needs either upstream kernel work landing
  with sm_80 support or new iron.
- **Why `gen/second-next`:** it is an architectural option that needs **sizing** (the Â§4
  curve) and a feasibility decision before it can become active product work â€” the G2
  definition verbatim.
- **Re-promotion triggers (closed set):**
  1. upstream llama.cpp's sparse indexer lands with **sm_80-capable** kernels â†’ re-test as
     a rebuild + config lever (`gen/now`);
  2. the measured Â§4 curve shows a **â‰¥1.3Ã— cliff at 32k** â†’ the prize justifies scoping
     sm_80-feasible indexer kernel work (`gen/next`);
  3. the lab gains **sm_90+** iron â†’ the fused-kernel path becomes a config/port lever.
- **Demotion/retirement:** a **flat curve to 32k** â‡’ retire L7 as *measured-no-prize* on
  this hardware generation (the curve is the retirement evidence, not a guess).

## 4. The honest witness â€” the full-MLA ctx-scaling curve (sizes the prize)

Per the lever map (L7's prior triage of record), the measurable artifact now is the
**full-MLA ctx curve**: decode (+ TTFT / prefill as reported by the witness) tok/s at ctx
{2k, 8k, 32k} on the resident serve. Flat â‡’ deprioritize; a cliff at depth â‡’ quantifies the
DSA headroom. Run **downstream of L1 (#3075)** at a **fixed FA setting** (record which),
node **GPU server 3 Â· Lane E**. Tools verified present in-tree:
`tools/glm52_mgpu_serve.sh` honors `CTX`/`PORT`/`DEVICES` (â†’ `--ctx-size`);
`tools/glm52_serving_witness.py` takes `--context-length` / `--out`.

```bash
for CTX in 2048 8192 32768; do
  CTX=$CTX PORT=8002 DEVICES=1,2,3,4,5,6,7 setsid bash tools/glm52_mgpu_serve.sh &
  until grep -q GLM52_MGPU_READY /tmp/glm52_mgpu/PHASE; do sleep 10; done
  python3 tools/glm52_serving_witness.py --base-url http://127.0.0.1:8002/v1 --model glm-5.2 \
    --context-length $CTX \
    --out experiments/benchmark/runs/by-machine/<gpu-server-3-id>/<UTCstamp>-glm52-l7-mla-ctx-curve/result-ctx$CTX.json
  pkill -f "llama-server.*port 8002" || true
done
```

| ctx | decode tok/s (full-MLA) | prefill/TTFT | scaling vs 2k |
|--:|--:|--:|--:|
| 2,048 | PENDING | PENDING | 1.00Ã— (def) |
| 8,192 | PENDING | PENDING | PENDING (ESTIMATED ~0.9â€“1.0Ã—, lever map) |
| 32,768 | PENDING | PENDING | PENDING â€” **this cell is the decision** |

Artifact layout (evidence convention of the cohort):

```
experiments/benchmark/runs/by-machine/<gpu-server-3-node-id>/<UTCstamp>-glm52-l7-mla-ctx-curve/
  manifest.json   # $schema: benchmark/run-manifest.v1; machine_id, timestamp, git rev/branch/dirty,
                  #   model {name: GLM-5.2 UD-Q4_K_M, precision: served}, config.claim_class: WITNESSED, scrubbed: true
  result-ctx*.json# one witness output per ctx point
  RESULTS.md      # the ctx table above, filled + the FA setting + the serve launch line per point
```

Headline to report on the issue: **the curve shape** (the prize L7 would capture) â€” not a
sparse-vs-dense speedup, which no run on this iron can honestly produce (Â§2). The curve
also feeds the long-context rows of the roofline dashboard (#3090), which is the issue's
own "why now."

## 5. The acceptance gate this worker cannot reach

**Blocker (host capability).** This worker runs on the GPU-less Windows dev box. The curve
needs the 433.82 GiB checkpoint served resident across GPU server 3's 8 GPUs, reached only
through the operator-gated `fak-private` control bridge
([docs/private-comms-channel.md](../private-comms-channel.md)) â€” no bridge session is
available in this session, and standing up the resident serve is an operator hardware
action. Inventing tok/s cells would violate the WITNESSED bar and the no-fabricated-pass
rule, so the Â§4 cells stay **PENDING**. Same gate as every sibling in the #3073 cohort.

**Smallest next step.** A GPU server-3 operator (post-L1, fixed FA) runs the Â§4 loop and
commits the artifact with a `(fak docs)` / `(fak <run-leaf>)` commit citing `#3088`.
**What closes #3088** is that artifact **plus** either (a) a real sparse arm becoming
runnable (re-promotion triggers, Â§3) and the A/B landing, or (b) an explicit re-scope of
the issue to the curve-only witness â€” **not this triage note**.

> **Do not auto-close #3088 on this note.** If a close-resolved / close-batch arm binds a
> #3088 reference here (or to the issue comment mirroring this triage), treat it as a false
> close and reopen until a recorded `experiments/benchmark/runs` artifact lands.

## 6. Cohort note

The #3073 lever cohort shares this shape â€” one lever, one WITNESSED
`experiments/benchmark/runs` artifact, triage note first when the witness is
operator-gated: [#3076 L4](GLM52-GPU-SERVER-LANEA-L4-FA-CUDAGRAPH-TRIAGE-2026-07-06.md) Â·
[#3077 L5](GLM52-L5-QUANT-SWEEP-TRIAGE-2026-07-06.md) Â·
[#3079 L2](GLM52-GPU-SERVER-LANEB-L2-CONTBATCH-TRIAGE-2026-07-06.md) Â·
[#3080 KV budget](GLM52-GPU-SERVER-LANEB-KV-BUDGET-TRIAGE-2026-07-06.md). L7 differs from all of
them in one way that matters: their levers are config flips awaiting an operator; **L7's
lever does not exist on this hardware**, so its only honest now-artifact is the curve that
decides whether the lever is worth building. Lowest priority of the resident levers (Pri 10,
lever map Â§3) â€” measure it downstream of everything cheaper.

*Companions:* [ceiling + lever map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) Â·
[costed lever map (L7 Â§5)](GLM52-L1-L10-COSTED-LEVER-MAP-2026-07-06.md) Â·
[8-GPU resident serve (23.2 tok/s WITNESSED)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md) Â·
[generation contract](../generation.md).
