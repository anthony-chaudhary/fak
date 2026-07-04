---
title: "vLLM EP/EPLB MoE serving baseline runbook"
description: "The reproducible serve command, load-generation path, and artifact schema for running a GLM/Qwen/DeepSeek-class MoE through vLLM with Expert Parallelism (EP) and the Expert Parallel Load Balancer (EPLB) enabled — the SOTA serving floor that native fak EP-sharded GGUF work must beat or explain away. Pending measurement: commands and schema, not results."
---

# vLLM EP/EPLB MoE serving baseline runbook

> **Status: pending measurement.** This document carries the **reproducible serve
> command, the load-generation path, and the artifact schema — not results.** No
> TTFT/TPOT/ITL, expert-balance, or throughput number may be quoted until the JSON
> artifact named in [Artifact schema](#artifact-schema) is produced on a real EP-capable
> GPU node and linked from [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).

Resolves [#1733](https://github.com/anthony-chaudhary/fak/issues/1733). Parent context:
[#1050](https://github.com/anthony-chaudhary/fak/issues/1050),
[#1476](https://github.com/anthony-chaudhary/fak/issues/1476),
[#1728](https://github.com/anthony-chaudhary/fak/issues/1728).

## Why this is the baseline, not a fak result

Before fak's native MoE expert placement is claimed as a win, the SOTA path it competes
with must be run and recorded first: **vLLM serving a large MoE with EP + EPLB enabled.**
vLLM is the reference implementation of expert parallelism for MoE serving (it ships a
concrete EPLB policy in `vllm/distributed/eplb/policy/default.py` and documents the path in
`docs/serving/expert_parallel_deployment.md`, local vLLM snapshot `28242824e`). This runbook
is the **benchmark floor**: the native fak EP-sharded GGUF track
([GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29](../notes/GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md))
must cite this baseline, or explain why the model/hardware path is not comparable, before any
"native EP beats X" claim is made.

This is **not** the [#870 GLM-5.2/vLLM agentic battery](GPU-SERVER-GLM52-VLLM-AGENTIC-BENCHMARKS.md)
(that measures gateway tax + agentic resolve-rate over a served endpoint). This runbook measures
the **raw serving profile of the MoE expert-parallel path itself** — decode/prefill latency and
routed-expert load balance — which is the quantity native fak EP must move.

## What the baseline captures

| axis | metric | source |
|---|---|---|
| decode/prefill latency | TTFT, TPOT, ITL (p50/p99), end-to-end throughput | `vllm bench serve` summary JSON |
| routed expert balance | per-expert token counts / EPLB balancedness, **before vs after** EPLB rearrangement | EPLB `log_balancedness` (see [expert-balance capture](#expert-load-balance-capture)) |
| cache behavior | prefix-cache hit rate, KV utilisation under the load sweep | vLLM Prometheus `/metrics` scrape at run end |
| failure modes | OOM, expert-map init failure, EPLB rebalance stalls, sm_XX kernel refusal | serve stderr + preflight verdict, recorded verbatim |

## Reproduce on an EP-capable GPU node

Run on a multi-GPU node whose per-GPU VRAM × count holds the MoE checkpoint's expert bulk
resident (see the capacity table in the
[native EP note](../notes/GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md#the-benchmark-this-makes-answerable-does-glm-52-fit-resident-across-8-gpus)).
Stock vLLM DSA kernels require sm_90+; an Ampere sm_80 node is expected to fail the GLM-5.2
preflight (Qwen3/DeepSeek FP8 paths are the sm_80-viable fallbacks).

EP width = `data_parallel_size × tensor_parallel_size`; the example below serves one node with
DP=8, TP=1 (8 EP ranks). Flag names are pinned to current vLLM (the consolidated `--eplb-config`
JSON form; the PR-era `--num-redundant-experts` / `--eplb-window-size` flags are superseded).
Confirm against the model card and the pinned snapshot before quoting a result.

```bash
set -euo pipefail
: "${MOE_MODEL:?set MOE_MODEL, e.g. deepseek-ai/DeepSeek-V3-0324 | Qwen/Qwen3-30B-A3B | zai-org/GLM-5.2-FP8}"
: "${OUT_DIR:?set OUT_DIR, e.g. experiments/vllm-ep-eplb/<model>-<date>}"
: "${DP:=8}"; : "${TP:=1}"; : "${PORT:=8000}"
mkdir -p "$OUT_DIR"

# 0. Preflight — fail closed if this node cannot serve the MoE with EP at all.
#    Record the verdict (READY / READY_PENDING_INSTALL / UNSUPPORTED) verbatim; an
#    UNSUPPORTED verdict is itself a recorded failure mode, not a skipped step.
python tools/glm52_serve_preflight.py \
  --engine vllm --require-ready \
  --out "$OUT_DIR/preflight.json" \
  --markdown "$OUT_DIR/preflight.md" || echo "preflight non-ready — record verdict, do not quote results"

# 1. ARM A — EP ON, EPLB OFF (the "before EPLB" balance reference).
#    --enable-expert-parallel shards the experts across the DP×TP ranks.
vllm serve "$MOE_MODEL" \
  --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --port "$PORT" \
  > "$OUT_DIR/serve-armA-eplb-off.log" 2>&1 &

# 2. ARM B — EP ON, EPLB ON (the "after EPLB" arm). log_balancedness:true exposes the
#    per-step expert-load balancedness so the harness can record before/after. If a
#    given vLLM build does not surface the metric, mark expert_load_balance.available=false
#    in the artifact — do NOT synthesise a number (acceptance item 2).
vllm serve "$MOE_MODEL" \
  --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
  --enable-expert-parallel --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}' \
  --port "$((PORT+1))" \
  > "$OUT_DIR/serve-armB-eplb-on.log" 2>&1 &

# 3. Drive the same request load against each arm and capture TTFT/TPOT/ITL.
for arm in "armA:$PORT" "armB:$((PORT+1))"; do
  name="${arm%%:*}"; p="${arm##*:}"
  vllm bench serve \
    --backend openai-chat --model "$MOE_MODEL" \
    --base-url "http://127.0.0.1:${p}" \
    --dataset-name random --num-prompts 512 \
    --random-input-len 1024 --random-output-len 256 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-filename "$OUT_DIR/bench-${name}.json"
  # cache + KV behaviour at run end
  curl -s "http://127.0.0.1:${p}/metrics" > "$OUT_DIR/metrics-${name}.prom"
done
```

## Expert-load balance capture

Acceptance item 2 is the honest one: **record expert-load balance before/after EPLB when
vLLM exposes it, or explicitly mark it unavailable.** Two exposure paths, in order of
preference:

1. **EPLB `log_balancedness`** — with `"log_balancedness":true` in `--eplb-config`, vLLM
   logs a per-step balancedness scalar (mean load / max load across experts). Parse it from
   `serve-armB-eplb-on.log`; the ARM-A log (EPLB off) is the "before" reference. Record both.
2. **Prometheus per-expert counters** — where the build exports them, scrape `/metrics` for
   the routed-expert token histogram and compute the imbalance ratio directly.

If **neither** is exposed by the build under test, the artifact records
`expert_load_balance.available = false` with `source = "unavailable"` and a one-line reason
(e.g. `"log_balancedness not surfaced in vLLM <commit>"`). An unavailable balance metric is a
recorded, honest state — never a fabricated ratio.

## Artifact schema

One JSON artifact per (model, hardware) run, `fak.vllm-ep-eplb-moe-baseline.v1`, written to
`$OUT_DIR/vllm-ep-eplb-baseline.json`:

```json
{
  "schema": "fak.vllm-ep-eplb-moe-baseline.v1",
  "issue": 1733,
  "result_claim_allowed": false,
  "model": "deepseek-ai/DeepSeek-V3-0324",
  "vllm_commit": "28242824e",
  "hardware": {"gpu": "H200", "count": 8, "vram_gib_per_gpu": 141},
  "parallel": {"data_parallel_size": 8, "tensor_parallel_size": 1, "ep_width": 8},
  "vllm_compile": {"class": "tuned|cold-start|diagnostic", "note": "see BENCHMARK-CONTRACT-MAP.md#serving-baseline-provenance-the-vllm_compile-block-1731"},
  "arms": {
    "eplb_off": {"ttft_ms": {"p50": null, "p99": null}, "tpot_ms": {"p50": null, "p99": null}, "itl_ms": {"p50": null, "p99": null}, "output_tok_per_s": null, "prefix_cache_hit_rate": null},
    "eplb_on":  {"ttft_ms": {"p50": null, "p99": null}, "tpot_ms": {"p50": null, "p99": null}, "itl_ms": {"p50": null, "p99": null}, "output_tok_per_s": null, "prefix_cache_hit_rate": null}
  },
  "expert_load_balance": {
    "available": false,
    "source": "log_balancedness|prometheus|unavailable",
    "before_eplb": null,
    "after_eplb": null,
    "metric": "balancedness (mean_load/max_load across experts; 1.0 = perfectly balanced)",
    "reason_if_unavailable": "log_balancedness not surfaced in vLLM <commit>"
  },
  "eplb_config": {"window_size": 1000, "step_interval": 3000, "num_redundant_experts": 2, "log_balancedness": true},
  "failure_modes": [],
  "artifacts": ["bench-armA.json", "bench-armB.json", "metrics-armA.prom", "metrics-armB.prom", "serve-armA-eplb-off.log", "serve-armB-eplb-on.log", "preflight.json"]
}
```

`null` fields stay `null` until the real run fills them. `result_claim_allowed` flips to `true`
only when the completion bar below is met and the numbers are linked from
`BENCHMARK-AUTHORITY.md`. Every quoted-baseline row must also carry the `vllm_compile` tuned-gate
block from [BENCHMARK-CONTRACT-MAP.md](BENCHMARK-CONTRACT-MAP.md#serving-baseline-provenance-the-vllm_compile-block-1731)
— a silently cold vLLM baseline poisons any downstream fak comparison.

## Completion bar

The baseline is complete only when all of these hold:

- `preflight.json` verdict is `READY` (or `READY_PENDING_INSTALL`); an `UNSUPPORTED`
  verdict is recorded as the result, and no latency/balance number is quoted.
- Both arms (`eplb_off`, `eplb_on`) have populated TTFT/TPOT/ITL percentiles from
  `vllm bench serve` over the **same** prompts, input/output lengths, and concurrency.
- `expert_load_balance` is either populated `before_eplb` **and** `after_eplb`, or carries
  `available:false` with a source reason — never a synthesised number.
- Each arm's `vllm_compile.class == "tuned"` (cache warm, CUDA-graph captured, no
  request-time compilation inside the measured window).
- Every number copied into docs is linked from `BENCHMARK-AUTHORITY.md` with the artifact
  path and reproduce command.

Until then, the honest claim is only: **the vLLM EP/EPLB MoE baseline is wired and ready to run
on an EP-capable serving node.**

## Comparability boundary (do not confound)

- This baseline is **full external-engine serving** on an FP8 (or model-card-native) checkpoint.
  Do **not** compare it directly to fak's synthetic reduced-scale native GLM kernel tok/s, or to
  the host cpu-offload / host-DistComm EP numbers — those are device-kernel-cost and
  above-the-device-line residency proofs, not full-checkpoint served throughput.
- A native fak EP claim must serve the **same model family, hardware, precision, context, and
  concurrency** before a delta is attributable to fak; otherwise the row is `[NOT COMPARABLE]`
  and must say so (native EP is still gated on the on-box multi-GPU `-tags cuda,nccl` binary and a
  live device-NCCL reduce — the residual named in the native EP note).

## Cross-links

- Native EP-sharded track (the side that must cite this floor):
  [GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29](../notes/GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md).
- Serving-baseline tuned-gate provenance: [BENCHMARK-CONTRACT-MAP.md](BENCHMARK-CONTRACT-MAP.md#serving-baseline-provenance-the-vllm_compile-block-1731) (`internal/vllmcompile`).
- Sibling vLLM plans: [GPU-SERVER-GLM52-VLLM-AGENTIC-BENCHMARKS.md](GPU-SERVER-GLM52-VLLM-AGENTIC-BENCHMARKS.md) (agentic/gateway-tax), [VLLM-HEADTOHEAD-RESULTS.md](VLLM-HEADTOHEAD-RESULTS.md) (engine-tax scaffold).
- Authority ledger (the only place a number becomes authoritative): [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
