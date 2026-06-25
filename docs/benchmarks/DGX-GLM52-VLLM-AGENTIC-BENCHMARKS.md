---
title: "GPU-server GLM-5.2 vs vLLM agentic benchmark plan"
description: "A pending-measurement plan for running GLM-5.2 on vLLM through fak's gateway across the serving-agent benchmarks and a 20-task SWE-bench Verified slice, with artifact requirements and honesty fences."
---

# GPU-server GLM-5.2 vs vLLM — agentic benchmark plan

> **Status: pending measurement.** This document carries **commands and gates, not
> results**. No GLM-5.2/vLLM resolve-rate, tax, or throughput number may be quoted
> until the JSON artifacts named below are produced on a real serving node and
> linked from `BENCHMARK-AUTHORITY.md`.

## What We Are Comparing

There are two different comparisons, and the reports must keep them separate:

| axis | fair comparison | required artifact |
|---|---|---|
| **Gateway tax / completion** | GLM-5.2 served by **raw vLLM** vs the same raw vLLM endpoint behind `fak serve` | `experiments/vllm/adjudication-tax-witness.json`; SWE-bench `COMPARE-PREFLIGHT.json` + `compare.json` |
| **Native fak kernel vs stock engines** | same GLM-family checkpoint, same hardware, same precision/context/batch, fak native vs vLLM/SGLang/llama.cpp | future B2/B3 artifacts; the current `glmdsatput` numbers are synthetic kernel-cost only |

Do not compare fak's synthetic GLM kernel tok/s directly to a full-size vLLM
serving tok/s. The live vLLM run is full external-engine serving; the current
native fak GLM numbers are reduced-scale device-kernel cost.

## Run Matrix

| rung | benchmark | question answered | command / artifact |
|---|---|---|---|
| 0 | readiness | can this node serve GLM-5.2 with vLLM at all? | `tools/glm52_serve_preflight.py` -> `preflight.json` |
| 1 | serving witness | does GLM-5.2 answer direct, through fak, and quarantine a poisoned tool result? | `tools/glm52_serving_witness.py` -> `experiments/glm52/full-size-serving-witness.json` |
| 2 | vLLM adjudication tax | what latency/decode tax does `fak serve` add over raw vLLM? | `tools/vllm_tax_witness.py` -> `experiments/vllm/adjudication-tax-witness.json` |
| 3 | SWE-bench Verified 20 | does the agent finish/resolve the same 20 Verified tasks raw-vLLM vs fak-gateway? | `tools/dgx_swebench_compare.py --preflight-only`, then `--verified-count 20` -> `COMPARE-PREFLIGHT.json` + `compare.json` + `COMPARE.md` + `DONE.rc` |
| 4 | fak-native agentic floors | what are the deterministic turn-tax/session/fanout/cache-reuse floors independent of live model variance? | `fak swebench compare`, `turntax`, `sessionbench`, `fanbench`, `radixbench` artifacts |

Rungs 0-3 are the GLM-5.2/vLLM live-serving comparison. Rung 4 is the fak-native
agentic mechanism series; it explains what fak should move, but it is not a raw
vLLM head-to-head unless the workload also drives the same served endpoint.

## Reproduce On A GLM-5.2-Capable Node

Run this on a Hopper/Blackwell-class serving node. An Ampere sm_80 node is expected
to fail the stock vLLM GLM-5.2 preflight because stock DSA kernels require sm_90+.

```bash
# The fak-native SWE-bench floor needs bench's official difficulty map or a full
# SWE-bench Verified dataset export. The live resolve-rate arm below still grades
# through the official harness; this file is for the deterministic floor geometry.
: "${FAK_SWEBENCH_DIFFICULTY:?set FAK_SWEBENCH_DIFFICULTY to swebench_verified_difficulty.json}"

# Optional: write the auditable command manifest and check any existing artifacts.
# The generated run.sh contains the same commands below; run it instead of
# copy/pasting the rest of this block if you want a single runner. It runs the
# strict artifact gate at the end and fails if any result is missing or stale.
# On success, that final gate writes FINAL-CHECK.md and a guarded
# BENCHMARK-AUTHORITY-DRAFT.md snippet; no draft is written while pending.
python tools/glm52_vllm_agentic_battery.py \
  --out experiments/vllm/glm52-agentic-battery/manifest.json \
  --markdown experiments/vllm/glm52-agentic-battery/MANIFEST.md \
  --script experiments/vllm/glm52-agentic-battery/run.sh \
  --swebench-difficulty "$FAK_SWEBENCH_DIFFICULTY" \
  --allow-pending

# Before starting vLLM, set GLM52_TOOL_CALL_PARSER to the parser name required
# by the vLLM build/model card. This guard refuses to continue if it is unset.
: "${GLM52_TOOL_CALL_PARSER:?set GLM52_TOOL_CALL_PARSER to the vLLM parser name}"

# 0. Preflight the node and fail closed if GLM-5.2/vLLM is not viable.
python tools/glm52_serve_preflight.py \
  --engine vllm --quant fp8 --require-ready \
  --out experiments/vllm/glm52-vllm-preflight.json \
  --markdown experiments/vllm/glm52-vllm-preflight.md

# 1. Start raw vLLM for the GLM-5.2 fp8 checkpoint. Add the model-specific parser flags
# required by the vLLM build/model card via ENGINE_ARGS.
ENGINE=vllm SERVED_NAME=glm-5.2 PORT=8000 \
  ENGINE_ARGS="--enable-auto-tool-choice --tool-call-parser ${GLM52_TOOL_CALL_PARSER}" \
  bash tools/glm52_sglang_vllm_serve.sh

# 2. Prove direct + fak-gateway + quarantine behavior.
python tools/glm52_serving_witness.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model glm-5.2 \
  --engine-cache-engine vllm \
  --context-length 131072 \
  --out experiments/glm52/full-size-serving-witness.json \
  --markdown experiments/glm52/full-size-serving-witness.md

# 3. Measure the fak-over-vLLM gateway tax.
python tools/vllm_tax_witness.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model glm-5.2 \
  --count 8 \
  --record \
  --out experiments/vllm/adjudication-tax-witness.json \
  --markdown experiments/vllm/adjudication-tax-witness.md

# 4a. Fail fast before the long SWE-bench Verified comparison. This checks the
# runner/grader install, raw endpoint, GPU label, and fak binary.
python tools/dgx_swebench_compare.py \
  --engine vllm \
  --model zai-org/GLM-5.2-FP8 \
  --served-model-name glm-5.2 \
  --raw-base-url http://127.0.0.1:8000/v1 \
  --verified-count 20 \
  --skip-engine-serve \
  --require-tool-calls \
  --require-grade \
  --require-gpu-name H200 \
  --run-dir /tmp/swe-glm52-vllm-20 \
  --preflight-only

# 4b. Run the 20-task SWE-bench Verified comparison. This reuses raw vLLM and
# lets the driver start a fresh fak gateway in front of it.
python tools/dgx_swebench_compare.py \
  --engine vllm \
  --model zai-org/GLM-5.2-FP8 \
  --served-model-name glm-5.2 \
  --raw-base-url http://127.0.0.1:8000/v1 \
  --verified-count 20 \
  --skip-engine-serve \
  --require-tool-calls \
  --require-grade \
  --require-gpu-name H200 \
  --run-dir /tmp/swe-glm52-vllm-20
```

Use `--require-gpu-name B200` or an empty `--require-gpu-name ""` when the node is
not an H200. The requirement is an anti-mislabel guard, not the hardware rule; the
hardware rule is the preflight verdict.

## Agentic Floor Series

These commands can run before or after the live vLLM run. They should be recorded
as fak-native mechanism evidence, not as vLLM results:

```bash
go run ./cmd/fak swebench compare --difficulty "$FAK_SWEBENCH_DIFFICULTY" \
  --workers 1,2,4,8 --limit 20 --with-adjudication \
  --out experiments/vllm/swebench-20-fak-floor.json \
  --md experiments/vllm/swebench-20-fak-floor.md

go run ./cmd/fak turntax --suite turntax-airline \
  --out experiments/vllm/turntax-airline.json
go run ./cmd/sessionbench -synthetic smollm2-135m -turns 50 -agents 5 \
  -prefix 2048 -decode 32 -result 64 \
  -out experiments/vllm/sessionbench-synthetic.json
go run ./cmd/fanbench -profile research -trials 12 \
  -out experiments/vllm/fanbench-research.json \
  -csv experiments/vllm/fanbench-research.csv
go run ./cmd/radixbench -live=false \
  -out experiments/vllm/radixbench-synthetic.json
```

The SWE-bench floor uses the same 20-instance scale as the live run, but it is
deterministic geometry/adjudication evidence. The live `dgx_swebench_compare.py`
artifact is the resolve-rate evidence.

## Completion Bar

The benchmark is complete only when all of these are true:

- `preflight.json` says the serving node is `READY` or `READY_PENDING_INSTALL`.
- `full-size-serving-witness.json` has `summary.full_size_serving_witness == "PASS"`.
- `adjudication-tax-witness.json` reports measured raw-vLLM and fak-gateway legs.
- SWE-bench `COMPARE-PREFLIGHT.json` passes and records `config` plus `runtime`
  metadata for the runner, SWE-bench harness, vLLM, endpoints, and GPU guard.
  `compare.json` is for the `zai-org/GLM-5.2-FP8` checkpoint served as
  `glm-5.2` on raw `vllm`, with two arms, `raw-vllm` and `fak-gateway`, each
  with 20 submitted instances, a matching `selection_instance_ids` list, a
  passing OpenAI tool-call self-test, and official harness grades with
  `grade_rc == 0`, arm-specific `run_id`, `submitted == 20`, `report_path`,
  `grade_log`, and `resolved_ids` drawn from the selected instance IDs. The same
  run directory has `DONE.rc` equal to `0`.
- The fak-native floor artifacts pass their identity checks: SWE-bench floor has
  20 instances and workers `1,2,4,8`; turntax is `turntax-airline` with safety
  floor intact; sessionbench is synthetic `smollm2-135m` at `T=50,C=5,P=2048`;
  fanbench is the research grid with 12 trials; radixbench is the synthetic
  cache-hit/policy-eviction witness.
- Any number copied into docs is linked from `BENCHMARK-AUTHORITY.md` with the
  artifact path and reproduce command. The generated runner's final gate emits
  `experiments/vllm/glm52-agentic-battery/BENCHMARK-AUTHORITY-DRAFT.md` only
  after every required artifact passes.

Until then, the honest claim is only: the benchmark is wired and ready to run on a
GLM-5.2-capable serving node.
