# GLM-5.2 vLLM Agentic Benchmark Battery

- Generated: `2026-06-25T23:16:18.571294Z`
- Model family: `zai-org/GLM-5.2`
- Checkpoint: `zai-org/GLM-5.2-FP8` served as `glm-5.2`
- Raw vLLM endpoint: `http://127.0.0.1:8000/v1`
- Status: **`PENDING_MEASUREMENT`**
- Required artifacts passed: `3/10`

> Pending measurement: this manifest contains commands and artifact checks, not benchmark results.

| step | kind | artifact | status | detail |
|---|---|---|---|---|
| `preflight` | live-readiness | `experiments/vllm/glm52-vllm-preflight.json` | FAIL | node_verdict='BLOCKED_ARCH'; vllm.ready=False; expected GLM-5.2 vLLM READY/PENDING preflight |
| `serve_raw_vllm` | manual-live-serving | `(manual)` | MANUAL | manual serving step; witnessed by downstream artifacts |
| `serving_witness` | live-serving | `experiments/glm52/full-size-serving-witness.json` | MISSING | missing |
| `vllm_tax` | live-serving | `experiments/vllm/adjudication-tax-witness.json` | MISSING | missing |
| `swebench_compare_preflight` | live-agentic-preflight | `/tmp/swe-glm52-vllm-20/COMPARE-PREFLIGHT.json` | MISSING | missing |
| `swebench_verified_20` | live-agentic | `/tmp/swe-glm52-vllm-20/compare.json` | MISSING | missing |
| `swebench_floor_20` | fak-native-floor | `experiments/vllm/swebench-20-fak-floor.json` | MISSING | missing |
| `turntax_airline` | fak-native-floor | `experiments/vllm/turntax-airline.json` | PASS | turntax-airline turns_saved=9 |
| `sessionbench_synthetic` | fak-native-floor | `experiments/vllm/sessionbench-synthetic.json` | MISSING | missing |
| `fanbench_research` | fak-native-floor | `experiments/vllm/fanbench-research.json` | PASS | fanbench research grid/trials present |
| `radixbench_synthetic` | fak-native-floor | `experiments/vllm/radixbench-synthetic.json` | PASS | radixbench agents hit_rate=0.8666666666666667 |

## Commands

### preflight

Fail-closed GLM-5.2/vLLM node readiness gate.

```bash
python tools/glm52_serve_preflight.py --engine vllm --quant fp8 --require-ready --out experiments/vllm/glm52-vllm-preflight.json --markdown experiments/vllm/glm52-vllm-preflight.md
```

### serve_raw_vllm

Start raw vLLM for GLM-5.2 before the live witnesses.

```bash
: "${GLM52_TOOL_CALL_PARSER:?set GLM52_TOOL_CALL_PARSER to the vLLM parser name}" && ENGINE=vllm SERVED_NAME=glm-5.2 PORT=8000 ENGINE_ARGS="--enable-auto-tool-choice --tool-call-parser ${GLM52_TOOL_CALL_PARSER}" bash tools/glm52_sglang_vllm_serve.sh
```

### serving_witness

Direct + fak-gateway + quarantine serving witness.

```bash
python tools/glm52_serving_witness.py --base-url http://127.0.0.1:8000/v1 --model glm-5.2 --engine-cache-engine vllm --context-length 131072 --out experiments/glm52/full-size-serving-witness.json --markdown experiments/glm52/full-size-serving-witness.md
```

### vllm_tax

Measure fak gateway adjudication tax over raw vLLM.

```bash
python tools/vllm_tax_witness.py --base-url http://127.0.0.1:8000/v1 --model glm-5.2 --count 8 --record --out experiments/vllm/adjudication-tax-witness.json --markdown experiments/vllm/adjudication-tax-witness.md
```

### swebench_compare_preflight

Fail fast before the long SWE-bench agent run.

```bash
python tools/dgx_swebench_compare.py --engine vllm --model zai-org/GLM-5.2-FP8 --served-model-name glm-5.2 --raw-base-url http://127.0.0.1:8000/v1 --verified-count 20 --skip-engine-serve --require-tool-calls --require-grade --run-dir /tmp/swe-glm52-vllm-20 --require-gpu-name H200 --preflight-only
```

### swebench_verified_20

Run raw-vLLM vs fak-gateway on a 20-task SWE-bench Verified slice.

```bash
python tools/dgx_swebench_compare.py --engine vllm --model zai-org/GLM-5.2-FP8 --served-model-name glm-5.2 --raw-base-url http://127.0.0.1:8000/v1 --verified-count 20 --skip-engine-serve --require-tool-calls --require-grade --run-dir /tmp/swe-glm52-vllm-20 --require-gpu-name H200
```

### swebench_floor_20

Record the deterministic fak-native SWE-bench geometry floor at the same 20-task scale.

```bash
go run ./cmd/fak swebench compare --workers 1,2,4,8 --limit 20 --with-adjudication --out experiments/vllm/swebench-20-fak-floor.json --md experiments/vllm/swebench-20-fak-floor.md
```

### turntax_airline

Measure the turn-tax safety/control floor.

```bash
go run ./cmd/fak turntax --suite turntax-airline --out experiments/vllm/turntax-airline.json
```

### sessionbench_synthetic

Measure the multi-agent session value stack on a weightless synthetic shape.

```bash
go run ./cmd/sessionbench -synthetic smollm2-135m -turns 50 -agents 5 -prefix 2048 -decode 32 -result 64 -out experiments/vllm/sessionbench-synthetic.json
```

### fanbench_research

Measure fan-out prefix sharing and cost geometry.

```bash
go run ./cmd/fanbench -profile research -trials 12 -out experiments/vllm/fanbench-research.json -csv experiments/vllm/fanbench-research.csv
```

### radixbench_synthetic

Measure RadixAttention-style prefix-cache hit-rate evidence.

```bash
go run ./cmd/radixbench -live=false -out experiments/vllm/radixbench-synthetic.json
```

Missing required artifacts: `serving_witness`, `vllm_tax`, `swebench_compare_preflight`, `swebench_verified_20`, `swebench_floor_20`, `sessionbench_synthetic`.
Failed required artifacts: `preflight`.

No GLM-5.2/vLLM benchmark number is quotable until every required artifact passes and any copied number is linked from BENCHMARK-AUTHORITY.md.
