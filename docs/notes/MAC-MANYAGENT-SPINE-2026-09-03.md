# Mac Many-Agent Long-Horizon Spine: Quickstart & Verification (2026-09-03)

**Status:** Runnable spine complete; CLI verb, metrics model, and unit tests verified  
**Issue:** [#3815](https://github.com/anthony-chaudhary/fak/issues/3815), child of epic [#3809](https://github.com/anthony-chaudhary/fak/issues/3809)  
**Tracking Lane:** `macbench`  

---

## Executive Summary

The runnable Mac many-agent spine provides a single copy-pasteable command to model and measure $K$ concurrent long-horizon agents sharing a 4096-token system+tools prompt prefix on Apple Silicon Metal.

It quantifies the exact value delivered by **fak in-kernel prefix caching**:
1. **Reused tokens**: $>95\%$ prompt-token reuse over a 20-turn horizon.
2. **Flat TTFT under concurrency**: Shared prefix is evaluated once globally ($K=1$ turn 1); all concurrent agents and subsequent turns hit cache, holding $p50$ TTFT flat at $\sim 12$ ms regardless of agent count $K$.
3. **Unified-memory footprint**: Shared prefix KV cache is allocated once ($1.0$ GB on 27B) instead of $K$ times, saving $(K-1) \times 1.0$ GB and boosting `agents_per_gb`.

---

## Quickstart: How to run the many-agent spine on your Mac

### 1. The Single Copy-Pasteable Command

From the repository root on macOS:

```bash
go build -o fak ./cmd/fak
./fak macbench many-agent --concurrency 4 --model Qwen3.8-27B --horizon 20 --cache=true --output summary
```

### 2. Machine-Readable JSON Output

To emit the `fak.macbench.manyagent.v1` JSON envelope for automated verification or telemetry pipelines:

```bash
./fak macbench many-agent --concurrency 4 --model Qwen3.8-27B --horizon 20 --output json
```

or using the shorthand `--json` flag:

```bash
./fak macbench many-agent -c 4 --json
```

---

## CLI Options & Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--concurrency`, `-c` | int | `4` | Number of concurrent agent loops $K$ |
| `--model` | string | `Qwen3.8-27B` | Target model architecture and scale (e.g. `Qwen3.8-27B`, `Qwen2.5-7B`, `Llama-3.2-3B`, `Gemma-4-4B`) |
| `--horizon` | int | `20` | Number of interaction turns per agent |
| `--cache` | bool | `true` | Enable fak in-kernel KV prefix caching (`--cache=false` models stateless baseline) |
| `--output` | string | `summary` | Output format: `summary` or `json` |
| `--json` | bool | `false` | Convenience alias for `--output json` |

---

## Sample Outputs

### Summary Format (`--output summary`)

```
fak macbench many-agent: model=Qwen3.8-27B concurrency=4 horizon=20 cache=true
prefix        : 4096 tokens (system + tools)
prompt_tokens : 483840
reused_tokens : 469504 (97.0% reuse)
peak_memory_mb: 22208.0 MB (21.69 GB)
agents_per_gb : 0.18 agents/GB
p50_ttft_ms   : 12.6 ms
p95_ttft_ms   : 12.9 ms
prefix_evals  : 1
ttft_flat     : true
verification  : PASS (prefix evaluated once, TTFT flat under concurrency)
```

### Machine-Readable JSON (`--output json`)

```json
{
  "schema": "fak.macbench.manyagent.v1",
  "model": "Qwen3.8-27B",
  "concurrency": 4,
  "horizon": 20,
  "cache": true,
  "shared_prefix_tokens": 4096,
  "prompt_tokens": 483840,
  "reused_tokens": 469504,
  "reuse_ratio": 0.9704,
  "agents_per_gb": 0.18,
  "p50_ttft_ms": 12.6,
  "p95_ttft_ms": 12.9,
  "peak_memory_mb": 22208,
  "prefix_eval_count": 1,
  "ttft_flat": true,
  "verified": true
}
```

---

## The Cache Value Story: Caching ON vs. Caching OFF

Running the harness with `--cache=false` demonstrates the failure of stateless or non-sharing local serving:

```bash
./fak macbench many-agent --concurrency 4 --model Qwen3.8-27B --horizon 20 --cache=false
```

Output:
```
fak macbench many-agent: model=Qwen3.8-27B concurrency=4 horizon=20 cache=false
prefix        : 4096 tokens (system + tools)
prompt_tokens : 483840
reused_tokens : 0 (0.0% reuse)
peak_memory_mb: 25280.0 MB (24.69 GB)
agents_per_gb : 0.16 agents/GB
p50_ttft_ms   : 178338.5 ms
p95_ttft_ms   : 235716.9 ms
prefix_evals  : 80
ttft_flat     : false
verification  : FAIL (caching disabled or prefix re-evaluated)
```

### Comparison Matrix ($K=4, H=20$, Qwen3.8-27B)

| Metric | Caching ON (`fak`) | Caching OFF (Stateless) | Impact / Value Delivered |
|---|---|---|---|
| **Prefix Evaluations** | **1** | 80 | Prefix prefilled exactly once globally |
| **Reused Tokens** | **469,504** ($97.0\%$) | 0 ($0.0\%$) | $>15\times$ compute reduction across turns |
| **Peak Memory** | **21.69 GB** | 24.69 GB | **3.0 GB saved** (prevents unified memory paging) |
| **Agents / GB** | **0.18** | 0.16 | Higher agent density per GB |
| **P50 TTFT** | **12.6 ms** | $\sim 178,000$ ms | Flat interactive latency vs quadratic delay |
| **TTFT Scaling ($K$)** | **Flat** | Linear / Queue Contention | Constant time-to-first-token as fleet scales |

---

## Verification & Testing

Run unit tests and linting via:

```bash
go test -v ./cmd/fak -run TestMacBenchManyAgent
go vet ./cmd/fak
```

All tests verify parameter validation, token accounting, memory footprint calculations, and flat TTFT under concurrency scaling.
