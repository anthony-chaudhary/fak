# Multi-Turn Agentic Quality Gate for Inference Core

## Why single-turn gates miss multi-turn inference regressions

Single-turn logit, perplexity, and KL-divergence checks (`internal/logitoracle`) catch static numeric drift, but they miss regressions that only emerge across a multi-turn agentic trajectory:

- **KV-cache & prefix-reuse drift across turns:** quantized KV (`q4_0`/`int4`) or radix/prefix cache eviction bugs can corrupt attention state on turn 4+ while turn 1 logits look clean.
- **Tool-call argument & schema truncation:** speculative decoding (MTP) or output-token caps can cut off JSON tool arguments mid-stream, breaking tool-call parsing across replayed turns.
- **Non-consecutive repetition cycles:** degraded quants or sampling settings often enter 2- to 3-step loops (`Read` -> `Bash` -> `Read` -> `Bash` -> `Read`) that exhaust turn budgets without triggering consecutive `repeat_loop` detectors.

## Industry benchmarks evaluated

| Benchmark | License | What it measures | Role in fak |
| --- | --- | --- | --- |
| **`fak bench agent tasks`** (`internal/agentbench`) | Apache-2.0 (in-repo) | 8-task (`default`) / 35-task (`extended`) multi-turn Go bug-fix trajectories with hidden oracles, tool-floor enforcement, and cycle detection against any OpenAI-compatible `/v1/chat/completions` endpoint | **Tier 1 gate** (fast, zero-Docker, deterministic) |
| **BFCL v3 Multi-Turn** | Apache-2.0 | Multi-turn tool selection, missing-parameter recovery, and stateful execution across external APIs | **Tier 2 external standard** (scheduled/release candidate) |
| **SWE-bench Verified** (`fak swebench`) | MIT | Long-horizon multi-file repository bug fixing in Docker | **Tier 2 release qualification** (heavy, multi-hour) |
| **tau-bench** | MIT | Multi-turn domain policy + DB state | Deferred (user-simulator LLM introduces judge variance) |
| **Terminal-Bench** | Apache-2.0 | Containerized CLI/terminal workflows | Tier 2 complementary (container-heavy) |
| **InferenceX AgentX** (`fak agentx`) | — | Serving throughput/latency replay (scores no answer quality) | Throughput/latency only; not a quality gate |

## Two-tier inference-core gate architecture

### Tier 1: Fast per-change / kernel / quant / KV-cache gate (`fak bench agent`)

Run the multi-turn Go coding suite against any OpenAI-compatible inference endpoint (native `fak serve`, llama.cpp, vLLM, SGLang, or any agent harness endpoint), then evaluate the summary with `fak bench agent gate`:

```bash
# 1. Run the multi-turn agentic suite (8 tasks default, or --suite extended --include-heldout for 35 tasks)
fak bench agent tasks \
  --endpoint http://127.0.0.1:8080/v1 \
  --model <model-id> \
  --out _scratch/agentbench-candidate \
  --temperature 0 --top-p 1 --top-k 0 --max-tokens 1024 --seed 42

# 2. Gate on absolute floors (standalone)
fak bench agent gate --candidate _scratch/agentbench-candidate

# 3. Gate against a matched baseline (floor + paired task regression check)
fak bench agent gate \
  --candidate _scratch/agentbench-candidate \
  --baseline _scratch/agentbench-baseline \
  --require-pinned-sampling
```

### Tier 2: Scheduled & release qualification

Scheduled GPU qualification pairs Tier 1 `fak bench agent tasks --suite extended --include-heldout` (35 tasks) with `fak swebench` and planned BFCL v3 Multi-Turn conformance runs.

## Gate contract (`fak.agentbench.quality-gate.v1`)

`fak bench agent gate` is **fail-closed**:

- **Exit codes:** `0` = `PASS`, `1` = `FAIL`, `3` = `INCONCLUSIVE` (missing/unmeasured preconditions), `2` = CLI usage or I/O error.
- **Preconditions (`INCONCLUSIVE`):** `RUN_ERROR`, `INSUFFICIENT_REPS`, `INSUFFICIENT_TRIALS`, `UNMEASURED_TOOL_CALLS`, `SAMPLING_UNPINNED`, `SUITE_MISMATCH`, `NO_PAIRED_TASKS`.
- **Absolute floors (`FAIL`):**
  - `ACCEPT_RATE_BELOW_FLOOR` (default `min-accept-rate = 0.75`)
  - `TOOL_VALIDITY_BELOW_FLOOR` (default `min-tool-validity = 0.98`)
  - `LOOP_RATE_ABOVE_CEILING` (default `max-loop-rate = 0.20`)
  - `CYCLIC_REPEAT_RATE_ABOVE_CEILING` (default `max-cyclic-repeat-rate = 0.25`, catching non-consecutive 3+ identical calls)
- **Paired baseline regression checks (`FAIL` when `--baseline` is provided):**
  - `ACCEPT_RATE_REGRESSION` (default `max-accept-drop = 0.05`)
  - `TOOL_VALIDITY_REGRESSION` (default `max-tool-validity-drop = 0.02`)
  - `LOOP_RATE_REGRESSION` (default `max-loop-increase = 0.05`)
  - `PAIRED_TASK_REGRESSION` (exact two-sided binomial sign test over discordant tasks, `p <= 0.05` with candidate worse than baseline)
