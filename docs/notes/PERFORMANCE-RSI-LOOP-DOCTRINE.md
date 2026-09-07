---
title: "Performance-RSI loop doctrine: continuous acceleration toward the 100x target"
description: "Operational doctrine for accelerating the performance RSI loop toward the unsaturated 100x target: 16 canonical dimensions, dominant bottleneck derivation, empirical evidence requirements without simulated results, CLI invocation, and loop turn telemetry."
date: 2026-08-28
---

# Performance-RSI loop doctrine: continuous acceleration toward the 100x target

This doctrine establishes the operational, architectural, and verification invariants for the **Performance Recursive Self-Improvement (RSI) loop** within the Fused Agent Kernel (`fak`). It provides the decision framework for autonomous agents and human performance engineers continuously compounding inference speed, efficiency, and resource utilization toward the explicit, unsaturated target multiplier of **100.0x** (`TargetMultiplier = 100.0`).

The core premise of the performance-RSI loop is empirical rigor: **engineering is building loops, and an RSI loop is only as fast as its dominant bottleneck**. Improvements cannot be established through narrative estimates, simulated benchmarks, or unconstrained comparisons. Every advancement must be backed by typed, versioned, machine-readable receipts from genuine execution environments.

---

## 1. The 16 Canonical Dimensions

The performance-RSI loop evaluates progress across **16 canonical dimensions**, divided into five owning receipt families. Over-performing in one dimension does not compensate for deficits or unmeasured debt in another.

| # | Dimension ID | Owning Section / Schema | Direction | Canonical Unit | Definition and Derivation |
|---|---|---|---|---|---|
| 1 | `cycle_time` | Cycle (`fak-performance-rsi-cycle/1`) | `lower` | `hours` (or `seconds`) | Total elapsed wall time from hypothesis conception (`idea_at`) to learning capture (`learning_at`). Measures end-to-end velocity of the optimization loop. |
| 2 | `improvement_yield` | Improvement (`fak-performance-rsi-improvement/1`) | `higher` | `percent` | Net true gain of the optimization: `(baseline - (candidate + overhead)) / baseline * 100`. Must strictly deduct wrapper, runtime, or measurement overhead. |
| 3 | `evaluation_latency` | Cycle (`fak-performance-rsi-cycle/1`) | `lower` | `seconds` | Elapsed duration of benchmark and verification evaluation (`evaluation_at - execution_at`). Minimizing evaluation latency directly accelerates the inner iteration loop. |
| 4 | `receipt_coverage` | Improvement (`fak-performance-rsi-improvement/1`) | `higher` | `percent` | Percentage of performance claims corroborated by independently verifiable, schema-valid JSON receipts. Unreceipted assertions contribute 0%. |
| 5 | `quality_gate_coverage` | Improvement (`fak-performance-rsi-improvement/1`) | `higher` | `percent` | Rigorous quality and regression parity: baseline and candidate must both pass functional, accuracy, and safety gates with strict parity (`baseline_passed && candidate_passed && parity`). |
| 6 | `experiment_throughput` | Cycle (`fak-performance-rsi-cycle/1`) | `higher` | `experiments_per_day` | Number of distinct empirical experiments evaluated per unit time over the cycle duration (`1 / (cycle_seconds * period)`). |
| 7 | `hypothesis_calibration` | Learning (`fak-performance-rsi-learning/1`) | `higher` | `percent` | Calibration accuracy between predicted improvement and observed outcome across iterations, confidence-weighted: `100 - (sum(confidence * |predicted - observed|) / sum(confidence))`. |
| 8 | `discovery_freshness` | Provenance (`fak-performance-rsi-provenance/1`) | `lower` | `hours` | Time elapsed from external upstream/literature publication to internal discovery and triage (`discovery_at - source.published_at`). |
| 9 | `adaptation_speed` | Provenance (`fak-performance-rsi-provenance/1`) | `lower` | `hours` | Wall time elapsed from starting adaptation of known art to landing the change (`production.landed_at - adaptation_started_at`). |
| 10 | `reuse_ratio` | Provenance (`fak-performance-rsi-provenance/1`) | `higher` | `percent` | Proportion of architectural mechanisms borrowed/adapted versus reinvented from scratch: `reused / (reused + reinvented) * 100`. |
| 11 | `learning_retention` | Learning (`fak-performance-rsi-learning/1`) | `higher` | `percent` | Systematic capture and reuse of negative and positive results: `valid_reuses / eligible_recurrences * 100`, penalized by repeated failures on the same recurrence key. |
| 12 | `production_transfer` | Provenance (`fak-performance-rsi-provenance/1`) | `lower` | `hours` | Total elapsed duration from discovery of an external technique to its verified landing on the main branch (`production.landed_at - discovery_at`). |
| 13 | `hardware_utilization` | Hardware (`fak-performance-rsi-hardware/1`) | `higher` | `percent` | Duration-weighted active accelerator utilization across measured runs: `sum(active_utilization * duration) / sum(duration)`. Terminal blockers (e.g. `local-no-gpu`) are rejected. |
| 14 | `attribution_quality` | Improvement (`fak-performance-rsi-improvement/1`) | `higher` | `percent` | Empirical causal attribution: requires an isolating ablation where `treatment_value < control_value` in milliseconds, proving the specific code delta drove the gain. |
| 15 | `automation_coverage` | Cycle (`fak-performance-rsi-cycle/1`) | `higher` | `percent` | Fraction of cycle execution executed without human operator intervention: `(1 - operator_active_seconds / cycle_seconds) * 100`. |
| 16 | `compounding_rate` | Learning (`fak-performance-rsi-learning/1`) | `higher` | `percent` | Velocity acceleration across successive iterations: reduction in cycle time when prior learnings are reused on related recurrence keys. |

---

## 2. Dominant Bottleneck Derivation

The performance-RSI loop adopts an algorithmic variant of Goldratt's Theory of Constraints and Amdahl's Law: **optimizing a non-bottleneck dimension yields zero acceleration for the compound system**.

### Normalized Ratio Calculation

Every dimension $d_i$ has a current measured value $c_i$, an explicit target $t_i$, and an optimization direction:
- For **higher-is-better** dimensions (`Direction = "higher"`):
  $$\text{ratio}_i = \frac{c_i}{t_i}$$
- For **lower-is-better** dimensions (`Direction = "lower"`):
  $$\text{ratio}_i = \begin{cases} \infty & \text{if } c_i = 0 \\ \frac{t_i}{c_i} & \text{if } c_i > 0 \end{cases}$$

### Status Classification

- **`MET`**: $\text{ratio}_i \ge 1.0$. The dimension meets or exceeds the target.
- **`BEHIND`**: $\text{ratio}_i < 1.0$. The dimension is measured but trails the target.
- **`UNKNOWN`**: $c_i$ or $t_i$ is `nil`. The dimension lacks empirical measurement, contributing infinite debt.

### Bottleneck Selection

The **dominant bottleneck** is deterministically derived as:
$$\text{dominant\_bottleneck} = \arg\min_{i} (\text{debt}_i)$$
where $\text{debt}_i = \text{ratio}_i$ when measured, or $+\infty$ when unmeasured. If any dimensions are `UNKNOWN`, the first unmeasured dimension in canonical ordering is selected as the dominant bottleneck.

When all dimensions are measured, the dimension with the lowest normalized ratio strictly dictates the next engineering priority. Optimizing a dimension with $\text{ratio} = 1.2$ while another languishes at $\text{ratio} = 0.15$ is a misallocation of optimization capacity.

### Loop Health and Performance-RSI Debt

Loop health grades the rigor and maturity of the measurement feedback loop itself, not whether the 100x performance target has been reached:
- Credit per dimension is capped at $1.0$: $\min(\text{ratio}_i, 1.0)$. Over-achieving in one dimension cannot mask unmeasured or lagging dimensions.
- **Loop-health score**: $100 \times \frac{\sum_{i} \min(\text{ratio}_i, 1.0)}{16}$.
- **Performance-RSI debt**: $\text{count}(\text{BEHIND}) + \text{count}(\text{UNKNOWN})$.
- **Clean status**: $\text{debt} == 0$.

---

## 3. Strict Evidence Requirements

Self-reported performance numbers, unverifiable summaries, and optimistic estimates are refused by the kernel. The following five invariants govern all accepted performance evidence:

### 1. No Simulated or Synthetic Results as Shipped Wins
Simulated execution, analytical approximations, and model-based performance estimates are valid only as design-time heuristics or claim ceilings. They may never substitute for empirical measurement in an improvement receipt or hardware ledger. Any claim of a shipped performance gain requires real physical hardware execution.

### 2. No `llama.cpp` Fallback in Native Claims
All benchmark, improvement, and hardware receipts claiming native engine performance must explicitly identify `engine: "fak-native"`. An uncompiled or failing native path that falls back to `llama.cpp` or an external runtime fails receipt validation immediately:
```text
dimension "improvement_yield": llama.cpp fallback is not native evidence
```
`llama.cpp` is permitted exclusively for external parity baseline comparisons, migration reference points, or borrow investigations—never as an undercover proxy for native execution.

### 3. Matched Operating Envelopes
To prevent apples-to-oranges comparisons, the baseline and candidate measurements in an improvement receipt must share an identical, fully specified `OperatingEnvelope`:
- `model`: Exact model name and parameter scale (e.g. `Qwen3.8-27B`).
- `quantization`: Exact quantization schema (e.g. `Q4_K_M`, `W4A8`).
- `hardware`: Specific accelerator/host specification (e.g. `1x NVIDIA L4`, `Apple M3 Pro 36GB`).
- `workload`: Workload profile and benchmark suite (e.g. `50-turn agentic coding trace`).
- `context_tokens`: Identical context length (e.g. `16384`).
- `batch_size`: Identical batch concurrency (e.g. `1`).

Any discrepancy between baseline and candidate envelopes invalidates the receipt.

### 4. Causal Ablations with Net True Gain
Every improvement claim must provide:
- A causal ablation demonstrating that the isolated code change was the sole driver of the performance delta (`treatment_value < control_value`).
- A net true gain calculation that strictly includes overhead:
  $$\text{net\_true\_gain} = \frac{\text{baseline} - (\text{candidate} + \text{overhead})}{\text{baseline}} \times 100$$
If an optimization saves 10 ms of decode time but introduces 12 ms of dispatch or memory-management overhead, the net true gain is negative and the change is rejected.

### 5. Measured Hardware Runs vs. Terminal Blockers
Hardware utilization receipts (`fak-performance-rsi-hardware/1`) require active accelerator sampling timestamps in UTC. Queuing delays and terminal blockers (such as `local-no-gpu`) are metadata or blockers; they are explicitly rejected if submitted as utilization measurements:
```text
hardware run 0 terminal_evidence type "local-no-gpu": local-no-GPU is a terminal blocker, not a hardware utilization measurement
```

---

## 4. The CLI Interface: `fak performance-rsi-scorecard`

The `fak performance-rsi-scorecard` command is the machine and operator interface for scoring, comparing, and composing performance RSI evidence.

### Command Syntax

```bash
# Evaluate and score a complete evidence document
fak performance-rsi-scorecard -input <file.json> [-json] [-markdown] [-prior <prior.json>]

# Compose independently produced receipts into a unified evidence document
fak performance-rsi-scorecard compose -snapshot <name> [receipts...]
```

### Flags and Options

- `-input string`: Path to a versioned performance RSI evidence JSON document (`fak-performance-rsi-evidence/1`). Required for evaluation.
- `-json`: Emits structured JSON (`fak-performance-rsi-scorecard/1`), containing dimensions, loop health, debt summary, invocation outcomes, and dominant bottleneck.
- `-markdown`: Emits a formatted GitHub-flavored Markdown table, suitable for PR reviews, issue updates, and automated reports.
- `-prior string`: Optional path to a prior scorecard JSON report. When provided, the command computes and displays delta trends (`PriorStatus → CurrentStatus`, ratio shifts) across cycles.
- `compose -snapshot string`: Assembles multiple independent single-owner section receipts (`cycle`, `improvement`, `provenance`, `learning`, `hardware`) into a single consolidated `fak-performance-rsi-evidence/1` artifact under the specified snapshot name.

---

## 5. Loop Turn Telemetry

Autonomous dispatch loops (such as agentic worker waves and continuous optimization loops) integrate performance-RSI scoring via automated loop turn telemetry.

### Telemetry Architecture

1. **Environment Entry Point**: The dispatch runner configures the environment variable:
   ```bash
   export FAK_PERFORMANCE_RSI_INPUT="/path/to/evidence.json"
   ```
2. **Fail-Closed, Non-Fatal Execution**: `ScoreLoopTurnFromEnvironment()` loads and scores the input via the strict `perfrsiscore.Load` and `perfrsiscore.Score` kernel paths. If the environment variable is unset or the file is missing/malformed, it returns a typed `LoopTurnReceipt` (`fak-performance-rsi-loop-turn/1`) with `status: "unavailable"` and reason `SCORE_INPUT_UNAVAILABLE` rather than panicking or failing the host turn. Auxiliary telemetry never replaces the completed dispatch's exit code.
3. **Bounded Usage Ledger**: `RecordLoopTurnUsage()` appends an adoption record (`fak-performance-rsi-usage/1`) to the repository-local ledger at `.fak/performance-rsi/usage.jsonl` (configurable via `FAK_PERFORMANCE_RSI_USAGE_LEDGER`), bounded by `jsonlledger.DefaultActiveBytes`.
4. **Scrubbed Observability**: Usage rows record timestamps, status, reason, snapshot, and invocation outcomes (`success`, `refusal`, `error`). Prompts, provider secrets, hostnames, and task outputs are strictly excluded from the telemetry ledger.
5. **Adoption Aggregation**: `FoldUsage` aggregates rows into ISO-week buckets (`fak-performance-rsi-usage-fold/1`), providing visibility into loop adoption, health trajectories, and neglected dimensions over time.
