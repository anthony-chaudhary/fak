# DeepSWE-bench Integration Design

## Overview

This document outlines the design for integrating DeepSWE-bench evaluation into the fleet benchmarking infrastructure. The goal is to:

1. **Setup DeepSWE-bench baseline** - Run DeepSWE-Preview predictions on SWE-bench Verified to establish a baseline
2. **Prove fleet gives a net lift** - Compare fleet's performance against the DeepSWE baseline
3. **Build automatic evaluation infrastructure** - Enable automated, repeatable SWE-bench evaluations

## Background: DeepSWE-Preview

DeepSWE-Preview is a state-of-the-art coding agent trained entirely with reinforcement learning on top of Qwen3-32B. Key results on SWE-bench Verified:

- **42.2% Pass@1** (single run)
- **59.0% with TTS** (test-time scaling, Best@16)
- **71.0% Pass@16** (theoretical optimal with 16 rollouts)

The DeepSWE pipeline involves:
1. **Multiple rollouts** - Running the agent 8-16 times per instance
2. **Execution-free verifier** - A Qwen3-14B model with LoRA adapter scores trajectories
3. **Execution-based verifier** - Runs tests to validate generated patches
4. **Hybrid selection** - Combines EF+EB scores to select the best patch

## Current Fleet SWE-bench Infrastructure

Fleet already has `fak/internal/swebench/` which provides:

- `instance.go` - SWE-bench Verified dataset loading
- `geometry.go` - Per-instance token/turn estimation
- `eval.go` - Official SWE-bench harness integration (resolve-rate grading)
- `report.go` - Four-family comparison metrics (prefill/KV-reuse, turns/tokens, adjudication cost, resolve-rate)
- `cost.go` - Token cost calculations

This infrastructure is designed to compare fleet against the external "N-Server Cache Benchmarking Tool" (the Benchmark repo).

## Design: Automatic SWE-bench Evaluation Infrastructure

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SWE-bench Evaluation Pipeline                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────┐      ┌─────────────────┐                       │
│  │  Dataset Loader│ ───> │ Instance Filter │                       │
│  │  (difficulty)  │      │  (smoke/l3)     │                       │
│  └────────────────┘      └─────────────────┘                       │
│           │                        │                                 │
│           v                        v                                 │
│  ┌────────────────┐      ┌─────────────────┐                       │
│  │ Agent Runners  │      │ DeepSWE Runner  │                       │
│  │                │      │                 │                       │
│  │ ┌────────────┐ │      │ ┌─────────────┐│                       │
│  │ │   Fleet    │ │      │ │ DeepSWE     ││                       │
│  │ │ (local/API)│ │      │ │ (VLLM/API)  ││                       │
│  │ └────────────┘ │      │ └─────────────┘│                       │
│  │ ┌────────────┐ │      │ ┌─────────────┐│                       │
│  │ │ Qwen Local │ │      │ │ TTS (N=16)  ││                       │
│  │ └────────────┘ │      │ └─────────────┘│                       │
│  └────────────────┘      └─────────────────┘                       │
│           │                        │                                 │
│           v                        v                                 │
│  ┌────────────────┐      ┌─────────────────┐                       │
│  │ Predictions    │      │ Predictions     │                       │
│  │ (predictions.  │      │ (predictions.    │                       │
│  │  json)         │      │  json)           │                       │
│  └────────────────┘      └─────────────────┘                       │
│           │                        │                                 │
│           └──────────┬─────────────┘                                 │
│                      v                                               │
│           ┌─────────────────┐                                        │
│           │ SWE-bench       │                                        │
│           │ Harness Grade   │                                        │
│           │ (resolve-rate)  │                                        │
│           └─────────────────┘                                        │
│                      │                                               │
│                      v                                               │
│           ┌─────────────────┐                                        │
│           │ Comparison      │                                        │
│           │ Report          │                                        │
│           └─────────────────┘                                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Dataset Management

**File**: `fak/cmd/swebench/dataset.go`

```go
// DatasetConfig controls dataset loading and filtering
type DatasetConfig struct {
    Source      string // "difficulty", "full", "custom"
    Path        string // Path to dataset file
    Filter      string // "smoke", "l3", "full"
    Limit       int    // Instance limit for testing
}

// LoadSWEbench loads and filters SWE-bench Verified instances
func LoadSWEbench(cfg DatasetConfig) (*swebench.Dataset, error)
```

**Filters**:
- `smoke` - Bench's l3_smoke subset (few instances for quick testing)
- `l3` - Larger subset for real evaluation
- `full` - All 500 SWE-bench Verified instances

#### 2. Agent Runners

**File**: `fak/cmd/swebench/runner/agent.go`

```go
// AgentConfig configures an agent runner
type AgentConfig struct {
    Type       string   // "fleet", "deepswe", "openai", "anthropic"
    Model      string   // Model identifier
    BaseURL    string   // API endpoint
    MaxTokens  int      // Max tokens per response
    MaxSteps   int      // Max agent steps per instance
    Temperature float64
}

// Runner runs an agent on SWE-bench instances
type Runner interface {
    Name() string
    Run(ctx context.Context, instances []swebench.Instance) <-chan Result
}

// Result is the outcome of running on one instance
type Result struct {
    InstanceID   string
    ModelPatch   string  // The generated patch
    Trajectory   []Turn  // Agent conversation
    Error        error
    Metrics      RunMetrics
}
```

**Runners**:
- `FleetRunner` - Uses fleet's in-process agent (local Qwen or API)
- `DeepSWERunner` - Runs DeepSWE-Preview via VLLM or API
- `OpenAIRunner` - For baseline comparison (GPT-4, etc.)

#### 3. DeepSWE Integration

**File**: `internal/swebench/runner.go`

The current DeepSWE path is an external adapter contract. `RunnerDeepSWE` does
not fabricate patches: it shells to a configured adapter and records the instance
as failed when no adapter is present.

Adapter selection:

- `FAK_DEEPSWE_RUNNER` names the adapter executable.
- `FAK_DEEPSWE_RUNNER_ARGS` supplies optional whitespace-separated arguments.
- If `FAK_DEEPSWE_RUNNER` is unset, `DeepSWERepo` may point at a checkout
  containing `fak-deepswe-runner` (or the Windows `.exe`/`.cmd`/`.bat` variants).

Request contract:

```json
{
  "schema": "fak.swebench.deepswe-request.v1",
  "runner": "deepswe",
  "model": "DeepSWE-Preview-or-endpoint",
  "max_steps": 50,
  "instance": {
    "instance_id": "django__django-10000",
    "repo": "django/django",
    "base_commit": "..."
  }
}
```

The adapter receives this JSON on stdin. It must write either a canonical
SWE-bench prediction object or a unified diff to stdout. JSON output is preferred:

```json
{
  "instance_id": "django__django-10000",
  "model_name_or_path": "DeepSWE-Preview",
  "model_patch": "diff --git ..."
}
```

The runner rejects mismatched instance ids and empty patches. This keeps the
official SWE-bench grader as the scoring source while making the DeepSWE/R2E-Gym
baseline pluggable.

The checked-in `cmd/fak-deepswe-runner --fixture` executable is a contract
fixture, not a model runner. It proves the request/adapter/prediction path with
`fak swebench run --agent deepswe`; the current witness lives at
`experiments/agent-live/deepswe-adapter-smoke-20260626/` and records two
grader-consumable prediction rows plus an honestly gated official-eval command.
Do not use that fixture packet as pass@1 or cost evidence.

DeepSWE has two modes the adapter should support:

**Single Pass (Pass@1)**:
```go
type DeepSWESingleConfig struct {
    ModelPath   string // Path to DeepSWE-Preview model
    VLLMArgs    []string // VLLM server arguments
    MaxContext  int    // 64K tokens
}

func (r *DeepSWERunner) RunSinglePass(ctx, instances) <-chan Result
```

**Test-Time Scaling (TTS)**:
```go
type DeepSWETTSConfig struct {
    Rollouts    int     // N=8 or N=16
    VerifierEF  string  // Execution-free verifier model
    VerifierEB  bool    // Run execution-based verifier
    HybridMode  bool    // Use hybrid EF+EB selection
}

func (r *DeepSWERunner) RunTTS(ctx, instances) <-chan Result
```

**Simplified approach for initial implementation**:
- Wrap Together AI's hosted DeepSWE-Preview API if available.
- Or download and serve via VLLM on a GPU server.
- Start with single-pass adapter output, then add TTS selection later.

#### 4. Evaluation Harness

**File**: `fak/cmd/swebench/eval.go`

```go
// EvalConfig configures the SWE-bench harness run
type EvalConfig struct {
    PredictionsPath string
    DatasetName     string
    RunID           string
    MaxWorkers      int
    Docker          bool    // Run in Docker (required for official grading)
}

// Run runs the official SWE-bench harness
func Run(cfg EvalConfig) (*swebench.EvalResult, error)
```

Leverages existing `swebench.RunEval()` from `fak/internal/swebench/eval.go`.

#### 5. Comparison Report

**File**: `fak/cmd/swebench/report.go`

```go
// CompareConfig generates a comparison between two agent runs
type CompareConfig struct {
    BaselineResults  string // predictions.json from baseline (e.g., DeepSWE)
    FleetResults     string // predictions.json from fleet
    OutputPath       string // Output directory
    IncludeCost      bool   // Include token/cost analysis
}

// Compare generates the comparison report
func Compare(cfg CompareConfig) (*ComparisonReport, error)
```

**Report sections**:
1. **Resolve Rate** - % instances resolved (via official harness)
2. **Token Efficiency** - Tokens used per resolved instance
3. **Wall-Clock Time** - Time to solution
4. **Cost Analysis** - API costs or compute time
5. **Per-Instance Breakdown** - Which instances each system resolved

### Command-Line Interface

**File**: `fak/cmd/fak/swebench.go`

```bash
# Run fleet on SWE-bench
fak swebench run \
    --agent fleet \
    --dataset smoke \
    --output ./runs/fleet-smoke \
    --model qwen3.6-27b

# Run DeepSWE baseline
fak swebench run \
    --agent deepswe \
    --dataset smoke \
    --output ./runs/deepswe-smoke \
    --tts-rollouts 16

# Grade predictions with official harness
fak swebench grade \
    --predictions ./runs/fleet-smoke/predictions.json \
    --run-id fleet-smoke

# Compare results
fak swebench compare \
    --baseline ./runs/deepswe-smoke/predictions.json \
    --fleet ./runs/fleet-smoke/predictions.json \
    --output ./comparison.md
```

## Integration with Existing Benchmark Catalog

The SWE-bench runs should integrate with `fak/experiments/benchmark/catalog.json`:

```json
{
  "runs": [
    {
      "run_id": "anthony-swebench-fleet-smoke-20260620",
      "machine_id": "anthony",
      "model": "Qwen3.6-27B",
      "tags": ["swebench", "fleet", "smoke-test"],
      "timestamp": "20260620T120000Z",
      "path": "fak\\experiments\\benchmark\\runs\\by-machine\\anthony\\20260620T120000Z-swebench",
      "swebench": {
        "dataset": "smoke",
        "resolved": 12,
        "total": 50,
        "resolve_rate_pct": 24.0,
        "baseline": "DeepSWE-Preview (42.2%)",
        "fleet_lift_pct": -18.2
      }
    }
  ]
}
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

1. **Dataset loader** - Implement `LoadSWEbench()` with smoke/l3 filters
2. **Fleet runner** - Implement `FleetRunner` using existing gateway
3. **Prediction format** - Implement SWE-bench predictions.json writer
4. **CLI skeleton** - `fak swebench run` with basic options

### Phase 2: DeepSWE Integration (Week 2)

1. **DeepSWE runner** - Single-pass mode via API or VLLM
2. **Grading integration** - Hook up `swebench.RunEval()`
3. **Catalog integration** - Add SWE-bench entries to catalog.json

### Phase 3: Comparison & Reporting (Week 3)

1. **Comparison report** - Fleet vs DeepSWE metrics
2. **Cost analysis** - Token/cost per resolved instance
3. **Per-instance breakdown** - Which instances each resolved

### Phase 4: Test-Time Scaling (Week 4+)

1. **TTS runner** - Multiple rollouts with verifier selection
2. **Hybrid mode** - EF+EB verifier integration
3. **Best-of-N selection** - Aggregate best predictions

## Proving Fleet Gives a Net Lift

The comparison should demonstrate fleet's advantages:

### 1. Token Efficiency (Prefill Elimination)

Fleet's shared KV cache eliminates redundant prefill work across agent turns:

```
DeepSWE (VLLM with prefix caching):
- Per-turn prefill: P tokens every turn
- Total prefill: T × P tokens

Fleet (shared KV):
- Initial prefill: P tokens (once)
- Per-turn reuse: ~0 tokens
- Total prefill: P tokens
- Savings: (T-1) × P tokens
```

For SWE-bench (P≈2500, T≈20): **~47,500 tokens saved per instance**

### 2. In-Process Adjudication

Fleet's in-process tool-call adjudication vs. DeepSWE's process-per-hook:

```
DeepSWE: Spawn process → Execute → Return result (~50-100ms)
Fleet: In-process call → Execute → Return result (~1-5ms)
Speedup: 10-100x per tool call
```

For typical SWE-bench instance (~100 tool calls): **~5-10 seconds saved**

### 3. Resource Utilization

Fleet's multi-agent session sharing vs. DeepSWE's per-instance isolation:

```
DeepSWE: One VLLM instance per rollout
Fleet: Shared session pool, N agents per session
Memory efficiency: N× improvement
```

### 4. Net Lift Calculation

```
Net Lift = (Fleet Resolve Rate - DeepSWE Resolve Rate) 
           + (Fleet Speed Factor - DeepSWE Speed Factor)
           + (Fleet Cost Efficiency - DeepSWE Cost Efficiency)
```

**Initial hypothesis**: Fleet may have slightly lower raw resolve rate but significantly better token/cost efficiency per resolved instance.

## Next Steps

1. **Implement Phase 1** - Core infrastructure (dataset, runner, predictions)
2. **Run smoke test** - Validate on 5-10 instances
3. **Measure baseline** - Run DeepSWE single-pass on same instances
4. **Generate comparison** - First fleet vs DeepSWE report
5. **Scale up** - Expand to full l3, then verified set

## References

- [DeepSWE Blog Post](https://www.together.ai/blog/deepswe)
- [R2E-Gym Repository](https://github.com/R2E-Gym/R2E-Gym)
- [DeepSWE TTS Reproduction Guide](https://github.com/R2E-Gym/R2E-Gym/blob/main/reproduction/DEEPSWE_TTS_REPRODUCTION.MD)
- [SWE-bench Leaderboards](https://www.swebench.com/)
- [Existing SWE-bench Code](../../internal/swebench/)
