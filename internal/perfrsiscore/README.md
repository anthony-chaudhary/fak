# Performance RSI Scorecard (`internal/perfrsiscore`)

`internal/perfrsiscore` implements the core scoring kernel, composition engine, presentation renderers, and usage telemetry ledger for the **Performance Recursive Self-Improvement (RSI) loop** in the Fused Agent Kernel (`fak`).

The package adjudicates versioned empirical evidence against an explicit, unsaturated **100.0x** target multiplier (`TargetMultiplier = 100.0`), calculates normalized dimension ratios, determines loop-health grades, accounts for performance-RSI debt, and identifies the dominant bottleneck limiting compounding iteration velocity.

---

## 1. Package Architecture

The package is partitioned into three focused modules:

```
internal/perfrsiscore/
├── perfrsiscore.go     # Core models, validation, dimensional derivation, bottleneck selection, composition, turn scoring
├── render.go           # Presentation layer: human CLI tables, Markdown reports, JSON marshalling, snapshot comparison
├── usage.go            # Adoption and telemetry ledger: JSONL persistence, ISO-week folding, scrubbed usage rows
├── usage_test.go       # Tests for JSONL ledger append, bounding, and ISO-week folding
├── determinism_test.go # Concurrency, determinism, and data race verification
├── perfrsiscore_test.go# Acceptance, validation, dimensional derivation, and refusal tests
└── testdata/
    └── complete.json   # Canonical versioned evidence fixture covering all 16 dimensions
```

### Core Responsibilities

1. **Strict Input Parsing and Validation**: Decodes versioned JSON documents with zero tolerance for unknown fields (`dec.DisallowUnknownFields()`), trailing JSON tokens, or missing required fields.
2. **Dimensional Derivation**: Derives metrics for canonical dimensions from specialized receipt sections (`Cycle`, `Improvement`, `Provenance`, `Learning`, `Hardware`).
3. **Bottleneck Derivation**: Finds the limiting dimension with the lowest normalized progress ratio to enforce Amdahl's Law in self-improvement loops.
4. **Composition (`ComposeV1`)**: Assembles independently validated, single-owner section receipts into a consolidated evidence document without cross-receipt contract contamination.
5. **Telemetry & Ledgering**: Evaluates dispatch turns safely via non-fatal observability (`ScoreLoopTurn`), recording bounded, scrubbed adoption facts to a local JSONL ledger.

---

## 2. Input and Output Schemas

The package defines 11 strict, versioned schemas:

| Schema Identifier | Go Type | Description |
|---|---|---|
| `fak-performance-rsi-evidence/1` | `Evidence` | Complete input document containing the snapshot name, 100x target multiplier, 16 dimensions, and optional receipt sections. |
| `fak-performance-rsi-composition/1` | `CompositionSchema` | Contract for assembling independent receipts into a unified evidence document. |
| `fak-performance-rsi-cycle/1` | `Cycle` | End-to-end iteration timeline ledger (`idea_at` through `learning_at`) and operator intervention time. |
| `fak-performance-rsi-improvement/1` | `Improvement` | Strict receipt for a measured, quality-preserving performance gain with matched operating envelopes and causal ablation. |
| `fak-performance-rsi-provenance/1` | `Provenance` | Research transfer receipt connecting an immutable upstream commit/source to a landed fak-native commit. |
| `fak-performance-rsi-learning/1` | `Learning` | Chronological history of prediction calibration, outcome observation, and recurrence-key learning reuse. |
| `fak-performance-rsi-hardware/1` | `Hardware` | Direct duration-weighted accelerator utilization measurements from executed native runs. |
| `fak-performance-rsi-scorecard/1` | `Report` | Output report containing scored dimensions, loop health, debt summary, invocation outcomes, and dominant bottleneck. |
| `fak-performance-rsi-loop-turn/1` | `LoopTurnReceipt` | Non-fatal telemetry receipt emitted by automated dispatch loops. |
| `fak-performance-rsi-usage/1` | `UsageRow` | Sanitized single-turn adoption record appended to the JSONL ledger. |
| `fak-performance-rsi-usage-fold/1` | `UsageFold` | Aggregated weekly adoption summary bucketed by ISO week. |

---

## 3. Scoring Model and Dimensional Rules

### Canonical Dimensions

The engine strictly mandates exactly **16 canonical dimensions** (`DimensionIDs()`):

```go
var dimensionIDs = []string{
    "cycle_time", "improvement_yield", "evaluation_latency", "receipt_coverage",
    "quality_gate_coverage", "experiment_throughput", "hypothesis_calibration", "discovery_freshness",
    "adaptation_speed", "reuse_ratio", "learning_retention", "production_transfer",
    "hardware_utilization", "attribution_quality", "automation_coverage", "compounding_rate",
}
```

Each dimension defines:
- **Direction**: `Higher` (`"higher"`) or `Lower` (`"lower"`).
- **Canonical Unit**: e.g., `hours`, `seconds`, `percent`, `experiments_per_day`.
- **Target**: Explicit numerical target matching the 100x improvement model.
- **Source & Next Action**: Actionable traceability strings.

### Normalized Ratio Derivation

For each dimension $d$:
- If `Direction == Higher`: $\text{ratio} = \frac{\text{Current}}{\text{Target}}$
- If `Direction == Lower`:
  - If $\text{Current} == 0$: $\text{ratio} = +\infty$
  - If $\text{Current} > 0$: $\text{ratio} = \frac{\text{Target}}{\text{Current}}$

### Status Classification

- **`MET`**: $\text{ratio} \ge 1.0$.
- **`BEHIND`**: $\text{ratio} < 1.0$.
- **`UNKNOWN`**: `Current` or `Target` is `nil`.

### Dominant Bottleneck Selection

The dominant bottleneck identifies the most severe constraint in the optimization loop:
- Dimensions with measurements compute $\text{debt} = \text{ratio}$.
- Dimensions with status `UNKNOWN` assign $\text{debt} = +\infty$.
- The engine chooses the dimension with the minimal debt value (`worst = min(debt)`).
- If multiple dimensions are `UNKNOWN`, the first unmeasured dimension in canonical index order is chosen.

### Loop Health and Debt Accounting

- **Loop-Health Score**: Evaluates measurement coverage and progress toward targets:
  $$\text{score} = \text{Round1}\left(100 \times \frac{\sum_{i=1}^{16} \min(\text{ratio}_i, 1.0)}{16}\right)$$
- **Grade**: Standard letter grade mapped via `scorecard.GradeStd(score)` (A, B, C, D, F).
- **Clean**: Boolean indicating whether performance-RSI debt is zero (`debt == 0`).
- **Performance-RSI Debt**: Total count of lagging or missing dimensions:
  $$\text{debt} = \text{Behind} + \text{Unknown}$$

---

## 4. Receipt Composition (`ComposeV1`)

`ComposeV1` and `LoadAndComposeV1` assemble independently validated receipts into a single `fak-performance-rsi-evidence/1` artifact:

1. **Section Ownership**:
   - `cycle` owns: `cycle_time`, `evaluation_latency`, `experiment_throughput`, `automation_coverage`.
   - `improvement` owns: `improvement_yield`, `receipt_coverage`, `quality_gate_coverage`, `attribution_quality`.
   - `provenance` owns: `discovery_freshness`, `adaptation_speed`, `reuse_ratio`, `production_transfer`.
   - `learning` owns: `hypothesis_calibration`, `learning_retention`, `compounding_rate`.
   - `hardware` owns: `hardware_utilization`.
2. **Exclusivity**: Exactly one receipt may provide each section. Providing duplicate owners for a section returns a deterministic composition error.
3. **Contract Alignment**: For unowned dimensions, all input receipts must agree on direction, unit, and target. The dimension is included with `Current = nil` so it reports honest `UNKNOWN` debt rather than inheriting unrelated values.
4. **Deterministic Sorting**: Inputs are sorted lexicographically by source path before processing.

---

## 5. Renderers and Comparison

The package provides multiple deterministic renderers in `render.go`:

- **Human Table (`RenderHuman`)**: Single-line summary with snapshot name, target multiplier, health score/grade, debt count, dominant bottleneck, followed by a fixed-width table of all 16 dimensions.
- **Markdown (`RenderMarkdown`)**: Complete GitHub-flavored Markdown document with bulleted metadata, key findings, and a structured markdown table.
- **JSON (`MarshalJSON`)**: Pretty-printed JSON representation adhering strictly to `fak-performance-rsi-scorecard/1`.
- **Comparison (`Compare`)**: Compares a current `Report` against a decoded prior `Report`, generating delta records (`PriorStatus → CurrentStatus`, ratio shifts) for regression and trend tracking.

---

## 6. Loop Turn Telemetry and Usage Ledger

Autonomous agent runners report loop health without failing dispatch turns via `usage.go`:

- `ScoreLoopTurn(inputPath)` / `ScoreLoopTurnFromEnvironment()`:
  - Reads input from `FAK_PERFORMANCE_RSI_INPUT`.
  - On missing input or read failure, emits a non-fatal `LoopTurnReceipt` with `status: "unavailable"` and `reason: "SCORE_INPUT_UNAVAILABLE"`.
  - On valid input, runs `Score` and returns scored health metrics.
- `RecordLoopTurnUsage(receipt)`:
  - Appends an adoption record to the path configured by `FAK_PERFORMANCE_RSI_USAGE_LEDGER` (default: `.fak/performance-rsi/usage.jsonl`).
  - Uses `jsonlledger.AppendBounded` to prevent unbounded disk growth.
  - Excludes raw prompts, code, diagnostics, and credentials.
- `FoldUsage(r)`:
  - Reads and aggregates usage rows into ISO-week buckets (`UsageWeek`), providing longitudinal adoption trends.
