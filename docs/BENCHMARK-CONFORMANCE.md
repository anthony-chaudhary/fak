# Benchmark Conformance

Conformance notes for fak's benchmark tooling: which invariants a conforming
benchmark scenario must hold, and where each invariant comes from.

## Scenario invariants (aiperf scenario/base)

> **NOTICE / attribution:** The invariants in this section are ported from
> `ai-dynamo/aiperf@ab610ad` (Apache-2.0), file
> `src/aiperf/common/scenario/base.py` (classes `ScenarioSpec`,
> `ScenarioViolation`, `ScenarioOutcome`, `ScenarioLockError`). Source pin:
> **ai-dynamo/aiperf@ab610ad** (full SHA
> `ab610add4e8876fca223197596c728e17365f381`), license Apache-2.0.

A benchmark scenario is a **frozen declaration of invariants** that a
conforming run must satisfy. The spec is immutable (`frozen=True`) and
rejects unknown fields (`extra="forbid"`): a scenario cannot be extended or
mutated after construction, and any field not declared in the spec is a
hard error. The invariants below are the ones a conforming benchmark
scenario must hold.

### Frozen scenario declaration

- The scenario spec is a frozen, extra-forbidding model: every invariant is
  declared up front, the spec cannot be mutated in place, and undeclared
  fields are rejected. A conforming scenario is fully described by its
  declared fields; nothing is read from or written to the spec at run time.

### Required timing mode

- `timing_mode`: the scenario names the one required timing mode. A run
  that resolves to any other timing mode violates the lock.

### require_ignore_eos

- `require_ignore_eos`: the scenario injects `ignore_eos=true` into the
  run's extra inputs. An **explicit `false` is an error** — the user may
  leave it unset (the scenario fills it in) but may not opt out.

### require_streaming

- `require_streaming`: streaming is forced on (`--streaming=true`,
  auto-enabled when unset). An **explicit `--no-streaming` is an error**.
  Streaming is required for the per-token latency metrics (TTFT, ITL) that
  are core to the benchmark; without it a run would silently report no
  first-token signal.

### forbid_ignore_trace_delays

- `forbid_ignore_trace_delays`: rejects `--ignore-trace-delays`. The
  scenario replays recorded trace timing; ignoring trace delays nulls every
  per-turn timestamp/delay in the loader, dispatching all turns
  back-to-back and **falsifying the workload** — while the run would
  otherwise still report `submission_valid=true`. This is the payload-
  integrity invariant: the recorded timing of the workload is part of the
  payload.

### forbid_input_truncation

- `forbid_input_truncation`: rejects client-side input-length truncation.
  Currently checks `--synthesis-max-isl`, which drops traces whose input
  length exceeds the cap. Truncating inputs changes the workload being
  measured, so it is a hard violation.

### require_loader

- `require_loader`: the required loader plugin name (e.g. `weka_trace`), or
  a tuple of equivalent loader names. The **detected loader must match any
  one** of the required names — useful when several loader plugins produce
  byte-identical data (e.g. file-based vs HF-hosted variants).

### Benchmark duration floor and auto-fill

- `min_benchmark_duration_seconds`: a floor (`> 0`) on
  `--benchmark-duration` in seconds.
- `default_benchmark_duration_seconds`: optional value auto-filled into
  `--benchmark-duration` when the user leaves it unset. **Explicit user
  values are honored, subject to the floor**; `None` disables auto-fill.

### Trajectory start ratio auto-fill

- `default_trajectory_start_min_ratio` / `default_trajectory_start_max_ratio`:
  optional values auto-filled into `--trajectory-start-min-ratio` /
  `--trajectory-start-max-ratio` when the user leaves them unset. Both are
  constrained to **[0, 1]**; explicit user values are honored; `None`
  disables auto-fill.

### Idle and inter-turn delay caps

- `inter_turn_delay_cap_seconds`: hard ceiling (seconds, `>= 0`) for trace
  inter-turn delays. `None` disables.
- `trace_idle_gap_cap_seconds`: hard ceiling (seconds) for observed runtime
  idle time within each AgentX trajectory tree. The idle clock covers
  initial future work and restarts when the final in-flight request across
  the root and all descendant streams completes.
- `system_idle_gap_cap_seconds`: hard ceiling (seconds) for globally idle
  replay time. Pending request timers shift uniformly only when no request
  is active.
- All three are **hard ceilings**: they clamp observed gaps, they do not
  rewrite the dataset. **Dataset timestamps are unchanged; request order,
  spawn/join dependencies, and replay barriers remain authoritative.**

### forbid_inter_turn_delay_cap

- `forbid_inter_turn_delay_cap`: rejects `inter_turn_delay_cap_seconds` so
  the scenario preserves each trace's **original inter-turn timing**. A
  scenario that sets this flag must not also carry a cap.

### require_cache_bust

- `require_cache_bust`: when set, `prompt.cache_bust.target` **must equal
  this value**. A mismatch is rejected unless `--unsafe-override` is also
  set, which **stamps `submission_valid=false`** on the run.

### minimum_profile_metric_coverage_ratio

- `minimum_profile_metric_coverage_ratio`: minimum fraction (`> 0`, `<= 1`)
  of each duration-based profiling phase that must contain TTFT or
  inter-token-latency observations. A run that falls below the threshold
  **exits non-zero and is not a valid scenario submission**. `None`
  disables the post-run check.

### require_use_think_time_only

- `require_use_think_time_only` (default `false`): forces
  `--use-think-time-only=true` to exclude response time from inter-turn
  delays.

### Outcome model

- **`ScenarioOutcome`** is the result of applying a scenario lock against a
  resolved run:
  - `submission_valid`: `True` when all invariants are satisfied (after
    auto-fills), `False` under `--unsafe-override` when violations were
    downgraded to warnings, `None` when no scenario is set.
  - `applied_locks`: short tags for each invariant lock that was applied
    (auto-filled or validated), e.g. `timing_mode`, `streaming`,
    `cache_bust`; order reflects application order.
  - `violations`: all scenario invariant conflicts, **collected in one
    pass**. Non-empty only under `--unsafe-override` (otherwise a
    `ScenarioLockError` is raised).
  - `submission_invalid_reasons`: short tags explaining why
    `submission_valid` is `False` (e.g. `unsafe_override`).
- **`ScenarioViolation`** records one conflict: the user-facing flag, the
  value the user provided, the value the scenario requires, and a
  human-readable explanation.
- **`ScenarioLockError`** is raised when a lock is violated and the run
  cannot proceed. It is **bypassable** by default (the message suggests
  `--unsafe-override`, which converts violations to warnings and marks the
  run `submission_valid=false`); with `bypassable=False` the violations
  include **hard locks** that the override cannot downgrade, and the error
  instead names the required remedy (e.g. providing the required public
  dataset).

## Recipe catalog (aiperf search_recipes/builtins)

> **NOTICE / attribution:** The recipe catalog in this section is adapted
> from `ai-dynamo/aiperf@ab610ad` (Apache-2.0), file
> `src/aiperf/search_recipes/builtins.py` (classes
> `MaxThroughputUnderTTFTSLA`, `MaxThroughputUnderITLSLA`,
> `ConcurrencyRamp`, `PrefillTTFTCurve`, `DecodeITLCurve`;
> `MaxConcurrencyUnderSLA` re-exported from
> `src/aiperf/search_recipes/_max_concurrency_under_sla.py`). Source pin:
> **ai-dynamo/aiperf@ab610ad** (full SHA
> `ab610add4e8876fca223197596c728e17365f381`), license Apache-2.0.

A search recipe is a **named preset that compiles at the CLI converter
boundary into canonical sweep fields** — the recipe name never reaches the
benchmark config itself. Each builtin below compiles to exactly one of an
adaptive-search (Bayesian-optimization) sweep or a grid sweep, plus optional
SLA filters and, for grid recipes, a post-process handler that emits a
derived artifact under `sweep_aggregate/`. Concurrency bounds are `[1, 1000]`
unless stated.

### max-throughput-ttft-sla

- Maximizes `output_token_throughput` at the highest concurrency where p95
  TTFT stays under `--ttft-sla-ms`.
- Bayesian-optimized over profiling concurrency in `[1, 1000]` with a
  **log-uniform prior** (30 max iterations, 5 initial points); bounds are
  overridable via `--concurrency-min` / `--concurrency-max`.
- The SLA constraint lands as an SLA filter (`time_to_first_token`, stat
  `p95`, op `lt`) that feeds the planner's lexicographic feasibility
  scoring — feasibility ranks before the throughput objective.
- **Streaming is required** (TTFT is a streaming-only metric); non-streaming
  configs are rejected at expand time rather than failing mid-search.

### max-throughput-itl-sla

- The inter-token-latency twin: maximizes `output_token_throughput` at the
  highest concurrency where p95 ITL stays under `--itl-sla-ms` (alias
  `--tpot-sla-ms`; passing both with different values is an error).
- Same BO wiring as the TTFT recipe: log-uniform over `[1, 1000]`, 30
  iterations, 5 initial points; SLA filter on `inter_token_latency` p95.
- Streaming required (ITL is streaming-only).

### max-concurrency-under-sla

- Finds the **highest concurrency where all SLA filters pass** — the
  boundary itself, not a throughput-maximizing interior point.
- Five search styles (`--search-style`):
  - `smooth_isotonic` (**default**): PAVA + PCHIP denoised isotonic
    regression — a 1-D SLA-saturation search that denoises before fitting
    and supports replicate-based bootstrap CIs; drop-in replacement for
    `monotonic`.
  - `monotonic`: 1-D binary search over `[1, 1000]` (~10-20 iterations);
    mirrors perf_analyzer's `--binary-search`.
  - `bo`: penalty Bayesian optimization with the same SLA filters and
    `output_token_throughput` as the maximize objective — lands at the
    throughput-maximizing feasible point rather than the boundary. Uses a
    **log-uniform prior**: concurrency spans three orders of magnitude, so
    uniform sampling would place ~1.6% of initial points in the `[1, 16]`
    knee region and the planner would latch onto an infeasible but
    high-throughput surface.
  - `optuna`: same penalty-BO formulation routed through the Optuna
    planner (TPE / GP / BoTorch samplers via `--optuna-sampler`);
    log-uniform prior for the same reason as `bo`.
  - `grid`: log-spaced 8-step sweep plus an `sla_breach_knee` post-process
    that emits `sla_breach.json` with the boundary report.
- SLA filters compose from any of `--ttft-sla-ms` (p95 TTFT),
  `--itl-sla-ms` / `--tpot-sla-ms` (p95 ITL), `--e2e-sla-ms` (p99
  `request_latency`), and `--error-rate-sla` (avg error rate; the CLI takes
  a fraction, the filter compares against the run-level percentage); at
  least one is required.
- **Streaming is required only when a selected filter references a
  streaming-only metric** (TTFT or ITL/TPOT); `--e2e-sla-ms` and
  `--error-rate-sla` work without streaming.

### concurrency-ramp

- Ramps concurrency on a **log scale** — default 8-step log-spaced grid over
  `[1, 1000]` (overridable via `--concurrency-min` / `--concurrency-max` /
  `--concurrency-steps`) — and detects the **latency degradation knee**.
- The post-process handler reports the first concurrency where the chosen
  metric/stat exceeds `baseline * (1 + threshold)`: default
  `--degradation-threshold 0.20` (20%), default metric `request_latency`
  stat `p99`; both flow through to the knee detector's params.
- Streaming is **not** required by default (`request_latency` is end-to-end);
  selecting a streaming-only metric (e.g. `time_to_first_token`) via
  `--degradation-metric-tag` does require streaming.

### prefill-ttft-curve

- Sweeps ISL log-spaced over `[256, 32768]` with **concurrency fixed at 1**,
  isolating prefill cost from queueing effects (the single-element sweep
  list collapses the cartesian product, so concurrency stays 1 on every
  row).
- The post-process handler fits `TTFT = a * ISL + b` and falls back to a
  **quadratic fit when r^2 < 0.85**; emits `prefill_curve.json`.
- Streaming required (TTFT is streaming-only).

### decode-itl-curve

- Sweeps a **concurrency x OSL grid**: concurrency 6 log-spaced points in
  `[1, 200]` against OSL 4 log-spaced points in `[64, 1024]` (all
  overridable).
- The post-process handler fits an ITL surface and emits
  `decode_itl_surface.json` with the raw measured points and a bilinear-grid
  surface.
- Streaming required.

### Grid helper and remaining exports

- `_logspace_int_steps` builds the log-spaced integer grids: endpoints are
  forced into the result (the lowest/highest swept value is exactly `lo` /
  `hi`), rounding collisions are deduped, order is ascending, and a
  single-point ramp is rejected (`steps >= 2`) because post-process can't
  compute a baseline.
- The catalog module also re-exports `pareto-sweep` and
  `max-goodput-under-slo`; they are not treated in this section.

## Streaming usage reporting (aiperf openai_chat)

> **NOTICE / attribution:** Ported from `ai-dynamo/aiperf@ab610ad`
> (Apache-2.0), file `src/aiperf/endpoints/openai_chat.py`
> (`format_payload`, `_ensure_include_usage`).

- A streaming chat request (`stream: true`) carries
  `stream_options.include_usage: true`. OpenAI-compatible servers (vLLM,
  SGLang) emit the trailing `usage` chunk only when asked, so without it a
  streamed run reports completion tokens as unknown instead of measured.
- Caller-supplied `stream_options` are never clobbered: an explicit
  `include_usage: false` is honored.
- A non-streaming request carries no `stream_options`; servers reject the
  field when `stream` is false.
- Stored payloads replayed from a dataset are sent byte-for-byte; the rule
  applies to the synthetic request builder only.

In fak, `cmd/loadgen` applies the rule in `syntheticChatRequest` /
`withIncludeUsage`; witness `TestIncludeUsageConformance`
(`go test ./cmd/loadgen -run ^TestIncludeUsageConformance$ -count=1`).
