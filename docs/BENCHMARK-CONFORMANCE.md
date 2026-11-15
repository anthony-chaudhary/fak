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
