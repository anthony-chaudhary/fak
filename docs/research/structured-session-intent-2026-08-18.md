# Structured session intent — inventory, field study, and spine

**Observed:** 2026-08-18. **Status:** working concept plus executable declaration spine.

## Verdict

Session requests need a typed **intent envelope**, not more prompt wording. The smallest useful model separates:

1. **Start eligibility** — now, at a civil time, or after an event.
2. **Effort bounds** — minimum, target, and maximum, each measured against an explicit clock.
3. **Completion** — objective/evidence decides done; time alone does not.
4. **Recurrence** — cadence plus timezone, overlap, and missed-run policy.
5. **Lifecycle reactions** — bounded hooks at named events, with timeout and failure behavior.
6. **Authority** — the declaration requests behavior but grants no tools, compute, or permission.

The executable spine is `go run ./cmd/sessionintentdemo -selfcheck`. It emits one validated `fak.session-intent/v1alpha1` declaration that keeps “work actively for at least 2 hours” distinct from “stop after 10 elapsed hours.”

## Value frame and problem check

- **Centrality:** Enabling. It makes common operator intent machine-readable but does not itself improve model inference.
- **For:** operators launching one interactive, background, or fleet-backed session.
- **Problem:** free text conflates effort, wall-clock lifetime, completion, scheduling, and callbacks; runtimes cannot enforce or explain the result consistently.
- **Today:** harness-specific flags, cron records, hooks, and prompt prose each own fragments of the same request.
- **Better because:** one validated envelope can be translated into each harness while preserving semantics and refusing contradictions.
- **Witness:** the demo's selfcheck plus `internal/sessionintent` validation tests.

| FAK problem check | Effect |
|---|---|
| P1 managed context | Replaces repeated steering prose with a compact declaration; hooks can inject context only at typed events. |
| P2 net-true efficiency | A target guides planning; a maximum bounds waste; explicit clocks prevent idle time from masquerading as work. Runtime savings remain unmeasured in this first spine. |
| P3 bounded adaptation | Minimum is stop eligibility, maximum is forced timeout, and completion remains evidence-based; none silently changes another. |
| P4 integrated operations | Trigger, recurrence, hooks, and effort share IDs/policies that a scheduler, guard, journal, and fleet runner can observe. |

## Own-prompt inventory

Source: the local Codex profile's human prompt history (`history.jsonl`), read 2026-08-18. This is private operational evidence, so examples are summarized rather than copied into a public fixture. Across 264 prompts, a deliberately broad phrase scan found:

| Need family | Matching prompts | Representative operator form |
|---|---:|---|
| dependency/event sequencing | 21 | after this finishes, do the next stage; spawn more workers as needed |
| lifecycle hooks | 15 | before stopping, verify or report; hook the session boundary |
| overnight/long-running work | 14 | work all night; run the next ten hours |
| fleet/fan-out policy | 11 | spawn fleets as needed |
| minimum/persistence | 7 | keep working; continue until the acceptance condition |
| maximum/bounded run | 5 | use at most N minutes; work for the next N hours |
| schedule/cadence | 3 | run on a schedule or cadence |

Counts overlap and are evidence of vocabulary, not prevalence estimates. The active request adds explicit examples of a five-minute maximum and a two-hour minimum. The local Codex goal store already records `token_budget` and observed `time_used_seconds`, but its goal declaration has no minimum/target/maximum time fields, trigger, recurrence, or lifecycle-hook schema. FAK already has extensive session telemetry, stop guards, nightrun, watchdog, and DOS loop machinery; the missing piece is a provider-neutral declaration joining those seams.

## Concept inventory

### Time is three different things

| Concept | Meaning | Enforcement |
|---|---|---|
| `minimum` | Normal completion is not yet eligible before this effort. It never requires pointless busywork after a hard failure or operator cancel. | stop gate |
| `target` | Planning horizon / desired investment, not a success claim and not a kill switch. | planner feedback |
| `maximum` | Hard ceiling; on expiry the runtime cancels, checkpoints, and emits a timeout receipt. | watchdog/cancel |

Every bound names a **clock**:

- `active`: time actually executing admitted work; excludes queueing and suspension.
- `elapsed`: monotonic wall time since activation; includes waits and backoff.

A civil timestamp (`deadline`, `trigger.at`) is different again: it needs a timezone at input and converts to an absolute instant. A completion predicate is not a duration. “Until tests pass, max 2h” is therefore a predicate plus a ceiling, not one overloaded timeout. If a minimum is declared, early completion evidence is retained but normal completion remains ineligible until that minimum; explicit cancel, failure, and hard maximum still terminate without burning time.

### Activation and recurrence

- **Trigger:** `immediate`, absolute `at`, or named external `event`.
- **Recurrence:** exactly one interval or cron expression.
- **Required schedule policies:** timezone for cron; overlap `allow|forbid|replace`; missed-run `skip|catch_up_one`.
- **Later operating-envelope fields:** jitter, bounded catch-up window, calendar exclusions, retries/backoff, run count/end date, and idempotency key.

### Lifecycle hooks

The portable event vocabulary starts with `on_start`, `before_tool`, `after_tool`, `before_stop`, `on_complete`, `on_timeout`, and `on_failure`. A hook names an action already registered elsewhere, a timeout, and `continue|block` failure behavior. It does **not** embed an arbitrary shell command or acquire authority. Executor-specific events can remain namespaced extensions rather than polluting the common core.

### State and receipts

A later executor should expose `declared -> waiting -> eligible -> running -> stop_eligible -> completing -> completed|timed_out|failed|cancelled`. Every transition should carry reason, monotonic and civil timestamps, active/elapsed counters, trigger occurrence, hook outcomes, and checkpoint/evidence references. Minimum expiry changes eligibility; it is not completion. Target expiry is an observation. Maximum expiry is a terminal control event.

## External field study

Observations and URLs were checked 2026-08-18; release/event dates below distinguish recent activity from long-lived documentation.

| Source | Observation | Borrow / disposition |
|---|---|---|
| [Codex scheduled tasks](https://developers.openai.com/codex/app/automations) | Scheduled tasks combine instructions, an optional skill, and a schedule; runs appear in an inbox, require the desktop app to be running, use local checkout/worktrees, and support daily/weekly schedules. | `PRESENT` as fragmented local orchestration; borrow task/schedule separation and run receipts. `DEFAULT`: typed trigger. `WATCH`: richer calendar UI. |
| [Codex hooks](https://developers.openai.com/codex/hooks) | Codex now documents lifecycle hooks separately from scheduled tasks. | Keep hooks orthogonal to scheduling. `DEFAULT`: named lifecycle events with bounded execution. |
| [Claude Code hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) | Rich event surface includes session, prompt, tool, stop/subagent, notification, compaction, and config events; command hooks default to a 10-minute timeout and can run asynchronously. | Borrow event taxonomy, timeout, and explicit failure policy. `OPTIONAL-MODULE`: executor-specific matchers/async hooks. |
| [Claude Code v2.1.234](https://github.com/anthropics/claude-code/releases/tag/v2.1.234), released 2026-08-17 | Recent release added `InstructionsLoaded`, subagent model overrides, and worktree metadata; it also fixed hook timeouts and a `Stop` hook edge case. | Recent evidence that hook schemas and receipts are live compatibility surfaces. `WATCH`: provider extension map. |
| [Background tasks need deadlines and typed cancel handlers](https://github.com/anthropics/claude-code/issues/87025), opened 2026-08-15 | Proposal identifies a worker-without-clock failure: background tasks need deadline plus typed cancellation/finalization. | `DEFAULT`: maximum + timeout receipt; cancellation handler belongs to executor policy, not embedded intent. |
| [Scheduled-task sessions never terminate](https://github.com/anthropics/claude-code/issues/82023), opened 2026-07-28 | Reported idle disconnect/re-arm loop demonstrates that scheduler lifetime and session terminal state can diverge. | `DEFAULT`: terminal states and no implicit re-arm. |
| [Cross-session message not submitted](https://github.com/anthropics/claude-code/issues/86069), opened 2026-08-12 | Delivery to a composer is not proof an event was consumed. | `DEFAULT`: trigger occurrence and acknowledgement receipts. |
| [Temporal Schedules](https://docs.temporal.io/schedule) | Mature schedules separate action, spec, overlap, catch-up window, pause, backfill, and jitter. | `DEFAULT`: overlap and misfire policy. `OPTIONAL-MODULE`: backfill/jitter/catch-up windows. |
| [Kubernetes CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/) | CronJobs expose timezone, concurrency policy, starting deadline, suspension, and missed-schedule behavior; scheduling is approximate and jobs should be idempotent. | Borrow explicit timezone, overlap, missed-run handling, and idempotency guidance. |
| [GitHub Actions workflow triggers](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows) | Trigger families distinguish manual dispatch, repository events, schedules, and workflow-completion dependencies. | `RECIPE`: adapters from named external events into `trigger.event`; do not bake vendor payloads into core. |

Recent ecosystem releases also show active work in adjacent durable execution systems: Temporal Go SDK v1.47.0 (2026-07-28), Trigger.dev v4.5.11 (2026-08-13), and LangGraph 1.2.11 (2026-08-11). They support studying durable retries/checkpoints later, but release recency alone is not evidence to import their APIs.

## FAK gap map

| Capability | Current FAK | Gap |
|---|---|---|
| Session observation, journals, stop guards | `PRESENT` | No common declaration that explains why a stop is early, on target, or timed out. |
| Goal persistence and token budget | `PARTIAL` | Stores observed elapsed seconds and optional token budget; no typed effort envelope. |
| Nightrun, watchdog, recurring loops, scheduled host tasks | `PARTIAL` | Each mechanism owns schedule/lifetime semantics independently. |
| Lifecycle hooks | `PARTIAL` | Guard/harness hooks exist, but there is no small provider-neutral session event contract. |
| Event-triggered chaining | `PARTIAL` | DOS/fleet workflows chain work, but operator intent is not one inspectable trigger declaration. |
| Structured session intent | `ABSENT` before this spine | `internal/sessionintent` now validates the declaration only; execution adapters remain. |

## Spine and boundary

`internal/sessionintent` validates and renders declarations and makes deterministic stop decisions from executor-owned counters; it does not schedule, sleep, launch a model, execute hooks, or grant tools. That is the smallest end-to-end seam we can witness locally without pretending a full executor exists.

```powershell
go run ./cmd/sessionintentdemo -selfcheck
```

Expected final lines:

```text
DECISIONS: 1h_active=continue 2h_active=eligible 10h_elapsed=timeout
SELFCHECK PASS: minimum and target govern stop eligibility/planning; maximum governs forced timeout
```

The schema rejects contradictory same-clock bounds, duplicate bounds, ambiguous recurrence, cron without timezone, unknown overlap/misfire behavior, hooks without a timeout/failure policy, and deadlines at or before a deferred start.

## Gold-plating boundary

Do not add a natural-language parser first: lossless declarations and receipts are the prerequisite. Do not invent a second scheduler: adapters should target the existing goal continuation, DOS loops, nightrun, host scheduled tasks, and native harness APIs. Do not treat minimum runtime as a command to burn tokens, maximum runtime as evidence of completion, or hook registration as permission to execute arbitrary code.

## Follow-on order

1. Integrate this declaration with the native goal store and expose it through the supported goal/session creation surface.
2. Add a runtime stop decision (`too_early`, `eligible`, `timed_out`) driven by monotonic active/elapsed counters and captured tests.
3. Add scheduler adapters with timezone, overlap, missed-run, idempotency, and acknowledgement receipts.
4. Map provider hook events into the common core while retaining namespaced extensions and capability checks.
5. Only then add conservative natural-language extraction that always shows the normalized declaration for confirmation.



