---
title: "Agentic detail in fak manage live status: the deterministic-first"
description: "Audits which fak manage live-status agent signals are already witnessed, which need aggregation, and which aren't yet deterministically attributable."
---

# Agentic detail in `fak manage` live status: the deterministic-first ladder (2026-07-04)

**Operator goal (2026-07-04):** make `fak manage` live status — the `fak info`
agents pane the 20% `--split` strip runs — show *more agentic detail*: what each
agent is doing, how many are active, and how many tokens each sub-agent is
using. "Think about deterministic items first."

This note records which of those signals are **already witnessed** at the
gateway/session registry (so surfacing them is a pure projection, not new
measurement), which need a small deterministic aggregation, and which are **not
deterministically attributable yet** and must stay honestly fenced. It maps each
axis to an existing issue (#2250) or a net-new, contract-validated, ready-to-file
body (parked verbatim below — the filing session's guard policy refuses direct
`gh issue create`; any authorized session or producer files it verbatim, same
convention as `MANAGED-CACHE-PROVING-GROUND-2026-07-03.md`).

Read-only audit; as of HEAD `809aa0e9`.

## What the pane shows today

The agents sub-pane already exists end-to-end and is payload-free by
construction:

```
SessionState (internal/gateway/gateway.go:468)
  → debugSessions/debugSessionVars (internal/gateway/debug.go:460,476)
  → /debug/vars sessions[]
  → guardInfoSession decode (cmd/fak/info.go:133)
  → guardInfoAgentsPanelRows (cmd/fak/info_panels.go:254)
```

Each row carries `trace_id`, `run` state, `parent_trace`/`generation` lineage,
`priority`, wall-clock `elapsed_seconds`, and the **remaining** budget axes
(`turns_left`, `tokens_left`, `context_tokens_left`), plus an `assumptions`
count. The one-row mini form (`guardInfoAgentsSummary`, info_panels.go:313)
already answers **"how many active"**: `"N active (M continued, deepest gK), S
spawned"`. (Corrected since this note was written: `parent_trace`/`generation`
are *re-continuation* lineage — the same agent after a context reset — not spawn
lineage, so the summary reports them as continuations and sources its sub-agent
count from `spawn_count` instead. See `guardInfoAgentsSummary`.)

So of the operator's three asks:

- **how many active** — SHIPPED (the summary line above).
- **how many tokens each sub-agent is using** — filed as **#2250**, but the
  issue proposes building a *new* per-trace counter map and is unaware the data
  is already computed (see Axis B). Enrich, don't re-file.
- **what each agent is doing** — only `run` state + lineage today; the cheap,
  deterministic, payload-free *activity* signal (last adjudicated tool +
  in-flight/idle) is a genuine gap. Net-new issue below (Axis C).

## The deterministic ladder

Ordered most-deterministic (data already computed, pure projection) → least.

| Rung | Signal | Already computed? | Where | Surfacing gap |
|---|---|---|---|---|
| 0 | per-**session** tokens *used*, windowed + spike ratio | **yes** | `session.State.Cost` = the #756 `CostRing`/`CostSummary` (`internal/session/costring.go`), pushed every debited turn incl. unbounded sessions | dropped at `toGatewaySessionStateAt` (cmd/fak/main.go:1205); never crosses the wire |
| 0 | per-**trace** *cumulative* output/resident tokens + peak | **yes** | `sessionCtxValue.totalOutput` / `totalResidentTurns` / `peakResident` (`internal/gateway/ctxvalue.go:91`), queryable at `/v1/fak/ctxvalue` | not mirrored onto `sessions[]` / the pane |
| 1 | per-session **spawn count** (subagents spawned) | detection yes, counter no | spawn shape recognized at `adjudicateProposed` (`ToolAdjudication.Tool`, wire.go:576) via `subagentSpawnShapes` (subagent_witness.go:40) | `AdjudicationSummary` (metrics.go:912) buckets by verdict/reason only — no `byTool`/spawn counter |
| 1 | per-session **last tool** + **in-flight/idle age** ("what doing") | tool name yes (payload-free), inflight yes (aggregate) | tool name per call at `adjudicateProposed`; aggregate inflight at `/debug/vars` `inflight_max_age_seconds` | no *per-trace* last-tool/inflight attribution kept |
| 1 | subagent **lifecycle** (spawned Task, runtime, LIVE/STALLED) | **yes** | toolproc journal `.fak/toolproc/journal.jsonl` — `spawn`/`exit`/`session_end` rows with `Tool`, owner session, runtime, liveness (`fak toolproc ps`) | a separate surface; not joined into the guard pane; no tokens/PID in the schema |
| 3 | true per-**sub-agent** *token* attribution | **NO — not wired** | — | gateway `trace_id` and toolproc harness `session_id` are separate namespaces with **no parent↔child link on the request path**; `State.ParentTrace` is budget-reset lineage only (session.go:250). A subagent gets a distinct trace row only if its requests carry a distinct `X-Trace-Id`; `fak manage` sets one `DefaultTraceID` for the wrapped CLI. See #2397 (agentgraph subagent registry). |

**The honest fence (rung 3).** The operator's phrase "how many tokens each
**sub-agent** is using" is only literally answerable once a subagent's requests
are attributable to a distinct trace. Until that linkage exists, rungs 0–1
deliver **per-session / per-trace** consumption — which equals per-sub-agent
*iff* the harness issues distinct trace ids per subagent. Every artifact below
states this fence rather than over-claiming.

## Axis A — "how many active": SHIPPED

`guardInfoAgentsSummary` (info_panels.go:313) already renders active/sub/deepest.
No work; listed for completeness. A small enrichment (per-generation breakdown,
spawn count) rides Axis C's spawn counter.

## Axis B — "tokens each sub-agent is using": ENRICH #2250, don't re-file

**#2250** ("fak info agents pane: per-sub-agent token consumption attribution")
already owns this axis and is open. Its rung 1 proposes a *new* per-trace-id
counter map in the gateway metrics — but two per-trace token sources are
**already computed** and merely dropped before the wire:

1. **`session.State.Cost`** — the #756 cost ring: last-8-turn `TurnCost`
   (output+context) + `CostSummary{Latest,Previous,Delta,SpikeRatio}`. The spike
   ratio is the runaway-loop tell the ring was built for. Dropped at
   `toGatewaySessionStateAt` (main.go:1205) alongside `Intent`/`Goal`/`LastActive`.
2. **`sessionCtxValue.totalOutput` / `totalResidentTurns` / `peakResident`**
   (ctxvalue.go:91) — cumulative per-trace totals, already served at
   `/v1/fak/ctxvalue`.

So #2250's rung 1 is cheaper than filed: **project the existing ring +
ctxvalue totals** through `SessionState` → `debugSessionVars`
(`tokens_used`, `context_tokens_used`, `spike_ratio`, `cache_read_tokens`) →
`guardInfoSession` → two more cells on the agents row. No new counter map, no
new hot-path accounting.

**Definition of Done for #2250 (the tokens axis):**
- `debugSessionVars` gains `tokens_used`, `context_tokens_used`, `spike_ratio`
  (and optionally `cache_read_tokens`), each `omitempty`/`omitzero` so an older
  gateway or a never-debited session marshals byte-identically (the established
  `sessions[]` discipline).
- Values are the projection of `State.Cost.CostSummary()` (+ ctxvalue totals),
  NOT a new accumulator; `toGatewaySessionStateAt` carries `Cost`.
- The agents pane row shows consumed tokens next to remaining budget, and flags a
  session whose `spike_ratio` crosses a threshold (the runaway tell).
- `go test ./internal/gateway ./internal/session ./cmd/fak` green; a pane
  fixture pins the new cells (info_panels_test.go / info_visual_test.go); the
  wire shape for a zero-cost session is unchanged (a golden `/debug/vars` test).
- **Honesty fence recorded on the issue:** the row is per-*trace*; it equals
  per-*sub-agent* only under distinct-trace subagents (rung 3 / #2397).

Delivery: a `gh issue comment` on #2250 carrying the above (attempted this
session; parked here if refused).

## Axis C — "what each agent is doing": NET-NEW, contract-validated below

The deterministic, payload-free activity signal — **spawn count + last
adjudicated tool + in-flight/idle age, per session** — is filed by no existing
issue (#2537 is a heavyweight transcript-audit *behavioral scorer* in the
trajctl epic, not a pane cell). Tool *names* are safe metadata (the tool
identifier, never its arguments/prompt), so this keeps the pane's
redaction-by-construction property. Body parked verbatim below.

---

### Ready-to-file issue (contract verdict: `ready`, score 100 — re-validate any edit with `fak issue contract --from-issues`)

Filing: `gh issue create --title <title> --body-file <body>` with labels
`enhancement`, `area/agentic-serving`, `priority/P2`; then link from #2250 and
the sidecar epic #2209.

<!-- fak-agentic-live-status-key: agentic-live-status/activity-cell -->

**Title:** `feat(gateway): per-session activity cell — last adjudicated tool + in-flight/idle + spawn count on the fak info agents pane`

**Agentic serving** · the pane shows each agent's budget and lineage but never what it is doing right now.

#### Parent context
Sibling of #2250 (per-sub-agent token consumption; the tokens axis of the same agents pane) and a child of the sidecar epic #2209 ("one pane, every agent"). The agents sub-pane shipped in `cmd/fak/info_panels.go` (985ca742) fed by the `/debug/vars` `sessions` block (53848166): one payload-free row per live session with run state, lineage, wall-clock, and remaining budget.

#### Current state
A session row (`debugSessionVars`, `internal/gateway/debug.go:460`) carries run state + budgets-remaining but nothing about the session's live activity. The gateway already sees the answer, unaggregated: `adjudicateProposed` (`internal/gateway/adjudicate_proposed.go:214`) produces one `ToolAdjudication{Tool,...}` per proposed call (`internal/gateway/wire.go:576`), and subagent-spawn tool shapes are already recognized deterministically (`subagentSpawnShapes`/`subagentSpawnTool`, `internal/gateway/subagent_witness.go:40`). But `AdjudicationSummary` (`internal/gateway/metrics.go:912`) buckets only by verdict/reason — there is no per-session last-tool, no per-session spawn count, and no per-session in-flight/idle age. So the pane cannot answer "what is this agent doing right now".

#### Why now
"How many active" is shipped and "how many tokens" is #2250; the missing third of the operator's live-status goal (2026-07-04) is a cheap, deterministic activity signal. It is the one that turns the pane from a budget readout into an operator's "who is hot / who is stuck / who is idle" view, and it composes with #2250's spike flag to localize a runaway to the session that is looping.

#### Working spine
Keep a bounded per-trace activity record beside the session registry read: on each `adjudicateProposed`, stamp the trace's `last_tool` (the tool name of the last admitted call), increment a `spawn_count` when `subagentSpawnTool` matches, and record `last_activity_unix`. Derive `in-flight age` from the request the trace currently holds and `idle age` from `now - last_activity`. Project these onto `debugSessionVars` (`last_tool`, `spawn_count`, `inflight_seconds`, `idle_seconds`), which `guardInfoSession` (`cmd/fak/info.go:133`) already decodes field-for-field, and render one extra clause on the agents row (`guardInfoAgentText`, `cmd/fak/info_panels.go:279`). All new fields `omitempty`/`omitzero`.

#### In scope
The per-trace activity record + its bounded lifecycle (cap traces, drop on session stop); the four `debugSessionVars` fields; the `guardInfoSession` decode + the agents-row/summary rendering; unit tests in `internal/gateway` and a pane fixture in `cmd/fak`.

#### Out of scope
Per-sub-agent TOKEN attribution (#2250 owns tokens; the parent↔child trace link is #2397). Joining the toolproc lifecycle journal (`.fak/toolproc/journal.jsonl`) into the pane (a separate follow-on). Any prompt/argument/result text — the pane stays payload-free; only the tool NAME crosses. `/metrics` labels (per-trace cardinality stays off Prometheus by design — snapshot only).

#### Done condition
A live `fak manage --split -- claude` session shows, per agent row, its last adjudicated tool, its spawn count when > 0, and an in-flight or idle age; a session with no activity yet renders exactly as today (fields omitted); the `/debug/vars` wire shape for a pre-activity session is byte-identical.

#### Witness
`go test ./internal/gateway ./cmd/fak` green; a gateway unit test asserts last-tool/spawn-count/inflight/idle from a synthesized adjudication sequence; a pane fixture (`info_panels_test.go`/`info_visual_test.go`) pins the new row clause at full and mini levels; a golden `/debug/vars` test proves the zero-activity wire shape is unchanged.

#### Acceptance gate
Same as Done condition; plus `make ci` (`scripts/ci.ps1` on Windows hosts) green on the ship commit, and the scorecard control-pane `--check` does not regress.

#### Work unit
One worker owns the per-trace record + the four wire fields + the pane render + tests in one sitting.

#### Expected steps
5

#### Assumptions
- The per-trace activity record can hang off the same seam `listSessions`/`debugSessions` already reads, so no second registry is introduced.
- Tool names are safe to surface (they are the tool identifier, not its arguments) — consistent with the pane's existing payload-free guarantee.
- Bounding the record (cap traces, fold stopped traces) keeps a wide sub-agent fan-out from growing gateway memory, mirroring the `sessions[]` stopped-session drop.

#### Confusion risks
- Do NOT surface tool ARGUMENTS or any result text — only the tool NAME. The pane is redaction-safe by construction; a leak here would break that.
- `spawn_count` counts subagent-shaped tool CALLS proposed, not confirmed live children — it is an activity signal, not a live-child census (that is #2397's registry). Label it as such.
- In-flight age vs idle age are distinct: in-flight means a request is open now; idle means time since the last adjudication with nothing open. Do not collapse them.
- This is per-TRACE; it equals per-sub-agent only under distinct-trace subagents — carry the same fence as #2250.

#### Coordination
- `internal/gateway` and `cmd/fak` are live multi-session lanes; verify the lane via `dos_arbitrate` before writing.
- Lands independently of #2250 but shares the `debugSessionVars`/`guardInfoSession` row — land whichever first, rebase the other's field additions.

#### Trigger
Manual decomposition of the 2026-07-04 operator goal ("fak manage live status shows more agentic detail"). Not a scheduled producer.

#### Batch policy
One issue; dedupes on the `fak-agentic-live-status-key` marker above (a rerun updates rather than re-files). Deduped against #2250 (tokens, not activity), #2537 (trajctl behavioral scorer, not a pane cell), and #2050 (hardware resource row, not agentic) — this is the activity-cell slice only.

#### Likely files
`internal/gateway/adjudicate_proposed.go`, `internal/gateway/debug.go`, `internal/gateway/gateway.go`, `cmd/fak/info.go`, `cmd/fak/info_panels.go`

#### Lane
`gateway`

#### Closure binding
Closed only by a trunk ship commit whose body carries `Closes #<this issue>` and whose subject ends with the `(fak gateway)` trailer so the `dos verify` referee can bind it; the pane fixture + the golden zero-activity wire test are the binding witnesses. No self-report closure.

#### Ship discipline
- Trunk only; commit by explicit path; Conventional-Commits subject + `(fak gateway)` stamp.
- Honest-scope fence: this surfaces per-TRACE activity; per-sub-agent token attribution stays #2250/#2397's claim, and the signal is what-was-proposed, not proof a child is live.

_Self-contained: depends only on already-landed adjudication + agents-pane code._

---

## Scope fences

- This note grades **witness reachability + surfacing**, not a display design. The
  agents-pane rendering follows the existing 3-step panel recipe
  (`cmd/fak/info_panels.go:8-22`).
- Prometheus stays aggregate-only by design; per-trace detail belongs on the
  `/debug/vars` snapshot (the `vcache_families` precedent, `debug.go:452-459`).
- `gh issue create`/`comment` is guard-refused in this session class, so the
  Axis C follow-on is parked above with a `ready`/100 contract verdict and the
  Axis B enrichment is parked as a comment body; an authorized session files
  both verbatim.

## Not to be confused with

- **#2250** owns the *tokens* axis of this same pane; this note's net-new issue
  owns the *activity* axis. They share a row, not a scope.
- **#2537 / the trajctl epic (#2533)** score *activity-vs-progress divergence*
  from a transcript audit (a stall SCORER); this is a cheap live pane cell from
  the adjudication seam, not a behavioral score.
- **#2050** adds a *hardware* resource row (CPU/RSS/IO) to the pane; this adds an
  *agentic* activity cell. Different families.
- **#2209** is the cross-agent *sidecar* epic (one pane, every agent/surface);
  both axes here are single-gateway agents-pane cells that the sidecar later
  rolls up.
