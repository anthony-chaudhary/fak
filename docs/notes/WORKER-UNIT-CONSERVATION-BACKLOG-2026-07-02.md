# Worker-unit conservation: the leak map and its dispatch-ready backlog (2026-07-02)

**The goal this serves:** if N open issues meet N worker-units inside a window,
all N get done — every unit ends in a witnessed ship or a graded, routed
refusal, and nothing leaks. This note records the 2026-07-02 lifecycle audit of
the issue-dispatch worker loop, what shipped the same day, and the three
remaining holes as **contract-validated, ready-to-file issues** (each reviewed
`ready`, score 100/100, by `fak issue contract --from-issues`). They are parked
here because the auditing session's guard policy refuses direct `gh issue
create`/`close`; any authorized session or producer can file them verbatim.

## What shipped from this audit (2026-07-02)

- **Resume durability** — #2368 (stale resume-took latch: a session whose
  resume TOOK and which then died again was skipped forever) and #2367
  (targeted resume plan raced by a concurrent registry refresh). Fixed in
  `8fd97557` (re-death revival kernel, `internal/resume`), `40e0d670` (watchdog
  wiring + `--plan` targeted runs), `870ce341` (bare `request timed out` is a
  transient), `1f29b395` (the `resume` lane declared in `dos.toml`). Both
  issues closed-by-commit.
- **Conservation meter** — `tools/dispatch_conservation.py` (`ceb0f7bf`): over
  a window, `spent = shipped_witnessed + committed_unwitnessed +
  no_commit{reason} + spawn_failed + leaked_unswept`, plus windowed closes,
  contract-hold pressure, and re-storm churn; `--fail-on-leak N` is the CI
  gate. First live reading (24h window): **0 resolve units spent while 233
  contract-hold rows kept 225 distinct issues out of dispatch** (open 699 vs
  baseline 483) — that window's capacity went ungroomed, not unspent. The
  contract-repair dispatch (`84f78417`, peer lane) is the counter-move.

## The leak map (what the audit found, stage by stage)

1. **Spawn** — the 5s probe misses deaths after its window or with a banner
   already written; on `SPAWN_FAILED` the just-acquired lane lease has no early
   release (`lease_held_until_ttl`, ~30min+ of lane downtime); the target issue
   is not re-queued. *Backlog issue C below.*
2. **Run** — no live-stall detection: a worker alive but writing nothing
   (retry livelock, hung tool) burns seat + lease until the 30-min wall clock.
   `dispatch_status.silent_workers` detects the signature; nothing acts on it
   (#750 is fenced record-only, #2277 detection-only). *Backlog issue A below.*
3. **Finish** — no-commit/unwitnessed exits only time-cool the issue (120min)
   and re-storm forever; the per-issue attempt budget shipped for #1777
   (`internal/attemptbudget`, `fak attempt-budget`, commits `3ab7ad8c` +
   `33a0f82a`) but **nothing in the Python tick consults it** —
   built-but-unwired. `CLAIM_UNWITNESSED` issues never close (close arm takes
   `OPEN_WITNESSED` only) and are not re-attempted. *Backlog issue B below.*
4. **Crash** — dead `claude -p` workers are re-dispatched only after cooldown
   (the resume watchdog covers autonomous/supervised sessions, not one-shot
   workers); partial work is lost by design. Accepted for now; the attempt
   budget bounds the waste.
5. **Issue side** — contract-held thin issues were invisible until groomed;
   the same-day contract-repair pipeline (peer lane) now grooms them, and the
   `updatedAt` re-admission (`d3112804` family) reopens them after repair.
6. **Measurement** — was: rate-only (`dispatch_throughput`). Now: the
   conservation identity above (`dispatch_conservation`).

## Ready-to-file issues (contract verdict: `ready`, 100/100 each)

Filing instructions: `gh issue create --title <title> --body-file <body>` with
labels `enhancement` + `dispatch`, milestone **Scale the dispatch fleet to 100
workers**. Bodies are verbatim below; re-validate any edit with
`fak issue contract --from-issues` before syncing. Dedupe fences are inside
each body's Batch policy.

---

### Issue B — `feat(dispatch): wire the shipped per-issue attempt budget (#1777) into the tick picker`

**Work conservation** · the re-storm half of the worker-unit conservation goal: an issue that keeps failing must stop burning fresh units without a human noticing.

#### Parent context
Dispatch worker-unit conservation: with N workers and N open issues in a window, every unit must end in a ship or a graded, ROUTED refusal. #1777 (closed) shipped the per-issue attempt budget as the `internal/attemptbudget` leaf + `fak attempt-budget` verb (3ab7ad8c, 33a0f82a) — but nothing in the Python dispatch tick consults it, so it is built-but-unwired.

#### Current state
`tools/issue_resolve_dispatch.py` gates re-dispatch of a failed issue with a flat time cooldown (`recently_attempted_issues`, default 120min) plus a one-tick hold for re-blockable no-commit reasons (`held_no_commit_issues`; the hold evaporates because the witness sweep skips already-witnessed logs on later ticks). `rg -i "attempt.budget" tools/` returns zero rows: no per-issue attempt counter, no failure-class-aware cooldown, no escalation ceiling. `tools/dispatch_conservation.py` (ceb0f7bf) now surfaces the symptom as `churn: issues burning 2+ units in one window`.

#### Why now
Fleet math: the seat cap is ~3-4 concurrent workers, so one re-storming issue can eat a double-digit share of a 6h window's capacity. Conservation reads today show open_now 699 vs baseline 483 — capacity must stop leaking into repeat losers while the backlog grows.

#### Working spine
In the tick's issue-pick path, before spawning on an issue, consult the shipped budget: count that issue's prior finished units from the durable artifacts the conservation meter already parses (`.witness` sidecars / worker logs for the issue number), classify the last failure (the `.witness` `reason` bucket), and refuse the spawn with a typed verdict (e.g. `ATTEMPT_BUDGET_HELD`) once the per-class budget is spent — routing the issue to a durable held ledger row (same idiom as `contract-holds.jsonl`) instead of the flat cooldown. `fak attempt-budget` / `internal/attemptbudget` already own the class→budget/cooldown policy; the tick supplies the counts and honors the verdict.

#### In scope
Wiring inside `tools/issue_resolve_dispatch.py`'s pick path + a small attempt-count reader over existing artifacts; the typed refuse verdict in the tick payload; a held ledger append; unit tests mirroring `issue_resolve_dispatch_test.py` idioms.

#### Out of scope
Changing `internal/attemptbudget` policy itself; the close arm; contract-gate behavior; any lane-lease change (that is the SPAWN_FAILED lease issue); GitHub-side labels/comments.

#### Done condition
A fixture issue with N prior `.witness` failures of one class is refused with `ATTEMPT_BUDGET_HELD` (not re-spawned), appears in the held ledger with its failure class, and a fresh issue is picked instead; `python tools/dispatch_conservation.py` churn count for the fixture window goes to zero once the budget binds.

#### Witness
`python tools/issue_resolve_dispatch_test.py` green including the new cases; a dry-run tick JSON showing `ATTEMPT_BUDGET_HELD` for the exhausted issue; ship commit passes `dos commit-audit`.

#### Acceptance gate
Same as Done condition; plus no regression in `tools/dispatch_conservation_test.py`.

#### Work unit
One worker owns the pick-path wiring + tests end to end in one sitting.

#### Expected steps
4

#### Assumptions
- `internal/attemptbudget`'s policy surface (classes, budgets, cooldowns) is authoritative; the tick only feeds it counts and honors verdicts.
- The `.witness` sidecar family (CLAIM_* + reason buckets) remains the durable failure record.

#### Confusion risks
- Do not double-gate: the flat time cooldown stays as the fast path; the budget binds only after repeated same-class failures.
- A budget hold must be a TYPED, ledgered verdict — silent skips recreate the invisibility this issue exists to end.

#### Coordination
- Touches `tools/issue_resolve_dispatch.py`, which is under active peer edit (contract-repair pipeline, 2026-07-02); rebase onto the landed repair work before starting.
- Lane `tools` — serialize with sibling dispatcher issues; verify via `dos_arbitrate` before writing.

#### Trigger
Filed from the 2026-07-02 worker-lifecycle leak audit: no-commit/unwitnessed finishes re-storm forever; the shipped attempt budget (#1777) is unwired.

#### Batch policy
One issue for the wiring; deduped against #1777 (leaf shipped, wiring absent) and #1396 (reason recording, shipped); update this issue rather than re-filing.

#### Likely files
`tools/issue_resolve_dispatch.py`, `tools/issue_resolve_dispatch_test.py`

#### Lane
`tools`

#### Closure binding
Closed by the ship commit wiring the budget into the pick path, stamped `(fak tools)` and referencing this issue; `dos commit-audit` verdict is the binding witness.

#### Ship discipline
- Trunk only; commit by explicit path; Conventional-Commits subject + `(fak tools)` stamp.
- Honest-scope fence: no claim that re-storms are eliminated until the churn meter shows it on live data.

_Self-contained: depends only on already-landed #1777 artifacts._

---

### Issue C — `feat(dispatch): release the lane lease on SPAWN_FAILED and grade header-only dead workers`

**Work conservation** · the spawn-failure half of worker-unit conservation: a unit that dies at spawn must free its lane immediately and be visible past the 5-second probe.

#### Parent context
Dispatch worker-unit conservation. Prior art: #1398 (closed) reaps a DEAD no-op worker's lane lease after exit; #1324 (closed) added the pre-spawn lane-lease gate. The residual is the SPAWN_FAILED path itself.

#### Current state
`tools/issue_resolve_dispatch.py` acquires the lane lease BEFORE spawning; on `SPAWN_FAILED` (probe catches an exit inside `DEFAULT_SPAWN_PROBE_S = 5.0` with an empty log) the tick records the verdict but the just-acquired lease has no early release — the tick payload reports `lease_held_until_ttl`, blocking that lane for worker_timeout + `LEASE_TTL_MARGIN_S` (~30min+). Separately, a worker that dies AFTER the 5s window (or after writing only its `# fak-spawn` banner) is never marked SPAWN_FAILED at all; `tools/dispatch_conservation.py` (ceb0f7bf) now counts these as `spawn_failed`/`leaked_unswept` after the fact, but the lane stays blocked and the issue is not re-queued.

#### Why now
With a ~3-4 seat cap, one dead-at-spawn worker can idle a whole lane for a double-digit share of a 6h window. Conservation reads show capacity leaking while open_now (699) outruns baseline (483).

#### Working spine
On a SPAWN_FAILED verdict, release the lane lease in the same tick (the witnessed-release idiom the normal exit path uses — never a blind delete: verify the lease holder is this tick's spawn before releasing). Extend spawn-death detection: on each tick, a worker log that is header-only/empty with a dead pid is graded SPAWN_FAILED retroactively (same evidence rule the conservation meter uses), releasing its lease and exempting the issue from the attempt cooldown so the unit is retried instead of silently lost.

#### In scope
Lease early-release on spawn failure; retroactive header-only/dead-pid grading in the reap/witness sweep; cooldown exemption for spawn-failure issues; tests.

#### Out of scope
The per-issue attempt budget wiring (sibling issue); changing probe duration semantics; lease TTL policy; any Go lease plumbing (`refs/fak/locks/*` verbs already exist).

#### Done condition
A fixture SPAWN_FAILED tick releases the lane lease in-tick (next tick can dispatch that lane) and the target issue is re-pickable immediately; a header-only log with a dead pid is graded SPAWN_FAILED on the next live tick; `python tools/dispatch_conservation.py` counts it under spawn_failed, not leaked_unswept.

#### Witness
`python tools/issue_resolve_dispatch_test.py` green including new cases; a two-tick dry-run trace showing the lane free on tick 2; ship commit passes `dos commit-audit`.

#### Acceptance gate
Same as Done condition; no regression in `tools/dispatch_conservation_test.py`.

#### Work unit
One worker owns spawn-failure lease release + retroactive grading + tests in one sitting.

#### Expected steps
4

#### Assumptions
- The lease acquire/release verbs (`fak` lease-lane) are stable; only the dispatcher's call sites change.
- The `# fak-spawn` header remains the first line every spawn writes.

#### Confusion risks
- Never release a lease the tick cannot prove it owns (a peer's live worker may hold the same lane name) — verify holder identity first.
- Retroactive SPAWN_FAILED grading must not reclassify a slow starter: require BOTH header-only log AND dead pid.

#### Coordination
- Touches `tools/issue_resolve_dispatch.py` (under active peer edit 2026-07-02, contract-repair pipeline); rebase onto landed work first.
- Lane `tools` — serialize with sibling dispatcher issues via `dos_arbitrate`.

#### Trigger
Filed from the 2026-07-02 worker-lifecycle leak audit: SPAWN_FAILED search returns zero issues; the lease leak and probe blind spot are untracked.

#### Batch policy
One issue; deduped against #1398/#1324 (both closed, different halves); update rather than re-file.

#### Likely files
`tools/issue_resolve_dispatch.py`, `tools/issue_resolve_dispatch_test.py`

#### Lane
`tools`

#### Closure binding
Closed by the ship commit adding the release + retroactive grading, stamped `(fak tools)`, referencing this issue; `dos commit-audit` is the binding witness.

#### Ship discipline
- Trunk only; explicit-path commits; Conventional-Commits subject + `(fak tools)` stamp.
- Honest-scope fence: claim lane-latency recovery only with the two-tick trace as witness.

_Self-contained: composes with the attempt-budget wiring issue but does not depend on it._

---

### Issue A — `feat(dispatch): reap live-but-stalled workers to free seat and lane before the 30-min wall clock`

**Work conservation** · the live-stall half of worker-unit conservation: a worker that is alive but not progressing must not burn its seat and lane for the full 30-minute wall clock.

#### Parent context
Epic #2269 (no-babysit loop). #2277 (open) ships the pid-liveness probe that classifies DEAD vs ALIVE_SILENT and explicitly does NOT act on ALIVE_SILENT ("a liveness-beat problem — do NOT re-dispatch"); epic #750 is fenced to record-only ("it records and grades, it does not gate admission"). This issue is the enforcement dual: ACT on a proven live-stall, in the dispatcher, where the seat and lease live.

#### Current state
The only bound on a live dispatch worker is `reap_timed_out_workers` (`tools/issue_resolve_dispatch.py`, `DEFAULT_WORKER_TIMEOUT_S = 1800`). A worker in a retry livelock or hung tool call holds its seat (cap ~3-4) and lane lease for the full 30 minutes; `dispatch_status.silent_workers` already DETECTS the signature (live pid + log silent past a floor) but nothing consumes it. The guard has no session stall detection (epic #1193 is visibility/control verbs only).

#### Why now
One stalled worker is ~25% of fleet capacity for half an hour, invisible until the wall clock. The conservation meter (ceb0f7bf) makes spent-vs-shipped visible; this issue stops the largest per-unit waste it can currently only report.

#### Working spine
In the tick, before the wall-clock reaper, fold the `silent_workers` evidence (live pid, log bytes unchanged for >= a stall floor, e.g. 10min, pid-reuse-safe identity match) into a typed reap decision: kill the worker tree (the existing `terminate_issue_worker_tree`), release its lane lease with holder verification, grade the log via the witness sweep as usual (its no-commit reason records the loss), and EXEMPT the issue from the attempt cooldown so the unit is retried. Dry-run ticks report `would_reap_stalled` without killing. Floor and enablement are flags (`--stall-reap-min`, default off for one bake window, then default on).

#### In scope
The stall-reap decision + kill + lease release + cooldown exemption in `tools/issue_resolve_dispatch.py`, reusing `dispatch_status.silent_workers` evidence; flags; tests.

#### Out of scope
Guard/session-side stall detection (#1193); taskmgr liveness grading (#750 fence); resume-watchdog behavior; changing the 30-min wall clock.

#### Done condition
A fixture worker with a live pid and a log silent past the floor is reaped on a live tick with a typed `reaped_stalled` verdict, its lane is dispatchable next tick, and the issue is re-pickable; a worker inside one long tool call but under the floor is untouched; dry-run only reports.

#### Witness
`python tools/issue_resolve_dispatch_test.py` green including the new stall cases; a dry-run tick JSON showing `would_reap_stalled` with the evidence fields; ship commit passes `dos commit-audit`.

#### Acceptance gate
Same as Done condition; false-positive fence proven by the under-floor test; no regression in `tools/dispatch_status_test.py`.

#### Work unit
One worker owns the stall-reap fold + tests end to end in one sitting.

#### Expected steps
5

#### Assumptions
- `dispatch_status.silent_workers`'s pid-reuse-safe identity rule is the authoritative stall evidence; this issue only ACTS on it.
- Long single operations (builds, model calls) are covered by the floor: log-silent means the harness wrote nothing, and 10min of nothing with a live pid is the documented livelock signature (#2159 data: verbatim-retry clear rate 0%).

#### Confusion risks
- Do not reap on transcript idleness alone — require the pid-identity match AND the byte-stable log, or a hung-but-healthy worker mid-tool gets killed (the #1984-class false positive from another fleet).
- Keep the floor well under 1800s but well over the longest observed healthy silent gap; start dry-run.

#### Coordination
- Touches `tools/issue_resolve_dispatch.py` (under active peer edit 2026-07-02); rebase onto landed work; lane `tools`, serialize via `dos_arbitrate`.
- Cross-reference #2277 (detection dual) so the two verdict vocabularies stay aligned.

#### Trigger
Filed from the 2026-07-02 worker-lifecycle leak audit: live-stall enforcement is unscoped in any open issue (#750 record-only fence, #2277 detection-only fence).

#### Batch policy
One issue; deduped against #2277/#750/#1193 (detection/visibility, not enforcement); update rather than re-file.

#### Likely files
`tools/issue_resolve_dispatch.py`, `tools/issue_resolve_dispatch_test.py`

#### Lane
`tools`

#### Closure binding
Closed by the ship commit adding the stall-reap fold, stamped `(fak tools)`, referencing this issue; `dos commit-audit` is the binding witness.

#### Ship discipline
- Trunk only; explicit-path commits; Conventional-Commits subject + `(fak tools)` stamp.
- Honest-scope fence: enable-by-default only after a bake window with zero false-positive reaps on live data.

_Self-contained: composes with the attempt-budget and spawn-failure issues; depends on neither._

---

## Provenance

Audit + fixes + meter: the 2026-07-02 goal session ("100 issues, 100 units,
6 hours — all done"). Contract validation: `fak issue contract --from-issues`,
all three `ready` at 100/100, 0 refused. Dedupe research: open/closed issue
sweep on 2026-07-02 (verdicts embedded in each Batch policy above).
