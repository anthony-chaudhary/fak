---
title: "Dispatch session observability + auto session-analysis tickets"
description: "The operating plan for giving fak operators default visibility into every dispatch session and auto-filing session-analysis tickets from what it finds."
---

# Dispatch session observability + automated session-analysis tickets — operating plan

**Goal (two halves).** (1) Give operators *default* visibility into all dispatch
work — **especially the actual worker sessions themselves**, not just backend
health. (2) Run *automatic* processes that analyze that session activity and
open GitHub tickets to improve the dispatch machinery.

This plan is spine-first (`docs/spine-first-defaults.md`): the minimal working
end-to-end surface ships this session; the hardening backlog is filed as issues
at creation time under the epic.

---

## What exists today (grounding)

A dispatch **worker** is not a bare agent CLI — the tick wraps the agent in
`fak manage` (`dispatch_tick.go:912` → `dispatchtick.GuardedLaunchCommand`), which
runs an in-process adjudicating gateway. That produces **two disjoint
observability planes that are never cross-referenced**:

| Plane | Keyed by | Written by | Read by |
|---|---|---|---|
| `.dispatch-runs/` sidecars (`resolve-<issue>-<stamp>.{log,pid,backend,lease-id,lease-tree.json,…}`) | issue + timestamp, PID liveness | `spawnDispatchIssueWorker` (`dispatch_tick.go:1043`) | `dispatch status` / `audit` / `evidence` / `progress` |
| `guard_sessions.jsonl` (handle, trace-id, agent, pid, cwd, audit-path) | handle / trace-id | `recordGuardSessionIndex` (`guard_sessions.go:35`) | `fak manage sessions` only |

Read verbs today:
- **`fak dispatch status`** — live-worker card (issue/lane/pid/worker), `fleet-dispatch-status/1`. Pure fold over the runs dir via `liveResolutionScopes` (`dispatch_tick_livescan.go:100`). Liveness is *derived* (log matches `resolve-*.log` + `.pid` alive + not a banner-noop), never self-reported.
- **`fak dispatch audit`** — per-worker outcome classification (`SHIPPED/RUNNING/WASTED_SPAWN/QUOTA_WALLED/RETRY_STORM/NO_OP/ERRORED`) + per-backend waste rollup + fingerprinted findings; `--file-issues` files a gh issue per NEW fingerprint (dedup by marker + open-title); `--heartbeat` appends to the loop ledger.
- **`fak dispatch evidence`** — wedged-worker partial evidence (scrubbed transcript tail, age from log mtime).
- **`fak dispatch progress`** — issue-close throughput + weekly retro (computes lane wedges + top blockers but **files nothing**).

Auto-analysis→ticket machinery today:
- `fak dispatch audit --file-issues` (Go, per-worker waste) — exists, **not scheduled in Actions**.
- `tools/dispatch_log_audit.py` (#1300) — log-signature failure detector (banner-noop, hook-storm, panic/Traceback, OFF_TRUNK, auth-wall) → gh issues. Runs **only** as a host-local Windows Scheduled Task (`FleetDispatchLogAudit`), never in Actions.
- `tools/idea_scout.py` — external research→backlog, also a Windows Scheduled Task.
- CI signal feeders (`bench-signal`, `gate-signal`, `score-signal`, `dogfood`) — Actions-scheduled, file with marker-dedup + max-issues + dry-run/live arm.

### Gaps this plan closes

**Observability (goal 1):**
- No single default place shows *the sessions themselves*: `status` never resolves a worker to its **audit outcome**, its **age / last-activity**, or its **guard session** (handle/trace/audit-path). You must run `status` + `audit` separately and hand-join.
- **No token/cost per worker** in the dispatch snapshot.
- `WallMinutes` is an *error-span*, not run duration — a clean worker reports `wall_minutes=0`.
- `dispatchaudit.go:36` hardcodes the literal `".dispatch-runs"` instead of `dispatchProgressRunsDir`.

**Automation (goal 2):**
- The two dispatch-session analyzers run only on one Windows host — no GitHub Actions coverage.
- **Genuine greenfield:** nothing mines per-session **token waste / retry counts / cost** into tickets. `trajctl-signal.yml` analyzes trajectories but files nothing; `knownbad` shares failure signatures cross-trace but does not ticket.

---

## Track A — Observability spine: `fak dispatch sessions` (build this session)

The minimal working end-to-end surface: **one command that folds both planes into
a per-session view for live *and* recently-completed workers.** Pure fold over the
filesystem (same inputs → same snapshot), so it is hermetically testable by
planting sidecars — exactly the `dispatch status` pattern.

Per-session row (`fleet-dispatch-session/1` inside `fleet-dispatch-sessions/1`):
- identity: `issue`, `lane`, `backend`, `worker`, `pid`, `pid_alive`, `live`
- outcome: `outcome` (`dispatchaudit.Classify`), `reason`, structured `evidence` (safe — no raw transcript)
- timing: `age_seconds` (now − log mtime), `started` (spawn-header stamp)
- lease: `lease_id`, `tree`
- **cross-plane join:** `guard` = `{handle, trace_id, audit_path}` resolved by PID against `guardsessions.Load(resolveSweepRegDir(""))` — the first command to bridge the two planes.

Surfaces: human table (default), `--markdown`, `--json`. Injectable `--now` for
determinism. Registered as `fak dispatch sessions` in `dispatch_order.go`.

**Witness (LCD bar):** `go test ./cmd/fak -run DispatchSessions` (hermetic, plants
sidecars + a guard row, asserts the join + outcome + age) **plus** one captured
`fak dispatch sessions --json` run over the repo's own `.dispatch-runs`.

### Track A fan-out backlog (filed at creation)
- **A1 observability** — per-session token/cost accounting (fold gateway usage/headroom by trace into the row).
- **A2 product** — make it the default: bare `fak dispatch` / `dispatch status` points at the session view; CLI reference + LCD demo entry.
- **A3 integration** — resolve each worker's `.guard-audit.json` + gateway trace and deep-link the audit journal.
- **A4 observability** — real run-duration wall-clock (fix `WallMinutes` error-span-only; derive from spawn stamp → exit/now).
- **A5 qa** — determinism/edge sweep: recycled PID, missing sidecars, banner-noop, cross-plane join misses, ambiguous PID.
- **A6 dogfood** — self-run on the repo's own `.dispatch-runs`, usage ledger.
- **A7 bugfix** (good-first) — `dispatchaudit.go` use `dispatchProgressRunsDir`, not the literal.
- **A8 docs** — doctrine + doc-map linkage (`INDEX.md`, this plan, `docs/run-the-demos.md`).
- **A9 observability** — `--watch`/follow rolling view + `--tail <handle>` single-session scrubbed transcript.

---

## Track B — Automated session-analysis → improvement tickets

**Spine ticket (filed `gen/now`, its own working spine required):** a GitHub
Actions workflow `dispatch-session-audit-feed.yml` that runs the *existing* Go
analyzer `fak dispatch audit --file-issues` on a schedule (dry-run default,
`workflow_dispatch live=true` arm), mirroring `bench-signal.yml`. Because
`.dispatch-runs/` is host-local, the spine must first decide the evidence-source
seam (ship a runs-dir artifact, or run the analyzer on the fleet host and push
findings) — that decision *is* the spine and is named in the ticket.

### Track B fan-out backlog
- **B1** — session **token-waste / retry-count / cost** analyzer → fingerprinted tickets (the greenfield lens `dispatch_log_audit.py` lacks). Reuses the `dispatchaudit` fingerprint+dedup substrate and `fak issue create`.
- **B2** — port `tools/dispatch_log_audit.py` signatures into the Go `dispatchaudit` taxonomy so one analyzer covers both log-signatures and waste.
- **B3** — weekly-retro → ticket: turn `dispatch progress --weekly` lane-wedge / top-blocker output into a filed improvement issue (deduped).
- **B4** — cloud coverage: a runs-dir evidence artifact so the analyzer runs in Actions, not just one Windows host.
- **B5 qa** — anti-spam sweep: marker dedup, open-title collision, max-issues cap, dry-run default (never file on a clean fleet).
- **B6 docs** — document the analyzer→ticket loop and its dedup contract.

---

## Sequencing
1. This plan doc (spine witness for the fan-out). ✅
2. File epic + Track A/B children (`fak issue create`, milestone + labels at creation).
3. Build + test + commit the Track A spine (`fak dispatch sessions`).
4. Note the spine commit on the observability-spine child.
