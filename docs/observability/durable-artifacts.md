---
title: "fak durable artifacts inventory"
description: "Everything the fak binary writes that survives process exit (logs, ledgers, journals, registries, locks) with a path and line writer citation for every claim."
---

# fak durable artifacts — inventory

_Refreshed 2026-07-10. A living reference: what the `fak` binary writes to disk that
survives process exit — logs, ledgers, journals, registries, sidecars, and lock/lease
state. In-memory-only state, `_test.go` fixtures, and per-launch temp scratch are out of
scope. Writer citations are `path:line` of the primary write site so the claim is
checkable against the tree, not this doc._

## How to read this

fak's durable state splits along four axes. Knowing which axis an artifact is on tells you
where it lives, who may delete it, and whether it is safe to commit:

| Axis | Root | Committed? | Lifetime |
|---|---|---|---|
| **Repo-local operator telemetry** | `.dispatch-runs/`, `.goal-runs/`, `.fak/` (cwd-relative) | No — gitignored | Regenerated each tick; host-local |
| **Repo-tracked analytics ledgers** | `docs/nightrun/*`, `docs/{cadence,programs,milestones,dojo}/` | Yes — tracked | Grows; folded/cut to bound |
| **Fleet / machine-local state** | `$FLEET_REG_DIR`, `$FLEET_STATE_DIR`, `UserConfigDir/fak`, `%LOCALAPPDATA%\Fleet` | No — out of repo | Fleet-shared / crash-recovery |
| **Arbitration substrate (git refs)** | `refs/fak/locks/*` in `.git` | Rides git object store | CAS + TTL; reaped on expiry |

Two facts to internalize up front, both surfaced by the mapping and easy to get wrong:

- **fak does not write the `.dos/` journals.** `.dos/verdict-journal.jsonl`,
  `.dos/lane-journal.jsonl`, `.dos/metrics/observations.jsonl`, `.dos/runs/*.jsonl`, etc.
  are emitted by the **`dos` kernel / dos hook**. fak only *scans* them
  (`internal/conceptusage`, `cmd/loophealth`, `cmd/fak/hooklat.go`, `internal/relay`). See
  [What fak reads but does not write](#what-fak-reads-but-does-not-write).
- **fak's lock/lane/intent arbitration is not a file** — it is persisted as **git refs**
  under `refs/fak/locks/*`, written through `git update-ref` compare-and-swap. See
  [Arbitration substrate](#5-arbitration-substrate--git-refs-not-files).

### Configurable roots (env overrides, in precedence order)

| Env var | Controls | Fallback chain |
|---|---|---|
| `FLEET_REG_DIR` | fleet registry dir (`sessions.json`, `resume_ledger.jsonl`, `probe_ledger.jsonl`, `guard_sessions.jsonl`) | `$FLEET_STATE_DIR/registry` → `%LOCALAPPDATA%\Fleet\registry` → `%TEMP%\Fleet\registry` → `<repo>/tools/_registry` |
| `FLEET_STATE_DIR` | cooldown store + registry parent | `%LOCALAPPDATA%\Fleet` → `%TEMP%\Fleet` → `<repo>/tools/_registry` |
| `FLEET_POLICY_PATH` / `FLEET_POLICY_DIR` | `accounts_policy.json` (read-only) | `<repo>/tools/_registry/accounts_policy.json` |
| `FAK_AUDIT_JOURNAL` | guard decision journal (boot-time enable) | `--audit PATH` flag, else `.dispatch-runs/guard-audit/...` |
| `FAK_SESSION_JOURNAL` | session crash-journal | `UserConfigDir/fak/session-journal.jsonl` → `.fak/session-journal.jsonl` |
| `FAK_TOOLPROC_JOURNAL` / `FAK_REPO_GUARD_TOOLPROC_JOURNAL` | toolproc journal | `.fak/toolproc/journal.jsonl` |
| `FAK_REPO_GUARD_DECISIONS` | repoguard decision journal | `.fak/repoguard/decisions.jsonl` |
| `FAK_GUARD_TRAJCTL_LEDGER` | trajectory-control ledger | `docs/nightrun/trajctl.jsonl` (`off` disables) |
| `FAK_MEMORY_COTRAVEL_LEDGER` | memory co-travel ledger | `~/.claude/fak-memory-cotravel-ledger.jsonl` |
| `FAK_STALL_DIR` | stallscan self-monitor ledger | `%LOCALAPPDATA%\Fleet\stallscan.jsonl` → `~/.fak/stallscan.jsonl` |
| `FAK_WATCHDOG_AUTOHEAL_DIR` | watchdog autoheal state dir | `UserConfigDir/fak/watchdog-autoheal` → `%TEMP%\fak-watchdog-autoheal` |
| `FAK_WATCHDOG_LOG_DIR` | resume-watchdog operational logs | `<repo>/tools/_watchdog` |

`.dispatch-runs/`, `.goal-runs/`, and `.fak/` are **cwd/workspace-relative and not
env-configurable** (the runs dir is `--workspace`-relative). The `docs/nightrun/*` family
anchors to the real repo root via `nightrunLedgerPath` (`cmd/fak/nightrun_ledger_path.go:23`,
a `go.mod` upward walk) regardless of process cwd.

---

## 1. Repo-local operator telemetry — `.dispatch-runs/`

Root: `dispatchtick.RunsDirName = ".dispatch-runs"`, resolved `filepath.Join(<workspace>, ".dispatch-runs")`.
Gitignored (`.gitignore:330`) — host-local, regenerated each dispatch tick.

| Artifact | Path pattern | Format | Writer |
|---|---|---|---|
| Worker session log | `resolve-<issue>-<UTCstamp>.log` | plain text (stdout+stderr tee) | `cmd/fak/dispatch_tick.go:1287` |
| Worker slot sidecars | `<stem>.{pid,backend,lease-id,lease-tree.json,basesha,worktree,account,wave,model,witness,prompt.txt,startup.json,partial-evidence.json}` | mixed text/JSON | `dispatch_tick.go:1324–1353`, `dispatch_model_policy.go:22`, `dispatch_tick_witness.go:152`, `dispatch_worker_evidence.go:245` |
| Last-resolve-tick snapshot | `last-resolve-tick.json`, `last-resolve-tick-<backend>.json` | single-JSON overwrite | `dispatch_tick.go:1499` |
| Skip ledger | `skip-ledger.jsonl` | JSONL append | `cmd/fak/skip_ledger.go:91` |
| Timeout ledger | `timeout-ledger.jsonl` | JSONL append | `cmd/fak/timeout_ledger.go:125` |
| Route-health ledger | `route-health.jsonl` | JSONL append | `cmd/fak/dispatch_route_health.go:375` |
| Progress log + baseline | `progress.jsonl` (append) + `progress-baseline.json` (overwrite) | JSONL / single-JSON | `dispatch_progress.go:786` / `:636` |
| Canary run sidecars | `canary/<runID>/{metadata,route-health,lease}.json, prompt.txt, transcript.jsonl, guard-audit.jsonl` | JSON + JSONL | `cmd/fak/dispatch_canary.go:295–370` |
| Dispatch loop ledger | `--ledger` value (or `defaultLoopLedger()`), joined to root | JSONL append (loopmgr) | `dispatch_progress.go:835` |
| Guard-audit journal | `guard-audit/interactive-<pid>-<hash>.jsonl` | hash-chained JSONL | see §2.1 |
| Status doc | `dispatch-status.md` | markdown (python `tools/dispatch_status.py --md`) | operator tool, not the Go binary |

Related repo-local roots: `.goal-runs/*.pid` (detached goal-loop workers,
`dispatch_tick_preflight.go:1260`); `.fak/toolproc/` and `.fak/repoguard/` (see §2).

---

## 2. Guard / session / hooks

The `fak guard` wrapper and the Claude Code lifecycle hooks it installs.

### 2.1 Guard decision journal — the primary durable witness

Hash-chained, tamper-evident, append-only WORM ledger: one JSONL row per kernel
adjudication (`DECIDE | DENY | RESULT_DENY | QUARANTINE | VDSO_HIT | CAP_FAULT | CAP_EVICT |
CAP_VERSION_BIND`), flushed per write, each row carrying the prior row's chained hash
(`internal/journal/journal.go` — package doc at `:1`, `Row` schema at `:62`, append at
`:161`/`:192`).

- **Off by default.** Enabled via `FAK_AUDIT_JOURNAL=<path>` (boot) or `journal.Enable(path)`.
  `fak guard` defaults it **on** (`guardEnableAudit`, `cmd/fak/guard_support.go:530`; wired at
  `guard.go:477`), writing to `.dispatch-runs/guard-audit/interactive-<pid>-<hash>.jsonl`
  where `<hash>` = first 12 hex of sha256(abs repo root) (`guardDefaultAuditPath`,
  `guard_support.go:461`). `--audit PATH` overrides; `--audit off` disables.
- **Rotation:** `Cut` archives a sealed segment to a sibling `<path>.cut-<finalSeq>` without
  breaking the chain (`internal/journal/rotate.go:43`); a rotated set verifies end-to-end.
- Fleet workers name theirs `<lane>-<backend>-<pid>-<id>.jsonl` in the same dir;
  `fak audit verify <path>` re-reads and recomputes the chain.

### 2.2 Other guard/session/hook artifacts

| Artifact | Path | Format | Writer | Notes |
|---|---|---|---|---|
| Refusal carry-forward | `<auditPath>.refusals.json` | single-JSON (atomic tmp+rename) | `guard_carryforward.go:237` | top refusals → next session's banner |
| Guard gateway log | `--log FILE` (off by default; `-`/`stderr` → stderr) | plain-text log | `guardLogSink`, `guard_support.go:430` | operator diagnostics |
| Toolproc containment journal | `.fak/toolproc/journal.jsonl` | JSONL append (compacted at stop) | `cmd/fak/toolproc.go:181` | spawn/pulse/exit/kill per tool call; `fak toolproc ps` |
| Console-fault snapshot | `.fak/toolproc/console-faults.jsonl` | JSONL **overwrite** (idempotent projection) | `consolefault_ingest.go:240` | pwsh HostException / `__fastfail` from event log |
| Host-fault snapshot | `.fak/toolproc/host-faults.jsonl` | JSONL overwrite | `hostfault_ingest.go` (`toolproc.go:411`) | Windows Update / GPU TDR / app hangs |
| Repoguard decision journal | `.fak/repoguard/decisions.jsonl` | JSONL append | `internal/repoguard/decisions.go:87` | one row per deny/advisory/record; out-of-tree-write witness |
| Trajectory-control ledger | `docs/nightrun/trajctl.jsonl` | JSONL append | `internal/trajctl/trajctl.go:139` | per-turn scores + compaction boundaries + detours (Stop/PreCompact hooks) |
| Session crash-journal | `session-journal.jsonl` (see roots table) | JSONL append | `internal/sessionjournal/sessionjournal.go:92` | open/beat/close vs. boot epoch → LIVE/CRASHED/STALE/CLOSED |
| Durable session registry | `UserConfigDir/fak/session-registry.json` (or `.fak/session-registry.json`) | single-JSON overwrite | `cmd/fak/session_durable.go:104` | mirror of serve/guard session table |
| Guard session index | `<regDir>/guard_sessions.jsonl` | JSONL append | `internal/guardsessions/guardsessions.go:90` | **⚠ writer present but the live-launch trigger is not wired** — `recordGuardSessionIndex` (`guard_sessions.go:35`) is referenced only from `guard_sessions_test.go`. `fak guard sessions` reads it; a live `fak guard` may not populate it in this tree. |
| Harness-resources ledger | `.fak/nightrun/harness-resources.jsonl` | JSONL append | `cmd/fak/harness_resources.go:18` | per-session CPU/RSS/disk-IO on guard/serve exit (`--resource-stats`, on by default) |

**Excluded (per-launch temp only):** the hook *installers* write Claude Code `--settings`
JSON into `os.MkdirTemp("fak-guard-*")` dirs scoped to the one launched child
(`guard_precompact.go:217`, `guard_sessionstart.go:212`, via `writeGuardSettingsFileAtomic`)
— not durable state.

---

## 3. Runtime analytics ledgers and published snapshots

Background-written trend ledgers are anchored under the gitignored `.fak/nightrun`
runtime state root. The same-named tracked files under `docs/nightrun` are historical
publication snapshots, not live writer targets. This keeps guard/serve/cron ticks
from dirtying the shared tree. A fresh clone has zero live rows until its first tick;
the readers below tolerate absence. Most writers share the `appendLedgerFile[T]` seam
(`cmd/fak/ledgerio.go:28`) and take a `--ledger` override.

| Ledger | Default path | Schema | Writer | Purpose |
|---|---|---|---|---|
| Nightrun collection | `.fak/nightrun/collected.jsonl` | — | `internal/nightrun/run.go:482` | one row per collected bench task (run-it-all-night) |
| Fleet status history | `.fak/nightrun/fleet-status-history.jsonl` | `fleet-trend/1` | `internal/fleettrend`, `tools/fleet_trend.py` | bounded status-tick trend for fleet/Slack views |
| Gateway usage | `.fak/nightrun/gateway-usage.jsonl` | `fak-gateway-usage-ledger/1` | `internal/gatewayusageledger/ledger.go:201` | per-session token/usage counters (`serve.go:478`) |
| Cache-value (Track 1) | `docs/nightrun/cache-value.jsonl` | `fak-cache-value-ledger/1` | `internal/cachevalueledger/ledger.go:90` | witnessed prompt-cache KV-reuse (`serve.go:401`, `run_model.go:81`) |
| Cache-savings (Track 2) | `.fak/nightrun/cache-savings.jsonl` | — | `internal/cachevaluereport/track2.go:582` | observed-$ provider-cache + compaction savings |
| Known-bad signatures | `docs/nightrun/known-bad.jsonl` | `fak.known-bad.v1` | `cmd/fak/knownbad.go:962` | fleet-wide record/claim/resolve/revoke of known-bad trees |
| Harness-resources | `.fak/nightrun/harness-resources.jsonl` | `fak-harness-resources/1` | `cmd/fak/harness_resources.go:18` | per-session resource footprint |
| Memory-value / recall | `docs/nightrun/memory-value.jsonl` | — | `cmd/fak/memory_recall.go:338` | recall-event / memory-value rows (`off`/`none` disables) |
| Trajectory-control | `docs/nightrun/trajctl.jsonl` | `fak-trajctl/1` | `internal/trajctl/trajctl.go:139` | see §2.2 |
| Cadence history | `docs/cadence/history.jsonl` | — | `cmd/fak/cadence.go:102` | dev-cadence scores/maturity trend |
| Program history | `docs/programs/history.jsonl` | — | `cmd/fak/program.go:158` | program frontier metrics per tick |
| Milestone history | `docs/milestones/history.jsonl` | — | `cmd/fak/milestone.go:126` | milestone trend rows |
| Dojo history | `docs/dojo/history.jsonl` | — | `cmd/fak/dojo.go:191` | dojo/RSI episode scoring |
| Cache-frontier review | `--append-ledger` JSONL + `--markdown-out` md (e.g. `docs/cache-frontier/review-ledger.jsonl`) | — | `cmd/fak/cachevalue_review.go:276`/`:291` | durable machine + human cache-frontier review |

**Retired:** `docs/nightrun/fleet-outcome-health.jsonl` never had a writer or reader in-tree
(one-off WIP snapshot, frozen `CRIT` rows from 2026-07-08); retired per #5149 — its content is
now a single tombstone row and the file is pending deletion. Do not fold it as loop-health.

**Bounding:** the gateway-usage ledger is compacted in place by `Cut`
(`internal/gatewayusageledger/cut.go:130`, atomic tmp+rename) folding rows older than the
newest N into counter-preserving carryforward rows — operator door `fak nightrun cut --apply`
(`cmd/fak/nightrun.go:438`). The decision journal (§2.1) rotates via `.cut-<seq>` segments.

Nightrun also writes **per-task output logs** `<ArtifactDir>/<box>/<UTCstamp>-<taskID>.log`
(plain text, kept even on timeout; `internal/nightrun/run.go:420`) and a pre-launch
watch-context descriptor (`watchctx.go:99`).

---

## 4. Fleet / accounts / machine-local state

Out-of-repo state under `$FLEET_REG_DIR` / `$FLEET_STATE_DIR` / `UserConfigDir` (see roots
table). Fleet-shared or crash-recovery; never committed.

| Artifact | Path | Format | Writer | Purpose |
|---|---|---|---|---|
| Accounts runtime registry | `<regDir>/sessions.json` | single-JSON, **atomic** (tmp+rename) | `internal/accounts/accounts.go:1123` | canonical seat/home/session registry |
| Account cooldown store | `<stateDir>/account-cooldown.json` | single-JSON, atomic | `internal/accounts/cooldown.go:134` | per-account usage/429 cooldown windows |
| Per-seat account homes | `<home>/.claude-<name>/{.oauth-token,.claude.json,.credentials.json}` | token (0600) + JSON | `accounts_add.go:260`/`:1255`, `credbackup.go:253` | isolated per-seat Claude credential homes |
| Credential backups | atomic 0600 snapshots | JSON | `internal/accounts/credbackup.go:253` | pre-mutation credential restore points |
| Resume/launch ledger | `<regDir>/resume_ledger.jsonl` | JSONL append | `cmd/fak/resume_watchdog_cli.go:764` | launch/phase rows; read by `fak resume status/self/why` + gateway history |
| Session identity store | `<regDir>/resume_identity.jsonl` | JSONL append (never swept, no TTL) | `cmd/fak/guard_sessionstart.go:162`/`:175` | transcript-UUID ↔ gateway-trace join, plus the **driver pid** the SessionStart hook witnessed; read by `fak resume identity` and the `fak resume stopped` liveness probe |
| Resume tick lock | `<regDir>/resume_watchdog.lock` | `O_CREATE|O_EXCL` (TTL 2m) | `internal/resume/ticklock.go:49` | single-flight mutual exclusion |
| Watchdog autoheal state | `<autohealDir>/<id>.json` + `<id>.lock` + `autoheal.log` | single-JSON / lock / text | `watchdog_autoheal.go:661`/`:295` | restart backoff/debounce (`fak.watchdog-autoheal.v1`) |
| Resume-watchdog logs | `<logDir>/{resume_watchdog.log, notifications.log, resume-<sid>-<unix>.log/.err, notified.json}` | text + JSON | `resume_watchdog_cli.go:886`/`:1040` | operational + per-spawn child output |
| Stallscan self-monitor | `<FAK_STALL_DIR>/stallscan.jsonl` | JSONL (`fak.stallscan.v1`) | `fak stallscan --watch` (`stallscan.go`) | churn/stall self-monitor; read by dispatch preflight |
| Probe ledger | `<regDir>/probe_ledger.jsonl` | JSONL | **python fleet probe** (Go reads via `accountprobe.ReadLedger`) | account reachability probes |
| Fleet fold/janitor ledger | `--ledger` value (no default) | JSONL append | `cmd/fak/fleet.go:566` | folded worker rows / janitor decisions (`fak fleet fold\|sweep\|replace --write`) |
| Fleet sweep sample-state | `--state` value (no default) | single-JSON overwrite | `cmd/fak/fleet.go:720` | prev-sample line/CPU deltas |

### A session with no recorded driver pid is not reclaimable, and no pass will backfill it

`fak resume stopped` only moves a mid-tool row off DEFER when it can decide whether the
driver that owned the transcript is still running, and the only handle it has on that is a
pid something durably **recorded**. There are exactly two writers: the launch ledger (for a
session the resume watchdog itself resumed) and the SessionStart hook above (for a
first-generation `claude -p …` worker, which carries no session id on its argv). If neither
wrote one, the row stays `MIDTOOL_UNKNOWN` and defers — the fail-safe working, not a bug to
wait out.

**A session that is not running under `fak guard` will never gain one.** fak installs its
hooks only into the `--settings` file `fak guard` appends to the wrapped agent's argv at
launch; it reads `~/.claude/settings.json` (`fak guard allow --from-claude-settings`) but
never writes hooks there. So a session started by hand has no SessionStart hook, no Stop
hook, and no other in-process moment where fak could witness the driver. **There is no
backfill: restart the session under `fak guard` if you want it reclaimable.** Waiting will
not produce a recorded pid.

A session that *is* under `fak guard` but predates the pid witness needs no new mechanism —
the SessionStart matcher is deliberately empty (`guard_sessionstart.go:469`), so the hook
re-fires on `clear` / `compact` / `resume` in the same driver process and appends a fresh pid
row then; the fold takes the last row per transcript. Until that next SessionStart source
event fires, it defers.

---

## 5. Arbitration substrate — git refs, not files

fak's lock/lane/intent layer (the analog of the DOS lease layer) is **not** on the
filesystem as loose files. It is persisted as **git refs** under `refs/fak/locks/*`, each
pointing at a JSON blob, written through `git update-ref` compare-and-swap — so persistence
rides the git object store in `.git` and converges across clones via fetch/push. Store dir
selectable with `--dir` (default: git discovery from cwd). CLI: `cmd/fak/leaseref.go`.

| Ref | Record | Writer |
|---|---|---|
| `refs/fak/locks/<id>` | lock lease | `AcquireFenced`/`Renew`/`Release`, `internal/leaseref/fence.go` |
| `refs/fak/locks/intent-<key>` | `IntentRecord` (TTL 3600s) | `Store.ClaimIntent`, `internal/leaseref/intent.go:202` |
| `refs/fak/locks/session-<id>` | liveness heartbeat descriptor | `PublishSession`, `internal/leaseref/liveness.go` |

Reaping deletes expired refs of all three kinds.

---

## 6. Memory, finding sinks, benchmark & scorecard outputs

| Artifact | Path | Format | Writer | Purpose |
|---|---|---|---|---|
| Memory co-travel ledger | `~/.claude/fak-memory-cotravel-ledger.jsonl` (`FAK_MEMORY_COTRAVEL_LEDGER`) | JSONL (atomic rewrite) | `internal/memorycotravel/memorycotravel.go:~211` | fact-file co-travel; also copies `*.md` facts into a dest memory dir |
| Memory store | `.claude/memory/` (`MEMORY.md` + `*.md`) | markdown | agent-authored; fak writes only via co-travel copy | recall corpus; scanned by `internal/memq` |
| Checkpoint findings ledger | `<dir>/.fak/checkpoint-findings.jsonl` | JSONL **content-addressed upsert** (atomic rewrite) | `internal/findingsink/findingsink.go:273` | converging findings keyed by finding Key + emit count |
| Macbench watch log + result | `--log` (append) + `--result` (overwrite) | multi-object indented JSON | `cmd/fak/macbench.go:245`/`:290` | benchmark health stream + latest result |
| Product-scorecard docs | `--markdown-dir <dir>` (e.g. `README.md`) | rendered markdown overwrite | `cmd/fak/productscorecard.go:80` | rendered scorecard doc folder |

---

## What fak reads but does not write

Frequently mistaken for fak outputs — these are authored by **other** processes; fak only
reads/scans/verifies them:

- **`.dos/` kernel journals** — `.dos/verdict-journal.jsonl`, `.dos/lane-journal.jsonl`,
  `.dos/metrics/observations.jsonl`, `.dos/rsiloop-journal.jsonl`, `.dos/runs/*.jsonl`,
  `.dos/stop-failures`, `.dos/streams`. Written by the `dos` kernel / dos hook
  (`dos improve --observe`). fak reads them: `internal/conceptusage/conceptusage.go:230`,
  `cmd/loophealth/loophealth.go`, `cmd/fak/hooklat.go:103`, `internal/relay`, `internal/growthgate`.
- **`resume_plan.json` / `sessions.json`** in the registry dir when refreshed by the
  **python `tools/fleet_sessions.py registry`** child (invoked from `rwRefreshRegistry`,
  `resume_watchdog_cli.go:811`) — fak reads via `rwLoadPlan`. (Note `sessions.json` *is* also
  written by the Go accounts registry in §4; the python path is the resume-watchdog refresh.)
- **`accounts_policy.json`** — operator-authored (`FLEET_POLICY_PATH`); read via `LoadPolicy`.
- **Relay progress files** — `internal/relay/progress_file.go` is a fail-closed *reader* of
  run/intent ledgers (`.dos/runs/*.jsonl`); the only writes are in `_test.go`.
- **Terminalbench contract artifacts** — `internal/terminalbench/contract.go` only *names*
  expected paths (`harbor-test-results.json`, `fak-command-evidence.jsonl`,
  `fak-gateway-witness.json`); the harbor/agent harness writes them.
- **Audit-receipt ledger** — `cmd/auditreceipt/` is a *verifier* only (reads `LEDGER.jsonl`,
  emits verdict JSON to stdout); the ledger itself is produced by `internal/modelroute`
  in the model-routing/gateway path.
- **Chat relay** (`internal/chatrelay`) is network-only (Slack HTTP); no disk artifacts.

## Slack transactional outbox (network-backed, but durable spool)

`internal/slackoutbox` persists a crash-safe send queue as append-only JSONL: a live HEAD
(`spool.jsonl` / `state.jsonl`) producers append to, a sealed transient (`*.seal.jsonl`), a
compacted archive (`*.arch.jsonl`), and a `drain.lock`. Load replays archive → seal → head so
a restart resumes exactly where the last process stopped (`internal/slackoutbox/outbox.go:29`).
Dir is caller-supplied (the Slack relay config), ensured via `pathutil.EnsureDir`.

---

_Method: this inventory was rebuilt by fanning out four parallel read-only surveys (fleet/
dispatch/accounts, guard/session/hooks, DOS/trajectory/memory/resume, and the misc long
tail) over `cmd/fak/` and `internal/`, cross-checked against `.gitignore` and the path
resolvers. To refresh: re-run the survey and diff against the writer citations above — a
citation that no longer resolves is the signal that an artifact moved or was removed._
