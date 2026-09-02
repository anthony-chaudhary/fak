---
name: resume-watchdog-audit
description: Audit and recover crashed Claude and Codex sessions through one dry-run-first cohort surface, with exact provider identity and post-launch transcript/thread advancement; then audit the scheduler tower that keeps automatic recovery alive. Use when this named workflow matches the task.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, PowerShell, Grep, Glob, Write
argument-hint: "[--fix] (no args = read-only cross-provider preview; --fix offers the operator-gated visible live drain)"
---

# /resume-watchdog-audit — is the watchdog itself alive, or silently stalled?

> **The recursion.** `fak session recover` is the cross-provider recovery surface;
> the resume watchdog (`fleet_resume_watchdog.ps1 -Live`, task
> `FleetResumeWatchdog`) is the **n¹** automatic Claude layer. But a watchdog that dies takes the fleet down
> **silently** — every dead session just stays dead and nothing complains. This skill
> is the **n²** layer (watch the watchdog) and the **n³** layer (watch the watcher of
> the watchdog). It answers from artifacts — ledger mtimes, transcript turns, and
> Task Scheduler exit codes — **never** from the watchdog's own status text.

The one rule: **silence is the failure mode.** A stalled watchdog and a healthy-with-
nothing-to-do watchdog look identical from the outside. This pass distinguishes them.

## Run it (one authoritative cohort command)

```powershell
fak session journal-audit --since 24h --json           # read-only launch -> exact provider-journal proof
fak session recover --json                         # preview only; writes a durable run witness, launches nothing
# After reviewing every row and completing any named login:
fak session recover --live --all --json            # all actionable exact sessions; add --limit N only to cap the wave
```

The journal audit sources recent launch identities only from the resolved live
`resume_identity.jsonl`, discovers every local Claude project root and Codex home,
and exact-joins each full identity to provider-owned transcript/turn cursors. Its
versioned JSON distinguishes `advanced`, `missing_transcript`, and
`present_no_post_launch_progress`; RED/nonzero on any unproven row or unreadable authority is deliberate.
Use its rows instead of a hand-written cross-root join or a newest-session guess.

The recovery preview is the selection authority for the Claude transcript cohort and the Codex
guard/host-resurrection cohort. Its versioned JSON retains probes, live rows, reset
waiters, and identity-blocked rows instead of hiding them; only substantive actionable
rows receive argv. Codex argv uses `codex exec --cd <exact cwd> resume <exact UUID>` and never a picker,
`--last`, prefix, newest-row guess, or guard handle. Read the reported `witness_path`
after a live run: a window/process alone remains `launched_unproven`; productive means
the Claude assistant transcript or Codex thread/turn cursor advanced after launch.

**Current operating limit:** until #9742 closes, a timed-out or non-advancing live
launch is not guaranteed to have its recovery-created process tree reaped automatically.
Keep the default `--limit 1`, inspect the exact `witness_path`, and clean up only the
PID/start-identity tree named by that run. Never treat `launched_unproven` as a
successful recovery or use it as proof that the original session is active.

The scheduler/tower check is separate and remains read-only:

```powershell
pwsh -NoProfile -File tools/watchdog_watchdog_audit.ps1        # human-readable tower verdict
pwsh -NoProfile -File tools/watchdog_watchdog_audit.ps1 -Json  # machine tower verdict; exit 0/2/3
```

The sections below expand both checks. **This audit must
never share the failure mode it detects: run it as an agent `/loop` or an S4U task,
never as an Interactive scheduled task** (that would die exactly the way the watchdog did).

---

## Layer 0 — locate the LIVE registry (do not trust `tools/_registry`)

`resolveSweepRegDir("")` resolves in order: `$FLEET_REG_DIR` → `$FLEET_STATE_DIR\registry`
→ `%LOCALAPPDATA%\Fleet\registry` (if it exists) → `%TEMP%\Fleet\registry` → repo
`tools/_registry` (fallback only). **The repo's `tools/_registry` is usually a stale
copy** — its `sessions.json` may be fresh (the dispatcher writes it) while its
`resume_ledger.jsonl` is days old. Always audit the resolved live dir.

The same preview reads Codex's configured `state_5.sqlite` in read-only/query-only
mode. That database's full thread UUIDs are the identity authority. A host-resurrection
row joins only by a full UUID already in argv or the durable trace→UUID identity ledger,
and is then verified back against `state_5.sqlite`. CWD, timestamps, ordering, prefixes,
and "newest" are never identity evidence; a missing or conflicting exact identity is
`identity_blocked`.

```powershell
$reg = @("$env:FLEET_REG_DIR","$env:FLEET_STATE_DIR\registry","$env:LOCALAPPDATA\Fleet\registry","$env:TEMP\Fleet\registry") |
  Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
"live regDir = $reg"
Get-ChildItem $reg -Filter 'resume_*' | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
```

## Layer 1 (n²) — the drain verdict + the stall check

**The authoritative read-only verdict** (returns exit 3 on RED — that is by design):

```powershell
& (Get-Command fak).Source resume watchdog --status --json
```

Fields that matter: `verdict` (green/amber/red), `auto_resume_depth` (queue depth),
`silent_seconds` (oldest unrecovered queued row's silence), and `mttr_sessions[]` with
`status: launched_unproven` (resumed but no real transcript turn after launch within
`--unproven-minutes`, default 10). **Caveat:** `mode` echoes YOUR invocation, so a plain
`--status` reports `DRY-RUN` and may add a spurious *"watchdog is DRY-RUN with queued
rows"* reason — ignore that one; the load-bearing reasons are `launched_unproven` and
`silent >= silent-hours`.

**The stall check the verdict alone won't give you** — has the watchdog *ticked at all*
recently? Compare ledger mtimes to now:

```powershell
$now = Get-Date
foreach ($f in 'resume_ledger.jsonl','resume_watchdog_status.jsonl','resume_plan.json') {
  $p = Join-Path $reg $f; if (Test-Path $p) {
    $m = (Get-Item $p).LastWriteTime; "{0,-28} {1}  ({2:n0} min ago)" -f $f,$m,($now-$m).TotalMinutes } }
```

> **RED if the newest of these is > ~15 min old.** The watchdog ticks on a short cron
> (default every 10 min); no write in 15+ min means it is not running. This is the
> single most important signal and the `--status` verdict does **not** encode it —
> `--status` reads whatever the last tick left behind and cannot tell "healthy + quiet"
> from "dead since 11:42".

## Layer 1b — WHY it stalled: exit code **paired with `Principal.LogonType`**

If it stalled, the cause is almost always the launch context, not the logic. Read the
task family's last result **and its LogonType together** — the LogonType is the
load-bearing column:

```powershell
Get-ScheduledTask | Where-Object { $_.TaskName -match 'Resume|Supervisor|Watchdog|Guard|Seat|Stranded|Dispatch' } | ForEach-Object {
  $i = Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath -EA SilentlyContinue
  [pscustomobject]@{ Task=$_.TaskName; LogonType="$($_.Principal.LogonType)"; Result=('0x{0:X}' -f $i.LastTaskResult); LastRun=$i.LastRunTime } } |
  Sort-Object LogonType | Format-Table -AutoSize
```

**Known fault (2026-07-09), CORRECTED diagnosis:** `0x800710E0` = "The operator or
administrator has refused the request" appears on **every task with
`Principal.LogonType=Interactive`**, while every **S4U** sibling returns `0x0` — on this
same headless / RDP-accessed box, at the same time, under the same launcher. That split
**is** the diagnosis: **Interactive-logon tasks are refused when there is no true
interactive console** (they stay refused even while an RDP session shows `Active` in
`qwinsta`, and die outright when it disconnects); **S4U** tasks ("run whether logged on
or not"; session 0, windowless, still AS THIS USER) are immune. The clean natural
experiment: `FleetResumeWatchdog` (Interactive → `0x800710E0`) sitting next to
`FleetStrandedRecovery` (S4U → `0x0`).

> **The `conhost.exe --headless` shim is a RED HERRING — do NOT chase it.** An earlier
> version of this skill blamed conhost. It was empirically disproved: `FleetResumeWatchdog`
> kept returning `0x800710E0` *after* conhost was removed and it launched `powershell.exe`
> directly. Removing conhost does nothing; only the LogonType matters. (`ExploitGuard MDM
> policy Refresh` / `SafeguardsReconciliation` are unrelated `ServiceAccount` system tasks
> that are always `0x0` — not the culprit.)

The failing set is not one task — enumerate ALL Interactive tasks (fleet-wide) so the
remediation covers the whole outage, not just the watchdog:

```powershell
Get-ScheduledTask | Where-Object { $_.Principal.LogonType -eq 'Interactive' } |
  ForEach-Object { $i=Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath -EA SilentlyContinue
    [pscustomobject]@{ Task=$_.TaskName; Result=('0x{0:X}' -f $i.LastTaskResult) } } |
  Where-Object { $_.Result -eq '0x800710E0' } | Format-Table -AutoSize
```

## Layer 2 — was the cross-provider response PRODUCTIVE? (not just "did it launch")

Launching a resume is cheap. The durable `fak session recover` run witness carries each
row's provider, category, action, identity provenance, argv, launch time, baseline/post
cursors, `advanced`, and evidence source. The run witness is written before the first
window opens and updated atomically after each launch and observation. Claude requires a real non-error assistant transcript record newer
than launch; Codex requires a newer exact thread/turn cursor. A visible wrapper, idle
shell, guard receipt, or process is liveness context only and can never set `advanced`.

Reconcile the recent launch cohort natively before interpreting the longer-horizon
watchdog ledger:

```powershell
fak session journal-audit --since 24h --json
```

Require `verdict: green`. For RED, repair every `missing_transcript` or
`present_no_post_launch_progress` row and any named unreadable authority; an empty result
produced by a failed identity/provider read is never healthy.

The automatic Claude watchdog's longer-horizon status ledger remains useful for drain
capacity. Quantify today's automatic work:

```powershell
$st = Join-Path $reg 'resume_watchdog_status.jsonl'; $today = Get-Content $st | Where-Object { $_ -match (Get-Date -Format 'yyyy-MM-dd') }
$depths = foreach ($l in $today) { try { $o=$l|ConvertFrom-Json; if ($o.phase -eq 'status'){ [int]$o.auto_resume_depth } } catch {} }
$prog = @($today | Where-Object { $_ -match '"phase":"progress"' })
"peak backlog depth = {0}   drained to = {1}" -f ($depths|Measure-Object -Maximum).Maximum, ($depths|Measure-Object -Minimum).Minimum
"progress-witness rows = {0}   distinct sessions revived-with-witnessed-progress = {1}" -f $prog.Count, (@($prog|% { ($_|ConvertFrom-Json).session })|Sort-Object -Unique).Count
```

A healthy post-crash run shows the peak depth **draining toward single digits** and a
large distinct-session witness count. A backlog that grows monotonically across ticks
(`--monotonic-ticks`) is a resume storm, not recovery — also RED.

Growth is not the only backlog RED. A backlog that merely **stays deep past the throttle
reset** is the other one — `BOTTLENECK-MAP-2026-07-09.md` §7's "if `auto_resume` is still
>= 20 with 0 throttled accounts, the cap is the real limiter." Nobody eyeballs that at
every reset any more: `--status` folds it as a standing gate (#3582) and pages when depth
holds above `--backlog-threshold` (default 20) for `--backlog-ticks` (default 3)
consecutive ticks **and** the roster reports zero throttled seats. Deep-*and*-throttled is
the transient account pressure §4 already expects to clear itself, so it deliberately does
NOT page; it is the backlog that **outlives** the throttle that proves recovery capacity —
not account pressure — is what binds. The gate fails closed on an unreadable roster,
because "0 throttled" is also exactly what a roster you failed to read looks like.

The page is deduped by signature in `$reg\_paged.json` — one occurrence-counted record per
gate, refreshed rather than re-notified, so a gate that stays tripped never spams one page
per tick. Read it to answer "has this been paging, and for how long?":

```powershell
Get-Content (Join-Path $reg '_paged.json') | ConvertFrom-Json | % { $_.PSObject.Properties } |
  % { "{0}  x{1}  depth={2} > {3}  first={4}  last={5}" -f $_.Name, $_.Value.count, $_.Value.last_depth, $_.Value.threshold, $_.Value.first_seen, $_.Value.last_seen }
```

A record whose `count` keeps climbing while `last_depth` never falls is §7's signal that
**recovery capacity**, not account throttling, needs raising — the evidence an operator
(or the MaxPerTick auto-scaler) acts on.

## Layer 2b — triage each `launched_unproven` straggler

Re-run the same authority instead of hand-matching a Claude file or guessing a Codex
thread:

```powershell
fak session recover --json
```

Use the row's `reason`, `baseline_*`, `post_*`, `advanced`, and `progress_evidence`.
`probe` is deliberately retained but never launched; `identity_blocked` requires the
named login/exact-identity repair; `live` is left alone; a substantive row with
`launched_unproven` has not recovered yet even if its wrapper is still visible.

## Layer 3 (n³) — who watches this audit?

The n² check is worthless if it only runs when a human remembers. Verify the meta-layer
is itself standing:

- Is `resume-watchdog-audit` wired to a **/loop cadence** or a scheduled task, so it fires
  without a human? (This repo's `run-it-all-night` / `super-loop` are candidate hosts.)
- Does a RED verdict **escalate** — a `notifications.log` toast, a Slack beat
  (`FleetSlackStatus`), or an issue — rather than just printing and exiting?
- If this audit is the ONLY thing that would notice a dead watchdog, then a dead
  auditor is a silent double-fault. Note it explicitly in the verdict.

---

## Verdict rubric

| Verdict | Condition |
|---|---|
| **GREEN** | newest ledger write < 15 min ago **and** no `launched_unproven` past threshold **and** backlog draining **and** `FleetResumeWatchdog` LastResult `0x0`. |
| **AMBER** | one straggler unproven, or a single missed tick (< ~25 min silent), backlog flat. |
| **RED** | no ledger write > 15 min (**stall**) **or** any task LastResult non-zero (e.g. `0x800710E0`) **or** ≥1 `launched_unproven` with a dead pid past threshold **or** monotonic backlog growth **or** an open `resume_backlog_persists_after_reset` page in `_paged.json` (backlog outlived the throttle reset — recovery capacity is the limiter, #3582). |

State the verdict as: layer (n²/n³), the artifact that decided it (mtime / exit code /
transcript), and the one action. Do not soften a stall into "probably fine".

## Remediation — OPERATOR-GATED (has side effects; confirm first)

The previews above are read-only. Live recovery spawns real Claude/Codex processes
and consumes account quota, so **surface the preview and get explicit go-ahead; never run live
as part of the audit**:

1. **Fix the stall's ROOT CAUSE first** — if the audit shows down tasks with
   `LogonType=Interactive / 0x800710E0`, migrate their principal to **S4U**. This
   **requires an ELEVATED shell** (setting an S4U principal is "Access is denied"
   non-elevated). The helper does the whole fleet idempotently, dry-run by default:
   ```powershell
   # from an Administrator PowerShell:
   .\tools\migrate_fleet_tasks_to_s4u.ps1                    # dry-run: review the plan
   .\tools\migrate_fleet_tasks_to_s4u.ps1 -Apply -VerifyRun  # migrate all failing + force-run each -> expect 0x0
   ```
   Do **not** re-register to `powershell.exe`-direct as a "fix" — that changes the
   launcher, not the LogonType, and leaves the task Interactive and still failing.
2. **Resolve identity blockers first.** Complete the named Claude login; repair a Codex
   exact-UUID join rather than selecting a nearby thread. Re-run the preview until the row
   is substantive/actionable.
3. **Drain visibly:** `fak session recover --live --all --json`. Inspect the durable
   witness and require provider advancement, not merely a window or pid.
4. **Close the reinstall regression** — most `register_*.ps1` installers still create
   Interactive (schtasks default), so a reinstall reintroduces the bug. `register_resume_watchdog.ps1`
   and `register_issue_dispatch.ps1` are already S4U; migrate the rest.
5. Re-run `fak session recover --json` (no actionable crashed row remains), then the
   tower audit (`tools/watchdog_watchdog_audit.ps1`) and expect GREEN.

## Cross-provider host/login crash recovery

For a host or login crash, use `fak session recover` as the authoritative cohort-selection surface. It joins Claude transcripts, Codex `state_5.sqlite`, guard rows, the Fleet host-resurrection cohort, and OpenCode registry evidence. Keep the default dry run until every substantive row has an exact durable identity.

A visible shell, exit code 0, thread timestamp growth, or `task_complete` is not recovery proof. Require a post-launch assistant message or successful tool call/output tied to the original identity. Inspect the transcript for terminal provider errors.

Launch live work only with the explicit live switch so each substantive row has a visible wrapper window. Exclude semantic probes, close failed/probe wrappers after evidence is captured, and preserve active substantive sessions plus the operator terminal.
