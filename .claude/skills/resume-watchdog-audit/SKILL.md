---
name: resume-watchdog-audit
description: The watchdog-watchdog — one read-only pass that proves the resume watchdog (the n¹ layer that revives dead autonomous Claude sessions) is itself ALIVE, TICKING, and revising PRODUCTIVE instances, versus silently STALLED. Distrusts self-report: reads the live Fleet registry ledgers (%LOCALAPPDATA%\Fleet\registry) and the scheduled-task exit codes, never the watchdog's own "I'm fine". Verifies the n² layer (is the watchdog ticking, is its backlog draining, are resumes witnessed as real transcript turns, or are they launched_unproven) and the n³ layer (is THIS audit itself scheduled and does it escalate when the watchdog is down — who watches the watchman's watchman). Catches the exact failure this repo hit on 2026-07-09: after a 10:46 boot the watchdog drained a 65-deep backlog and witnessed progress on 122 sessions, then stalled at 11:42 because its `conhost.exe --headless` scheduled-task launch shim started returning 0x800710E0 ("operator or administrator refused the request"), leaving 2 stragglers resumed-but-dead-and-unproven. Read-only by default; the live drain is operator-gated. Use after a crash/reboot, when sessions look stuck, or on a /loop cadence as the standing meta-watchdog.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, PowerShell, Grep, Glob, Write
argument-hint: "[--fix] (no args = read-only audit; --fix offers the operator-gated live drain)"
---

# /resume-watchdog-audit — is the watchdog itself alive, or silently stalled?

> **The recursion.** The resume watchdog (`fleet_resume_watchdog.ps1 -Live`, task
> `FleetResumeWatchdog`) is the **n¹** layer: it revives dead autonomous Claude
> sessions across accounts. But a watchdog that dies takes the whole fleet down
> **silently** — every dead session just stays dead and nothing complains. This skill
> is the **n²** layer (watch the watchdog) and the **n³** layer (watch the watcher of
> the watchdog). It answers from artifacts — ledger mtimes, transcript turns, and
> Task Scheduler exit codes — **never** from the watchdog's own status text.

The one rule: **silence is the failure mode.** A stalled watchdog and a healthy-with-
nothing-to-do watchdog look identical from the outside. This pass distinguishes them.

---

## Layer 0 — locate the LIVE registry (do not trust `tools/_registry`)

`resolveSweepRegDir("")` resolves in order: `$FLEET_REG_DIR` → `$FLEET_STATE_DIR\registry`
→ `%LOCALAPPDATA%\Fleet\registry` (if it exists) → `%TEMP%\Fleet\registry` → repo
`tools/_registry` (fallback only). **The repo's `tools/_registry` is usually a stale
copy** — its `sessions.json` may be fresh (the dispatcher writes it) while its
`resume_ledger.jsonl` is days old. Always audit the resolved live dir.

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

## Layer 1b — WHY it stalled: the scheduled-task exit codes

If it stalled, the cause is almost always the launcher, not the logic. Read the task
family's last results:

```powershell
foreach ($t in 'FleetResumeWatchdog','FleetOwnerSeatResume','FleetSupervisorWatchdog','FleetProcResourceGuard','FleetStrandedRecovery','FleetIssueDispatch') {
  $i = Get-ScheduledTaskInfo -TaskName $t -ErrorAction SilentlyContinue
  if ($i) { [pscustomobject]@{Task=$t; LastRun=$i.LastRunTime; Result=('0x{0:X}' -f $i.LastTaskResult); Next=$i.NextRunTime} } } | Format-Table -AutoSize
```

Decode a non-zero result with `[System.ComponentModel.Win32Exception]0x<low16>`.
Known fault seen in this repo: **`0x800710E0` = "The operator or administrator has
refused the request"** on every task whose action is `conhost.exe --headless
powershell …`, while sibling tasks launched *without* the conhost shim return `0x0`.
That split **is** the diagnosis: the `conhost --headless` launch shim is being refused
(post-boot Windows/MDM policy — suspect `ExploitGuard MDM policy Refresh` /
`SafeguardsReconciliation`). Confirm harmlessly — a working shim propagates the child
exit code, a broken one returns empty:

```powershell
$null = & conhost.exe --headless powershell.exe -NoProfile -Command "exit 7" 2>&1
"conhost --headless exit = $LASTEXITCODE  (7 = ok; empty/other = shim refused)"
```

## Layer 2 — was the response PRODUCTIVE? (not just "did it launch")

Launching a resume is cheap; **witnessing a real transcript turn after it** is the proof
the session came back to life. The status ledger records both. Quantify today's work:

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

## Layer 2b — triage each `launched_unproven` straggler

For every session `--status` flags unproven, decide *stuck vs. just-not-witnessed-yet*
by going to ground truth — the transcript and the process table:

```powershell
$sid = '<uuid-from-status>'
$t = Get-ChildItem "$env:USERPROFILE\.claude*\projects\*\$sid.jsonl" -EA SilentlyContinue | Sort LastWriteTime -Desc | Select -First 1
"transcript {0}  ({1:n0} min idle)  last={2}" -f $t.FullName, ((Get-Date)-$t.LastWriteTime).TotalMinutes, (Get-Content $t.FullName -Tail 1).Substring(0,60)
# a `last-prompt` tail + a dead resume pid + idle > unproven-minutes = the watchdog stalled before it could re-revive this one
```

If the tail is the injected re-entry prompt (`type:last-prompt`), the resume pid is gone,
and it's been idle past the threshold, the straggler needs another tick the stalled
watchdog can't give it — see remediation.

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
| **RED** | no ledger write > 15 min (**stall**) **or** any task LastResult non-zero (e.g. `0x800710E0`) **or** ≥1 `launched_unproven` with a dead pid past threshold **or** monotonic backlog growth. |

State the verdict as: layer (n²/n³), the artifact that decided it (mtime / exit code /
transcript), and the one action. Do not soften a stall into "probably fine".

## Remediation — OPERATOR-GATED (has side effects; confirm first)

The audit above is read-only. These act — they spawn real `claude --resume` processes
and consume account quota, so **surface them and get explicit go-ahead; never run them
as part of the audit**:

1. **Drain the stragglers now** (one live tick, honors the per-tick cap + source governor):
   `pwsh -NoProfile -File tools/fleet_resume_watchdog.ps1 -Live`
2. **Fix the stall's root cause** — if `0x800710E0` on the conhost shim: re-register the
   task to launch `powershell.exe` (or `fak` / `pwsh`) **directly** instead of via
   `conhost.exe --headless`, then confirm the next scheduled run returns `0x0` and the
   ledger mtime advances.
3. Re-run this audit; expect GREEN (ledger fresh, backlog drained, no unproven).
