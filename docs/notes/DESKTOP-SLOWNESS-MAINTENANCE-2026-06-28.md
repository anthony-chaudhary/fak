# Desktop slowness: maintenance runbook (the fleet workstation)

Date: 2026-06-28. Scope: why the Windows fleet workstation (16-core/32-thread x86-64,
256 GB RAM, 3.7 TB disk) "feels slow", and the recurring maintenance that actually
fixes it. Audited live at ~40 h uptime under a ~24-session agent fleet.

## TL;DR verdict

It is NOT resource exhaustion. Measured at audit time: CPU sustained ~32%, RAM 25%
used (191 GB free), PhysicalDisk queue length 0 / %disk-time ~1%, CPU clock pinned at
4300/4300 MHz (no thermal throttle), memory pages/sec ~9 (no thrash), no pending
reboot. The drag is two long-uptime accumulations, both of which are cleared by a
**reboot** and are NOT killable as ordinary processes:

1. **WindowsTerminal single-process chokepoint.** One WindowsTerminal process renders
   every fleet pane (measured: ~24 bash + ~18 pwsh login shells + claude tabs, with 51
   conhost children). At 40 h it held a measured 2.2 GB working set / 14,719 handles /
   ~1.9 CPU-hours. This is the interactive lag felt while typing, scrolling, switching
   panes.
2. **TermService (Remote Desktop) handle+memory leak.** The svchost hosting TermService
   leaks over uptime on RDP-accessed hosts. Measured growth within a single ~30 min
   session: 2,525 MB / 20,712 handles -> 2,963 MB / 23,536 handles. It only resets on
   reboot; restarting the service drops the live RDP session, so do not do it remotely.

## Recurring maintenance actions

### 1. Reboot on a cadence (highest impact)
A reboot clears both leaks above and resets process sprawl in one move. Trigger it when
TermService handles climb past ~30k or its working set past ~4 GB, or roughly weekly
under heavy fleet use. It kills all live agent sessions, so it is an operator decision
scheduled for a quiet window -- it is not something an agent should do unattended.

### 2. Windows Defender exclusions for the build tree (preventive, survives reboot)
Real-time scanning taxes the constant compile/git churn and the ~18 GB
`%LOCALAPPDATA%\go-build` cache. Note: Defender was NOT observed spiking during the
audit (MsMpEng was idle in the sample), so this is preventive -- it helps most during
build bursts, not idle typing. Run elevated (UAC prompt):

    Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-NoExit','-Command',
      'Add-MpPreference -ExclusionPath "C:\work","C:\Users\USER\AppData\Local\go-build","C:\Users\USER\go";
       Add-MpPreference -ExclusionProcess "go.exe","gopls.exe","git.exe","python.exe","node.exe","fak.exe","claude.exe","compile.exe","link.exe";
       (Get-MpPreference).ExclusionPath'

Trade-off: excluding `C:\work` lowers AV coverage on that tree (standard for dev boxes).
Reverse with `Remove-MpPreference -ExclusionPath ...` / `-ExclusionProcess ...`.
Viewing exclusions also requires an elevated shell (`Get-MpPreference` returns
"Must be an administrator" otherwise).

## What NOT to do (verified failure modes)

- **Do not bulk-kill "idle" python/pwsh by a CPU/age filter.** A crude
  `CPU<3 & age>0.5h` filter flagged ~36 processes as "leaked"; a process-TREE check
  (parent liveness via `Win32_Process.ParentProcessId`) showed they are nearly all LIVE
  workers: ~13 python under `dos-mcp` parents (the DOS MCP server's own workers), ~6
  pwsh login shells under `WindowsTerminal`, plus opencode/node children. Killing them
  takes out live MCP tooling and sessions. Always classify by parent liveness first.
- **Do not `go clean -cache`.** The ~18 GB go-build cache is shared across all fleet
  sessions; clearing it forces every session's next build to recompile from scratch.
- **The one genuine orphan is not worth chasing.** A single 1 MB python husk with a dead
  parent (PID was 51300 this session) refused `Stop-Process -Force` AND `taskkill /F`
  ("Access is denied" -- needs elevation). It costs 1 MB / 0 CPU and clears on reboot.

## Re-running the audit

Read-only PowerShell snippets (run via `powershell.exe -NoProfile -File <script>` to
avoid Git-Bash mangling `$_`/`{}`; keep scripts ASCII so the cp1252 provenance hook does
not choke):

- Baseline: `Get-CimInstance Win32_OperatingSystem` (mem), `Win32_Processor` (clock /
  throttle: compare CurrentClockSpeed vs MaxClockSpeed), `Win32_LogicalDisk` (space).
- Live pressure: `Get-Counter` on `\PhysicalDisk(_Total)\Current Disk Queue Length`,
  `\Processor(_Total)\% Processor Time`, `\Memory\Pages/sec`.
- The two leaks: `Get-Process WindowsTerminal` and the TermService svchost
  (`Get-CimInstance Win32_Service -Filter "Name='TermService'"` -> its ProcessId ->
  `Get-Process -Id <pid>`), reading `WorkingSet64` and `HandleCount`.
- Orphan classification: enumerate `Win32_Process`, map each `ParentProcessId` to a live
  PID, treat only dead/reused-parent processes as orphans.
