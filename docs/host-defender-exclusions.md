---
title: "Windows Defender exclusion baseline for fleet hosts"
description: "The per-spawn Defender scan tax on an agent fleet host, the narrow exclusion set that removes it, the exact elevated Add-MpPreference paste to apply it, how to verify, and the honest security trade-off."
---

# Windows Defender exclusion baseline for fleet hosts

A fleet host running a city of agents pays a synchronous Windows Defender
real-time scan on **every process spawn and every temp-file write**. An agent
tool call is a process spawn, so the scan latency serializes straight into the
agent's inner loop. This is the one-paste baseline that removes that tax on a
Windows fleet host, plus the honest note on what it costs in coverage.

Apply it from an **elevated** shell (it needs administrator rights, and so does
reading it back). It survives reboot.

## The measured tax (why bother)

On the primary agent host (32 cores) the process churn was measured at **≥108
process births/min** — plausibly 150–300/min once you count the sub-20-second
tool-call lifetimes the coarse sampler misses (that wider range is the issue's
estimate, not a direct count). Defender's engine, `MsMpEng`, sat **sustained at
~21% of one core** just tracking that churn. That ~21%/core is the load-bearing
number here; everything else in this doc is the fix for it.

Two things make this easy to miss until someone measures:

- On a many-core box, one saturated core is only `100/ncores`% of total CPU
  (~3% of a 32-core host), so no top-line CPU gauge ever trips.
- Exclusions are **not auditable from a non-elevated session** —
  `Get-MpPreference` returns *"N/A: Must be an administrator"* — so nobody knows
  a host is unexcluded until they open an elevated shell and look.

## The exclusion baseline

Two axes: the **paths** that carry the constant compile / git / scratch churn,
and the **processes** an agent tool call spawns over and over.

| Exclusion | What it is | Why |
| --- | --- | --- |
| `C:\work\fak` | the repo tree | constant compile + git churn on every ship |
| `C:\Program Files\Git` | the Git install | `git.exe` / `bash.exe` spawn on every tool call |
| `%LOCALAPPDATA%\Temp\claude` | the agent scratch dir | per-tool-call temp writes, each otherwise scanned |
| process `claude.exe` | the agent harness | spawned per session/subagent |
| process `fak.exe` | the kernel / gateway | spawned per guarded turn |
| process `bash.exe` | the shell behind tool calls | spawned per Bash/shell tool call |
| process `python.exe` | the tool + MCP-server runtime | spawned per Python tool / MCP server |

The path list matches this checkout's root (`C:\work\fak`); adjust it to your
clone root if you keep the tree elsewhere.

## Apply it (elevated)

From an **administrator** PowerShell:

```powershell
Add-MpPreference -ExclusionPath 'C:\work\fak','C:\Program Files\Git',"$env:LOCALAPPDATA\Temp\claude"
Add-MpPreference -ExclusionProcess 'claude.exe','fak.exe','bash.exe','python.exe'
```

`Add-MpPreference` **appends** — it does not replace an existing list — so
re-running it is safe and additive.

From a non-elevated shell you can raise the prompt with:

```powershell
Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-NoExit','-Command',
  "Add-MpPreference -ExclusionPath 'C:\work\fak','C:\Program Files\Git',`"$env:LOCALAPPDATA\Temp\claude`";
   Add-MpPreference -ExclusionProcess 'claude.exe','fak.exe','bash.exe','python.exe';
   Get-MpPreference | Select-Object ExclusionPath,ExclusionProcess"
```

### The scope decision: exclude `Temp\claude`, NOT all of `%TEMP%`

Exclude the **narrow** `%LOCALAPPDATA%\Temp\claude` scratch subtree, not the
temp root. Droppers and staged payloads land in the temp **root**; keeping the
root scanned is the whole point of leaving Defender on. The agent's scratch
writes are the hot path, so a scratch-dir-scoped exclusion buys back nearly all
the throughput without opening the door a malware dropper walks through.

### Optional: the Go build cache (inferred, not from the measurement)

If a host also does heavy Go builds, the `%LOCALAPPDATA%\go-build` cache (tens of
GB, rewritten constantly) is a second hot path worth excluding. This is an
**inferred** extension for build-heavy hosts, not part of the ~21%/core spawn
measurement above; see the broader
[desktop-slowness runbook](notes/DESKTOP-SLOWNESS-MAINTENANCE-2026-06-28.md) for
the build-tree exclusion set and the reboot-cadence maintenance it pairs with.

```powershell
# build-heavy hosts only (optional):
Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\go-build"
Add-MpPreference -ExclusionProcess 'go.exe','gopls.exe'
```

## Verify (elevated)

Reading the exclusions back **also requires an elevated shell** — from a normal
session `Get-MpPreference` errors with *"Must be an administrator"*, which is why
an unexcluded host stays silently unexcluded until someone checks:

```powershell
Get-MpPreference | Select-Object ExclusionPath,ExclusionProcess
```

A host is at baseline when the four paths and four processes above appear in the
output. That is the whole acceptance test: a new host reaches baseline from a
single elevated paste, and this command proves it.

## The honest trade-off

Every exclusion is a **security-surface-for-throughput trade**, and this doc will
not pretend otherwise:

- Excluding `C:\work\fak` means Defender no longer real-time-scans files written
  under the repo tree. On a dev/build box that is a standard, accepted trade; on
  a host that also handles untrusted input it is a real reduction in coverage.
- The `Temp\claude` scope is deliberately narrow precisely because the temp
  **root** is where droppers land — do not widen it to all of `%TEMP%`.
- Excluding a **process** (e.g. `python.exe`) is broader than excluding a path:
  it exempts that image wherever it runs, not just under the repo. Keep the
  process list short and specific to the images the agent loop actually spawns.

Reverse any of it with the matching `Remove-MpPreference` (elevated):

```powershell
Remove-MpPreference -ExclusionPath 'C:\work\fak','C:\Program Files\Git',"$env:LOCALAPPDATA\Temp\claude"
Remove-MpPreference -ExclusionProcess 'claude.exe','fak.exe','bash.exe','python.exe'
```

## See also

- [Runaway-process guard & leak audit](perf-runaway-guard.md) — the other
  standing host-cost backstop (a process whose thread/handle/CPU level runs away).
- [Desktop-slowness maintenance runbook](notes/DESKTOP-SLOWNESS-MAINTENANCE-2026-06-28.md)
  — the broader "why the fleet workstation feels slow" audit and its reboot-cadence
  and build-tree-exclusion maintenance.
