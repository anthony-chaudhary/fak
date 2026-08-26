# Windows hostdiag census operations

`FakHostdiagCensus` records a privacy-safe census of running `fak.exe` processes every five minutes so a later Windows resource event can be tied to a process identity that existed before the event. It observes only; it never terminates or restarts a process, and growth is not itself proof of a leak.

## Contract

The tracked task definition is [`tools/scheduled-tasks/FakHostdiagCensus.xml`](../../tools/scheduled-tasks/FakHostdiagCensus.xml).

- `S4U` runs without an interactive desktop or stored password.
- `PT5M` is the measured conservative cadence.
- `IgnoreNew` prevents overlapping census instances.
- Each invocation has a two-minute execution limit and catches up once after downtime.
- The ledger is `%LOCALAPPDATA%\fak-hostdiag\hostdiag.jsonl`, bounded to 16 MiB while preserving complete JSONL rows.
- Census rows contain process identity, executable hash, bounded command class, session pseudonym, private/working bytes, and thread count. They do not contain raw argv, environment, output, or secrets.
- A failed invocation is visible as the Scheduled Task result; there is no retry loop and no process-control side effect.

## Install and verify

Registration changes machine scheduler state and therefore requires an elevated PowerShell session. Export an existing task before replacing it:

```powershell
$taskName = 'FakHostdiagCensus'
$xmlPath = 'C:\work\fak\tools\scheduled-tasks\FakHostdiagCensus.xml'
$rollback = Join-Path $env:LOCALAPPDATA 'Fleet\FakHostdiagCensus.pre-install.xml'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
  Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath $rollback
}
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$xml = (Get-Content -LiteralPath $xmlPath -Raw).Replace('%FLEET_TASK_USER_SID%', $sid)
Register-ScheduledTask -TaskName $taskName -Xml $xml -Force | Out-Null
Enable-ScheduledTask -TaskName $taskName | Out-Null
Start-ScheduledTask -TaskName $taskName
```

Read the installed contract back instead of treating registration success as deployment proof:

```powershell
$t = Get-ScheduledTask -TaskName 'FakHostdiagCensus'
$i = Get-ScheduledTaskInfo -TaskName 'FakHostdiagCensus'
[pscustomobject]@{
  state       = $t.State
  logon_type  = $t.Principal.LogonType
  action      = $t.Actions.Execute + ' ' + $t.Actions.Arguments
  interval    = $t.Triggers[0].Repetition.Interval
  concurrency = $t.Settings.MultipleInstances
  last_result = ('0x{0:X8}' -f [uint32]$i.LastTaskResult)
  next_run    = $i.NextRunTime
}
```

Acceptance requires `S4U`, `PT5M`, `IgnoreNew`, the expected bounded action, two successful scheduled runs, advancing complete JSONL rows, and no raw argv/secret fields. A `LastTaskResult` of zero alone is not proof that the ledger advanced.

## Roll back or remove

Disable first so no new invocation starts, then wait for any running invocation to finish:

```powershell
Disable-ScheduledTask -TaskName 'FakHostdiagCensus'
while ((Get-ScheduledTask -TaskName 'FakHostdiagCensus').State -eq 'Running') { Start-Sleep -Seconds 1 }
Unregister-ScheduledTask -TaskName 'FakHostdiagCensus' -Confirm:$false
```

If a pre-install export exists, restore it with `Register-ScheduledTask -TaskName FakHostdiagCensus -Xml (Get-Content -LiteralPath $rollback -Raw) -Force`. Removing the task does not remove the evidence ledger. Delete `%LOCALAPPDATA%\fak-hostdiag` separately only when its retained diagnostic evidence is no longer needed.
