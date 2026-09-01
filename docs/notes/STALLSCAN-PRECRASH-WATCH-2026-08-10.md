# FakStallscanWatch — retained pre-crash evidence

The shared Windows workstation runs `FakStallscanWatch` at logon. It invokes the
installed `fak` binary headlessly and appends one host/process pressure frame per
minute to `%LOCALAPPDATA%\Fleet\stallscan.jsonl`:

```powershell
conhost.exe --headless "$HOME\bin\fak.exe" stallscan --watch `
  --interval 60s --log "$env:LOCALAPPDATA\Fleet\stallscan.jsonl" `
  --max-bytes 16777216
```

## What the watcher records

The 16 MiB bound is enforced by atomic newest-record retention. Every current
record includes boot time, committed bytes/limit, available bytes, process and
thread census, system handles, top holders, and the explainable stall verdict.
After an unexpected reboot, compare the newest record whose `sample.boot_time`
precedes the new boot with Windows Event 6008/Kernel-Power 41.

## Install or repair the scheduled task

Install or repair the task from an ordinary PowerShell session. The five-minute
trigger is a self-heal for a watcher process that exits after logon; `IgnoreNew`
means it cannot create a second watcher while the first is healthy:

```powershell
$exe = "$HOME\bin\fak.exe"
$log = "$env:LOCALAPPDATA\Fleet\stallscan.jsonl"
$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\conhost.exe" `
  -Argument ('--headless "{0}" stallscan --watch --interval 60s --log "{1}" --max-bytes 16777216' -f $exe,$log)
$logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$selfHeal = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) `
  -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName FakStallscanWatch -Action $action `
  -Trigger @($logon,$selfHeal) -Settings $settings -Force
Start-ScheduledTask FakStallscanWatch
```

## Verify process state and effect

Verify both process state and effect; a task result alone is ambiguous because a
healthy running instance can refuse a periodic duplicate:

```powershell
Get-ScheduledTask FakStallscanWatch
Get-ScheduledTaskInfo FakStallscanWatch
Get-Item "$env:LOCALAPPDATA\Fleet\stallscan.jsonl" |
  Select-Object LastWriteTime,Length
Get-CimInstance Win32_Process |
  Where-Object CommandLine -match 'stallscan --watch'
```

The task must be `Running`, exactly one watcher command line must exist, and the
JSONL mtime must advance by the configured interval. See
[`INCIDENT-WINDOWS-UNEXPECTED-RESTART-2026-08-10.md`](INCIDENT-WINDOWS-UNEXPECTED-RESTART-2026-08-10.md)
for the incident that exposed the logon-only durability gap and the post-reboot
classification procedure.
