# FakStallscanWatch — retained pre-crash evidence

The shared Windows workstation runs `FakStallscanWatch` at logon. It invokes the
installed `fak` binary headlessly and appends one host/process pressure frame per
minute to `%LOCALAPPDATA%\Fleet\stallscan.jsonl`:

```powershell
conhost.exe --headless "$HOME\bin\fak.exe" stallscan --watch `
  --interval 60s --log "$env:LOCALAPPDATA\Fleet\stallscan.jsonl" `
  --max-bytes 16777216
```

The 16 MiB bound is enforced by atomic newest-record retention. Every current
record includes boot time, committed bytes/limit, available bytes, process and
thread census, system handles, top holders, and the explainable stall verdict.
After an unexpected reboot, compare the newest record whose `sample.boot_time`
precedes the new boot with Windows Event 6008/Kernel-Power 41.

Install or repair the task from an ordinary PowerShell session:

```powershell
$exe = "$HOME\bin\fak.exe"
$log = "$env:LOCALAPPDATA\Fleet\stallscan.jsonl"
$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\conhost.exe" `
  -Argument ('--headless "{0}" stallscan --watch --interval 60s --log "{1}" --max-bytes 16777216' -f $exe,$log)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName FakStallscanWatch -Action $action `
  -Trigger $trigger -Settings $settings -Force
Start-ScheduledTask FakStallscanWatch
```

Verify with `Get-ScheduledTask FakStallscanWatch` and confirm that the JSONL
mtime advances and its newest `sample.boot_time`, `commit_bytes`,
`commit_limit`, and `available_bytes` fields are populated.
