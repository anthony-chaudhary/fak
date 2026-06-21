<#
run_overnight_soak.ps1 -- unattended hourly FAK soak runner.

Default run:
  .\tools\run_overnight_soak.ps1

Smoke check:
  .\tools\run_overnight_soak.ps1 -Smoke

Outputs:
  tools\_registry\soak\<stamp>\

Each tick writes raw logs, JSON artifacts, endpoint state, cross-node comparison,
summary.json, and SUMMARY.md. Laptop nodes are folded from existing
fak/experiments/fleet-nodes/* artifacts and any new bundles copied in while the
soak is running.
#>
[CmdletBinding()]
param(
  [string]$FleetDir = 'C:\work\fleet',
  [int]$Hours = 8,
  [int]$Samples = 0,
  [int]$IntervalMinutes = 60,
  [string]$RunRoot = '',
  [switch]$Smoke,
  [switch]$SkipBoundaryGates,
  [int]$Workers = 0
)

$ErrorActionPreference = 'Stop'

function New-Dir([string]$Path) {
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
  }
}

function ConvertTo-PlainText($Value) {
  if ($null -eq $Value) { return '' }
  if ($Value -is [System.Array]) { return ($Value -join "`n") }
  return [string]$Value
}

function Invoke-LoggedProcess {
  param(
    [string]$Name,
    [string]$FilePath,
    [string[]]$ArgumentList,
    [string]$WorkingDirectory,
    [string]$LogPath,
    [int]$TimeoutSec = 0,
    [hashtable]$Env = @{}
  )

  function Quote-ProcessArg([string]$Arg) {
    if ($null -eq $Arg) { return '""' }
    if ($Arg -notmatch '[\s"]') { return $Arg }
    return '"' + ($Arg -replace '\\(?=")', '\\' -replace '"', '\"') + '"'
  }

  $started = Get-Date
  $stdout = ''
  $stderr = ''
  $exitCode = 1
  $timedOut = $false
  $errorText = $null

  try {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.Arguments = ($ArgumentList | ForEach-Object { Quote-ProcessArg ([string]$_) }) -join ' '
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($k in $Env.Keys) { $psi.EnvironmentVariables[$k] = [string]$Env[$k] }

    $p = [System.Diagnostics.Process]::Start($psi)
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    if ($TimeoutSec -gt 0) {
      if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        $timedOut = $true
        try { $p.Kill($true) } catch { try { $p.Kill() } catch {} }
        $p.WaitForExit()
      }
    } else {
      $p.WaitForExit()
    }
    $outTask.Wait(5000) | Out-Null
    $errTask.Wait(5000) | Out-Null
    $stdout = ConvertTo-PlainText $outTask.Result
    $stderr = ConvertTo-PlainText $errTask.Result
    $exitCode = if ($timedOut) { 124 } else { $p.ExitCode }
  } catch {
    $errorText = $_.Exception.Message
    $exitCode = 1
  }

  $ended = Get-Date
  $durationSec = [Math]::Round(($ended - $started).TotalSeconds, 3)
  $cmd = "$FilePath " + ($ArgumentList -join ' ')
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# $Name")
  $lines.Add("started_utc: $($started.ToUniversalTime().ToString('o'))")
  $lines.Add("ended_utc: $($ended.ToUniversalTime().ToString('o'))")
  $lines.Add("duration_sec: $durationSec")
  $lines.Add("exit_code: $exitCode")
  if ($timedOut) { $lines.Add("timed_out: true") }
  $lines.Add("cwd: $WorkingDirectory")
  $lines.Add("command: $cmd")
  if ($errorText) {
    $lines.Add("")
    $lines.Add("## error")
    $lines.Add($errorText)
  }
  $lines.Add("")
  $lines.Add("## stdout")
  $lines.Add($stdout)
  $lines.Add("")
  $lines.Add("## stderr")
  $lines.Add($stderr)
  $lines | Set-Content -Path $LogPath -Encoding UTF8

  return [pscustomobject]@{
    name = $Name
    command = $cmd
    exit_code = $exitCode
    timed_out = $timedOut
    duration_sec = $durationSec
    log = $LogPath
  }
}

function Read-JsonSafe([string]$Path) {
  if (-not (Test-Path $Path)) { return $null }
  try {
    return Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json
  } catch {
    return $null
  }
}

function Get-GitFacts([string]$Root) {
  $facts = [ordered]@{}
  try { $facts.branch = (& git -C $Root branch --show-current 2>$null | Select-Object -First 1) } catch {}
  try { $facts.rev = (& git -C $Root rev-parse --short HEAD 2>$null | Select-Object -First 1) } catch {}
  try { $facts.status = (& git -C $Root status --short --branch 2>$null) } catch {}
  return $facts
}

function Write-JsonFile([string]$Path, $Object, [int]$Depth = 20) {
  $Object | ConvertTo-Json -Depth $Depth | Set-Content -Path $Path -Encoding UTF8
}

function MetricOrNull($Object, [string]$Name) {
  if ($null -eq $Object) { return $null }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $null
}

function Build-MetricSummary([string]$TickDir) {
  $fakbench = Read-JsonSafe (Join-Path $TickDir 'fakbench-tau2-smoke.json')
  $turntax = Read-JsonSafe (Join-Path $TickDir 'turntax-airline.json')
  $agent = Read-JsonSafe (Join-Path $TickDir 'agent-offline.json')
  $modelbench = Read-JsonSafe (Join-Path $TickDir 'modelbench-q8.json')
  $batchbench = Read-JsonSafe (Join-Path $TickDir 'batchbench-q8.json')
  $nodes = Read-JsonSafe (Join-Path $TickDir 'node-compare.json')

  $speedup = $null
  if ($fakbench -and $fakbench.spawned_hook_baseline.p50_ns -and $fakbench.vdso_on.p50_ns) {
    $speedup = [Math]::Round([double]$fakbench.spawned_hook_baseline.p50_ns / [double]$fakbench.vdso_on.p50_ns, 1)
  }

  $batchPeak = $null
  if ($batchbench -and $batchbench.peak) {
    $batchPeak = [ordered]@{
      batch = $batchbench.peak.batch
      agg_tok_per_sec = $batchbench.peak.agg_tok_per_sec
      speedup_vs_naive_serial = $batchbench.peak.speedup_vs_naive_serial
    }
  }

  return [ordered]@{
    fakbench = [ordered]@{
      gate = MetricOrNull $fakbench 'gate_primary'
      p50_ns = if ($fakbench) { $fakbench.vdso_on.p50_ns } else { $null }
      spawned_p50_ns = if ($fakbench) { $fakbench.spawned_hook_baseline.p50_ns } else { $null }
      speedup_x = $speedup
      vdso_hit_rate = if ($fakbench) { $fakbench.kpis.vdso_hit_rate } else { $null }
    }
    turntax = [ordered]@{
      turns_saved = if ($turntax) { $turntax.net.turns_saved } else { $null }
      tokens_saved = if ($turntax) { $turntax.net.tokens_saved } else { $null }
      dollars_saved = if ($turntax) { $turntax.net.dollars_saved } else { $null }
      consistency_check = MetricOrNull $turntax 'consistency_check'
      injections_fak = if ($turntax) { $turntax.safety_floor.injections_admitted_fak } else { $null }
      destructive_fak = if ($turntax) { $turntax.safety_floor.destructive_executed_fak } else { $null }
    }
    agent_offline = [ordered]@{
      both_completed = MetricOrNull $agent 'both_completed'
      turns_saved = MetricOrNull $agent 'turns_saved'
      tokens_saved = MetricOrNull $agent 'tokens_saved'
      fak_injection_in_context = if ($agent) { $agent.fak.injection_in_context } else { $null }
      baseline_injection_in_context = if ($agent) { $agent.baseline.injection_in_context } else { $null }
      fak_destructive_executed = if ($agent) { $agent.fak.destructive_executed } else { $null }
      baseline_destructive_executed = if ($agent) { $agent.baseline.destructive_executed } else { $null }
    }
    modelbench_q8 = [ordered]@{
      decode_tok_per_sec = if ($modelbench) { $modelbench.decode.tok_per_sec } else { $null }
      workload_cases = if ($modelbench -and $modelbench.workload) { $modelbench.workload.cases } else { $null }
      workload_prefill_cap = if ($modelbench -and $modelbench.workload) { $modelbench.workload.prefill_cap } else { $null }
    }
    batchbench_q8 = $batchPeak
    node_compare = [ordered]@{
      nodes = if ($nodes) { @($nodes).Count } else { 0 }
      hosts = if ($nodes) { @($nodes | ForEach-Object { $_.host }) } else { @() }
    }
  }
}

function Write-TickMarkdown([string]$TickDir, $Summary) {
  $m = $Summary.metrics
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Soak tick $($Summary.tick)")
  $lines.Add("")
  $lines.Add("- started UTC: $($Summary.started_utc)")
  $lines.Add("- ended UTC: $($Summary.ended_utc)")
  $lines.Add("- duration: $($Summary.duration_sec) sec")
  $lines.Add("- boundary heavy: $($Summary.boundary_heavy)")
  $lines.Add("- failures: $($Summary.failures)")
  $lines.Add("")
  $lines.Add("## Key metrics")
  $lines.Add("")
  $lines.Add("- fakbench: gate=$($m.fakbench.gate), p50=$($m.fakbench.p50_ns) ns, spawned=$($m.fakbench.spawned_p50_ns) ns, speedup=$($m.fakbench.speedup_x)x, vdso_hit=$($m.fakbench.vdso_hit_rate)")
  $lines.Add("- turntax: turns_saved=$($m.turntax.turns_saved), tokens_saved=$($m.turntax.tokens_saved), consistency=$($m.turntax.consistency_check), fak injections=$($m.turntax.injections_fak), fak destructive=$($m.turntax.destructive_fak)")
  $lines.Add("- agent offline: both_completed=$($m.agent_offline.both_completed), turns_saved=$($m.agent_offline.turns_saved), baseline_injection=$($m.agent_offline.baseline_injection_in_context), fak_injection=$($m.agent_offline.fak_injection_in_context)")
  if ($m.modelbench_q8.decode_tok_per_sec) {
    $lines.Add("- modelbench q8: decode_tok_per_sec=$($m.modelbench_q8.decode_tok_per_sec), workload_cases=$($m.modelbench_q8.workload_cases), cap=$($m.modelbench_q8.workload_prefill_cap)")
  }
  if ($m.batchbench_q8) {
    $lines.Add("- batchbench q8: peak=$($m.batchbench_q8.agg_tok_per_sec) tok/s at B=$($m.batchbench_q8.batch), speedup_vs_naive=$($m.batchbench_q8.speedup_vs_naive_serial)x")
  }
  $lines.Add("- node compare: $($m.node_compare.nodes) node(s): $(@($m.node_compare.hosts) -join ', ')")
  $lines.Add("")
  $lines.Add("## Step status")
  foreach ($s in $Summary.steps) {
    $lines.Add("- $($s.name): exit=$($s.exit_code), duration=$($s.duration_sec)s, log=$([System.IO.Path]::GetFileName($s.log))")
  }
  $lines | Set-Content -Path (Join-Path $TickDir 'SUMMARY.md') -Encoding UTF8
}

Push-Location $FleetDir
try {
  if ($Smoke) {
    if ($Samples -le 0) { $Samples = 1 }
    $IntervalMinutes = 0
    $SkipBoundaryGates = $true
  } elseif ($Samples -le 0) {
    $Samples = $Hours + 1
  }
  if ($Samples -lt 1) { $Samples = 1 }

  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmssZ')
  if (-not $RunRoot) {
    $RunRoot = Join-Path $FleetDir "tools\_registry\soak\$stamp"
  }
  $RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
  $fakDir = Join-Path $FleetDir 'fak'
  $binDir = Join-Path $RunRoot 'bin'
  New-Dir $RunRoot
  New-Dir $binDir

  $envOverrides = @{}
  if ($Workers -gt 0) { $envOverrides['FAK_WORKERS'] = $Workers }

  $matrix = [ordered]@{
    schema = 'fak.overnight-soak.v1'
    run_root = $RunRoot
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    samples = $Samples
    interval_minutes = $IntervalMinutes
    smoke = [bool]$Smoke
    git = Get-GitFacts $FleetDir
    hourly_artifacts = @(
      'fakbench-tau2-smoke.json',
      'turntax-airline.json',
      'agent-offline.json',
      'q8kernel.txt',
      'modelbench-q8.json',
      'batchbench-q8.json',
      'fleetbench.json',
      'endpoints.json',
      'node-compare.json'
    )
    laptop_node_policy = 'Fold existing fak/experiments/fleet-nodes/* each hour; endpoint probe records READY/ONLINE/SSH state. Direct remote launch is intentionally not assumed.'
    boundary_policy = if ($SkipBoundaryGates) { 'skipped' } else { 'go vet and go test at first and final tick' }
  }
  Write-JsonFile (Join-Path $RunRoot 'RUN.json') $matrix

  @"
# Overnight FAK soak

- run root: $RunRoot
- samples: $Samples
- interval minutes: $IntervalMinutes
- smoke: $([bool]$Smoke)
- boundary gates: $($matrix.boundary_policy)

Hourly matrix:
- endpoint probe and cross-node comparison
- fak bench tau2-smoke
- fak turntax turntax-airline
- fak agent offline A/B
- q8kernel GEMV
- modelbench Q8 workload slice
- batchbench Q8 synthetic-prompt batch curve
- fleetbench read-heavy grid

Benchmark nodes:
- Private benchmark-node results are folded from fak/experiments/fleet-nodes/*.
- Live endpoint use requires serve port readiness; unreachable or disabled endpoints are recorded as skips, not failures.
"@ | Set-Content -Path (Join-Path $RunRoot 'RUN.md') -Encoding UTF8

  $buildSteps = @()
  $buildSteps += Invoke-LoggedProcess -Name 'build fak' -FilePath 'go' -ArgumentList @('build', '-o', (Join-Path $binDir 'fak.exe'), './cmd/fak') -WorkingDirectory $fakDir -LogPath (Join-Path $RunRoot 'build-fak.log') -TimeoutSec 300 -Env $envOverrides
  foreach ($cmd in @('q8kernel', 'modelbench', 'batchbench', 'fleetbench')) {
    $buildSteps += Invoke-LoggedProcess -Name "build $cmd" -FilePath 'go' -ArgumentList @('build', '-o', (Join-Path $binDir "$cmd.exe"), "./cmd/$cmd") -WorkingDirectory $fakDir -LogPath (Join-Path $RunRoot "build-$cmd.log") -TimeoutSec 300 -Env $envOverrides
  }
  Write-JsonFile (Join-Path $RunRoot 'build-summary.json') $buildSteps

  $start = Get-Date
  $allSummaries = @()
  for ($i = 0; $i -lt $Samples; $i++) {
    $tick = 'h{0:D2}' -f $i
    $tickDir = Join-Path $RunRoot $tick
    New-Dir $tickDir
    $tickStart = Get-Date
    $boundary = (-not $Smoke) -and ($i -eq 0 -or $i -eq ($Samples - 1))
    $steps = @()

    Write-Host "[$tick] starting at $($tickStart.ToString('s')) boundary=$boundary"
    Write-JsonFile (Join-Path $tickDir 'meta.json') ([ordered]@{
      tick = $tick
      index = $i
      started_utc = $tickStart.ToUniversalTime().ToString('o')
      boundary_heavy = $boundary
      git = Get-GitFacts $FleetDir
      computer = $env:COMPUTERNAME
      user = $env:USERNAME
      workers = if ($Workers -gt 0) { $Workers } else { 'default' }
    })

    $endpointJson = Join-Path $tickDir 'endpoints.json'
    $steps += Invoke-LoggedProcess -Name 'endpoint probe json' -FilePath 'powershell' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $FleetDir 'tools\fleet_endpoints.ps1'), '-FleetDir', $FleetDir, '-Probe', '-All', '-Json') -WorkingDirectory $FleetDir -LogPath (Join-Path $tickDir 'endpoints.log') -TimeoutSec 45
    $endpointText = Get-Content -Path (Join-Path $tickDir 'endpoints.log') -Raw -Encoding UTF8
    $match = [regex]::Match($endpointText, '(?s)\## stdout\s*(.*?)\s*\## stderr')
    if ($match.Success) { $match.Groups[1].Value.Trim() | Set-Content -Path $endpointJson -Encoding UTF8 }

    if ($boundary -and -not $SkipBoundaryGates) {
      $steps += Invoke-LoggedProcess -Name 'go vet boundary' -FilePath 'go' -ArgumentList @('vet', './...') -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'go-vet.log') -TimeoutSec 900 -Env $envOverrides
      $steps += Invoke-LoggedProcess -Name 'go test boundary' -FilePath 'go' -ArgumentList @('test', './...') -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'go-test.log') -TimeoutSec 1800 -Env $envOverrides
    }

    $fak = Join-Path $binDir 'fak.exe'
    $q8 = Join-Path $binDir 'q8kernel.exe'
    $modelbench = Join-Path $binDir 'modelbench.exe'
    $batchbench = Join-Path $binDir 'batchbench.exe'
    $fleetbench = Join-Path $binDir 'fleetbench.exe'

    $baselineN = if ($Smoke) { '5' } else { '30' }
    $steps += Invoke-LoggedProcess -Name 'fakbench tau2-smoke' -FilePath $fak -ArgumentList @('bench', '--suite', 'tau2-smoke', '--baseline-n', $baselineN, '--out', (Join-Path $tickDir 'fakbench-tau2-smoke.json')) -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'fakbench-tau2-smoke.log') -TimeoutSec 240 -Env $envOverrides
    $steps += Invoke-LoggedProcess -Name 'turntax airline' -FilePath $fak -ArgumentList @('turntax', '--suite', 'turntax-airline', '--out', (Join-Path $tickDir 'turntax-airline.json')) -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'turntax-airline.log') -TimeoutSec 120 -Env $envOverrides
    $steps += Invoke-LoggedProcess -Name 'agent offline ab' -FilePath $fak -ArgumentList @('agent', '--offline', '--max-turns', '12', '--out', (Join-Path $tickDir 'agent-offline.json'), '--log', (Join-Path $tickDir 'agent-offline-trace.txt')) -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'agent-offline.log') -TimeoutSec 120 -Env $envOverrides

    $q8Reps = if ($Smoke) { '3' } else { '15' }
    $steps += Invoke-LoggedProcess -Name 'q8kernel' -FilePath $q8 -ArgumentList @('-reps', $q8Reps) -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'q8kernel.txt') -TimeoutSec 180 -Env $envOverrides

    $workload = Join-Path $fakDir 'experiments\agent-live\production-workload.json'
    $prefillReps = if ($Smoke) { '1' } elseif ($boundary) { '3' } else { '2' }
    $decodeReps = if ($Smoke) { '1' } elseif ($boundary) { '3' } else { '2' }
    $decodeSteps = if ($Smoke) { '4' } elseif ($boundary) { '32' } else { '16' }
    $prefillCap = if ($Smoke) { '64' } elseif ($boundary) { '0' } else { '512' }
    $batchReps = if ($Smoke) { '1' } elseif ($boundary) { '3' } else { '2' }
    $batchSteps = if ($Smoke) { '4' } elseif ($boundary) { '32' } else { '8' }
    $batches = if ($Smoke) { '1,4,8' } else { '1,4,8,16,32,64,128,256' }

    $steps += Invoke-LoggedProcess -Name 'modelbench q8' -FilePath $modelbench -ArgumentList @('-quant', '-prefill-reps', $prefillReps, '-decode-reps', $decodeReps, '-decode-steps', $decodeSteps, '-workload', $workload, '-workload-prefill-cap', $prefillCap, '-out', (Join-Path $tickDir 'modelbench-q8.json')) -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'modelbench-q8.log') -TimeoutSec 1800 -Env $envOverrides
    $steps += Invoke-LoggedProcess -Name 'batchbench q8' -FilePath $batchbench -ArgumentList @('-quant', '-reps', $batchReps, '-decode-steps', $batchSteps, '-batches', $batches, '-out', (Join-Path $tickDir 'batchbench-q8.json')) -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'batchbench-q8.log') -TimeoutSec 600 -Env $envOverrides

    $turnMax = if ($Smoke) { '4' } else { '16' }
    $agentMax = if ($Smoke) { '4' } else { '16' }
    $trials = if ($Smoke) { '2' } else { '8' }
    $steps += Invoke-LoggedProcess -Name 'fleetbench read-heavy' -FilePath $fleetbench -ArgumentList @('--grid', 'log', '--turn-max', $turnMax, '--agent-max', $agentMax, '--trials', $trials, '--out', (Join-Path $tickDir 'fleetbench.json'), '--csv', (Join-Path $tickDir 'fleetbench.csv')) -WorkingDirectory $fakDir -LogPath (Join-Path $tickDir 'fleetbench.log') -TimeoutSec 600 -Env $envOverrides

    $steps += Invoke-LoggedProcess -Name 'node compare text' -FilePath 'python' -ArgumentList @((Join-Path $FleetDir 'tools\fak_node_compare.py')) -WorkingDirectory $FleetDir -LogPath (Join-Path $tickDir 'node-compare.txt') -TimeoutSec 60
    $steps += Invoke-LoggedProcess -Name 'node compare json' -FilePath 'python' -ArgumentList @((Join-Path $FleetDir 'tools\fak_node_compare.py'), '--json') -WorkingDirectory $FleetDir -LogPath (Join-Path $tickDir 'node-compare-json.log') -TimeoutSec 60
    $nodeText = Get-Content -Path (Join-Path $tickDir 'node-compare-json.log') -Raw -Encoding UTF8
    $nodeMatch = [regex]::Match($nodeText, '(?s)\## stdout\s*(.*?)\s*\## stderr')
    if ($nodeMatch.Success) { $nodeMatch.Groups[1].Value.Trim() | Set-Content -Path (Join-Path $tickDir 'node-compare.json') -Encoding UTF8 }

    $tickEnd = Get-Date
    $failures = @($steps | Where-Object { $_.exit_code -ne 0 }).Count
    $summary = [ordered]@{
      schema = 'fak.overnight-soak.tick.v1'
      tick = $tick
      index = $i
      started_utc = $tickStart.ToUniversalTime().ToString('o')
      ended_utc = $tickEnd.ToUniversalTime().ToString('o')
      duration_sec = [Math]::Round(($tickEnd - $tickStart).TotalSeconds, 3)
      boundary_heavy = $boundary
      failures = $failures
      steps = $steps
      metrics = Build-MetricSummary $tickDir
    }
    Write-JsonFile (Join-Path $tickDir 'summary.json') $summary
    Write-TickMarkdown $tickDir $summary
    $allSummaries += $summary
    Write-JsonFile (Join-Path $RunRoot 'summary.json') ([ordered]@{
      schema = 'fak.overnight-soak.summary.v1'
      run_root = $RunRoot
      updated_utc = (Get-Date).ToUniversalTime().ToString('o')
      completed_samples = @($allSummaries).Count
      requested_samples = $Samples
      failures = @($allSummaries | ForEach-Object { $_.failures } | Measure-Object -Sum).Sum
      ticks = $allSummaries
    })

    Write-Host "[$tick] done in $($summary.duration_sec)s failures=$failures -> $tickDir"

    if ($i -lt ($Samples - 1) -and $IntervalMinutes -gt 0) {
      $nextTarget = $start.AddMinutes($IntervalMinutes * ($i + 1))
      $sleepSec = [Math]::Floor(($nextTarget - (Get-Date)).TotalSeconds)
      if ($sleepSec -gt 0) {
        Write-Host "[$tick] sleeping $sleepSec seconds until $($nextTarget.ToString('s'))"
        Start-Sleep -Seconds $sleepSec
      } else {
        Write-Host "[$tick] next target already due; continuing"
      }
    }
  }

  $done = Get-Date
  @"
# Overnight FAK soak summary

- run root: $RunRoot
- started UTC: $($start.ToUniversalTime().ToString('o'))
- ended UTC: $($done.ToUniversalTime().ToString('o'))
- samples completed: $(@($allSummaries).Count) / $Samples
- total failures: $(@($allSummaries | ForEach-Object { $_.failures } | Measure-Object -Sum).Sum)

Latest tick summaries:
$(@($allSummaries | ForEach-Object { "- $($_.tick): failures=$($_.failures), duration=$($_.duration_sec)s, fakbench=$($_.metrics.fakbench.gate), nodes=$($_.metrics.node_compare.nodes)" }) -join "`n")
"@ | Set-Content -Path (Join-Path $RunRoot 'SUMMARY.md') -Encoding UTF8

  Write-Host "soak complete -> $RunRoot"
}
finally {
  Pop-Location
}
