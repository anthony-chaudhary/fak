param(
  [string]$OutDir = "fak\experiments\agent-live\transcript-adapter-sweep",
  [int]$MaxTurns = 12,
  [int]$Trials = 1,
  [string]$ApiBaseUrl = "https://gateway.glama.ai/v1",
  [string]$ApiKeyEnv = "GLAMA_API_KEY",
  [string[]]$ApiModels = @(
    "zai/glm-4.7-flash",
    "openai/gpt-4.1-nano-2025-04-14",
    "deepseek/deepseek-v4-flash"
  ),
  [string[]]$LocalShimModels = @(),
  [string]$Python = "python",
  [int]$LocalPlannerTimeoutS = 600,
  [switch]$AllowNoApiKey,
  [switch]$SkipApi,
  [switch]$SkipOffline,
  [switch]$SkipLocalShim,
  [switch]$SkipMicrobench,
  [switch]$FailFast
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FakDir = Join-Path $Root "fak"
$OutDirAbs = Join-Path $Root $OutDir
$BinDir = Join-Path $Root "tools\.bin"
$FakBin = Join-Path $BinDir "fak.exe"
$Shim = Join-Path $Root "fak\experiments\agent-live\local_shim.py"
$GeneratedAt = (Get-Date).ToString("o")

$ApiPriceHints = @{
  "zai/glm-4.7-flash" = @{ input = 0.07; output = 0.40; source = "glama-models-2026-06-18" }
  "openai/gpt-4.1-nano-2025-04-14" = @{ input = 0.10; output = 0.40; source = "glama-models-2026-06-18" }
  "deepseek/deepseek-v4-flash" = @{ input = 0.14; output = 0.28; source = "glama-models-2026-06-18" }
  "mistral/ministral-14b-2512" = @{ input = 0.20; output = 0.20; source = "glama-models-2026-06-18" }
  "google-vertex/gemini-2.5-flash" = @{ input = 0.15; output = 0.60; source = "glama-models-2026-06-18" }
  "xai/grok-4-fast-non-reasoning" = @{ input = 0.20; output = 0.50; source = "glama-models-2026-06-18" }
}

function New-Slug([string]$Value) {
  $slug = $Value -replace '[^A-Za-z0-9._-]+', '-'
  $slug = $slug.Trim('-')
  if ($slug.Length -eq 0) { return "case" }
  return $slug
}

function Invoke-InDir([string]$Cwd, [string]$Exe, [string[]]$ArgList, [string]$LogPath) {
  Push-Location $Cwd
  $oldEAP = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $output = & $Exe @ArgList 2>&1
    $exit = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $oldEAP
    Pop-Location
  }
  $output | Set-Content -Path $LogPath -Encoding UTF8
  return @{ exit = $exit; output = ($output -join [Environment]::NewLine) }
}

function Build-Fak {
  New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
  $log = Join-Path $OutDirAbs "build-fak.log"
  $res = Invoke-InDir $FakDir "go" @("build", "-o", $FakBin, "./cmd/fak") $log
  if ($res.exit -ne 0) {
    throw "go build failed; see $log"
  }
}

function Run-AdapterMicrobench([string]$MicrobenchPath) {
  $testBin = Join-Path $BinDir "agent.test.exe"
  $buildLog = Join-Path $OutDirAbs "adapter-microbench-build.log"
  $build = Invoke-InDir $FakDir "go" @("test", "-c", "-o", $testBin, "./internal/agent") $buildLog
  if ($build.exit -ne 0) {
    return $build
  }
  return Invoke-InDir $FakDir $testBin @("-test.run=^$", "-test.bench=Benchmark(PreSendQuarantine|TranscriptAdapters)", "-test.benchmem") $MicrobenchPath
}

function Read-Report([string]$Path) {
  if (!(Test-Path $Path)) {
    return $null
  }
  try {
    return Get-Content -Raw -Path $Path | ConvertFrom-Json
  } catch {
    return $null
  }
}

function Add-RunRow(
  [System.Collections.Generic.List[object]]$Rows,
  [string]$Kind,
  [string]$Name,
  [string]$Provider,
  [string]$BaseUrl,
  [string]$Model,
  [int]$Trial,
  [string]$Status,
  [int]$ExitCode,
  [long]$ElapsedMs,
  [string]$OutPath,
  [string]$TracePath,
  [string]$StdoutPath,
  [object]$Report,
  [string]$ErrorText
) {
  $price = $ApiPriceHints[$Model]
  $row = [ordered]@{
    generated_at = $GeneratedAt
    kind = $Kind
    name = $Name
    provider = $Provider
    base_url = $BaseUrl
    model = $Model
    trial = $Trial
    status = $Status
    exit_code = $ExitCode
    elapsed_ms = $ElapsedMs
    out = $OutPath
    trace = $TracePath
    stdout = $StdoutPath
    price_hint_input_usd_per_mtok = $null
    price_hint_output_usd_per_mtok = $null
    price_hint_source = $null
    live = $null
    transcript_sha = $null
    fak_turns = $null
    baseline_turns = $null
    fak_prompt_tokens = $null
    baseline_prompt_tokens = $null
    fak_completion_tokens = $null
    baseline_completion_tokens = $null
    both_completed = $null
    fak_completed = $null
    baseline_completed = $null
    poison_blocked = $null
    fak_quarantines = $null
    fak_repairs = $null
    fak_vdso_hits = $null
    error = $ErrorText
  }
  if ($price -ne $null) {
    $row.price_hint_input_usd_per_mtok = $price.input
    $row.price_hint_output_usd_per_mtok = $price.output
    $row.price_hint_source = $price.source
  }
  if ($Report -ne $null) {
    $row.live = $Report.live
    $row.transcript_sha = $Report.transcript_sha
    $row.fak_turns = $Report.fak.turns
    $row.baseline_turns = $Report.baseline.turns
    $row.fak_prompt_tokens = $Report.fak.prompt_tokens
    $row.baseline_prompt_tokens = $Report.baseline.prompt_tokens
    $row.fak_completion_tokens = $Report.fak.completion_tokens
    $row.baseline_completion_tokens = $Report.baseline.completion_tokens
    $row.both_completed = $Report.both_completed
    $row.fak_completed = $Report.fak.task_completed
    $row.baseline_completed = $Report.baseline.task_completed
    $row.poison_blocked = ($Report.baseline.injection_in_context -and -not $Report.fak.injection_in_context)
    $row.fak_quarantines = $Report.fak.quarantines
    $row.fak_repairs = $Report.fak.repairs
    $row.fak_vdso_hits = $Report.fak.vdso_hits
  }
  $Rows.Add([pscustomobject]$row) | Out-Null
}

function Run-AgentCase(
  [System.Collections.Generic.List[object]]$Rows,
  [string]$Kind,
  [string]$Name,
  [string]$Provider,
  [string]$BaseUrl,
  [string]$Model,
  [string]$ApiKeyEnv,
  [bool]$Offline
) {
  for ($trial = 1; $trial -le $Trials; $trial++) {
    $slug = New-Slug "$Kind-$Name-t$trial"
    $out = Join-Path $OutDirAbs "$slug.json"
    $trace = Join-Path $OutDirAbs "$slug-trace.txt"
    $stdout = Join-Path $OutDirAbs "$slug.stdout.txt"
    $args = @("agent", "--model", $Model, "--max-turns", [string]$MaxTurns, "--out", $out, "--log", $trace)
    if ($Offline) {
      $args += "--offline"
    } else {
      $args += @("--provider", $Provider, "--base-url", $BaseUrl, "--api-key-env", $ApiKeyEnv)
    }

    Write-Host "[sweep] $Kind $Name trial $trial -> $out"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $res = Invoke-InDir $FakDir $FakBin $args $stdout
    $sw.Stop()

    $report = Read-Report $out
    $status = "ok"
    $err = $null
    if ($res.exit -ne 0) {
      $status = "failed"
      $err = $res.output
      if (!(Test-Path $out)) {
        [ordered]@{
          status = "failed"
          kind = $Kind
          provider = $Provider
          base_url = $BaseUrl
          model = $Model
          trial = $trial
          exit_code = $res.exit
          error = $err
        } | ConvertTo-Json -Depth 6 | Set-Content -Path $out -Encoding UTF8
      }
      if ($FailFast) {
        throw "$Kind $Name failed; see $stdout"
      }
    }
    Add-RunRow $Rows $Kind $Name $Provider $BaseUrl $Model $trial $status $res.exit $sw.ElapsedMilliseconds $out $trace $stdout $report $err
  }
}

function Wait-LocalShim([int]$Port, [System.Diagnostics.Process]$Proc, [string]$ErrLog) {
  for ($i = 0; $i -lt 150; $i++) {
    if ($Proc.HasExited) {
      throw "local shim exited during load; see $ErrLog"
    }
    try {
      Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2 | Out-Null
      return
    } catch {
      Start-Sleep -Seconds 2
    }
  }
  throw "local shim did not become ready on port $Port; see $ErrLog"
}

function Run-LocalShimCase([System.Collections.Generic.List[object]]$Rows, [string]$Spec) {
  $idx = $Spec.LastIndexOf("@")
  if ($idx -lt 1) {
    throw "Local shim spec must be '<hf-model-id>@<port>'; got '$Spec'"
  }
  $model = $Spec.Substring(0, $idx)
  $port = [int]$Spec.Substring($idx + 1)
  $slug = New-Slug "local-shim-$model-$port"
  $shimOut = Join-Path $OutDirAbs "$slug.shim.stdout.txt"
  $shimErr = Join-Path $OutDirAbs "$slug.shim.stderr.txt"
  $oldTimeout = $env:FAK_PLANNER_TIMEOUT_S
  $env:TRANSFORMERS_OFFLINE = "1"
  $env:HF_HUB_OFFLINE = "1"
  $env:FAK_PLANNER_TIMEOUT_S = [string]$LocalPlannerTimeoutS

  Write-Host "[sweep] starting local shim $model on :$port"
  $proc = Start-Process -FilePath $Python -ArgumentList @($Shim, "--model", $model, "--port", [string]$port) -PassThru -WindowStyle Hidden -RedirectStandardOutput $shimOut -RedirectStandardError $shimErr
  try {
    Wait-LocalShim $port $proc $shimErr
    Run-AgentCase $Rows "local-shim" $model "openai" "http://127.0.0.1:$port/v1" $model "NONE_LOCAL" $false
  } catch {
    $err = $_.Exception.Message
    if (Test-Path $shimErr) {
      $detail = Get-Content -Raw -Path $shimErr
      if (![string]::IsNullOrWhiteSpace($detail)) {
        $err = $err + [Environment]::NewLine + $detail
      }
    }
    $failOut = Join-Path $OutDirAbs "$slug-load-failed.json"
    [ordered]@{
      status = "failed"
      kind = "local-shim"
      provider = "openai"
      base_url = "http://127.0.0.1:$port/v1"
      model = $model
      trial = 1
      exit_code = 1
      error = $err
    } | ConvertTo-Json -Depth 6 | Set-Content -Path $failOut -Encoding UTF8
    Add-RunRow $Rows "local-shim" $model "openai" "http://127.0.0.1:$port/v1" $model 1 "failed" 1 0 $failOut "" $shimErr $null $err
    if ($FailFast) {
      throw
    }
  } finally {
    if (!$proc.HasExited) {
      Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    $env:FAK_PLANNER_TIMEOUT_S = $oldTimeout
  }
}

function Write-Summaries([System.Collections.Generic.List[object]]$Rows, [string]$MicrobenchPath) {
  $summaryJson = Join-Path $OutDirAbs "sweep-summary.json"
  $summaryMd = Join-Path $OutDirAbs "SWEEP.md"
  ConvertTo-Json -InputObject @($Rows.ToArray()) -Depth 12 | Set-Content -Path $summaryJson -Encoding UTF8

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Transcript Adapter Sweep") | Out-Null
  $lines.Add("") | Out-Null
  $lines.Add("Generated: $GeneratedAt") | Out-Null
  $lines.Add("") | Out-Null
  $lines.Add("Scope: adapter microbenchmarks plus ``fak agent`` A/B runs through low-cost API models, offline mock, and any requested local shim models.") | Out-Null
  $lines.Add("") | Out-Null
  if ($MicrobenchPath -ne "") {
    $lines.Add("Microbenchmarks: ``$MicrobenchPath``") | Out-Null
    $lines.Add("") | Out-Null
  }
  $lines.Add("| kind | model | status | provider | base turns | fak turns | both completed | poison blocked | quarantines | output |") | Out-Null
  $lines.Add("|---|---|---|---|---:|---:|:---:|:---:|---:|---|") | Out-Null
  foreach ($r in $Rows) {
    $lines.Add("| $($r.kind) | ``$($r.model)`` | $($r.status) | $($r.provider) | $($r.baseline_turns) | $($r.fak_turns) | $($r.both_completed) | $($r.poison_blocked) | $($r.fak_quarantines) | ``$($r.out)`` |") | Out-Null
  }
  $lines.Add("") | Out-Null
  $lines.Add("When present, price hints are only used to order the API sweep; the benchmark JSON remains the source of measured behavior.") | Out-Null
  $lines | Set-Content -Path $summaryMd -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $OutDirAbs | Out-Null
Build-Fak

$rows = New-Object System.Collections.Generic.List[object]
$microbenchPath = ""
if (!$SkipMicrobench) {
  $microbenchPath = Join-Path $OutDirAbs "adapter-microbench.txt"
  Write-Host "[sweep] running adapter microbenchmarks"
  $benchRes = Run-AdapterMicrobench $microbenchPath
  if ($benchRes.exit -ne 0 -and $FailFast) {
    throw "adapter microbenchmarks failed; see $microbenchPath"
  }
}

if (!$SkipOffline) {
  Run-AgentCase $rows "offline" "mock" "mock" "" "offline-mock" "NONE_LOCAL" $true
}

if (!$SkipApi) {
  $noKeyRequested = $AllowNoApiKey -or [string]::IsNullOrWhiteSpace($ApiKeyEnv) -or $ApiKeyEnv -in @("NONE", "NONE_LOCAL")
  if (!$noKeyRequested -and [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($ApiKeyEnv))) {
    Write-Host "[sweep] skipping API runs: env $ApiKeyEnv is empty"
  } else {
    foreach ($model in $ApiModels) {
      Run-AgentCase $rows "api" $model "openai" $ApiBaseUrl $model $ApiKeyEnv $false
    }
  }
}

if (!$SkipLocalShim) {
  foreach ($spec in $LocalShimModels) {
    Run-LocalShimCase $rows $spec
  }
}

Write-Summaries $rows $microbenchPath
Write-Host "[sweep] summary: $(Join-Path $OutDirAbs 'SWEEP.md')"
