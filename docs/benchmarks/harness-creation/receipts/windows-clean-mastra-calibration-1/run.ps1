param(
  [string]$RunID = 'windows-clean-mastra-calibration-1'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$base = Join-Path $env:TEMP ("fak-mastra-baseline-" + [guid]::NewGuid().ToString('N'))
$cache = Join-Path $base 'npm-cache'
$product = Join-Path $base 'product'
$transcript = Join-Path $base 'transcript.txt'
New-Item -ItemType Directory -Force $base,$cache | Out-Null
$oldCache = $env:npm_config_cache
$env:npm_config_cache = $cache
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = [Collections.Generic.List[string]]::new()
function Record([string]$line) { $lines.Add($line); Write-Host $line }
function Timed([string]$name, [scriptblock]$action) {
  $timer = [Diagnostics.Stopwatch]::StartNew(); & $action; $timer.Stop()
  Record ("{0}_seconds={1:N3}" -f $name,$timer.Elapsed.TotalSeconds)
}
try {
  Record "run_id=$RunID"
  Record 'participant_id=maintainer-baseline-calibration'
  Record 'participant_class=maintainer-calibration independent=false'
  Record ("start={0:o}" -f [DateTimeOffset]::UtcNow)
  Record 'cache_state=empty npm cache'
  Record ("node_host={0}" -f (& node --version))
  Record 'generator=create-mastra@1.25.0 runtime=@mastra/core@1.59.0'
  Push-Location $base
  Timed install { & npx --yes create-mastra@1.25.0 product --empty --no-git --no-skills 2>&1 | ForEach-Object { Record "install: $_" }; if ($LASTEXITCODE) { throw "generator exit $LASTEXITCODE" } }
  Pop-Location
  Copy-Item (Join-Path $PSScriptRoot 'user-harness.ts') (Join-Path $product 'src\mastra\user-harness.ts')
  Copy-Item (Join-Path $PSScriptRoot 'selfcheck.ts') (Join-Path $product 'src\mastra\selfcheck.ts')
  $before = (Get-FileHash (Join-Path $product 'src\mastra\user-harness.ts') -Algorithm SHA256).Hash.ToLower()
  Push-Location $product
  Timed selfcheck { $out=& node --experimental-strip-types src/mastra/selfcheck.ts 2>&1; $out | ForEach-Object { Record "selfcheck: $_" }; if ($LASTEXITCODE -or ($out -notmatch 'BASELINE_SELFCHECK ok')) { throw 'selfcheck failed' } }
  Timed build { & npm run build 2>&1 | ForEach-Object { Record "build: $_" }; if ($LASTEXITCODE) { throw "build exit $LASTEXITCODE" } }
  Timed rerun { & npm install 2>&1 | ForEach-Object { Record "rerun: $_" }; if ($LASTEXITCODE) { throw "rerun exit $LASTEXITCODE" } }
  Pop-Location
  $after = (Get-FileHash (Join-Path $product 'src\mastra\user-harness.ts') -Algorithm SHA256).Hash.ToLower()
  if ($before -ne $after) { throw 'user-owned file changed on rerun' }
  Record "user_sha_before=$before"
  Record "user_sha_after=$after"
  Record 'owned_files=2'
  Record 'rebuilds=1 failures=0 help_requests=0'
  Record 'upgrade_command=npm install --save-exact @mastra/core@1.59.0 && npm install --save-dev --save-exact mastra@1.25.0'
  Record 'rollback=restore package.json package-lock.json and task-card TypeScript files; npm ci'
  $sw.Stop(); Record ("elapsed_seconds={0:N3}" -f $sw.Elapsed.TotalSeconds)
  Record ("stop={0:o}" -f [DateTimeOffset]::UtcNow)
  Record 'outcome=success'
} catch {
  $sw.Stop(); Record ("elapsed_seconds={0:N3}" -f $sw.Elapsed.TotalSeconds); Record 'outcome=failure'; Record "error=$($_.Exception.Message)"
  $lines | Set-Content -Encoding utf8 $transcript
  throw
} finally {
  $lines | Set-Content -Encoding utf8 $transcript
  $env:npm_config_cache = $oldCache
  Record "transcript=$transcript"
}
