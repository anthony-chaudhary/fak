param(
    [Parameter(Mandatory = $true)]
    [string]$CodexArtifact,
    [Parameter(Mandatory = $true)]
    [string]$ClaudeArtifact,
    [string]$OutputDir = (Join-Path $PSScriptRoot "witnesses")
)

$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$playwright = Join-Path $PSScriptRoot "node_modules\.bin\playwright.cmd"
if (-not (Test-Path -LiteralPath $playwright)) {
    throw "Playwright is not installed; run npm ci --prefix docs/benchmarks/prompt-vault"
}
if (-not (Get-Command chrome -ErrorAction SilentlyContinue) -and
    -not (Test-Path -LiteralPath "C:\Program Files\Google\Chrome\Application\chrome.exe")) {
    throw "Google Chrome is required for the pinned Playwright channel"
}

$codexPath = (Resolve-Path -LiteralPath $CodexArtifact).Path
$claudePath = (Resolve-Path -LiteralPath $ClaudeArtifact).Path
$specName = "color-palette.spec.js"
$scratch = Join-Path ([IO.Path]::GetTempPath()) ("fak-prompt-vault-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratch | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Invoke-ColorPaletteGrade {
    param(
        [string]$ArtifactID,
        [string]$Artifact,
        [bool]$ShouldPass
    )

    $env:PALETTE_ARTIFACT_ID = $ArtifactID
    $env:PALETTE_ARTIFACT = $Artifact
    $env:PALETTE_REPORT = Join-Path $OutputDir ($ArtifactID + ".json")
    $env:PALETTE_SCREENSHOT = Join-Path $OutputDir ($ArtifactID + ".png")
    $env:PALETTE_BROWSER_CHANNEL = "chrome"
    $env:PLAYWRIGHT_LAST_RUN_OUTPUT_FILE = Join-Path $scratch ($ArtifactID + "-last-run.json")

    Push-Location $PSScriptRoot
    try {
        & $playwright test $specName `
            --workers=1 `
            --reporter=line `
            --output (Join-Path $scratch $ArtifactID)
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($ShouldPass -and $exitCode -ne 0) {
        throw "$ArtifactID did not pass the Color Palette fixture (exit $exitCode)"
    }
    if (-not $ShouldPass -and $exitCode -eq 0) {
        throw "$ArtifactID unexpectedly passed the Color Palette fixture"
    }
}

Invoke-ColorPaletteGrade -ArtifactID "fail-empty" `
    -Artifact (Join-Path $PSScriptRoot "testdata\empty\index.html") -ShouldPass $false
Invoke-ColorPaletteGrade -ArtifactID "fail-wrong" `
    -Artifact (Join-Path $PSScriptRoot "testdata\wrong\index.html") -ShouldPass $false
Invoke-ColorPaletteGrade -ArtifactID "codex" -Artifact $codexPath -ShouldPass $true
Invoke-ColorPaletteGrade -ArtifactID "claude" -Artifact $claudePath -ShouldPass $true

Get-ChildItem -LiteralPath $OutputDir -File | Sort-Object Name | Select-Object Name, Length
