# fetch-gguf.ps1 — download a GGUF model and tokenizer for the simple demo
#
# Usage:
#   .\scripts\fetch-gguf.ps1              # download default (1.5B Qwen2.5)
#   .\scripts\fetch-gguf.ps1 3b          # download 3B model
#   .\scripts\fetch-gguf.ps1 7b          # download 7B model
#   .\scripts\fetch-gguf.ps1 custom <url> # download custom GGUF
#
# Output: ~/.cache/fak-models/gguf/<model-file>.gguf
#         ~/.cache/fak-models/tokenizers/<tokenizer-dir>/

param(
    [Parameter(Position=0)]
    [string]$Size = "1.5b",

    [Parameter(Position=1)]
    [string]$CustomUrl = ""
)

$ErrorActionPreference = "Stop"

# Model configurations
$Models = @{
    "1.5b" = "Qwen2.5-1.5B-Instruct.Q8_0"
    "3b" = "Qwen2.5-3B-Instruct.Q8_0"
    "7b" = "Qwen2.5-7B-Instruct.Q4_K_M"
}

$Sizes = @{
    "1.5b" = "1.6GB"
    "3b" = "3.2GB"
    "7b" = "4.7GB"
}

$BaseUrl = "https://huggingface.co/mradermacher/Qwen2.5"

# Normalize size key
$Size = $Size.ToLower()

if ($CustomUrl -ne "") {
    $ModelName = [System.IO.Path]::GetFileNameWithoutExtension($CustomUrl)
    $TOK_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json"
    $GGUF_URL = $CustomUrl
} elseif (-not $Models.ContainsKey($Size)) {
    Write-Host "Error: unknown size '$Size'. Choose from: 1.5b, 3b, 7b, or 'custom <url>'" -ForegroundColor Red
    exit 1
} else {
    $ModelName = $Models[$Size]
    $SizeDisplay = $Sizes[$Size]
    # Use the original Qwen repo for tokenizer (same across Qwen2.5 family)
    $TOK_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json"
    # Fix URL: need to convert "1.5b" to "1.5B", "3b" to "3B", etc.
    if ($Size -eq "1.5b") { $SizeUpper = "1.5B" }
    elseif ($Size -eq "3b") { $SizeUpper = "3B" }
    elseif ($Size -eq "7b") { $SizeUpper = "7B" }
    else { $SizeUpper = $Size.ToUpper() }
    $GGUF_URL = "$BaseUrl-$SizeUpper-Instruct-GGUF/resolve/main/$ModelName.gguf"
}

# Output directories
$CacheDir = Join-Path $env:USERPROFILE ".cache\fak-models"
$GGUFDir = Join-Path $CacheDir "gguf"
$TOK_DIR = Join-Path $CacheDir "tokenizers\qwen2.5"

New-Item -ItemType Directory -Force -Path $GGUFDir | Out-Null
New-Item -ItemType Directory -Force -Path $TOK_DIR | Out-Null

$GGUFPath = Join-Path $GGUFDir "$ModelName.gguf"
$TOKPath = Join-Path $TOK_DIR "tokenizer.json"

Write-Host "📥 Fetching GGUF model..." -ForegroundColor Cyan
Write-Host "   Model: $ModelName"
if ($SizeDisplay) { Write-Host "   Size: $SizeDisplay" }
Write-Host "   Target: $GGUFPath"
Write-Host ""

# Use curl.exe for large files (more reliable than Invoke-WebRequest)
function Get-FileWithCurl {
    param($Url, $Output)
    $proc = Start-Process -FilePath "curl.exe" -ArgumentList "-L", "--progress-bar", $Url, "-o", $Output -PassThru -NoNewWindow
    $proc.WaitForExit()
    return $proc.ExitCode
}

# Check if already downloaded
if (Test-Path $GGUFPath) {
    Write-Host "✅ Model already exists at $GGUFPath" -ForegroundColor Green
    $Redownload = Read-Host "Re-download? [y/N]"
    if ($Redownload -eq "y" -or $Redownload -eq "Y") {
        Remove-Item $GGUFPath -Force
        Get-FileWithCurl -Url $GGUF_URL -Output $GGUFPath
    } else {
        Write-Host "Skipping model download."
    }
} else {
    Get-FileWithCurl -Url $GGUF_URL -Output $GGUFPath
}

Write-Host ""
Write-Host "📥 Fetching tokenizer..." -ForegroundColor Cyan

if (Test-Path $TOKPath) {
    Write-Host "✅ Tokenizer already exists at $TOKPath" -ForegroundColor Green
} else {
    Get-FileWithCurl -Url $TOK_URL -Output $TOKPath
}

Write-Host ""
Write-Host "✅ Done!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the demo:"
Write-Host "  go run ./cmd/simpledemo -gguf `$env:USERPROFILE\.cache\fak-models\gguf\$ModelName.gguf -tok `$env:USERPROFILE\.cache\fak-models\tokenizers\qwen2.5"
Write-Host ""
Write-Host "Or set environment variables:"
Write-Host "  `$env:FAK_GGUF = `"$GGUFPath`""
Write-Host "  `$env:FAK_TOKENIZER = `"$TOK_DIR`""
