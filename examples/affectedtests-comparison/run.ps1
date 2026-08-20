$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$fixture = Join-Path $PSScriptRoot 'go'
$expected = @(
    'example.com/diamond/cmd/app'
    'example.com/diamond/internal/a'
    'example.com/diamond/internal/b'
    'example.com/diamond/internal/c'
)

Push-Location $fixture
try {
    $selected = @(& fak affected --file internal/a/a.go --list)
    if ($LASTEXITCODE -ne 0) {
        throw "fak affected exited $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$difference = @(Compare-Object -ReferenceObject $expected -DifferenceObject $selected)
if ($difference.Count -ne 0 -or $selected.Count -ne $expected.Count) {
    throw "affected set mismatch: expected [$($expected -join ', ')], got [$($selected -join ', ')]"
}

Write-Output 'PASS affected set for internal/a/a.go'
Write-Output "selected: $($selected.Count)"
$selected | ForEach-Object { Write-Output "  $_" }
Write-Output 'excluded: example.com/diamond/internal/isolated'
