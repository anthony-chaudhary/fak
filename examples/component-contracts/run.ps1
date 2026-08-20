$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$contracts = @(
    (Join-Path $PSScriptRoot 'radix-cache.json')
    (Join-Path $PSScriptRoot 'paged-attention-kernel.json')
    (Join-Path $PSScriptRoot 'cuda-runtime.json')
)

$output = @(& fak component check `
    --contract $contracts[0] `
    --contract $contracts[1] `
    --contract $contracts[2] `
    --root cache.kv.radix@1 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "fak component check exited $LASTEXITCODE"
}

$receipt = $output -join "`n"
if ($receipt -notmatch 'ALLOW stack for component-compatibility-check') {
    throw 'component check did not allow the complete hard-requirement chain'
}
if ($receipt -notmatch 'RECOMMENDATION_UNMET: cache\.kv\.radix@1 wants runtime\.cuda\.graphs') {
    throw 'component check did not preserve the expected soft recommendation warning'
}

$output | Write-Output
