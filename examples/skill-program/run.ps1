$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$skill = Join-Path $PSScriptRoot 'SKILL.md'
$hiddenRaw = (& fak skill compile --json $skill | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "hidden skill compile exited $LASTEXITCODE"
}
$hidden = $hiddenRaw | ConvertFrom-Json

$shownRaw = (& fak skill compile --json --dialect codex --expose repo_search $skill | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "exposed skill compile exited $LASTEXITCODE"
}
$shown = $shownRaw | ConvertFrom-Json

$hiddenOmitted = @($hidden.model_view.omitted)
$shownTools = @($shown.model_view.tools)

if ($null -ne $hidden.model_view.tools) {
    throw 'registration leaked into the default model snapshot'
}
if ($hiddenOmitted.Count -ne 1 -or $hiddenOmitted[0].reason -ne 'NOT_SELECTED') {
    throw 'default snapshot did not carry the expected omission witness'
}
if ($shownTools.Count -ne 1 -or $shownTools[0].name -ne 'functions.shell_command') {
    throw 'Codex dialect alias is missing or ambiguous'
}
if ($shownTools[0].canonical_name -ne 'repo_search') {
    throw 'provider-visible tool lost its canonical identity'
}
if ($shownTools[0].PSObject.Properties.Name -contains 'executor') {
    throw 'host-only executor data leaked into the provider-visible tool'
}

Write-Output 'PASS hidden snapshot: tools=0 omitted=repo_search reason=NOT_SELECTED'
Write-Output 'PASS codex snapshot: name=functions.shell_command canonical=repo_search'
Write-Output 'PASS executor isolation: provider-visible tool contains no executor argv'
