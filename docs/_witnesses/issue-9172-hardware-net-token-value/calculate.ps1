[CmdletBinding()]
param(
    [string]$Scenarios = (Join-Path $PSScriptRoot 'scenarios.csv'),
    [string]$Sources = (Join-Path $PSScriptRoot 'sources.csv'),
    [string]$Results = (Join-Path $PSScriptRoot 'results.csv'),
    [switch]$Check
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture

function Number([object]$value, [string]$field, [string]$id) {
    $parsed = 0.0
    if (-not [double]::TryParse([string]$value, [Globalization.NumberStyles]::Float, $culture, [ref]$parsed)) {
        throw "$id has invalid $field '$value'"
    }
    return $parsed
}
function F([double]$value, [string]$format = '0.000000') { $value.ToString($format, $culture) }

$rows = @(Import-Csv -LiteralPath $Scenarios)
if ($rows.Count -eq 0) { throw 'scenarios.csv has no rows' }
$required = @('scenario_id','category','source_refs','bandwidth_gbps','compute_cap_tokens_s','logical_bytes_per_token_gb','byte_reduction_factor','capex_usd','amortization_years','power_w','energy_usd_per_kwh','utilization','accepted_quality','rejected_work_fraction','migration_cost_usd','migration_amortization_hours','availability','reliability_cost_usd_per_hour','assumption_set')
foreach ($name in $required) { if ($rows[0].PSObject.Properties.Name -notcontains $name) { throw "missing column $name" } }
if ((@($rows.scenario_id | Sort-Object -Unique)).Count -ne $rows.Count) { throw 'scenario_id values must be unique' }

$sourcesRows = @(Import-Csv -LiteralPath $Sources)
$sourceIDs = @{}
foreach ($source in $sourcesRows) {
    if ([string]::IsNullOrWhiteSpace($source.source_id)) { throw 'sources.csv contains an empty source_id' }
    if ($sourceIDs.ContainsKey($source.source_id)) { throw "duplicate source_id $($source.source_id)" }
    if ($source.source_kind -notin @('manufacturer','standards-body','primary-paper')) { throw "source $($source.source_id) is not primary" }
    $sourceIDs[$source.source_id] = $true
}
foreach ($row in $rows) {
    foreach ($sourceID in $row.source_refs.Split(';')) {
        if (-not $sourceIDs.ContainsKey($sourceID)) { throw "$($row.scenario_id) references unknown source_id $sourceID" }
    }
}
$out = foreach ($r in $rows) {
    $id = $r.scenario_id
    $bw = Number $r.bandwidth_gbps bandwidth_gbps $id
    $cap = Number $r.compute_cap_tokens_s compute_cap_tokens_s $id
    $logical = Number $r.logical_bytes_per_token_gb logical_bytes_per_token_gb $id
    $reduction = Number $r.byte_reduction_factor byte_reduction_factor $id
    $capex = Number $r.capex_usd capex_usd $id
    $years = Number $r.amortization_years amortization_years $id
    $power = Number $r.power_w power_w $id
    $energyRate = Number $r.energy_usd_per_kwh energy_usd_per_kwh $id
    $util = Number $r.utilization utilization $id
    $quality = Number $r.accepted_quality accepted_quality $id
    $reject = Number $r.rejected_work_fraction rejected_work_fraction $id
    $migration = Number $r.migration_cost_usd migration_cost_usd $id
    $migrationHours = Number $r.migration_amortization_hours migration_amortization_hours $id
    $availability = Number $r.availability availability $id
    $reliabilityCost = Number $r.reliability_cost_usd_per_hour reliability_cost_usd_per_hour $id
    foreach ($pair in @(@('bandwidth_gbps',$bw),@('compute_cap_tokens_s',$cap),@('logical_bytes_per_token_gb',$logical),@('byte_reduction_factor',$reduction),@('amortization_years',$years),@('migration_amortization_hours',$migrationHours))) {
        if ($pair[1] -le 0) { throw "$id requires $($pair[0]) > 0" }
    }
    foreach ($pair in @(@('utilization',$util),@('accepted_quality',$quality),@('availability',$availability))) {
        if ($pair[1] -lt 0 -or $pair[1] -gt 1) { throw "$id requires $($pair[0]) in [0,1]" }
    }
    if ($reject -lt 0 -or $reject -ge 1) { throw "$id requires rejected_work_fraction in [0,1)" }

    $physicalBytes = $logical * $reduction * 1e9
    $bandwidthTokens = ($bw * 1e9) / $physicalBytes
    $rawTokens = [Math]::Min($bandwidthTokens, $cap)
    $acceptedTokensHour = $rawTokens * 3600 * $util * $quality * (1 - $reject) * $availability
    $capexHour = $capex / ($years * 365 * 24)
    $energyHour = ($power / 1000) * $energyRate * $util
    $migrationHour = $migration / $migrationHours
    $totalHour = $capexHour + $energyHour + $migrationHour + $reliabilityCost
    $netValue = $acceptedTokensHour / $totalHour
    [pscustomobject][ordered]@{
        scenario_id = $id
        category = $r.category
        source_refs = $r.source_refs
        physical_bytes_per_token_gb = F ($physicalBytes / 1e9)
        bandwidth_limited_tokens_s = F $bandwidthTokens
        raw_tokens_s = F $rawTokens
        accepted_tokens_per_hour = F $acceptedTokensHour
        capex_usd_per_hour = F $capexHour
        energy_usd_per_hour = F $energyHour
        migration_usd_per_hour = F $migrationHour
        reliability_usd_per_hour = F $reliabilityCost
        total_usd_per_hour = F $totalHour
        net_accepted_tokens_per_usd = F $netValue
        assumption_set = $r.assumption_set
    }
}

$csv = ($out | ConvertTo-Csv -NoTypeInformation -UseQuotes AsNeeded) -join "`n"
$csv += "`n"
if ($Check) {
    if (-not (Test-Path -LiteralPath $Results)) { throw "missing results file: $Results" }
    $existing = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $Results)).Replace("`r`n", "`n")
    if ($existing -ne $csv) { throw 'results.csv is stale; run calculate.ps1 without -Check' }
    Write-Output "PASS: $($rows.Count) scenarios; results.csv is deterministic and current"
} else {
    [IO.File]::WriteAllText([IO.Path]::GetFullPath($Results), $csv, [Text.UTF8Encoding]::new($false))
    Write-Output "WROTE: $Results ($($rows.Count) scenarios)"
}

