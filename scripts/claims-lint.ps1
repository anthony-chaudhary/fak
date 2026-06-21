# claims-lint.ps1 — the unit-96 witness: every claim line in CLAIMS.md (a line
# starting with "- [") must carry EXACTLY ONE of the three tags
# [SHIPPED] / [SIMULATED] / [STUB]. Exit non-zero on any violation.
$ErrorActionPreference = "Stop"
$claims = Join-Path $PSScriptRoot "..\CLAIMS.md"
$lines = Get-Content $claims
$bad = 0
$count = 0
for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    if ($line -match '^- \[') {
        $count++
        $tags = 0
        foreach ($t in @('[SHIPPED]', '[SIMULATED]', '[STUB]')) {
            if ($line.Contains($t)) { $tags++ }
        }
        if ($tags -ne 1) {
            Write-Host ("VIOLATION line {0}: {1} tags -- {2}" -f ($i + 1), $tags, $line)
            $bad++
        }
    }
}
Write-Host ("claims-lint: {0} claim lines, {1} violations" -f $count, $bad)
if ($bad -gt 0 -or $count -eq 0) { exit 1 }
exit 0
