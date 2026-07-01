// Package amdgpu probes AMD GPU facts on Windows through PowerShell.
package amdgpu

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const ProbeScript = `
$ErrorActionPreference = 'SilentlyContinue'
$vc = Get-CimInstance Win32_VideoController |
      Where-Object { $_.Name -match 'Radeon|AMD' } | Select-Object -First 1
$dedicated = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples |
             Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum
$compute = (Get-Counter '\GPU Engine(*engtype_Compute)\Utilization Percentage').CounterSamples |
           Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum
$total = (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples |
         Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum
$byType = @{}
foreach ($s in (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples) {
  if ($s.InstanceName -match 'engtype_(\w+)') {
    $t = $Matches[1]
    if (-not $byType.ContainsKey($t)) { $byType[$t] = 0.0 }
    $byType[$t] += [double]$s.CookedValue
  }
}
$engines = ($byType.GetEnumerator() | ForEach-Object {
  [pscustomobject]@{ type = $_.Key; util_pct = [math]::Round($_.Value, 1) }
})
[pscustomobject]@{
  name            = $vc.Name
  driver_version  = $vc.DriverVersion
  adapter_ram     = [int64]$vc.AdapterRAM
  vram_used_bytes = [int64]$dedicated
  compute_util_pct = [math]::Round([double]$compute, 1)
  total_util_pct   = [math]::Round([double]$total, 1)
  engines          = @($engines)
} | ConvertTo-Json -Compress
`

type Runner func(script string, timeout time.Duration) (ok bool, stdout string, stderr string)

func PowerShellRunner(script string, timeout time.Duration) (bool, string, string) {
	exe, err := exec.LookPath("pwsh")
	if err != nil {
		exe, err = exec.LookPath("powershell")
	}
	if err != nil {
		return false, "", "no PowerShell (pwsh/powershell) on PATH"
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-NoProfile", "-NonInteractive", "-Command", script)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return false, "", fmt.Sprintf("timeout after %.0fs", timeout.Seconds())
	}
	if err != nil {
		return false, "", strings.TrimSpace(string(out))
	}
	return true, strings.TrimSpace(string(out)), ""
}

func Facts(nameFilter string, runner Runner) map[string]any {
	if runner == nil {
		runner = PowerShellRunner
	}
	ok, out, errText := runner(ProbeScript, 20*time.Second)
	if !ok || strings.TrimSpace(out) == "" {
		if errText == "" {
			errText = "GPU perf counters returned nothing"
		}
		return map[string]any{"available": false, "error": errText}
	}
	var facts map[string]any
	if err := json.Unmarshal([]byte(out), &facts); err != nil {
		return map[string]any{"available": false, "error": "probe JSON parse failed: " + err.Error(), "raw": out}
	}
	name, _ := facts["name"].(string)
	if nameFilter != "" && !strings.Contains(strings.ToLower(name), strings.ToLower(nameFilter)) {
		return map[string]any{"available": false, "error": fmt.Sprintf("no GPU matching %q", nameFilter), "saw": name}
	}
	used := number(facts["vram_used_bytes"])
	facts["vram_used_mib"] = round1(used / (1024 * 1024))
	facts["available"] = true
	facts["note"] = "adapter_ram is WMI-WORD-capped (~4GB) for >4GB cards; vram_used_bytes is exact. On AMD, Vulkan compute runs under the '3d' engine, so busiest_engine / total_util_pct track GPU work - compute_util_pct (engtype_Compute) reads ~0 even mid-decode."
	if engines := engineRows(facts["engines"]); len(engines) > 0 {
		sort.Slice(engines, func(i, j int) bool { return number(engines[i]["util_pct"]) > number(engines[j]["util_pct"]) })
		facts["busiest_engine"] = engines[0]["type"]
		facts["busiest_util_pct"] = engines[0]["util_pct"]
	}
	return facts
}

func SupportedPlatform() bool {
	return runtime.GOOS == "windows"
}

func engineRows(v any) []map[string]any {
	switch rows := v.(type) {
	case []any:
		out := make([]map[string]any, 0, len(rows))
		for _, row := range rows {
			if m, ok := row.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	case map[string]any:
		return []map[string]any{rows}
	default:
		return nil
	}
}

func number(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case int64:
		return float64(x)
	default:
		return 0
	}
}

func round1(v float64) float64 {
	if v >= 0 {
		return float64(int(v*10+0.5)) / 10
	}
	return float64(int(v*10-0.5)) / 10
}
