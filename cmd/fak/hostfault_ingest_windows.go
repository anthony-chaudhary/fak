//go:build windows

package main

// hostfault_ingest_windows.go — the Windows OS-reading layer for
// `fak toolproc host-faults --ingest`. It reads TWO event-log surfaces in one
// bounded PowerShell call (the stallscan shell-out idiom):
//
//   - System log, Microsoft-Windows-WindowsUpdateClient Event 20 — update
//     install failures (0x80073D02 package-in-use, etc.),
//   - Application log, Windows Error Reporting Event 1001 — pre-filtered to the
//     live-kernel / app-term / update-orchestrator signatures, because a raw WER
//     1001 stream is dominated by unrelated app crashes and GPU events run to
//     thousands.
//
// The call is -NonInteractive by design (an event-log read must never itself
// become an interactive console host). -MaxEvents bounds each source.

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostfault"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// winHostRecord is the JSON shape the probe emits per event-log record.
type winHostRecord struct {
	Provider string `json:"provider"`
	ID       int    `json:"id"`
	TimeMS   int64  `json:"time_ms"`
	App      string `json:"app"`
	Message  string `json:"message"`
}

// gatherWinHostFaultRecords collects the host-fault-relevant records from the
// last `since` window, at most maxPerSource per source. Returns (records, "") on
// success or (nil, errmsg) on a probe failure; an empty log is (nil, "").
func gatherWinHostFaultRecords(since time.Duration, maxPerSource int) ([]hostfault.WinFaultRecord, string) {
	days := int(since.Hours()/24) + 1
	if days < 1 {
		days = 1
	}
	if maxPerSource < 1 {
		maxPerSource = 1
	}
	script := strings.NewReplacer(
		"__DAYS__", fmt.Sprintf("%d", days),
		"__MAX__", fmt.Sprintf("%d", maxPerSource),
	).Replace(hostFaultIngestPS)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	windowgate.ConfigureBackgroundCommand(cmd)
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Sprintf("Get-WinEvent: %v", err)
	}
	trimmed := strings.TrimSpace(string(out))
	if trimmed == "" || trimmed == "[]" {
		return nil, ""
	}
	var recs []winHostRecord
	if err := json.Unmarshal([]byte(trimmed), &recs); err != nil {
		return nil, fmt.Sprintf("parse Get-WinEvent JSON: %v", err)
	}
	rows := make([]hostfault.WinFaultRecord, 0, len(recs))
	for _, r := range recs {
		rows = append(rows, hostfault.WinFaultRecord{
			Provider: r.Provider,
			ID:       r.ID,
			TimeMS:   r.TimeMS,
			App:      r.App,
			Message:  r.Message,
		})
	}
	return rows, ""
}

// hostFaultIngestPS is the self-contained probe. It emits a JSON array by hand so
// the shape is identical for 0, 1, or N rows across Windows PowerShell 5.1 and
// pwsh 7.x. __DAYS__ / __MAX__ are substituted by the Go caller.
const hostFaultIngestPS = `
$ErrorActionPreference='SilentlyContinue'
$since=(Get-Date).AddDays(-__DAYS__)
$rows=@()
function Add-Row($e,$app){
  $ms=[int64]([DateTimeOffset]$e.TimeCreated).ToUnixTimeMilliseconds()
  $script:rows += [pscustomobject]@{provider=[string]$e.ProviderName; id=[int]$e.Id; time_ms=$ms; app=[string]$app; message=[string]$e.Message}
}
# Windows Update install failures (System log).
$wu = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WindowsUpdateClient'; Id=20; StartTime=$since} -MaxEvents __MAX__ -ErrorAction SilentlyContinue
foreach($e in $wu){ Add-Row $e '' }
# WER 1001, pre-filtered to the host-fault signatures (Application log).
$wer = Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='Windows Error Reporting'; Id=1001; StartTime=$since} -MaxEvents __MAX__ -ErrorAction SilentlyContinue
foreach($e in $wer){
  $app = if($e.Properties.Count -gt 5){ [string]$e.Properties[5].Value } else { '' }
  if(($e.Message -match 'LiveKernelEvent|AppTermFailureEvent|MoUpdateOrchestrator|MoUsoCoreWorker') -or ($app -match 'MoUpdateOrchestrator|MoUsoCoreWorker')){
    Add-Row $e $app
  }
}
'[' + (($rows | ForEach-Object { $_ | ConvertTo-Json -Depth 3 -Compress }) -join ',') + ']'
`
