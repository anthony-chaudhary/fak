//go:build windows

package main

// consolefault_ingest_windows.go — the Windows OS-reading layer for
// `fak toolproc console-faults --ingest`. It reads the Application event log for
// two crash shapes via bounded PowerShell Get-WinEvent calls (the stallscan
// shell-out idiom): the .NET Runtime unhandled-exception dumps (Event 1026) that
// carry a console-host managed stack, and the WER Application Error banners
// (Event 1000) that carry a console-host __fastfail / FailFast (the 1026-less
// class, #3513). The call is -NonInteractive by design: the very finding this
// wiring serves is that only an INTERACTIVE console host enters the crashing
// InputLoop, so the reader must never itself become one.

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// gatherWinConsoleFaultRecords collects the console-host-relevant crash records
// from the last `since` window: .NET Runtime Event 1026 dumps AND WER Application
// Error Event 1000 console-host FailFasts and Windows Terminal renderer exits,
// in one merged record array. Returns (records, "")
// on success or (nil, errmsg) on a probe failure; an empty log is (nil, "").
func gatherWinConsoleFaultRecords(since time.Duration) ([]winEventRecord, string) {
	days := int(since.Hours()/24) + 1
	if days < 1 {
		days = 1
	}
	script := strings.Replace(consoleFaultIngestPS, "__DAYS__", fmt.Sprintf("%d", days), 1)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	// Get-WinEvent returns an empty stream under DETACHED_PROCESS on this host.
	// CREATE_NO_WINDOW keeps the probe invisible while retaining the console
	// semantics Windows PowerShell needs to initialize its event-log pipeline.
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Sprintf("Get-WinEvent: %v", err)
	}
	trimmed := strings.TrimSpace(string(out))
	if trimmed == "" || trimmed == "[]" {
		return nil, ""
	}
	var recs []winEventRecord
	if err := json.Unmarshal([]byte(trimmed), &recs); err != nil {
		return nil, fmt.Sprintf("parse Get-WinEvent JSON: %v", err)
	}
	return recs, ""
}

// consoleFaultIngestPS is the self-contained probe. It runs two cheap
// pre-filters in PowerShell (before marshalling) and emits ONE JSON array by
// hand so the shape is identical for 0, 1, or N rows across both Windows
// PowerShell 5.1 and pwsh 7.x (ConvertTo-Json alone would emit a bare object for
// a single row). The Go mapper (consoleFaultEventsFromWinRecords) is the
// authoritative classifier; these filters only bound volume, so they are the
// same shape the Go side accepts and never drop a row Go would keep:
//   - Event 1026 (.NET Runtime): console-host managed-stack signatures.
//   - Event 1000 (Application Error): a console-host/shell faulting app AND the
//     FailFast exception code (0xc0000409) — the 1026-less class (#3513).
//
// __DAYS__ is substituted by the Go caller.
const consoleFaultIngestPS = `
$ErrorActionPreference='SilentlyContinue'
$since=(Get-Date).AddDays(-__DAYS__)
$rows = @()
$dn = Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='.NET Runtime'; Id=1026; StartTime=$since} -ErrorAction SilentlyContinue
foreach($e in $dn){
  if($e.Message -match 'ConsoleHost|PSReadLine|ReadKeyThreadProc|Cannot read keys|GetConsoleScreenBufferInfo|HostException'){
    $ms=[int64]([DateTimeOffset]$e.TimeCreated).ToUnixTimeMilliseconds()
    $rows += [pscustomobject]@{provider=[string]$e.ProviderName; id=[int]$e.Id; time_ms=$ms; message=[string]$e.Message}
  }
}
$wer = Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='Application Error'; Id=1000; StartTime=$since} -ErrorAction SilentlyContinue
foreach($e in $wer){
  if((($e.Message -match '(?i)pwsh\.exe|powershell\.exe|conhost\.exe|OpenConsole\.exe|cmd\.exe') -and ($e.Message -match '(?i)0xc0000409')) -or ($e.Message -match '(?i)Faulting application name:\s*WindowsTerminal\.exe')){
    $ms=[int64]([DateTimeOffset]$e.TimeCreated).ToUnixTimeMilliseconds()
    $rows += [pscustomobject]@{provider=[string]$e.ProviderName; id=[int]$e.Id; time_ms=$ms; message=[string]$e.Message}
  }
}
'[' + (($rows | ForEach-Object { $_ | ConvertTo-Json -Depth 3 -Compress }) -join ',') + ']'
`
