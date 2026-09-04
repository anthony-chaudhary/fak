//go:build windows

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostfault"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func runHostEventScript(since time.Duration, psTemplate, label string) ([]byte, error) {
	millis := since.Milliseconds()
	script := strings.Replace(psTemplate, "__MILLIS__", strconv.FormatInt(millis, 10), 1)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	windowgate.ConfigureBackgroundCommand(cmd)
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, fmt.Errorf("Get-WinEvent %s: timed out after 30s", label)
	}
	if err != nil {
		return nil, fmt.Errorf("Get-WinEvent %s: %w", label, err)
	}
	return out, nil
}

func parseAndFilterEvents[T any](out []byte, label string, since time.Duration, getTimeMS func(T) int64) ([]T, error) {
	var events []T
	if err := json.Unmarshal(out, &events); err != nil {
		return nil, fmt.Errorf("parse %s JSON: %w", label, err)
	}
	cutoff := time.Now().Add(-since).UnixMilli()
	kept := events[:0]
	for _, event := range events {
		if getTimeMS(event) >= cutoff {
			kept = append(kept, event)
		}
	}
	return kept, nil
}

func gatherHostCrashEvents(since time.Duration) ([]hostfault.ApplicationError1000, error) {
	out, err := runHostEventScript(since, hostCrashEventPS, "Event 1000")
	if err != nil {
		return nil, err
	}
	return parseAndFilterEvents(out, "Event 1000", since, func(e hostfault.ApplicationError1000) int64 { return e.TimeMS })
}

func gatherHostSystemEvents(since time.Duration) ([]hostfault.WindowsSystemEvent, error) {
	out, err := runHostEventScript(since, hostSystemEventPS, "System incidents")
	if err != nil {
		return nil, err
	}
	return parseAndFilterEvents(out, "System incidents", since, func(e hostfault.WindowsSystemEvent) int64 { return e.TimeMS })
}

const hostSystemEventPS = `
$ErrorActionPreference='Stop'
$since=(Get-Date).AddMilliseconds(-__MILLIS__)
$rows=@(Get-WinEvent -FilterHashtable @{LogName='System';Id=41,6008,1001;StartTime=$since} -ErrorAction SilentlyContinue | ForEach-Object {
  $provider=[string]$_.ProviderName; $id=[int]$_.Id
  if(($id -eq 41 -and $provider -eq 'Microsoft-Windows-Kernel-Power') -or ($id -eq 6008 -and $provider -eq 'EventLog') -or ($id -eq 1001 -and $provider -eq 'Microsoft-Windows-WER-SystemErrorReporting')){
    $msg=[string]$_.Message; $code=''; $params=@(); $dump=''; $report=''
    if($id -eq 1001){
      $p=@($_.Properties | ForEach-Object { [string]$_.Value })
      if($p.Count -ge 1 -and $p[0] -match '^\s*(0x[0-9a-fA-F]+)\s*\(([^)]*)\)'){ $code=$Matches[1]; $params=@($Matches[2] -split '\s*,\s*') }
      if($p.Count -ge 2){ $dump=$p[1].Trim() }
      if($p.Count -ge 3){ $report=$p[2].Trim() }
    }
    [pscustomobject]@{time_ms=[int64]([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds();source=$provider;windows_event_id=$id;record_id=[string]$_.RecordId;bugcheck_code=$code;parameters=$params;dump_path=$dump;report_id=$report;message=$msg}
  }
})
'['+(($rows|ForEach-Object{$_|ConvertTo-Json -Compress}) -join ',')+']'
`

const hostCrashEventPS = `
$ErrorActionPreference='Stop'
$since=(Get-Date).AddMilliseconds(-__MILLIS__)
$rows=@(@(Get-WinEvent -FilterHashtable @{LogName='Application';ProviderName='Application Error';Id=1000;StartTime=$since} -ErrorAction SilentlyContinue) | ForEach-Object {
  $p=$_.Properties
  if($p.Count -ge 8){
    $app=[string]$p[0].Value
    if($app -match '(?i)^(WindowsTerminal|svchost|OpenConsole|pwsh|powershell)\.exe$'){
      [pscustomobject]@{time_ms=[int64]([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds();app=$app;module=[string]$p[3].Value;exception=[string]$p[6].Value;fault_offset=[string]$p[7].Value;process_id=if($p.Count -gt 8){[string]$p[8].Value}else{''};report_id=if($p.Count -gt 12){[string]$p[12].Value}else{[string]$_.RecordId}}
    }
  }
})
'['+(($rows|ForEach-Object{$_|ConvertTo-Json -Compress}) -join ',')+']'
`
