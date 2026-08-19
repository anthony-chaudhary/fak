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

func gatherHostCrashEvents(since time.Duration) ([]hostfault.ApplicationError1000, error) {
	millis := since.Milliseconds()
	script := strings.Replace(hostCrashEventPS, "__MILLIS__", strconv.FormatInt(millis, 10), 1)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	windowgate.ConfigureBackgroundCommand(cmd)
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, fmt.Errorf("Get-WinEvent Event 1000: timed out after 30s")
	}
	if err != nil {
		return nil, fmt.Errorf("Get-WinEvent Event 1000: %w", err)
	}
	var events []hostfault.ApplicationError1000
	if err := json.Unmarshal(out, &events); err != nil {
		return nil, fmt.Errorf("parse Event 1000 JSON: %w", err)
	}
	cutoff := time.Now().Add(-since).UnixMilli()
	kept := events[:0]
	for _, event := range events {
		if event.TimeMS >= cutoff {
			kept = append(kept, event)
		}
	}
	return kept, nil
}

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
