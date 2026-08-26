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

	"github.com/anthony-chaudhary/fak/internal/hostdiag"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func gatherHostdiagEvents(since time.Duration) ([]hostdiag.ResourceEvent, error) {
	script := strings.Replace(hostdiagEventPS, "__MILLIS__", strconv.FormatInt(since.Milliseconds(), 10), 1)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	windowgate.ConfigureBackgroundCommand(cmd)
	configureDispatchHelperCommand(cmd)
	output, err := cmd.Output()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, errors.New("Get-WinEvent resource diagnostics timed out after 30s")
	}
	if err != nil {
		return nil, fmt.Errorf("Get-WinEvent resource diagnostics: %w", err)
	}
	var events []hostdiag.ResourceEvent
	if err := json.Unmarshal(output, &events); err != nil {
		return nil, fmt.Errorf("parse resource diagnostics: %w", err)
	}
	for i := range events {
		if events[i].EventID == 2004 && events[i].Name == "LOW_VIRTUAL_MEMORY" {
			events[i].Culprits = hostdiag.ParseLowVirtualMemoryCulprits(events[i].Message)
		}
	}
	return events, nil
}

const hostdiagEventPS = `
$ErrorActionPreference='Stop'
$since=(Get-Date).AddMilliseconds(-__MILLIS__)
$rows=@()
$rows += @(Get-WinEvent -FilterHashtable @{LogName='Application';ProviderName='Windows Error Reporting';Id=1001;StartTime=$since} -ErrorAction SilentlyContinue | ForEach-Object {
  $msg=[string]$_.Message
  if($msg -match '(?im)^Event Name:\s*RADAR_PRE_LEAK_64\s*$' -and $msg -match '(?im)^P1:\s*fak\.exe\s*$'){
    $report=''; if($msg -match '(?im)^Report Id:\s*([^\r\n]+)'){ $report=$Matches[1].Trim() }
    [pscustomobject]@{time_ms=[int64]([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds();source=[string]$_.ProviderName;windows_event_id=[int]$_.Id;record_id=[string]$_.RecordId;event_name='RADAR_PRE_LEAK_64';report_id=$report;app='fak.exe';message=$msg}
  }
})
$rows += @(Get-WinEvent -FilterHashtable @{LogName='Application';ProviderName='Application Error';Id=1000;StartTime=$since} -ErrorAction SilentlyContinue | ForEach-Object {
  $xml=[xml]$_.ToXml(); $fields=@{}; foreach($d in $xml.Event.EventData.Data){ $fields[[string]$d.Name]=[string]$d.'#text' }
  $app=[string]$fields.AppName
  if($app -ieq 'pwsh.exe' -or $app -ieq 'powershell.exe' -or $app -ieq 'explorer.exe'){
    $fault=[pscustomobject]@{app_version=[string]$fields.AppVersion;module=[string]$fields.ModuleName;module_version=[string]$fields.ModuleVersion;exception_code=[string]$fields.ExceptionCode;fault_offset=[string]$fields.FaultingOffset}
    $pid=0; if([string]$fields.ProcessId -match '^0x([0-9a-f]+)$'){ $pid=[Convert]::ToInt32($Matches[1],16) }
    $created=0; if([string]$fields.ProcessCreationTime -match '^0x([0-9a-f]+)$'){ $created=[DateTimeOffset]::FromFileTime([Convert]::ToInt64($Matches[1],16)).ToUnixTimeMilliseconds() }
    if($app -ieq 'explorer.exe'){
      [pscustomobject]@{time_ms=[int64]([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds();source=[string]$_.ProviderName;windows_event_id=[int]$_.Id;record_id=[string]$_.RecordId;event_name='WINDOWS_SHELL_PROCESS_CRASH';report_id=[string]$fields.IntegratorReportId;app=$app;process_id=$pid;process_start_ms=$created;application_fault=$fault}
    } else {
      [pscustomobject]@{time_ms=[int64]([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds();source=[string]$_.ProviderName;windows_event_id=[int]$_.Id;record_id=[string]$_.RecordId;event_name='POWERSHELL_PROCESS_CRASH';report_id=[string]$fields.IntegratorReportId;app=$app;process_id=$pid;process_start_ms=$created;application_fault=$fault;message=[string]$_.Message}
    }
  }
})
$rows += @(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Resource-Exhaustion-Detector';Id=2004;StartTime=$since} -ErrorAction SilentlyContinue | ForEach-Object {
  $msg=[string]$_.Message
  [pscustomobject]@{time_ms=[int64]([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds();source=[string]$_.ProviderName;windows_event_id=[int]$_.Id;record_id=[string]$_.RecordId;event_name='LOW_VIRTUAL_MEMORY';report_id='';app='';message=$msg}
})
$rows += @(Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-Resource-Exhaustion-Resolver/Operational';Id=1014,1015;StartTime=$since} -ErrorAction SilentlyContinue | ForEach-Object {
  [pscustomobject]@{time_ms=[int64]([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds();source=[string]$_.ProviderName;windows_event_id=[int]$_.Id;record_id=[string]$_.RecordId;event_name=('RESOURCE_EXHAUSTION_'+[string]$_.Id);report_id='';app='';message=[string]$_.Message}
})
'['+(($rows|Sort-Object time_ms|ForEach-Object{$_|ConvertTo-Json -Compress}) -join ',')+']'
`
