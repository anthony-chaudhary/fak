//go:build windows

package tempartifact

import (
	"context"
	"encoding/json"
	"os/exec"
	"syscall"
)

type windowsProcess struct {
	ExecutablePath string `json:"ExecutablePath"`
	CommandLine    string `json:"CommandLine"`
}

func inspectLiveProcessPaths(ctx context.Context, candidates []string) Inspection {
	if len(candidates) == 0 {
		return Inspection{Complete: true, References: map[string]bool{}}
	}
	const script = `$ErrorActionPreference='Stop'; ConvertTo-Json -Compress -Depth 3 -InputObject @(Get-CimInstance Win32_Process | Select-Object ExecutablePath,CommandLine)`
	command := exec.CommandContext(ctx, "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script)
	configureDispatchHelperCommand(command)
	output, err := command.Output()
	if err != nil {
		return Inspection{Reason: ReasonInspectionUnavailable, References: map[string]bool{}}
	}
	var rows []windowsProcess
	if err := json.Unmarshal(output, &rows); err != nil {
		return Inspection{Reason: ReasonInspectionUnavailable, References: map[string]bool{}}
	}
	records := make([]processRecord, 0, len(rows))
	for _, row := range rows {
		records = append(records, processRecord{ExecutablePath: row.ExecutablePath, CommandLine: row.CommandLine})
	}
	return Inspection{Complete: true, References: referencesFromProcessRecords(records, candidates)}
}

func configureDispatchHelperCommand(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
}
