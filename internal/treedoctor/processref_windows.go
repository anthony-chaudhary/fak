//go:build windows

package treedoctor

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"syscall"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func listGoTmpProcesses() ([]GoTmpProcess, error) {
	const script = `$ErrorActionPreference='Stop'; ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine,ExecutablePath)`
	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)
	windowgate.ConfigureBackgroundCommand(cmd)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("enumerate Win32_Process CommandLine and ExecutablePath: %w", err)
	}
	var rows []struct {
		ProcessID      int     `json:"ProcessId"`
		CommandLine    *string `json:"CommandLine"`
		ExecutablePath *string `json:"ExecutablePath"`
	}
	if err := json.Unmarshal(output, &rows); err != nil {
		return nil, fmt.Errorf("decode Win32_Process snapshot: %w", err)
	}
	processes := make([]GoTmpProcess, 0, len(rows))
	for _, row := range rows {
		process := GoTmpProcess{PID: row.ProcessID}
		if row.CommandLine != nil {
			process.CommandLine = *row.CommandLine
		}
		if row.ExecutablePath != nil {
			process.ExecutablePath = *row.ExecutablePath
		}
		processes = append(processes, process)
	}
	sort.Slice(processes, func(i, j int) bool { return processes[i].PID < processes[j].PID })
	return processes, nil
}

func goTmpIsReparse(path string, info os.FileInfo) (bool, error) {
	ptr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return false, err
	}
	attrs, err := syscall.GetFileAttributes(ptr)
	if err != nil {
		return false, err
	}
	return attrs&syscall.FILE_ATTRIBUTE_REPARSE_POINT != 0, nil
}
