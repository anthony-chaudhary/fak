//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
)

type windowsProcessRow struct {
	ProcessID       int    `json:"ProcessId"`
	ParentProcessID int    `json:"ParentProcessId"`
	CreationDate    string `json:"CreationDate"`
	CommandLine     string `json:"CommandLine"`
}

func collectBackgroundProcesses() ([]loopfleet.Process, error) {
	script := `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress`
	out, err := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script).Output()
	if err != nil {
		return nil, fmt.Errorf("process inventory: %w", err)
	}
	raw := strings.TrimSpace(string(out))
	if raw == "" || raw == "null" {
		return nil, nil
	}
	var rows []windowsProcessRow
	if strings.HasPrefix(raw, "[") {
		err = json.Unmarshal([]byte(raw), &rows)
	} else {
		var row windowsProcessRow
		err = json.Unmarshal([]byte(raw), &row)
		rows = []windowsProcessRow{row}
	}
	if err != nil {
		return nil, fmt.Errorf("decode process inventory: %w", err)
	}
	got := make([]loopfleet.Process, 0, len(rows))
	for _, row := range rows {
		started, _ := time.Parse(time.RFC3339Nano, row.CreationDate)
		got = append(got, loopfleet.Process{PID: row.ProcessID, ParentPID: row.ParentProcessID, StartedAt: started, Command: row.CommandLine})
	}
	return got, nil
}
