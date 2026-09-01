//go:build windows

package treedoctor

import (
	"encoding/csv"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

func ActiveGoBuild() (bool, error) {
	out, err := exec.Command("tasklist", "/FO", "CSV", "/NH").Output()
	if err != nil {
		return false, fmt.Errorf("list processes: %w", err)
	}
	records, err := csv.NewReader(strings.NewReader(string(out))).ReadAll()
	if err != nil {
		return false, fmt.Errorf("parse process list: %w", err)
	}
	for _, record := range records {
		if len(record) == 0 {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(record[0])) {
		case "go.exe", "compile.exe", "link.exe", "asm.exe", "cgo.exe":
			return true, nil
		}
	}
	return false, nil
}

func goCacheProcessAlive(pid int) (bool, error) {
	if pid <= 0 {
		return false, fmt.Errorf("invalid pid %d", pid)
	}
	out, err := exec.Command("tasklist", "/FI", "PID eq "+strconv.Itoa(pid), "/FO", "CSV", "/NH").Output()
	if err != nil {
		return false, fmt.Errorf("query pid %d: %w", pid, err)
	}
	records, err := csv.NewReader(strings.NewReader(string(out))).ReadAll()
	if err != nil {
		return false, fmt.Errorf("parse pid %d: %w", pid, err)
	}
	for _, record := range records {
		if len(record) < 2 {
			continue
		}
		got, parseErr := strconv.Atoi(strings.TrimSpace(record[1]))
		if parseErr == nil && got == pid {
			return true, nil
		}
	}
	return false, nil
}
