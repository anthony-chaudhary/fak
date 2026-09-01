//go:build windows

package treedoctor

import (
	"encoding/csv"
	"fmt"
	"os/exec"
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
