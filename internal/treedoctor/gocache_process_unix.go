//go:build !windows

package treedoctor

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
)

func ActiveGoBuild() (bool, error) {
	out, err := exec.Command("ps", "-axo", "comm=").Output()
	if err != nil {
		return false, fmt.Errorf("list processes: %w", err)
	}
	for _, line := range strings.Split(string(out), "\n") {
		name := strings.ToLower(filepath.Base(strings.TrimSpace(line)))
		switch name {
		case "go", "compile", "link", "asm", "cgo":
			return true, nil
		}
	}
	return false, nil
}
