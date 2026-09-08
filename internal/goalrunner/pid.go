package goalrunner

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// IsProcessLive checks if the process with the given PID is currently alive.
func IsProcessLive(pid int) bool {
	return isProcessLive(pid)
}

// SweepDeadPidBreadcrumbs scans .goal-runs/*.pid in the workspace, removes dead PID breadcrumbs,
// preserves live ones, and returns the slice of swept dead PIDs.
func SweepDeadPidBreadcrumbs(workspace string) ([]int, error) {
	if workspace == "" {
		var err error
		workspace, err = os.Getwd()
		if err != nil {
			return nil, err
		}
	}

	logDir := filepath.Join(workspace, ".goal-runs")
	if filepath.Base(workspace) == ".goal-runs" {
		logDir = workspace
	}

	entries, err := os.ReadDir(logDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var sweptPIDs []int
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(entry.Name(), ".pid") {
			continue
		}

		pidFile := filepath.Join(logDir, entry.Name())
		data, err := os.ReadFile(pidFile)
		if err != nil {
			_ = os.Remove(pidFile)
			continue
		}

		raw := strings.TrimSpace(string(data))
		pid, err := strconv.Atoi(raw)
		if err != nil || pid <= 0 {
			// Not a valid PID breadcrumb; remove dead/corrupt entry
			_ = os.Remove(pidFile)
			continue
		}

		if IsProcessLive(pid) {
			// Process is live, preserve breadcrumb
			continue
		}

		// Process is dead, sweep it
		if err := os.Remove(pidFile); err == nil {
			sweptPIDs = append(sweptPIDs, pid)
		}
	}

	return sweptPIDs, nil
}
