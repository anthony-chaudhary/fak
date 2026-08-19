//go:build windows

package accounts

import (
	"os/exec"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// setRefreshSysProcAttr is a no-op on Windows: containment comes from the
// process-tree walk in refreshKillTree, not a process group.
func setRefreshSysProcAttr(*exec.Cmd) {}

// refreshKillTree terminates pid's whole process tree via taskkill /T /F — the
// same lever procguard.KillPID falls back to. Copied rather than imported:
// procguard -> dispatchtick -> accounts is a committed import cycle (#3105).
func refreshKillTree(pid int) error {
	if pid <= 0 {
		return nil
	}
	cmd := exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T", "/F")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Run()
}
