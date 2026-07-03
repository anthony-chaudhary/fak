//go:build windows

package procguard

import (
	"os/exec"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ConfigureProcessTreeCancel wires Cancel to taskkill /T so Windows launchers
// reap grandchildren as well as the direct child when a timeout fires.
func ConfigureProcessTreeCancel(cmd *exec.Cmd) {
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		pid := strconv.Itoa(cmd.Process.Pid)
		kill := exec.Command("taskkill", "/T", "/F", "/PID", pid)
		windowgate.ConfigureBackgroundCommand(kill)
		if err := kill.Run(); err != nil {
			return cmd.Process.Kill()
		}
		return nil
	}
}
