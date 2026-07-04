//go:build windows

package procguard

import "os/exec"

// ConfigureProcessTreeCancel wires Cancel to the native process-tree reaper so
// Windows launchers reap grandchildren as well as the direct child when a timeout
// fires, without taskkill /T's WMI-backed process-tree walk.
func ConfigureProcessTreeCancel(cmd *exec.Cmd) {
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		if ok, _ := KillPID(cmd.Process.Pid); !ok {
			return cmd.Process.Kill()
		}
		return nil
	}
}
