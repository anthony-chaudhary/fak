//go:build windows

package codexresume

import (
	"fmt"
	"os/exec"
)

func configureOwnedProcess(cmd *exec.Cmd) error { return nil }
func killOwnedProcessTree(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	killer := exec.Command("taskkill", "/PID", fmt.Sprint(cmd.Process.Pid), "/T", "/F")
	if err := killer.Run(); err != nil {
		return cmd.Process.Kill()
	}
	return nil
}
