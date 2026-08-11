//go:build windows

package codexresume

import (
	"fmt"
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func configureOwnedProcess(cmd *exec.Cmd) error {
	windowgate.ConfigureBackgroundCommand(cmd)
	return nil
}
func killOwnedProcessTree(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	killer := windowgate.Command("taskkill", "/PID", fmt.Sprint(cmd.Process.Pid), "/T", "/F")
	if err := killer.Run(); err != nil {
		return cmd.Process.Kill()
	}
	return nil
}
