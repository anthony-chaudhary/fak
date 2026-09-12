//go:build windows

package taskrun

import "os/exec"

func configureChildProcess(_ *exec.Cmd) {}

func killChildProcess(cmd *exec.Cmd) {
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}
