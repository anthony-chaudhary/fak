//go:build windows

package main

import (
	"errors"
	"os"
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

// launchHostSessionPlatform asks Windows Terminal to create a fresh window/tab.
// The S4U watchdog owns this call, so the launcher survives an RDP session teardown;
// wt.exe is only the replaceable presentation adapter above that control plane.
var hostSessionExecCommand = exec.Command

func launchHostSessionPlatform(req hostresurrect.Request) (int, error) {
	if len(req.Command) == 0 {
		return 0, errors.New("empty relaunch command")
	}
	args := []string{"-w", "new", "new-tab", "-d", req.CWD, req.Command[0]}
	args = append(args, req.Command[1:]...)
	cmd := hostSessionExecCommand("wt.exe", args...)
	cmd.Env = append(os.Environ(), "FAK_RESUME_HANDLE="+req.ResumeHandle, "FAK_HOST_CRASH_EVENT="+req.EventID)
	configureDispatchHelperCommand(cmd)
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	return cmd.Process.Pid, nil
}
