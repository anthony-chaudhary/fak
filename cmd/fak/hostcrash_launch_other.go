//go:build !windows

package main

import (
	"errors"
	"os"
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

func launchHostSessionPlatform(req hostresurrect.Request) (int, error) {
	if len(req.Command) == 0 {
		return 0, errors.New("empty relaunch command")
	}
	cmd := exec.Command(req.Command[0], req.Command[1:]...)
	cmd.Dir = req.CWD
	cmd.Env = append(os.Environ(), "FAK_RESUME_HANDLE="+req.ResumeHandle, "FAK_HOST_CRASH_EVENT="+req.EventID)
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	return cmd.Process.Pid, nil
}
