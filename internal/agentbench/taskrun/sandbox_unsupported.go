//go:build !darwin && !linux

package taskrun

import (
	"context"
	"os/exec"
	"runtime"
)

func platformSandboxAvailable() error {
	return &SandboxUnavailableError{Reason: "unsupported operating system " + runtime.GOOS}
}

func platformSandboxName() string { return "unsupported" }

func platformSandboxCommand(context.Context, sandboxCommandConfig) (*exec.Cmd, error) {
	return nil, platformSandboxAvailable()
}
