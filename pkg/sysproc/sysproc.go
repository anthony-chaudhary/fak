package sysproc

import (
	"context"
	"os/exec"
)

// Command returns an *exec.Cmd configured for background execution with
// console window suppression on Windows.
func Command(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)
	ConfigureBackground(cmd)
	return cmd
}

// CommandContext returns an *exec.Cmd with context cancellation configured
// for background execution with console window suppression on Windows.
func CommandContext(ctx context.Context, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...)
	ConfigureBackground(cmd)
	return cmd
}
