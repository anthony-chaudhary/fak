//go:build darwin

package macobs

import (
	"context"
	"os/exec"
)

// DefaultCommandRunner executes commands via os/exec.
func DefaultCommandRunner(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	return cmd.Output()
}

// CollectHardware collects Apple Silicon Mac hardware metrics using host tools.
func CollectHardware(ctx context.Context) HardwareTelemetry {
	return CollectHardwareWithRunner(ctx, DefaultCommandRunner)
}
