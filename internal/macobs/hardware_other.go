//go:build !darwin

package macobs

import "context"

// DefaultCommandRunner is a no-op on non-Darwin platforms.
func DefaultCommandRunner(ctx context.Context, name string, args ...string) ([]byte, error) {
	return nil, nil
}

// CollectHardware returns an unavailable hardware telemetry record on non-Darwin platforms.
func CollectHardware(ctx context.Context) HardwareTelemetry {
	return HardwareTelemetry{
		Available:    false,
		ThermalState: ThermalUnknown,
		PowerSource:  PowerUnknown,
	}
}
