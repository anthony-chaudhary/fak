//go:build !windows

package windowgate

import "time"

// RestoreTerminalWindow is a no-op off Windows. POSIX terminals do not have the
// Windows minimize/focus behavior this hook repairs.
func RestoreTerminalWindow() bool {
	return false
}

// StartTerminalRestorePulse is a no-op off Windows.
func StartTerminalRestorePulse(duration, interval time.Duration) {
	_, _ = duration, interval
}
