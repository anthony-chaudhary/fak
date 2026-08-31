//go:build !windows

package windowgate

// RestoreTerminalWindow is a no-op off Windows. POSIX terminals do not have the
// Windows minimize/focus behavior this hook repairs.
func RestoreTerminalWindow() bool {
	return false
}

// TerminalRestore is a no-op capture off Windows.
type TerminalRestore struct{}

func CaptureTerminalRestore() TerminalRestore { return TerminalRestore{} }

func (TerminalRestore) RepairAfterStart() bool { return false }
