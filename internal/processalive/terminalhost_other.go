//go:build !windows

package processalive

// TerminalHostPID is Windows-only; other platforms fail closed.
func TerminalHostPID(int) (int, bool) { return 0, false }
