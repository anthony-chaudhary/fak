//go:build !windows

package procguard

func yieldWorkingSets(pids ...int) {
	// Non-Windows platforms rely on runtime.GC() and debug.FreeOSMemory()
	// already executed by YieldMemory.
}
