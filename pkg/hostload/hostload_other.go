//go:build !windows

package hostload

func readWindows() (float64, bool) { return 0, false }
