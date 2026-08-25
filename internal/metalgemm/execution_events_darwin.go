//go:build darwin && arm64 && cgo

package metalgemm

func executionEventsAvailable() bool { return true }
