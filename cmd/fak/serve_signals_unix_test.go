//go:build !windows

package main

import (
	"syscall"
	"testing"
)

// #1359: on a real OS, closing the controlling terminal sends SIGHUP — the most common
// "I accidentally closed it" case. It must drain + flush like Ctrl-C, so it has to be in
// the registered set. (Windows has no SIGHUP and never delivers it, so this is non-Windows
// only; the cross-platform SIGINT/SIGTERM coverage lives in serve_signals_test.go.)
func TestTerminatingSignals_includesHUPonUnix(t *testing.T) {
	if !signalsContain(terminatingSignals(), syscall.SIGHUP) {
		t.Errorf("non-Windows terminatingSignals must include SIGHUP (terminal close); got %v", terminatingSignals())
	}
}
