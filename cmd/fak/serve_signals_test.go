package main

import (
	"os"
	"syscall"
	"testing"
)

func signalsContain(sigs []os.Signal, target os.Signal) bool {
	for _, s := range sigs {
		if s == target {
			return true
		}
	}
	return false
}

// #1359: the serve graceful-shutdown handler must drain on an orchestrator SIGTERM and on
// Ctrl-C (SIGINT/os.Interrupt), on every platform — not only the original Ctrl-C. (The
// SIGHUP "terminal closed" case is asserted separately on non-Windows, where the OS has it.)
func TestTerminatingSignals_alwaysIncludesInterruptAndTerm(t *testing.T) {
	sigs := terminatingSignals()
	if !signalsContain(sigs, os.Interrupt) {
		t.Errorf("terminatingSignals must include os.Interrupt (Ctrl-C); got %v", sigs)
	}
	if !signalsContain(sigs, syscall.SIGTERM) {
		t.Errorf("terminatingSignals must include SIGTERM (orchestrator stop); got %v", sigs)
	}
}
