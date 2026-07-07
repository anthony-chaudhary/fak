//go:build windows

package main

import (
	"os"
	"testing"
)

// TestDispatchPIDAliveSelf proves the zero-spawn liveness check reports the
// current process as alive (OpenProcess succeeds, GetExitCodeProcess == STILL_ACTIVE).
// This is the contract dispatch_tick_test.go relies on when it uses our own pid
// as a stand-in for a recycled, still-live worker pid.
func TestDispatchPIDAliveSelf(t *testing.T) {
	if !dispatchPIDAlive(os.Getpid()) {
		t.Fatalf("dispatchPIDAlive(self) = false, want true")
	}
}

// TestDispatchPIDAliveDead proves a pid that cannot be a live process reports
// not-alive rather than spawning anything. pid 0 and negatives are rejected up
// front; a very high, almost-certainly-unallocated pid exercises the OpenProcess
// failure path.
func TestDispatchPIDAliveDead(t *testing.T) {
	if dispatchPIDAlive(0) {
		t.Fatalf("dispatchPIDAlive(0) = true, want false")
	}
	if dispatchPIDAlive(-1) {
		t.Fatalf("dispatchPIDAlive(-1) = true, want false")
	}
	// A pid well above any realistic live pid. If by rare chance it is live, the
	// check is still correct — we only assert the call does not panic and returns
	// a bool cheaply (no spawn). Not asserting false to avoid a flaky pid clash.
	_ = dispatchPIDAlive(0x7ffffff0)
}
