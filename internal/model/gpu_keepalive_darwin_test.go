//go:build darwin && arm64 && cgo

package model

import (
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// fak-test:runtime medium est=2s lane=optin
func TestSessionGPUKeepAliveQ4Reachability(t *testing.T) {
	if testing.Short() || os.Getenv("FAK_METAL_KEEPALIVE_TEST") != "1" {
		t.Skip("physical session keepalive smoke requires FAK_METAL_KEEPALIVE_TEST=1 without -short")
	}
	lease, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if !metalgemm.Available() {
		t.Fatal("physical Metal device required")
	}
	assertInactive := func(before metalgemm.KeepAliveReceipt) {
		t.Helper()
		after := metalgemm.KeepAliveState()
		if after.SubmittedBuffers != before.SubmittedBuffers || after.Holders != 0 || after.InFlight != 0 {
			t.Fatalf("ineligible scope submitted GPU work: before=%+v after=%+v", before, after)
		}
	}
	t.Setenv("FAK_METAL_KEEPALIVE", "on")
	for _, session := range []*Session{nil, {}} {
		before := metalgemm.KeepAliveState()
		stop := session.BeginGPUKeepAlive()
		stop()
		assertInactive(before)
	}
	// Resident Q4 uses MetalQ4K independently of the f16 Metal flag. Exercise the
	// real session hook with that exact flag combination without allocating a model.
	q4 := &Session{MetalQ4K: true}
	t.Setenv("FAK_METAL_KEEPALIVE", "off")
	before := metalgemm.KeepAliveState()
	stopOff := q4.BeginGPUKeepAlive()
	stopOff()
	assertInactive(before)
	t.Setenv("FAK_METAL_KEEPALIVE", "on")
	stop := q4.BeginGPUKeepAlive()
	defer stop()
	active := metalgemm.KeepAliveState()
	if active.SubmittedBuffers <= before.SubmittedBuffers || active.Holders != 1 || active.InFlight > 2 || active.Failed {
		t.Fatalf("MetalQ4K session did not invoke bounded GPU keepalive: before=%+v active=%+v", before, active)
	}
	stop()
	deadline := time.Now().Add(time.Second)
	for {
		state := metalgemm.KeepAliveState()
		if state.Holders == 0 && state.InFlight == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("session keepalive did not drain: %+v", state)
		}
		time.Sleep(time.Millisecond)
	}
}
