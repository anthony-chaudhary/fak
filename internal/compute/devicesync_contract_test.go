package compute

import (
	"errors"
	"strings"
	"testing"
)

// deviceSyncFakeAccelerator is a minimal stand-in for a device hot path. Its
// forward is the unit under contract: a compliant accelerator performs only
// device work, a regressed one performs a scalar host readback (the `.item()` /
// `.cpu()` equivalent) to branch on a value.
type deviceSyncFakeAccelerator struct {
	guard *DeviceSyncGuard
	// scalarReadback models a hot path that syncs to make a host decision.
	scalarReadback bool
}

// forward is the representative accelerator hot path the contract governs.
func (a *deviceSyncFakeAccelerator) forward(x []float32) float32 {
	acc := float32(0)
	for _, v := range x {
		acc += v
	}
	if a.scalarReadback {
		// The regression: pull a scalar host-ward mid-forward. Under the
		// contract this is a named violation, not an invisible latency hit.
		a.guard.Synced("scalar-readback", "branched on a device scalar mid-forward")
		if acc > 1e9 {
			return 0
		}
	}
	return acc
}

// TestDeviceSyncFreeForward is the contract witness: a sync-free accelerator
// forward passes the device-sync-error mode, and a deliberately
// sync-injecting variant fails it with the typed violation.
func TestDeviceSyncFreeForward(t *testing.T) {
	t.Run("sync-free forward passes the contract", func(t *testing.T) {
		g := NewDeviceSyncGuard("fake-accelerator/forward")
		acc := &deviceSyncFakeAccelerator{guard: g}

		violation, err := WithDeviceSyncContract(g, func() { _ = acc.forward([]float32{1, 2, 3, 4}) })
		if err != nil {
			t.Fatalf("WithDeviceSyncContract err = %v, want nil", err)
		}
		if violation != nil {
			t.Fatalf("sync-free forward violation = %v, want nil", violation)
		}
		if g.Violated() {
			t.Fatal("guard Violated() = true after a sync-free forward, want false")
		}
		if g.Armed() {
			t.Fatal("guard Armed() = true after the contract returned, want false (must disarm on every path)")
		}
	})

	t.Run("injected scalar sync fails the contract", func(t *testing.T) {
		g := NewDeviceSyncGuard("fake-accelerator/forward")
		acc := &deviceSyncFakeAccelerator{guard: g, scalarReadback: true}

		violation, err := WithDeviceSyncContract(g, func() { _ = acc.forward([]float32{1, 2, 3, 4}) })
		if err != nil {
			t.Fatalf("WithDeviceSyncContract err = %v, want nil (violation is returned separately)", err)
		}
		if violation == nil {
			t.Fatal("injected scalar sync violation = nil, want a typed DeviceSyncViolationError")
		}
		var typed *DeviceSyncViolationError
		if !errors.As(violation, &typed) {
			t.Fatalf("violation type = %T, want *DeviceSyncViolationError", violation)
		}
		if typed.Path != "fake-accelerator/forward" {
			t.Fatalf("violation Path = %q, want fake-accelerator/forward", typed.Path)
		}
		if typed.Site != "scalar-readback" {
			t.Fatalf("violation Site = %q, want scalar-readback", typed.Site)
		}
		if !strings.Contains(typed.Error(), "device-sync contract violation") {
			t.Fatalf("violation text = %q, want it to name the contract", typed.Error())
		}
	})
}

// TestDeviceSyncGuardDisarmedSyncedIsNoop pins the production posture: an
// instrumented backend that calls Synced when no contract is armed records
// nothing, so the same code runs on a prod path without a report.
func TestDeviceSyncGuardDisarmedSyncedIsNoop(t *testing.T) {
	g := NewDeviceSyncGuard("fake-accelerator/forward")
	g.Synced("scalar-readback", "no guard armed")
	if g.Violated() {
		t.Fatal("disarmed guard recorded a violation, want no-op")
	}
}

// TestDeviceSyncContractDisarmsOnPanic proves the guard cannot leak armed state
// into a following callable even when the governed forward panics.
func TestDeviceSyncContractDisarmsOnPanic(t *testing.T) {
	g := NewDeviceSyncGuard("fake-accelerator/forward")

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected the governed callable to re-panic")
			}
		}()
		_, _ = WithDeviceSyncContract(g, func() { panic("device fault") })
	}()

	if g.Armed() {
		t.Fatal("guard Armed() = true after a panicking callable, want false")
	}
}

// TestDeviceSyncContractNilGuardFailsClosed pins the fail-closed edge: a nil
// guard is a named error, never a silently-ungoverned forward.
func TestDeviceSyncContractNilGuardFailsClosed(t *testing.T) {
	_, err := WithDeviceSyncContract(nil, func() {})
	if err == nil {
		t.Fatal("WithDeviceSyncContract(nil, ...) err = nil, want a named refusal")
	}
}
