package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestDeviceFaultLatchClearAllocationSuspectOOMRetention(t *testing.T) {
	t.Run("allocation-only reopens without spending reconstruction budget", func(t *testing.T) {
		l := NewDeviceFaultLatch("cuda", 1)
		l.Observe(DeviceFaultAllocation, "scratch-a", 0)
		l.Observe(DeviceFaultAllocation, "scratch-b", 0)
		probes := 0
		if err := l.ClearAllocationSuspect("release", func() error { probes++; return nil }); err != nil {
			t.Fatalf("allocation-only clear refused: %v", err)
		}
		if probes != 1 || l.Admit("next") != nil {
			t.Fatalf("probes %d admit %v, want one probe and admission", probes, l.Admit("next"))
		}
		// The budget of 1 is untouched, so later allocation misses can clear again.
		l.Observe(DeviceFaultAllocation, "scratch-c", 0)
		if err := l.ClearAllocationSuspect("release", func() error { return nil }); err != nil {
			t.Fatalf("second allocation-only clear refused: %v", err)
		}
		if s := l.Snapshot(); s.Attempts != 0 || s.Recoveries != 2 || s.Health != DeviceHealthy {
			t.Fatalf("snapshot attempts/recoveries/health = %d/%d/%s, want 0/2/healthy", s.Attempts, s.Recoveries, s.Health)
		}
	})
	t.Run("any non-allocation fault since healthy stays latched", func(t *testing.T) {
		l := NewDeviceFaultLatch("cuda", 3)
		l.Observe(DeviceFaultExecution, "gdn", 10002)
		l.Observe(DeviceFaultAllocation, "scratch", 0)
		called := false
		err := l.ClearAllocationSuspect("release", func() error { called = true; return nil })
		var fault *DeviceFaultError
		if !errors.As(err, &fault) || called || l.Health() != DevicePoisoned {
			t.Fatalf("err %v called %v health %s, want typed refusal without probe", err, called, l.Health())
		}
	})
	t.Run("nil or failing probe stays latched", func(t *testing.T) {
		l := NewDeviceFaultLatch("cuda", 3)
		l.Observe(DeviceFaultAllocation, "scratch", 0)
		var fault *DeviceFaultError
		if err := l.ClearAllocationSuspect("release", nil); !errors.As(err, &fault) {
			t.Fatalf("nil probe err = %v, want typed refusal", err)
		}
		if err := l.ClearAllocationSuspect("release", func() error { return errors.New("fence") }); !errors.As(err, &fault) {
			t.Fatalf("failed probe err = %v, want typed refusal", err)
		}
		if l.Health() != DevicePoisoned {
			t.Fatalf("health = %s, want poisoned", l.Health())
		}
	})
	t.Run("fault during probe invalidates the clear", func(t *testing.T) {
		l := NewDeviceFaultLatch("cuda", 3)
		l.Observe(DeviceFaultAllocation, "scratch", 0)
		err := l.ClearAllocationSuspect("release", func() error {
			l.Observe(DeviceFaultAllocation, "concurrent", 0)
			return nil
		})
		var fault *DeviceFaultError
		if !errors.As(err, &fault) || l.Health() != DevicePoisoned {
			t.Fatalf("err %v health %s, want refusal and poisoned", err, l.Health())
		}
	})
	t.Run("healthy and nil latches are no-ops", func(t *testing.T) {
		if err := NewDeviceFaultLatch("cuda", 3).ClearAllocationSuspect("release", nil); err != nil {
			t.Fatalf("healthy latch: %v", err)
		}
		var l *DeviceFaultLatch
		if err := l.ClearAllocationSuspect("release", nil); err != nil {
			t.Fatalf("nil latch: %v", err)
		}
	})
}
