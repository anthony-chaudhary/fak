//go:build darwin

package power

import "testing"

func TestPlatformDarwinCaffeinateFallback(t *testing.T) {
	ResetGlobalForTesting()
	defer ResetGlobalForTesting()

	SetDarwinForceCaffeinateForTesting(true)
	defer SetDarwinForceCaffeinateForTesting(false)

	lock, err := NewWakeLock("test-darwin-caffeinate", PreventSystemSleep|PreventDisplaySleep)
	if err != nil {
		t.Fatalf("caffeinate fallback NewWakeLock: %v", err)
	}
	if !lock.Active() {
		t.Fatal("expected caffeinate lock to be active")
	}

	if err := lock.Release(); err != nil {
		t.Fatalf("caffeinate lock Release: %v", err)
	}
	if lock.Active() {
		t.Fatal("expected caffeinate lock to be inactive after release")
	}
}
