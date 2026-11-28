package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fak-test:runtime fast est=50ms lane=default
func TestPopulateLockBypass(t *testing.T) {
	lockPath := filepath.Join(t.TempDir(), "cache.populate.lock")
	now := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)
	alive := map[int]bool{101: true, 202: false}

	lock := NewPopulateLock(lockPath, 30*time.Second)
	lock.now = func() time.Time { return now }
	lock.isAlive = func(pid int) bool { return alive[pid] }

	// Live PID 101 acquires lock cleanly.
	bypassed, err := lock.Acquire(101)
	if err != nil || bypassed {
		t.Fatalf("initial Acquire(101) = (%v, %v), want (false, nil)", bypassed, err)
	}
	// While PID 101 is alive and within TTL, PID 303 is refused.
	if _, err := lock.Acquire(303); err == nil {
		t.Fatal("expected Acquire(303) to fail while live PID 101 holds lock")
	}

	// Simulate SIGKILL of PID 101 -> dead PID tombstone is bypassed.
	alive[101] = false
	bypassed, err = lock.Acquire(303)
	if err != nil || !bypassed {
		t.Fatalf("Acquire(303) after SIGKILL = (%v, %v), want (true, nil)", bypassed, err)
	}
	_ = lock.Release()

	// Corrupt/empty tombstone (SIGKILL mid-write) is also bypassed.
	if err := os.WriteFile(lockPath, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	bypassed, err = lock.Acquire(404)
	if err != nil || !bypassed {
		t.Fatalf("Acquire(404) on corrupt tombstone = (%v, %v), want (true, nil)", bypassed, err)
	}
}
