package power

import (
	"context"
	"testing"
	"time"
)

func TestAcquireRelease(t *testing.T) {
	ResetGlobalForTesting()
	defer ResetGlobalForTesting()

	lock, err := NewWakeLock("test-acquire-release", PreventSystemSleep)
	if err != nil {
		t.Fatalf("NewWakeLock: %v", err)
	}
	if !lock.Active() {
		t.Fatal("expected lock to be active")
	}
	if lock.Reason() != "test-acquire-release" {
		t.Fatalf("unexpected reason: %s", lock.Reason())
	}
	if lock.Flags() != PreventSystemSleep {
		t.Fatalf("unexpected flags: %d", lock.Flags())
	}

	if err := lock.Release(); err != nil {
		t.Fatalf("Release error: %v", err)
	}
	if lock.Active() {
		t.Fatal("expected lock to be inactive after release")
	}

	// Idempotent release
	if err := lock.Release(); err != nil {
		t.Fatalf("subsequent Release error: %v", err)
	}
}

func TestWakeLockInstanceReentrant(t *testing.T) {
	ResetGlobalForTesting()
	defer ResetGlobalForTesting()

	lock, err := NewWakeLock("test-reentrant-instance", PreventSystemSleep)
	if err != nil {
		t.Fatalf("NewWakeLock: %v", err)
	}
	defer lock.Release()

	if got := lock.RefCount(); got != 1 {
		t.Fatalf("initial refcount = %d, want 1", got)
	}

	r1, err := lock.Acquire()
	if err != nil {
		t.Fatalf("r1 acquire: %v", err)
	}
	if got := lock.RefCount(); got != 2 {
		t.Fatalf("after r1 refcount = %d, want 2", got)
	}

	r2, err := lock.Acquire()
	if err != nil {
		t.Fatalf("r2 acquire: %v", err)
	}
	if got := lock.RefCount(); got != 3 {
		t.Fatalf("after r2 refcount = %d, want 3", got)
	}

	// Release r1: should still be active
	if err := r1.Release(); err != nil {
		t.Fatalf("r1 release: %v", err)
	}
	if !lock.Active() {
		t.Fatal("expected lock still active after releasing r1")
	}
	if got := lock.RefCount(); got != 2 {
		t.Fatalf("after releasing r1 refcount = %d, want 2", got)
	}

	// Release r1 again (idempotent)
	if err := r1.Release(); err != nil {
		t.Fatalf("r1 duplicate release: %v", err)
	}
	if got := lock.RefCount(); got != 2 {
		t.Fatalf("duplicate release should not change refcount, got %d", got)
	}

	// Release r2: should still be active
	if err := r2.Release(); err != nil {
		t.Fatalf("r2 release: %v", err)
	}
	if !lock.Active() {
		t.Fatal("expected lock still active after releasing r2")
	}
	if got := lock.RefCount(); got != 1 {
		t.Fatalf("after releasing r2 refcount = %d, want 1", got)
	}

	// Release original root lock: should become inactive
	if err := lock.Release(); err != nil {
		t.Fatalf("root release: %v", err)
	}
	if lock.Active() {
		t.Fatal("expected lock inactive after releasing root")
	}
	if got := lock.RefCount(); got != 0 {
		t.Fatalf("final refcount = %d, want 0", got)
	}

	// Acquiring after release should fail
	if _, err := lock.Acquire(); err == nil {
		t.Fatal("expected error acquiring released lock")
	}
}

func TestGlobalReentrantLock(t *testing.T) {
	ResetGlobalForTesting()
	defer ResetGlobalForTesting()

	if IsActive() {
		t.Fatal("expected global lock to start inactive")
	}

	r1, err := Acquire("test-global-1", PreventSystemSleep)
	if err != nil {
		t.Fatalf("r1 Acquire: %v", err)
	}
	if !IsActive() {
		t.Fatal("expected global lock to be active")
	}
	if got := GlobalRefCount(); got != 1 {
		t.Fatalf("initial global refcount = %d, want 1", got)
	}

	r2, err := Acquire("test-global-2", PreventSystemSleep)
	if err != nil {
		t.Fatalf("r2 Acquire: %v", err)
	}
	if got := GlobalRefCount(); got != 2 {
		t.Fatalf("global refcount after r2 = %d, want 2", got)
	}

	// Release r1
	if err := r1.Release(); err != nil {
		t.Fatalf("r1 Release: %v", err)
	}
	if !IsActive() {
		t.Fatal("expected global lock still active after r1 release")
	}
	if got := GlobalRefCount(); got != 1 {
		t.Fatalf("global refcount after r1 release = %d, want 1", got)
	}

	// Release r2
	if err := r2.Release(); err != nil {
		t.Fatalf("r2 Release: %v", err)
	}
	if IsActive() {
		t.Fatal("expected global lock inactive after r2 release")
	}
	if got := GlobalRefCount(); got != 0 {
		t.Fatalf("final global refcount = %d, want 0", got)
	}
}

func TestAcquireWithTimeout(t *testing.T) {
	ResetGlobalForTesting()
	defer ResetGlobalForTesting()

	r, err := AcquireWithTimeout("test-timeout", PreventSystemSleep, 50*time.Millisecond)
	if err != nil {
		t.Fatalf("AcquireWithTimeout: %v", err)
	}
	if !IsActive() {
		t.Fatal("expected lock active initially")
	}

	// Wait for timeout to expire and auto-release
	deadline := time.Now().Add(250 * time.Millisecond)
	for IsActive() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if IsActive() {
		t.Fatal("expected lock to be auto-released after timeout")
	}

	// Calling release after timeout should be a clean no-op
	if err := r.Release(); err != nil {
		t.Fatalf("release after timeout: %v", err)
	}
}

func TestAcquireContext(t *testing.T) {
	ResetGlobalForTesting()
	defer ResetGlobalForTesting()

	ctx, cancel := context.WithCancel(context.Background())
	r, err := AcquireContext(ctx, "test-context", PreventSystemSleep)
	if err != nil {
		t.Fatalf("AcquireContext: %v", err)
	}
	if !IsActive() {
		t.Fatal("expected lock active initially")
	}

	cancel()
	deadline := time.Now().Add(250 * time.Millisecond)
	for IsActive() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if IsActive() {
		t.Fatal("expected lock to release when context canceled")
	}

	if err := r.Release(); err != nil {
		t.Fatalf("release after cancel: %v", err)
	}
}

func TestWakeFlags(t *testing.T) {
	flags := PreventSystemSleep | PreventDisplaySleep
	lock, err := NewWakeLock("test-flags", flags)
	if err != nil {
		t.Fatalf("NewWakeLock: %v", err)
	}
	defer lock.Release()

	if lock.Flags()&PreventSystemSleep == 0 {
		t.Fatal("expected PreventSystemSleep flag to be set")
	}
	if lock.Flags()&PreventDisplaySleep == 0 {
		t.Fatal("expected PreventDisplaySleep flag to be set")
	}
}

func TestPlatformBehavior(t *testing.T) {
	ResetGlobalForTesting()
	defer ResetGlobalForTesting()

	r, err := Acquire("test-platform-behavior", PreventSystemSleep)
	if err != nil {
		t.Fatalf("platform Acquire: %v", err)
	}
	if !IsActive() {
		t.Fatal("expected lock to be active")
	}
	if err := r.Release(); err != nil {
		t.Fatalf("platform Release: %v", err)
	}
	if IsActive() {
		t.Fatal("expected lock to be inactive")
	}
}
