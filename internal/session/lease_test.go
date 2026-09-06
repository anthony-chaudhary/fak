package session

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestInputLeaseAcquisitionAndExpiry verifies the single-writer input lease coordinator's
// complete lifecycle: exclusive acquisition, conflict rejection, heartbeat renewal,
// explicit release, and expiration fallback (issue #11438).
func TestInputLeaseAcquisitionAndExpiry(t *testing.T) {
	coord := NewInputLeaseCoordinator()

	// 0. Pre-condition: initial state is empty
	if cur := coord.Current(); cur != nil {
		t.Fatalf("expected initial Current() to be nil, got %+v", cur)
	}
	if coord.Verify("arbitrary-token") {
		t.Fatalf("expected initial Verify to return false")
	}

	// 1. Exclusive acquisition
	const ttl = 40 * time.Millisecond
	lease1, err := coord.Acquire("surface-terminal", ttl)
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}
	if lease1 == nil {
		t.Fatalf("expected non-nil lease on Acquire")
	}
	if lease1.HolderID != "surface-terminal" {
		t.Fatalf("expected HolderID 'surface-terminal', got %q", lease1.HolderID)
	}
	if lease1.Token == "" {
		t.Fatalf("expected non-empty Token")
	}
	if !lease1.ExpiresAt.After(time.Now()) {
		t.Fatalf("expected ExpiresAt to be in the future, got %v", lease1.ExpiresAt)
	}
	if !coord.Verify(lease1.Token) {
		t.Fatalf("expected Verify(lease1.Token) to return true")
	}
	if coord.Verify("bogus-token") {
		t.Fatalf("expected Verify with bogus token to return false")
	}
	if cur := coord.Current(); cur == nil || cur.Token != lease1.Token || cur.HolderID != "surface-terminal" {
		t.Fatalf("Current() = %+v, want holder surface-terminal with token %s", cur, lease1.Token)
	}

	// 2. Conflict rejection while lease is active
	lease2, err := coord.Acquire("surface-web", ttl)
	if !errors.Is(err, ErrInputLeaseHeld) {
		t.Fatalf("expected ErrInputLeaseHeld on concurrent acquire, got err=%v, lease=%+v", err, lease2)
	}
	if err != ErrInputLeaseHeld {
		t.Fatalf("expected exact ErrInputLeaseHeld error, got %v", err)
	}

	// 3. Heartbeat Renewal
	renewTTL := 80 * time.Millisecond
	// Invalid token fails
	if _, err := coord.Renew("wrong-token", renewTTL); !errors.Is(err, ErrInputLeaseMismatch) {
		t.Fatalf("expected ErrInputLeaseMismatch on wrong token renewal, got %v", err)
	}
	// Invalid TTL fails
	if _, err := coord.Renew(lease1.Token, 0); !errors.Is(err, ErrInvalidLeaseTTL) {
		t.Fatalf("expected ErrInvalidLeaseTTL on non-positive TTL, got %v", err)
	}
	// Valid renewal extends expiration
	renewed, err := coord.Renew(lease1.Token, renewTTL)
	if err != nil {
		t.Fatalf("Renew failed: %v", err)
	}
	if renewed.Token != lease1.Token {
		t.Fatalf("expected renewed token %q, got %q", lease1.Token, renewed.Token)
	}
	if !renewed.ExpiresAt.After(lease1.ExpiresAt) {
		t.Fatalf("expected renewed expiry %v to be after initial expiry %v", renewed.ExpiresAt, lease1.ExpiresAt)
	}
	if !coord.Verify(lease1.Token) {
		t.Fatalf("expected lease1 token to remain valid after renewal")
	}

	// 4. Release
	// Mismatched token fails
	if err := coord.Release("wrong-token"); !errors.Is(err, ErrInputLeaseMismatch) {
		t.Fatalf("expected ErrInputLeaseMismatch on wrong token release, got %v", err)
	}
	// Empty token fails
	if err := coord.Release(""); !errors.Is(err, ErrInputLeaseMismatch) {
		t.Fatalf("expected ErrInputLeaseMismatch on empty token release, got %v", err)
	}
	// Valid release clears lease
	if err := coord.Release(lease1.Token); err != nil {
		t.Fatalf("Release failed: %v", err)
	}
	if coord.Verify(lease1.Token) {
		t.Fatalf("expected Verify to be false after release")
	}
	if cur := coord.Current(); cur != nil {
		t.Fatalf("expected Current() to be nil after release, got %+v", cur)
	}
	// Releasing already released lease returns ErrInputLeaseNotFound
	if err := coord.Release(lease1.Token); !errors.Is(err, ErrInputLeaseNotFound) {
		t.Fatalf("expected ErrInputLeaseNotFound on release of unheld lease, got %v", err)
	}

	// 5. Expiration fallback
	shortTTL := 25 * time.Millisecond
	leaseShort, err := coord.Acquire("surface-terminal", shortTTL)
	if err != nil {
		t.Fatalf("Acquire with short TTL failed: %v", err)
	}
	if !coord.Verify(leaseShort.Token) {
		t.Fatalf("expected leaseShort to be initially valid")
	}

	// Wait for expiration
	time.Sleep(35 * time.Millisecond)

	// Verify that the expired lease is rejected
	if coord.Verify(leaseShort.Token) {
		t.Fatalf("expected Verify to return false after lease expiration")
	}
	if cur := coord.Current(); cur != nil {
		t.Fatalf("expected Current() to return nil after expiration, got %+v", cur)
	}
	// Renewing expired lease returns ErrInputLeaseExpired
	if _, err := coord.Renew(leaseShort.Token, ttl); !errors.Is(err, ErrInputLeaseExpired) {
		t.Fatalf("expected ErrInputLeaseExpired on renewal of expired lease, got %v", err)
	}

	// Expiration fallback: surface-web can now acquire without error
	fallbackLease, err := coord.Acquire("surface-web", ttl)
	if err != nil {
		t.Fatalf("expected fallback Acquire to succeed after expiration, got %v", err)
	}
	if fallbackLease.HolderID != "surface-web" {
		t.Fatalf("expected HolderID 'surface-web', got %q", fallbackLease.HolderID)
	}
	if fallbackLease.Token == leaseShort.Token {
		t.Fatalf("expected fresh token for new lease, got reuse of %q", fallbackLease.Token)
	}
	if !coord.Verify(fallbackLease.Token) {
		t.Fatalf("expected fallback lease to be valid")
	}
	if coord.Verify(leaseShort.Token) {
		t.Fatalf("old expired token must remain invalid")
	}

	// Old holder releasing old token fails with ErrInputLeaseMismatch against new lease
	if err := coord.Release(leaseShort.Token); !errors.Is(err, ErrInputLeaseMismatch) {
		t.Fatalf("expected old holder release to fail with ErrInputLeaseMismatch, got %v", err)
	}
	// New holder can cleanly release
	if err := coord.Release(fallbackLease.Token); err != nil {
		t.Fatalf("fallback lease release failed: %v", err)
	}
}

// TestInputLeaseDeterministicClock uses a mock clock to verify nanosecond-exact lease boundaries.
func TestInputLeaseDeterministicClock(t *testing.T) {
	coord := NewInputLeaseCoordinator()
	now := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)
	coord.SetNowFn(func() time.Time { return now })

	const ttl = 10 * time.Second
	lease, err := coord.Acquire("mock-holder", ttl)
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}
	if !lease.ExpiresAt.Equal(now.Add(ttl)) {
		t.Fatalf("expected ExpiresAt = %v, got %v", now.Add(ttl), lease.ExpiresAt)
	}

	// Exactly at 9.999 seconds: still valid
	now = now.Add(9999 * time.Millisecond)
	if !coord.Verify(lease.Token) {
		t.Fatalf("lease should be valid before 10s")
	}
	if cur := coord.Current(); cur == nil {
		t.Fatalf("Current() should not be nil before 10s")
	}

	// At exactly 10.000 seconds: expired
	now = now.Add(1 * time.Millisecond)
	if coord.Verify(lease.Token) {
		t.Fatalf("lease should be expired at exactly 10s")
	}
	if cur := coord.Current(); cur != nil {
		t.Fatalf("Current() should be nil at exactly 10s")
	}

	// Conflict rejection no longer triggers; new holder can acquire
	newLease, err := coord.Acquire("second-holder", ttl)
	if err != nil {
		t.Fatalf("failed to acquire expired lease: %v", err)
	}
	if newLease.HolderID != "second-holder" {
		t.Fatalf("expected holder second-holder, got %q", newLease.HolderID)
	}
}

// TestInputLeaseTableWiring verifies Table integration and trace isolation.
func TestInputLeaseTableWiring(t *testing.T) {
	tbl := NewTableWithLimit(2)

	// Separate coordinators per session trace
	leaseA, err := tbl.AcquireInputLease("session-alpha", "holder-a", 100*time.Millisecond)
	if err != nil {
		t.Fatalf("AcquireInputLease alpha failed: %v", err)
	}
	leaseB, err := tbl.AcquireInputLease("session-beta", "holder-b", 100*time.Millisecond)
	if err != nil {
		t.Fatalf("AcquireInputLease beta failed: %v", err)
	}

	// Both traces hold their independent leases concurrently
	if !tbl.VerifyInputLease("session-alpha", leaseA.Token) {
		t.Fatalf("session-alpha lease must be verified")
	}
	if !tbl.VerifyInputLease("session-beta", leaseB.Token) {
		t.Fatalf("session-beta lease must be verified")
	}

	// Conflict rejection on same trace
	_, err = tbl.AcquireInputLease("session-alpha", "holder-c", 100*time.Millisecond)
	if !errors.Is(err, ErrInputLeaseHeld) {
		t.Fatalf("expected ErrInputLeaseHeld on same trace, got %v", err)
	}

	// Renewing via Table
	renewed, err := tbl.RenewInputLease("session-alpha", leaseA.Token, 200*time.Millisecond)
	if err != nil {
		t.Fatalf("RenewInputLease failed: %v", err)
	}
	if renewed.Token != leaseA.Token {
		t.Fatalf("expected same token on renew")
	}

	// Release via Table
	if err := tbl.ReleaseInputLease("session-alpha", leaseA.Token); err != nil {
		t.Fatalf("ReleaseInputLease failed: %v", err)
	}
	if tbl.VerifyInputLease("session-alpha", leaseA.Token) {
		t.Fatalf("session-alpha should no longer be verified after release")
	}

	// Resetting a trace clears its lease coordinator
	tbl.Reset("session-beta")
	coordBeta := tbl.InputLease("session-beta")
	if cur := coordBeta.Current(); cur != nil {
		t.Fatalf("expected reset trace to have clean coordinator, got %+v", cur)
	}
}

// TestInputLeaseConcurrency validates thread safety under heavy concurrent access.
func TestInputLeaseConcurrency(t *testing.T) {
	coord := NewInputLeaseCoordinator()
	const goroutines = 16
	const iterations = 50

	var (
		acquiredCount int64
		heldCount     int64
		wg            sync.WaitGroup
	)

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			holderID := "worker"
			for j := 0; j < iterations; j++ {
				lease, err := coord.Acquire(holderID, 5*time.Millisecond)
				if err == nil {
					atomic.AddInt64(&acquiredCount, 1)
					if !coord.Verify(lease.Token) {
						t.Errorf("acquired lease must be verifiable")
					}
					// Quick release or let expire
					if j%2 == 0 {
						_ = coord.Release(lease.Token)
					}
				} else if errors.Is(err, ErrInputLeaseHeld) {
					atomic.AddInt64(&heldCount, 1)
				}
				_ = coord.Current()
			}
		}(i)
	}
	wg.Wait()

	if acquiredCount == 0 {
		t.Fatalf("expected at least one acquisition during concurrent runs")
	}
	if heldCount == 0 {
		t.Fatalf("expected contention (heldCount > 0) under concurrent runs")
	}
}
