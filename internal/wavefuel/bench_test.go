package wavefuel

import (
	"testing"
	"time"
)

func TestWaveFuelAccounting(t *testing.T) {
	now := time.Now()
	deadline := now.Add(4 * time.Hour)
	acct := NewAccount(2_000_000, 30, deadline)

	slot, err := acct.AllocateSession(now)
	if err != nil {
		t.Fatalf("unexpected allocation failure: %v", err)
	}
	if slot != 1 {
		t.Fatalf("expected slot 1, got %d", slot)
	}

	if err := acct.Debit(100_000, now); err != nil {
		t.Fatalf("unexpected debit failure: %v", err)
	}
	if rem := acct.RemainingTokens(); rem != 1_900_000 {
		t.Fatalf("expected 1900000 remaining, got %d", rem)
	}

	// Fail-closed on exceeding budget
	if err := acct.Debit(2_000_000, now); err != ErrExhausted {
		t.Fatalf("expected ErrExhausted, got %v", err)
	}

	// Fail-closed on past deadline
	expired := deadline.Add(time.Second)
	if err := acct.Debit(1_000, expired); err != ErrDeadlineExceeded {
		t.Fatalf("expected ErrDeadlineExceeded, got %v", err)
	}
}

func BenchmarkWaveFuel(b *testing.B) {
	now := time.Now()
	deadline := now.Add(24 * time.Hour)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		acct := NewAccount(100_000_000, 10_000, deadline)
		for j := 0; j < 10; j++ {
			_, _ = acct.AllocateSession(now)
			_ = acct.Debit(10_000, now)
		}
		_ = acct.RemainingTokens()
		_ = acct.RemainingSessions()
	}
}
