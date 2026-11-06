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

	// Over-budget debit must be refused (fail-closed).
	if err := acct.Debit(2_000_000, now); err != ErrExhausted {
		t.Fatalf("expected ErrExhausted, got %v", err)
	}

	// Post-deadline debit must be refused (fail-closed).
	expired := deadline.Add(time.Second)
	if err := acct.Debit(1_000, expired); err != ErrDeadlineExceeded {
		t.Fatalf("expected ErrDeadlineExceeded, got %v", err)
	}
}

// Runtime estimate is unmeasured; this exercises only a fixed-clock ledger.
// fak-test:runtime fast est=5ms lane=default
func TestDebitRejectsOverflowWithoutConsumingBudget(t *testing.T) {
	const maxTokens = int64(1<<63 - 1)
	now := time.Unix(0, 0)
	acct := NewAccount(maxTokens, 1, time.Time{})
	if err := acct.Debit(maxTokens-1, now); err != nil {
		t.Fatalf("initial debit: %v", err)
	}
	if err := acct.Debit(2, now); err != ErrExhausted {
		t.Fatalf("overflowing debit = %v, want ErrExhausted", err)
	}
	if got := acct.RemainingTokens(); got != 1 {
		t.Fatalf("refused debit changed remaining budget: got %d, want 1", got)
	}
	if err := acct.Debit(1, now); err != nil {
		t.Fatalf("exact remaining budget refused: %v", err)
	}
	if got := acct.RemainingTokens(); got != 0 {
		t.Fatalf("exhausted budget = %d, want 0", got)
	}
	if err := acct.Debit(1, now); err != ErrExhausted {
		t.Fatalf("debit after exhaustion = %v, want ErrExhausted", err)
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
