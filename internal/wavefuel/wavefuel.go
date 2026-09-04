package wavefuel

import (
	"errors"
	"sync"
	"time"
)

var (
	// ErrExhausted indicates that the wave context or token budget has been fully consumed.
	ErrExhausted = errors.New("wave fuel: budget exhausted")
	// ErrDeadlineExceeded indicates that the scheduled landing cutoff timestamp has passed.
	ErrDeadlineExceeded = errors.New("wave fuel: landing deadline exceeded")
	// ErrInvalidAllocation indicates that a requested budget quantity is non-positive or malformed.
	ErrInvalidAllocation = errors.New("wave fuel: invalid allocation request")
)

// Account tracks cumulative token and concurrent worker session allowances for a wave.
//
// Invariant: wave fuel accounting is fail-closed and bounded across all fleet wave allocations.
// Guard: allocations and debits past the landing deadline or exceeding the budget are refused immediately.
type Account struct {
	mu           sync.Mutex
	maxTokens    int64
	usedTokens   int64
	maxSessions  int
	usedSessions int
	deadline     time.Time
}

// NewAccount initializes a bounded fuel ledger enforcing token, session, and deadline limits.
func NewAccount(maxTokens int64, maxSessions int, deadline time.Time) *Account {
	return &Account{
		maxTokens:   maxTokens,
		maxSessions: maxSessions,
		deadline:    deadline,
	}
}

// Debit consumes context tokens from the available budget, failing closed on overflow or expiry.
func (a *Account) Debit(tokens int64, now time.Time) error {
	if tokens <= 0 {
		return ErrInvalidAllocation
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.deadline.IsZero() && now.After(a.deadline) {
		return ErrDeadlineExceeded
	}
	if a.usedTokens+tokens > a.maxTokens {
		return ErrExhausted
	}
	a.usedTokens += tokens
	return nil
}

// AllocateSession reserves one worker execution slot if capacity remains under the configured cap.
func (a *Account) AllocateSession(now time.Time) (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.deadline.IsZero() && now.After(a.deadline) {
		return 0, ErrDeadlineExceeded
	}
	if a.usedSessions >= a.maxSessions {
		return 0, ErrExhausted
	}
	a.usedSessions++
	return a.usedSessions, nil
}

// RemainingTokens reports the unspent context allowance remaining in the wave budget.
func (a *Account) RemainingTokens() int64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	rem := a.maxTokens - a.usedTokens
	if rem < 0 {
		return 0
	}
	return rem
}

// RemainingSessions reports how many worker executions may still be dispatched before hitting the cap.
func (a *Account) RemainingSessions() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	rem := a.maxSessions - a.usedSessions
	if rem < 0 {
		return 0
	}
	return rem
}
