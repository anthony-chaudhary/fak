package session

// lease.go — single-writer InputLease coordinator for multi-surface session attachment (issue #11438).
//
// In multi-surface session environments (terminal, web, RPC clients), multiple surfaces
// may attach concurrently to observe events, but only one surface may hold the input lease
// (the single writer). InputLeaseCoordinator provides thread-safe, lease-based concurrency
// control with automatic expiration fallback, heartbeat renewal, and token verification.

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"sync"
	"time"
)

var (
	// ErrInputLeaseHeld indicates an active, unexpired input lease is already held.
	ErrInputLeaseHeld = errors.New("session: input lease already held")
	// ErrInputLeaseNotFound indicates no active lease exists or the lease has expired.
	ErrInputLeaseNotFound = errors.New("session: input lease not found")
	// ErrInputLeaseExpired indicates the lease has expired.
	ErrInputLeaseExpired = errors.New("session: input lease expired")
	// ErrInputLeaseMismatch indicates the provided lease token does not match the active lease.
	ErrInputLeaseMismatch = errors.New("session: input lease token mismatch")
	// ErrInvalidLeaseTTL indicates the requested lease duration is non-positive.
	ErrInvalidLeaseTTL = errors.New("session: invalid lease ttl")
)

// InputLease represents a single-writer input lease granted to a surface/client.
type InputLease struct {
	HolderID  string    `json:"holder_id"`
	ExpiresAt time.Time `json:"expires_at"`
	Token     string    `json:"token"`
}

// Clone returns a shallow copy of the lease.
func (l *InputLease) Clone() *InputLease {
	if l == nil {
		return nil
	}
	return &InputLease{
		HolderID:  l.HolderID,
		ExpiresAt: l.ExpiresAt,
		Token:     l.Token,
	}
}

// Expired reports whether the lease is expired at time t.
func (l *InputLease) Expired(at time.Time) bool {
	if l == nil {
		return true
	}
	return !at.Before(l.ExpiresAt)
}

// InputLeaseCoordinator coordinates single-writer input lease admission and lifecycle.
// All methods are safe for concurrent access.
type InputLeaseCoordinator struct {
	mu      sync.RWMutex
	current *InputLease
	nowFn   func() time.Time
}

// NewInputLeaseCoordinator constructs an empty InputLeaseCoordinator.
func NewInputLeaseCoordinator() *InputLeaseCoordinator {
	return &InputLeaseCoordinator{}
}

func (c *InputLeaseCoordinator) now() time.Time {
	if c.nowFn != nil {
		return c.nowFn()
	}
	return time.Now()
}

// SetNowFn overrides the time source (useful for deterministic tests).
func (c *InputLeaseCoordinator) SetNowFn(fn func() time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nowFn = fn
}

// Acquire grants an exclusive input lease to holderID for ttl.
// If an active unexpired lease is already held, ErrInputLeaseHeld is returned.
func (c *InputLeaseCoordinator) Acquire(holderID string, ttl time.Duration) (*InputLease, error) {
	if ttl <= 0 {
		return nil, ErrInvalidLeaseTTL
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	now := c.now()
	if c.current != nil && now.Before(c.current.ExpiresAt) {
		return nil, ErrInputLeaseHeld
	}

	token, err := mintLeaseToken()
	if err != nil {
		return nil, err
	}

	lease := &InputLease{
		HolderID:  holderID,
		ExpiresAt: now.Add(ttl),
		Token:     token,
	}
	c.current = lease
	return lease.Clone(), nil
}

// Renew extends the active lease for ttl if token matches and the lease has not expired.
func (c *InputLeaseCoordinator) Renew(token string, ttl time.Duration) (*InputLease, error) {
	if ttl <= 0 {
		return nil, ErrInvalidLeaseTTL
	}
	if token == "" {
		return nil, ErrInputLeaseMismatch
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.current == nil {
		return nil, ErrInputLeaseNotFound
	}
	if c.current.Token != token {
		return nil, ErrInputLeaseMismatch
	}
	now := c.now()
	if !now.Before(c.current.ExpiresAt) {
		return nil, ErrInputLeaseExpired
	}

	c.current.ExpiresAt = now.Add(ttl)
	return c.current.Clone(), nil
}

// Release relinquishes the active lease if token matches.
func (c *InputLeaseCoordinator) Release(token string) error {
	if token == "" {
		return ErrInputLeaseMismatch
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.current == nil {
		return ErrInputLeaseNotFound
	}
	if c.current.Token != token {
		return ErrInputLeaseMismatch
	}

	c.current = nil
	return nil
}

// Verify returns true if token matches the active lease and the lease has not expired.
func (c *InputLeaseCoordinator) Verify(token string) bool {
	if token == "" {
		return false
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.current == nil {
		return false
	}
	if c.current.Token != token {
		return false
	}
	return c.now().Before(c.current.ExpiresAt)
}

// Current returns a copy of the active lease, or nil if no unexpired lease is held.
func (c *InputLeaseCoordinator) Current() *InputLease {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.current == nil || !c.now().Before(c.current.ExpiresAt) {
		return nil
	}
	return c.current.Clone()
}

func mintLeaseToken() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

// InputLease returns the InputLeaseCoordinator for trace, allocating one if not yet present.
func (t *Table) InputLease(trace string) *InputLeaseCoordinator {
	if t == nil {
		return NewInputLeaseCoordinator()
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ensureLocked()
	coord, ok := t.inputLeases[trace]
	if !ok {
		coord = NewInputLeaseCoordinator()
		t.inputLeases[trace] = coord
	}
	return coord
}

// InputLeaseCoordinator returns the InputLeaseCoordinator for trace (alias for InputLease).
func (t *Table) InputLeaseCoordinator(trace string) *InputLeaseCoordinator {
	return t.InputLease(trace)
}

// AcquireInputLease acquires the single-writer input lease for trace.
func (t *Table) AcquireInputLease(trace, holderID string, ttl time.Duration) (*InputLease, error) {
	return t.InputLease(trace).Acquire(holderID, ttl)
}

// RenewInputLease extends the single-writer input lease for trace.
func (t *Table) RenewInputLease(trace, token string, ttl time.Duration) (*InputLease, error) {
	return t.InputLease(trace).Renew(token, ttl)
}

// ReleaseInputLease releases the single-writer input lease for trace.
func (t *Table) ReleaseInputLease(trace, token string) error {
	return t.InputLease(trace).Release(token)
}

// VerifyInputLease reports whether token holds the active input lease for trace.
func (t *Table) VerifyInputLease(trace, token string) bool {
	return t.InputLease(trace).Verify(token)
}
