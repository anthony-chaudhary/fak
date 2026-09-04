package dispatchcache

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"sync"
	"time"
)

// Invariant: in-memory store eviction is fail-closed; expired or uninitialized entries return zero values.

// Key content-addresses the routed-backlog inputs shared by successive ticks.
func Key(workspace, view string, issueLimit int) string {
	h := sha256.Sum256([]byte(workspace + "\x00" + view + "\x00" + strconv.Itoa(issueLimit)))
	return hex.EncodeToString(h[:])
}

type entry[T any] struct {
	value     T
	expiresAt time.Time
}

// Store is an in-memory TTL cache with an injectable clock.
type Store[T any] struct {
	mu  sync.Mutex
	now func() time.Time
	m   map[string]entry[T]
}

// New allocates an in-memory Store with an optional injectable clock function.
func New[T any](now func() time.Time) *Store[T] {
	if now == nil {
		now = time.Now
	}
	return &Store[T]{now: now, m: map[string]entry[T]{}}
}

// Get retrieves an unexpired entry for the given key, returning false if expired or missing.
func (s *Store[T]) Get(key string) (T, bool) {
	var zero T
	if s == nil {
		return zero, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.m[key]
	if !ok {
		return zero, false
	}
	if !s.now().Before(e.expiresAt) {
		delete(s.m, key)
		return zero, false
	}
	return e.value, true
}

// Put inserts or updates an entry under key with a specified time-to-live duration.
func (s *Store[T]) Put(key string, value T, ttl time.Duration) {
	if s == nil || ttl <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[key] = entry[T]{value: value, expiresAt: s.now().Add(ttl)}
}
