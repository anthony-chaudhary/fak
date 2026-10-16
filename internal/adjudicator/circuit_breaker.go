package adjudicator

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// DefaultCircuitBreakerThreshold is the default consecutive identical refusal threshold (K=3)
// before the in-kernel circuit breaker trips (#12523).
const DefaultCircuitBreakerThreshold = 3

type sessionRefusalState struct {
	tool    string
	reason  abi.ReasonCode
	argHash string
	count   int
	tripped bool
}

func callArgFingerprint(tool string, c *abi.ToolCall) string {
	var rawArgs map[string]any
	if c != nil {
		rawArgs = decodeArgs(context.Background(), c)
	}
	tokArgs := argsForToken(rawArgs)
	b, _ := json.Marshal(tokArgs)
	h := sha256.Sum256(append([]byte(strings.ToLower(tool)+"\x00"), b...))
	return hex.EncodeToString(h[:8])
}

type circuitBreakerTracker struct {
	mu        sync.Mutex
	sessions  map[string]*sessionRefusalState
	threshold int
}

func newCircuitBreakerTracker(threshold int) *circuitBreakerTracker {
	if threshold <= 0 {
		threshold = DefaultCircuitBreakerThreshold
	}
	return &circuitBreakerTracker{
		sessions:  make(map[string]*sessionRefusalState),
		threshold: threshold,
	}
}

func (a *Adjudicator) getCircuitBreaker() *circuitBreakerTracker {
	if a == nil {
		return nil
	}
	cb := a.circuitBreaker.Load()
	if cb == nil {
		newCb := newCircuitBreakerTracker(DefaultCircuitBreakerThreshold)
		if a.circuitBreaker.CompareAndSwap(nil, newCb) {
			return newCb
		}
		return a.circuitBreaker.Load()
	}
	return cb
}

// SetCircuitBreakerThreshold sets the threshold for consecutive identical refusals before tripping (#12523).
func (a *Adjudicator) SetCircuitBreakerThreshold(threshold int) {
	cb := a.getCircuitBreaker()
	if cb == nil {
		return
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()
	if threshold <= 0 {
		threshold = DefaultCircuitBreakerThreshold
	}
	cb.threshold = threshold
}

// ResetCircuitBreaker resets the refusal tracking and circuit breaker for a given session (#12523).
func (a *Adjudicator) ResetCircuitBreaker(sessionID string) {
	cb := a.getCircuitBreaker()
	if cb == nil {
		return
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()
	delete(cb.sessions, sessionID)
}

// CircuitBreakerState returns the current tool, reason, count, and tripped state for a session (#12523).
func (a *Adjudicator) CircuitBreakerState(sessionID string) (tool string, reason abi.ReasonCode, count int, tripped bool) {
	cb := a.getCircuitBreaker()
	if cb == nil {
		return "", 0, 0, false
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()
	if entry, ok := cb.sessions[sessionID]; ok {
		return entry.tool, entry.reason, entry.count, entry.tripped
	}
	return "", 0, 0, false
}
