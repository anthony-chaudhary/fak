package model

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// DefaultCollapseThreshold is the minimum number of consecutive identical tokens
// required to trip the circuit breaker.
const DefaultCollapseThreshold = 16

var (
	// ErrTokenCollapse is returned by OutputEntropyMonitor when >= 16 identical tokens
	// are emitted consecutively, signaling speculative degeneration (e.g. token-0 collapse).
	ErrTokenCollapse = errors.New("model: token collapse circuit breaker tripped: >= 16 identical consecutive tokens")

	// ErrCheckpointNotFound is returned when attempting to rollback to an unknown or pruned step.
	ErrCheckpointNotFound = errors.New("model: recurrent state checkpoint not found")
)

// RecurrentStateRollbackManager holds snapshot checkpoints of recurrent linear
// attention states (Gated DeltaNet / SSM state buffers S_t) prior to speculative
// drafting, enabling bit-identical rollback when speculative draft tokens are rejected.
type RecurrentStateRollbackManager struct {
	mu           sync.RWMutex
	checkpoints  map[int][]float32
	currentState []float32
}

// NewRecurrentStateRollbackManager creates a new manager for recurrent state checkpoints.
func NewRecurrentStateRollbackManager() *RecurrentStateRollbackManager {
	return &RecurrentStateRollbackManager{
		checkpoints: make(map[int][]float32),
	}
}

// Checkpoint takes an auxiliary snapshot prior to speculative drafting.
// It deep-copies the provided state buffer so subsequent speculative mutations
// do not corrupt the saved checkpoint.
func (m *RecurrentStateRollbackManager) Checkpoint(step int, state []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.checkpoints == nil {
		m.checkpoints = make(map[int][]float32)
	}

	snap := make([]float32, len(state))
	copy(snap, state)
	m.checkpoints[step] = snap

	m.currentState = make([]float32, len(state))
	copy(m.currentState, state)
}

// Rollback restores the recurrent state S_t to the exact verified step when a
// speculative draft token is rejected. It returns a deep copy of the restored state
// and prunes any speculative checkpoints recorded for steps > verifiedStep.
func (m *RecurrentStateRollbackManager) Rollback(verifiedStep int) ([]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.checkpoints == nil {
		return nil, fmt.Errorf("%w: step %d", ErrCheckpointNotFound, verifiedStep)
	}

	snap, ok := m.checkpoints[verifiedStep]
	if !ok {
		return nil, fmt.Errorf("%w: step %d", ErrCheckpointNotFound, verifiedStep)
	}

	// Restore current state
	m.currentState = make([]float32, len(snap))
	copy(m.currentState, snap)

	// Prune speculative checkpoints created beyond verifiedStep
	for step := range m.checkpoints {
		if step > verifiedStep {
			delete(m.checkpoints, step)
		}
	}

	res := make([]float32, len(snap))
	copy(res, snap)
	return res, nil
}

// Commit prunes checkpoints older than verified step (step < verifiedStep).
func (m *RecurrentStateRollbackManager) Commit(verifiedStep int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.checkpoints == nil {
		return
	}

	for step := range m.checkpoints {
		if step < verifiedStep {
			delete(m.checkpoints, step)
		}
	}
}

// State returns a deep copy of the current tracked recurrent state.
func (m *RecurrentStateRollbackManager) State() []float32 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.currentState == nil {
		return nil
	}
	res := make([]float32, len(m.currentState))
	copy(res, m.currentState)
	return res
}

// HasCheckpoint reports whether a checkpoint exists for the given step.
func (m *RecurrentStateRollbackManager) HasCheckpoint(step int) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.checkpoints == nil {
		return false
	}
	_, ok := m.checkpoints[step]
	return ok
}

// CheckpointCount returns the number of active checkpoints currently held.
func (m *RecurrentStateRollbackManager) CheckpointCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.checkpoints == nil {
		return 0
	}
	return len(m.checkpoints)
}

// RestoreTo copies the checkpoint at verifiedStep into dst without allocating.
func (m *RecurrentStateRollbackManager) RestoreTo(verifiedStep int, dst []float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.checkpoints == nil {
		return fmt.Errorf("%w: step %d", ErrCheckpointNotFound, verifiedStep)
	}

	snap, ok := m.checkpoints[verifiedStep]
	if !ok {
		return fmt.Errorf("%w: step %d", ErrCheckpointNotFound, verifiedStep)
	}
	if len(dst) != len(snap) {
		return fmt.Errorf("model: destination length %d does not match checkpoint length %d", len(dst), len(snap))
	}

	copy(dst, snap)

	m.currentState = make([]float32, len(snap))
	copy(m.currentState, snap)

	for step := range m.checkpoints {
		if step > verifiedStep {
			delete(m.checkpoints, step)
		}
	}
	return nil
}

// OutputEntropyMonitor tracks recent sequence of emitted token IDs and trips
// an automatic circuit breaker (ErrTokenCollapse) if >= 16 identical tokens
// (such as token 0 '!') are emitted sequentially.
type OutputEntropyMonitor struct {
	mu          sync.RWMutex
	threshold   int
	lastToken   int
	consecutive int
	hasTokens   bool
	history     []int
	tripped     bool
}

// NewOutputEntropyMonitor creates a monitor with an optional custom threshold
// (defaulting to DefaultCollapseThreshold = 16).
func NewOutputEntropyMonitor(threshold ...int) *OutputEntropyMonitor {
	th := DefaultCollapseThreshold
	if len(threshold) > 0 && threshold[0] > 0 {
		th = threshold[0]
	}
	return &OutputEntropyMonitor{
		threshold: th,
	}
}

// Record records one emitted token ID. If the sequence of identical consecutive
// tokens reaches or exceeds the collapse threshold (default 16), the circuit breaker
// trips and returns ErrTokenCollapse.
func (m *OutputEntropyMonitor) Record(token int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.history = append(m.history, token)

	if !m.hasTokens {
		m.hasTokens = true
		m.lastToken = token
		m.consecutive = 1
	} else if token == m.lastToken {
		m.consecutive++
	} else {
		m.lastToken = token
		m.consecutive = 1
	}

	th := m.threshold
	if th <= 0 {
		th = DefaultCollapseThreshold
	}

	if m.consecutive >= th {
		m.tripped = true
		return ErrTokenCollapse
	}
	if m.tripped {
		return ErrTokenCollapse
	}
	return nil
}

// Observe is an alias for Record.
func (m *OutputEntropyMonitor) Observe(token int) error {
	return m.Record(token)
}

// Step is an alias for Record.
func (m *OutputEntropyMonitor) Step(token int) error {
	return m.Record(token)
}

// Push is an alias for Record.
func (m *OutputEntropyMonitor) Push(token int) error {
	return m.Record(token)
}

// Track is an alias for Record.
func (m *OutputEntropyMonitor) Track(token int) error {
	return m.Record(token)
}

// RecordSequence records multiple tokens sequentially and halts on error.
func (m *OutputEntropyMonitor) RecordSequence(tokens []int) error {
	for _, tok := range tokens {
		if err := m.Record(tok); err != nil {
			return err
		}
	}
	return nil
}

// Tripped returns true if the circuit breaker has been tripped.
func (m *OutputEntropyMonitor) Tripped() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.tripped
}

// ConsecutiveCount returns the current count of consecutive identical tokens.
func (m *OutputEntropyMonitor) ConsecutiveCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.consecutive
}

// LastToken returns the most recently emitted token ID and true, or (0, false) if no tokens.
func (m *OutputEntropyMonitor) LastToken() (int, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if !m.hasTokens {
		return 0, false
	}
	return m.lastToken, true
}

// History returns a copy of the emitted token sequence.
func (m *OutputEntropyMonitor) History() []int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.history) == 0 {
		return nil
	}
	res := make([]int, len(m.history))
	copy(res, m.history)
	return res
}

// Reset clears history, consecutive tracking, and untrips the circuit breaker.
func (m *OutputEntropyMonitor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.consecutive = 0
	m.hasTokens = false
	m.tripped = false
	m.history = nil
}

// Entropy calculates the Shannon entropy (in bits) over the most recent window of tokens (up to 64 tokens).
// When identical consecutive tokens are emitted, entropy drops to 0.0.
func (m *OutputEntropyMonitor) Entropy() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.history) == 0 {
		return 0.0
	}
	window := m.history
	const maxWindow = 64
	if len(window) > maxWindow {
		window = window[len(window)-maxWindow:]
	}

	counts := make(map[int]int)
	for _, tok := range window {
		counts[tok]++
	}
	n := float64(len(window))
	var entropy float64
	for _, count := range counts {
		p := float64(count) / n
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}
