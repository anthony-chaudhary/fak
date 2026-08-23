package model

import "fmt"

// HybridSpecCheckpoint identifies one speculative suffix and its accepted prefix.
// A checkpoint belongs to exactly one state and can be consumed only once.
type HybridSpecCheckpoint struct {
	owner      *HybridSpecState
	generation uint64
	cursor     int
	kv         [][]byte
	recurrent  [][]byte
}

// HybridSpecState keeps attention KV and recurrent state at one token cursor.
// Speculative writes and restoration always move both state families together.
type HybridSpecState struct {
	kv         [][]byte
	recurrent  [][]byte
	cursor     int
	generation uint64
	active     bool
}

// NewHybridSpecState copies an accepted hybrid-state prefix.
func NewHybridSpecState(kv, recurrent [][]byte) (*HybridSpecState, error) {
	if len(kv) != len(recurrent) {
		return nil, fmt.Errorf("hybrid speculative state: KV length %d differs from recurrent length %d", len(kv), len(recurrent))
	}
	return &HybridSpecState{
		kv:        cloneHybridState(kv),
		recurrent: cloneHybridState(recurrent),
		cursor:    len(kv),
	}, nil
}

// Begin snapshots the accepted prefix before speculative writes.
func (s *HybridSpecState) Begin() HybridSpecCheckpoint {
	s.generation++
	s.active = true
	return HybridSpecCheckpoint{
		owner:      s,
		generation: s.generation,
		cursor:     s.cursor,
		kv:         cloneHybridState(s.kv),
		recurrent:  cloneHybridState(s.recurrent),
	}
}

// Append adds one speculative token to both state families.
func (s *HybridSpecState) Append(kv, recurrent []byte) error {
	if !s.active {
		return fmt.Errorf("hybrid speculative state: append requires an active checkpoint")
	}
	s.kv = append(s.kv, append([]byte(nil), kv...))
	s.recurrent = append(s.recurrent, append([]byte(nil), recurrent...))
	s.cursor++
	return nil
}

// Commit keeps accepted speculative tokens and discards the rejected suffix.
func (s *HybridSpecState) Commit(checkpoint HybridSpecCheckpoint, accepted int) error {
	if err := s.validate(checkpoint); err != nil {
		return err
	}
	speculative := s.cursor - checkpoint.cursor
	if accepted < 0 || accepted > speculative {
		return fmt.Errorf("hybrid speculative state: accepted %d tokens from suffix of %d", accepted, speculative)
	}
	cursor := checkpoint.cursor + accepted
	s.kv = cloneHybridState(s.kv[:cursor])
	s.recurrent = cloneHybridState(s.recurrent[:cursor])
	s.cursor = cursor
	s.active = false
	return nil
}

// Rollback restores both state families to the exact checkpointed prefix.
func (s *HybridSpecState) Rollback(checkpoint HybridSpecCheckpoint) error {
	if err := s.validate(checkpoint); err != nil {
		return err
	}
	s.kv = cloneHybridState(checkpoint.kv)
	s.recurrent = cloneHybridState(checkpoint.recurrent)
	s.cursor = checkpoint.cursor
	s.active = false
	return nil
}

// Cursor returns the shared accepted-plus-speculative token position.
func (s *HybridSpecState) Cursor() int { return s.cursor }

// KVState returns a copy of the attention state.
func (s *HybridSpecState) KVState() [][]byte { return cloneHybridState(s.kv) }

// RecurrentState returns a copy of the recurrent GDN state.
func (s *HybridSpecState) RecurrentState() [][]byte { return cloneHybridState(s.recurrent) }

func (s *HybridSpecState) validate(checkpoint HybridSpecCheckpoint) error {
	if checkpoint.owner != s {
		return fmt.Errorf("hybrid speculative state: foreign checkpoint")
	}
	if !s.active || checkpoint.generation != s.generation {
		return fmt.Errorf("hybrid speculative state: stale checkpoint")
	}
	if len(s.kv) != len(s.recurrent) || s.cursor != len(s.kv) || checkpoint.cursor > s.cursor {
		return fmt.Errorf("hybrid speculative state: inconsistent paired state")
	}
	return nil
}

func cloneHybridState(state [][]byte) [][]byte {
	if state == nil {
		return nil
	}
	cloned := make([][]byte, len(state))
	for i := range state {
		cloned[i] = append([]byte(nil), state[i]...)
	}
	return cloned
}
