package model

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

type countWeightCloser struct{ n atomic.Int32 }

func (c *countWeightCloser) Close() error { c.n.Add(1); return nil }

func TestCloseWeightsDefersForSessionAndRefusesNewSession(t *testing.T) {
	m := NewSynthetic(Config{VocabSize: 8, HiddenSize: 8, NumLayers: 0})
	c := &countWeightCloser{}
	m.SetWeightCloser(c)
	s := m.NewSession()
	err := m.CloseWeights()
	var active *WeightSessionsActiveError
	if !errors.As(err, &active) || active.Count != 1 {
		t.Fatalf("CloseWeights error=%v want one active session", err)
	}
	if c.n.Load() != 0 {
		t.Fatal("checkpoint owner closed while session remained")
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("NewSession admitted after close began")
			}
		}()
		_ = m.NewSession()
	}()
	s.Close()
	s.Close()
	if c.n.Load() != 1 {
		t.Fatalf("checkpoint closes=%d want 1 after double Session.Close", c.n.Load())
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatalf("completed CloseWeights: %v", err)
	}
}

func TestCloseWeightsConcurrentSessionsReleaseOnce(t *testing.T) {
	m := NewSynthetic(Config{VocabSize: 8, HiddenSize: 8, NumLayers: 0})
	c := &countWeightCloser{}
	m.SetWeightCloser(c)
	const n = 32
	sessions := make([]*Session, n)
	for i := range sessions {
		sessions[i] = m.NewSession()
	}
	if err := m.CloseWeights(); err == nil {
		t.Fatal("CloseWeights admitted with live sessions")
	}
	var wg sync.WaitGroup
	for _, s := range sessions {
		wg.Add(1)
		go func(s *Session) { defer wg.Done(); s.Close(); s.Close() }(s)
	}
	wg.Wait()
	if c.n.Load() != 1 {
		t.Fatalf("checkpoint closes=%d want 1", c.n.Load())
	}
}
