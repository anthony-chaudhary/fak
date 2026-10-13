package model

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type halLifetimeBackend struct {
	compute.Backend
	model        *Model
	newKV        int
	fault        any
	postTransfer bool
}

func (b *halLifetimeBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	b.newKV++
	if b.model.weightCloser == nil || b.model.weightCloser.sessions != 1 {
		panic("KV before model pin")
	}
	if !b.postTransfer {
		_ = b.model.CloseWeights()
		panic(b.fault)
	}
	return b.Backend.NewKV(cfg)
}
func (b *halLifetimeBackend) GraphReset() { _ = b.model.CloseWeights(); panic(b.fault) }
func (b *halLifetimeBackend) FlushBatch() { panic("secondary failed-construction cleanup") }

type halLifetimeOwnedBackend struct {
	*halLifetimeBackend
	owner sync.Mutex
}

func (b *halLifetimeOwnedBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

// fak-test:runtime fast est=100ms lane=default
func TestBackendSessionWeightLifetime(t *testing.T) {
	t.Run("closed refusal precedes validation and KV", func(t *testing.T) {
		m := v41ReducedModel(t)
		// Deliberately invalid generic LayerNorm architecture: closing wins.
		m.Cfg.ModelType, m.Cfg.DeepSeekV41, m.Cfg.LayerNorm = "unknown", nil, true
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		be := &halLifetimeBackend{Backend: compute.Default(), model: m}
		s, err := m.NewBackendSessionChecked(be)
		var closed *ModelWeightsClosedError
		if s != nil || !errors.As(err, &closed) || be.newKV != 0 {
			t.Fatalf("session=%v err=%v KV=%d", s, err, be.newKV)
		}
	})
	for _, owned := range []bool{false, true} {
		for _, transferred := range []bool{false, true} {
			t.Run(map[bool]string{false: "no-owner", true: "owned"}[owned]+map[bool]string{false: "-pre-transfer", true: "-post-transfer"}[transferred], func(t *testing.T) {
				m := v41ReducedModel(t)
				primary := errors.New("constructor operation")
				base := &halLifetimeBackend{Backend: compute.Default(), model: m, fault: primary, postTransfer: transferred}
				wrapper := &halLifetimeOwnedBackend{halLifetimeBackend: base}
				var be compute.Backend = base
				if owned {
					be = wrapper
				}
				retired := 0
				m.SetWeightCloser(v41EligibilityCloser(func() error {
					if owned {
						if !wrapper.owner.TryLock() {
							panic("retirement under owner")
						}
						wrapper.owner.Unlock()
					}
					retired++
					panic("secondary final retirement")
				}))
				var got any
				func() { defer func() { got = recover() }(); _, _ = m.NewBackendSessionChecked(be) }()
				if got != primary || retired != 1 || m.weightCloser.sessions != 0 {
					t.Fatalf("panic=%v retirements=%d pins=%d", got, retired, m.weightCloser.sessions)
				}
			})
		}
	}
}
