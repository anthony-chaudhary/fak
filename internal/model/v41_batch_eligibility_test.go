package model

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41EligibilityBackend struct {
	*executionOwnerTestBackend
	model  *Model
	probes int
	probe  func()
}

func (b *v41EligibilityBackend) V41SharedHeadAvailable(dt compute.Dtype, out, in, rows int) bool {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("unowned eligibility probe")
	}
	if !q8Mu.TryLock() {
		panic("metadata lock held during probe")
	}
	q8Mu.Unlock()
	state := b.model.ensureWeightCloser()
	if !state.mu.TryLock() {
		panic("lifetime lock held during probe")
	}
	pinned := state.sessions > 0
	state.mu.Unlock()
	if !pinned {
		panic("probe without pin")
	}
	b.probes++
	if b.probe != nil {
		b.probe()
	}
	return b.v41SharedHeadRecorder.V41SharedHeadAvailable(dt, out, in, rows)
}

type v41EligibilityCloser func() error

func (f v41EligibilityCloser) Close() error { return f() }

func v41EligibilityFixture(t *testing.T) (*Model, *v41EligibilityBackend) {
	t.Helper()
	m := v41ReducedModel(t)
	return m, &v41EligibilityBackend{executionOwnerTestBackend: newExecutionOwnerTestBackend(&sync.Mutex{}), model: m}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41BatchEligibilityLifetime(t *testing.T) {
	t.Run("exact probe and fallback retain one pin", func(t *testing.T) {
		for _, deny := range []bool{false, true} {
			m, be := v41EligibilityFixture(t)
			be.denyShared = deny
			e, err := m.AcquireV41BatchEligibility(context.Background(), be, 4)
			if err != nil || e == nil || e.Eligible() == deny || be.probes != 1 || be.calls.Load() != 0 {
				t.Fatalf("eligible=%v err=%v probes=%d operations=%d", e, err, be.probes, be.calls.Load())
			}
			closed := 0
			m.SetWeightCloser(v41EligibilityCloser(func() error {
				if !be.owner.TryLock() {
					panic("retired under owner")
				}
				be.owner.Unlock()
				closed++
				return nil
			}))
			var active *WeightSessionsActiveError
			if !errors.As(m.CloseWeights(), &active) || active.Count != 1 || closed != 0 {
				t.Fatal("request pin did not retain weights")
			}
			e.Finish(&err)
			e.Finish(&err)
			if closed != 1 {
				t.Fatalf("retirements=%d", closed)
			}
		}
	})
	t.Run("unsupported layout and missing owner", func(t *testing.T) {
		m, be := v41EligibilityFixture(t)
		meta := m.manifest["lm_head.weight"]
		meta.Shape = []int{m.Cfg.VocabSize, m.Cfg.HiddenSize + 1}
		m.manifest["lm_head.weight"] = meta
		e, err := m.AcquireV41BatchEligibility(context.Background(), be, 4)
		if err != nil || e.Eligible() || be.probes != 0 {
			t.Fatalf("unsupported: %v %v", e, err)
		}
		e.Finish(&err)
		e, err = m.AcquireV41BatchEligibility(context.Background(), compute.Default(), 4)
		if err != nil || e.Eligible() {
			t.Fatalf("missing owner: %v %v", e, err)
		}
		e.Finish(&err)
		be.owner = nil
		e, err = m.AcquireV41BatchEligibility(context.Background(), be, 4)
		if err == nil || e != nil || m.weightCloser.sessions != 0 {
			t.Fatal("nil owner failed open or leaked")
		}
	})
	t.Run("busy cancellation before probe", func(t *testing.T) {
		m, be := v41EligibilityFixture(t)
		be.owner.Lock()
		ctx, cancel := context.WithCancel(context.Background())
		be.query = make(chan struct{}, 1)
		done := make(chan error, 1)
		go func() { _, err := m.AcquireV41BatchEligibility(ctx, be, 4); done <- err }()
		select {
		case <-be.query:
		case <-time.After(time.Second):
			t.Fatal("no owner query")
		}
		cancel()
		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		case <-time.After(time.Second):
			t.Fatal("busy owner blocked cancellation")
		}
		be.owner.Unlock()
		if be.probes != 0 || be.calls.Load() != 0 || m.weightCloser.sessions != 0 {
			t.Fatal("cancel had effects or leaked pin")
		}
	})
	t.Run("primary survives final retirement", func(t *testing.T) {
		for _, kind := range []string{"error", "panic", "cleanup"} {
			m, be := v41EligibilityFixture(t)
			primary, cleanup := errors.New("operation"), errors.New("retirement")
			m.SetWeightCloser(v41EligibilityCloser(func() error {
				if !be.owner.TryLock() {
					panic("retired under owner")
				}
				be.owner.Unlock()
				panic(cleanup)
			}))
			var got error
			var escaped any
			func() {
				defer func() { escaped = recover() }()
				got = func() (err error) {
					e, err := m.AcquireV41BatchEligibility(context.Background(), be, 4)
					if err != nil {
						return err
					}
					defer e.Finish(&err)
					_ = m.CloseWeights()
					if kind == "panic" {
						panic(primary)
					}
					if kind == "error" {
						return primary
					}
					return nil
				}()
			}()
			if (kind == "error" && (got != primary || escaped != nil)) || (kind == "panic" && escaped != primary) || (kind == "cleanup" && escaped != cleanup) {
				t.Fatalf("%s: err=%v panic=%v", kind, got, escaped)
			}
			if m.weightCloser.sessions != 0 {
				t.Fatal("retirement leaked pin")
			}
		}
	})
	t.Run("probe panic unlocks before failed acquisition retirement", func(t *testing.T) {
		m, be := v41EligibilityFixture(t)
		primary := errors.New("probe")
		m.SetWeightCloser(v41EligibilityCloser(func() error {
			if !be.owner.TryLock() {
				panic("retired under owner")
			}
			be.owner.Unlock()
			panic("secondary")
		}))
		be.probe = func() { _ = m.CloseWeights(); panic(primary) }
		var got any
		func() {
			defer func() { got = recover() }()
			_, _ = m.AcquireV41BatchEligibility(context.Background(), be, 4)
		}()
		if got != primary || m.weightCloser.sessions != 0 {
			t.Fatalf("panic=%v pins=%d", got, m.weightCloser.sessions)
		}
	})
}
