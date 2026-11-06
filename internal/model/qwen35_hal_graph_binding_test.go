package model

import (
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Actual owner/COW and binding code, CPU buffers only. No GDN or graph operation
// is implemented by the embedded nil interface, and the fixture never calls it.
type graphBindingTestBackend struct {
	compute.Backend
	Qwen35GDNBackend
	domain sync.Mutex
	clones int
}

func (b *graphBindingTestBackend) ForwardExecutionOwner() *sync.Mutex { return &b.domain }
func (b *graphBindingTestBackend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	b.clones++
	return compute.NewF32(b, append([]int(nil), t.Shape...), append([]float32(nil), b.Backend.Read(t)...)), nil
}

func graphBindingFixture(t *testing.T) (*Session, *graphBindingTestBackend) {
	t.Helper()
	b := &graphBindingTestBackend{Backend: compute.Default()}
	cfg := Config{NumLayers: 3, LayerTypes: []string{"linear_attention", "full_attention", "linear_attention"},
		LinearNumKeyHeads: 1, LinearNumValueHeads: 1, LinearKeyHeadDim: 2, LinearValueHeadDim: 2, LinearConvKernelDim: 2}
	layout, err := ComputeQwen35StateLayout(cfg, "f32")
	if err != nil {
		t.Fatal(err)
	}
	q := &qwen35HALState{backend: b, layers: make([]qwen35HALLayerState, cfg.NumLayers)}
	for l := range q.layers {
		if cfg.isLinearAttnLayer(l) {
			q.layers[l] = newQwen35HALLayerState(b,
				compute.NewF32(b, []int{layout.ConvRows, layout.ConvDim}, make([]float32, layout.ConvElements)),
				compute.NewF32(b, []int{layout.RecurrentHeads, layout.KeyHeadDim, layout.ValueHeadDim}, make([]float32, layout.RecurrentElements)))
		}
	}
	kv := b.Backend.NewKV(compute.KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 2})
	s := &Session{M: &Model{Cfg: cfg}, Backend: b, halKV: kv, qwen35HAL: q}
	t.Cleanup(func() {
		// On a bounded stall failure, do not wait again on a stranded worker's locks.
		if !b.domain.TryLock() {
			return
		}
		defer b.domain.Unlock()
		if !q.mu.TryLock() {
			return
		}
		q.mu.Unlock()
		for i := range q.layers {
			if o := q.layers[i].owner; o != nil {
				if !o.mu.TryLock() {
					return
				}
				o.mu.Unlock()
			}
		}
		q.free(b)
		kv.Free()
	})
	return s, b
}

func graphBindingCapture(t *testing.T, s *Session) *qwen35GraphBinding {
	t.Helper()
	b, ok := s.bindQwen35GraphOwned()
	if !ok || b == nil || !b.matchesOwned(s) {
		t.Fatal("healthy current private owner did not bind")
	}
	return b
}

func graphBindingMutate(s *Session, b *graphBindingTestBackend, layer int) error {
	_, _, _, err := s.qwen35HAL.mutateLayer(b, layer, func(c, r compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
		b.Backend.Read(c)[0]++
		b.Backend.Read(r)[0]++
		return compute.Tensor{}, c, r, nil
	})
	return err
}

// fak-test:runtime fast est=150ms lane=default
func TestQwen35GraphOwnerBinding(t *testing.T) {
	t.Run("live-domain-no-recursive-acquisition", func(t *testing.T) {
		s, b := graphBindingFixture(t)
		if _, ok := s.bindQwen35GraphOwned(); ok {
			t.Fatal("unowned binding accepted")
		}
		done := make(chan struct{})
		var binding *qwen35GraphBinding
		var valid, returned bool
		go func() {
			defer close(done)
			scope := s.beginBackendExecution()
			complete := false
			defer scope.finishCompleted(&complete)
			binding, valid = s.bindQwen35GraphOwned()
			valid = valid && binding.matchesOwned(s) && !b.domain.TryLock()
			complete = true
			returned = true
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("binding reacquired the non-reentrant domain")
		}
		if !returned || !valid {
			t.Fatal("live scope did not bind while retaining its domain")
		}
		if binding.matchesOwned(s) {
			t.Fatal("expired scope accepted")
		}
		if !b.domain.TryLock() {
			t.Fatal("domain leaked")
		}
		b.domain.Unlock()
		scope := s.beginBackendExecution()
		defer scope.finish()
		if !binding.matchesOwned(s) {
			t.Fatal("same live domain in a later operation refused unchanged selection")
		}
	})
	t.Run("typed-nil-backend-before-capability", func(t *testing.T) {
		s, b := graphBindingFixture(t)
		scope := s.beginBackendExecution()
		defer scope.finish()
		binding := graphBindingCapture(t, s)
		defer func() { s.Backend = b }()
		s.Backend = (*graphBindingTestBackend)(nil)
		// ForwardExecutionOwner on this receiver dereferences b.domain. The guard
		// must refuse it before invocation, even though the existing scope is live.
		if qwen35GraphLiveDomain(s) != nil {
			t.Fatal("typed-nil selector retained a live domain")
		}
		if _, ok := s.bindQwen35GraphOwned(); ok {
			t.Fatal("typed-nil selector bound")
		}
		if binding.matchesOwned(s) {
			t.Fatal("typed-nil selector validated")
		}
		s.Backend = b
		if !binding.matchesOwned(s) {
			t.Fatal("unauthorized selector check mutated receipt state")
		}
	})
	t.Run("actual-mutation-and-sequence-epochs", func(t *testing.T) {
		s, b := graphBindingFixture(t)
		scope := s.beginBackendExecution()
		defer scope.finish()
		binding := graphBindingCapture(t, s)
		if err := graphBindingMutate(s, b, 0); err != nil {
			t.Fatal(err)
		}
		if binding.matchesOwned(s) || s.qwen35HAL.layers[0].owner.graphEpoch != 1 {
			t.Fatal("successful in-place mutation reused old binding")
		}
		fresh := graphBindingCapture(t, s)
		_, err := s.qwen35HAL.mutateSequence(b, func(states []compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
			for _, state := range states {
				if state.Conv.Buf() != nil {
					b.Backend.Read(state.Conv)[0]++
				}
			}
			return compute.Qwen35SequencePrefillResult{}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if fresh.matchesOwned(s) || s.qwen35HAL.layers[0].owner.graphEpoch != 2 || s.qwen35HAL.layers[2].owner.graphEpoch != 1 {
			t.Fatal("whole callback did not advance every participating pair")
		}
		graphBindingCapture(t, s)
		if !s.qwen35HAL.graphCaptureReady(s.M.Cfg) {
			t.Fatal("normal mutation disabled ordinary graph readiness")
		}
	})
	t.Run("snapshot-sharing-and-real-cow", func(t *testing.T) {
		s, b := graphBindingFixture(t)
		scope := s.beginBackendExecution()
		defer scope.finish()
		mutateSequence := func() {
			_, err := s.qwen35HAL.mutateSequence(b, func(states []compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
				for _, state := range states {
					if state.Conv.Buf() != nil {
						b.Backend.Read(state.Conv)[0]++
						b.Backend.Read(state.Recurrent)[0]++
					}
				}
				return compute.Qwen35SequencePrefillResult{}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		// Prime old epochs to one. A real COW replacement starts at zero and its
		// first callback advances to one, so epoch inequality cannot mask selection.
		mutateSequence()
		before := graphBindingCapture(t, s)
		unobserved := graphBindingCapture(t, s)
		snapshot, err := cloneQwen35HALState(s.qwen35HAL, b)
		if err != nil {
			t.Fatal(err)
		}
		defer snapshot.free(b)
		if before.matchesOwned(s) || s.qwen35HAL.graphCaptureReady(s.M.Cfg) {
			t.Fatal("shared owner accepted")
		}
		mutateSequence()
		if b.clones != 4 || unobserved.invalid || !s.qwen35HAL.graphCaptureReady(s.M.Cfg) {
			t.Fatal("COW control was masked by sticky invalidity or non-private current state")
		}
		for _, entry := range unobserved.layers {
			old := snapshot.layers[entry.layer].owner
			current := s.qwen35HAL.layers[entry.layer].owner
			if old != entry.owner || old == current || old.refs != 1 || current.refs != 1 ||
				old.graphEpoch != entry.epoch || current.graphEpoch != entry.epoch ||
				old.graphInvalid || old.graphMutating || current.graphInvalid || current.graphMutating ||
				!old.conv.Ready() || !old.recurrent.Ready() {
				t.Fatal("actual COW did not isolate selection while preserving equal healthy epochs and live snapshot state")
			}
		}
		// No failed validation touched this receipt while sharing was active.
		// Its first mismatch is the old owner still selected by the live snapshot.
		if unobserved.matchesOwned(s) {
			t.Fatal("old live snapshot satisfied current owner selection")
		}
		graphBindingCapture(t, s)
		snapshot.free(b)
		if before.matchesOwned(s) || unobserved.matchesOwned(s) {
			t.Fatal("invalid receipt refreshed after snapshot close")
		}
		graphBindingCapture(t, s)
	})
	t.Run("descriptor-selection-and-inventory", func(t *testing.T) {
		for _, mode := range []string{"shape", "layer-selection", "state-selector", "model-selector", "kv-selector", "epoch", "owner-alias"} {
			t.Run(mode, func(t *testing.T) {
				s, b := graphBindingFixture(t)
				scope := s.beginBackendExecution()
				defer scope.finish()
				binding := graphBindingCapture(t, s)
				restore := func() {}
				switch mode {
				case "shape":
					shape := s.qwen35HAL.layers[0].conv.Shape
					old := shape[0]
					shape[0]++
					restore = func() { shape[0] = old }
				case "layer-selection":
					s.M.Cfg.LayerTypes[0] = "full_attention"
					restore = func() { s.M.Cfg.LayerTypes[0] = "linear_attention" }
				case "state-selector":
					old := s.qwen35HAL
					s.qwen35HAL = &qwen35HALState{backend: b, layers: old.layers}
					restore = func() { s.qwen35HAL = old }
				case "model-selector":
					old := s.M
					s.M = &Model{Cfg: old.Cfg}
					restore = func() { s.M = old }
				case "kv-selector":
					old := s.halKV
					replacement := b.Backend.NewKV(compute.KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 2})
					defer replacement.Free()
					s.halKV = replacement
					restore = func() { s.halKV = old }
				case "epoch":
					owner := s.qwen35HAL.layers[0].owner
					owner.graphEpoch++
					restore = func() { owner.graphEpoch-- }
				case "owner-alias":
					old := s.qwen35HAL.layers[0].owner
					s.qwen35HAL.layers[0].owner = s.qwen35HAL.layers[2].owner
					restore = func() { s.qwen35HAL.layers[0].owner = old }
				}
				if binding.matchesOwned(s) {
					restore()
					t.Fatal("changed selection/descriptor accepted")
				}
				restore()
				if binding.matchesOwned(s) {
					t.Fatal("mismatch was not sticky")
				}
				graphBindingCapture(t, s)
			})
		}
	})
	t.Run("initial-role-alias-refusal", func(t *testing.T) {
		s, b := graphBindingFixture(t)
		scope := s.beginBackendExecution()
		defer scope.finish()
		old := s.qwen35HAL.layers[2]
		defer func() { s.qwen35HAL.layers[2] = old }()
		s.qwen35HAL.layers[2] = s.qwen35HAL.layers[0]
		if _, ok := s.bindQwen35GraphOwned(); ok {
			t.Fatal("same physical owner admitted twice")
		}
		s.qwen35HAL.layers[2] = newQwen35HALLayerState(b, s.qwen35HAL.layers[0].conv, old.recurrent)
		if _, ok := s.bindQwen35GraphOwned(); ok {
			t.Fatal("distinct owners with one shared tensor admitted")
		}
		s.qwen35HAL.layers[2] = old
		graphBindingCapture(t, s)
	})
	t.Run("exhaustion-and-cow-invalidity", func(t *testing.T) {
		s, b := graphBindingFixture(t)
		scope := s.beginBackendExecution()
		defer scope.finish()
		owner := s.qwen35HAL.layers[0].owner
		owner.graphEpoch = ^uint64(0) - 1
		if err := graphBindingMutate(s, b, 0); err != nil {
			t.Fatal(err)
		}
		if owner.graphEpoch != ^uint64(0) || owner.graphInvalid {
			t.Fatal("maximum epoch not issued exactly once")
		}
		graphBindingCapture(t, s)
		if err := graphBindingMutate(s, b, 0); err != nil {
			t.Fatal(err)
		}
		if owner.graphEpoch != ^uint64(0) || !owner.graphInvalid || s.qwen35HAL.graphCaptureReady(s.M.Cfg) {
			t.Fatal("exhaustion wrapped or stayed graph-eligible")
		}
		snapshot, err := cloneQwen35HALState(s.qwen35HAL, b)
		if err != nil {
			t.Fatal(err)
		}
		defer snapshot.free(b)
		if err := graphBindingMutate(s, b, 0); err != nil {
			t.Fatal(err)
		}
		if s.qwen35HAL.layers[0].owner == owner || !s.qwen35HAL.layers[0].owner.graphInvalid {
			t.Fatal("COW erased inherited graph invalidity")
		}
		if _, ok := s.bindQwen35GraphOwned(); ok {
			t.Fatal("invalid replacement rebound")
		}
	})
}

// The normal-return flag distinguishes Goexit and legacy panic(nil) without
// recovering them inside production code. Every timeout is bounded; done closes
// after all production defers and the outer execution scope have unwound.
// fak-test:runtime fast est=150ms lane=default
func TestQwen35GraphMutationFailureEpochs(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, sequence := range []bool{false, true} {
		for _, mode := range []string{"error", "panic", "nil-panic", "goexit"} {
			name := mode
			if sequence {
				name = "sequence-" + name
			}
			t.Run(name, func(t *testing.T) {
				s, b := graphBindingFixture(t)
				q := s.qwen35HAL
				marker := errors.New("callback failure")
				done := make(chan struct{})
				var returned bool
				var recovered any
				var gotErr error
				go func() {
					defer close(done)
					defer func() { recovered = recover() }()
					scope := s.beginBackendExecution()
					complete := false
					defer scope.finishCompleted(&complete)
					fail := func() error {
						switch mode {
						case "panic":
							panic(marker)
						case "nil-panic":
							panic(nil)
						case "goexit":
							runtime.Goexit()
						}
						return marker
					}
					if sequence {
						_, gotErr = q.mutateSequence(b, func([]compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
							return compute.Qwen35SequencePrefillResult{}, fail()
						})
					} else {
						_, _, _, gotErr = q.mutateLayer(b, 0, func(c, r compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
							return compute.Tensor{}, c, r, fail()
						})
					}
					complete = true
					returned = true
				}()
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Fatal("callback unwind or lock retirement stalled")
				}
				if returned != (mode == "error") || (mode == "error" && gotErr != marker) || (mode == "panic" && recovered != marker) || ((mode == "nil-panic" || mode == "goexit") && recovered != nil) {
					t.Fatalf("control transfer changed: returned=%v error=%v panic=%v", returned, gotErr, recovered)
				}
				if !q.mu.TryLock() {
					t.Fatal("state lock retained after unwind")
				}
				q.mu.Unlock()
				for _, l := range []int{0, 2} {
					o := q.layers[l].owner
					if !o.mu.TryLock() {
						t.Fatal("owner lock retained after unwind")
					}
					o.mu.Unlock()
					want := sequence || l == 0
					if o.graphInvalid != want || o.graphMutating || (want && o.graphEpoch != 1) || (!want && o.graphEpoch != 0) {
						t.Fatalf("layer %d failure/epoch flags invalid: %+v", l, o)
					}
				}
				if !b.domain.TryLock() {
					t.Fatal("outer domain retained after unwind")
				}
				b.domain.Unlock()
				scope := s.beginBackendExecution()
				defer scope.finish()
				if q.graphCaptureReady(s.M.Cfg) {
					t.Fatal("failed callback remained graph-ready")
				}
				if _, ok := s.bindQwen35GraphOwned(); ok {
					t.Fatal("failed callback rebound")
				}
				if err := graphBindingMutate(s, b, 0); err != nil {
					t.Fatal(err)
				}
				if !q.layers[0].owner.graphInvalid {
					t.Fatal("caller recovery/retry erased graph invalidity")
				}
			})
		}
	}
}
