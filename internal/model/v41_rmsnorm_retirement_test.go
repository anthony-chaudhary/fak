package model

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Real CPU RMSNorm and existing Session callbacks exercise owner retirement.
// The recorder does not add an exact-norm or query-prefix capability.
type v41RMSRetirementRecorder struct {
	*v41DenseTestBackend
	owner                     *Session
	leaf                      string
	outputs                   map[compute.Buffer]bool
	norms, reads, retireCalls int
	opsAtFault                int
	primary                   error
	badRead, closeFault       bool
}

func (b *v41RMSRetirementRecorder) RMSNorm(x, gain compute.Tensor, eps float32) compute.Tensor {
	target := b.owner != nil && gain.Buf() == b.owner.halW[b.leaf].Buf()
	if target {
		b.norms++
		if b.primary != nil {
			b.opsAtFault = len(b.ops)
			panic(b.primary)
		}
	}
	y := b.Backend.RMSNorm(x, gain, eps)
	b.live[y.Buf()] = true
	if target {
		b.outputs[y.Buf()] = true
	}
	return y
}

func (b *v41RMSRetirementRecorder) Read(x compute.Tensor) []float32 {
	values := b.v41DenseTestBackend.Read(x)
	if b.outputs[x.Buf()] {
		b.reads++
		if b.badRead {
			b.opsAtFault = len(b.ops)
			return values[:len(values)-1]
		}
	}
	return values
}

func (b *v41RMSRetirementRecorder) Free(x compute.Tensor) {
	delete(b.outputs, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

func (b *v41RMSRetirementRecorder) Recycle() {
	b.retireCalls++
	if b.closeFault {
		panic(errors.New("secondary generic RMSNorm Close failure"))
	}
}

func v41RMSRetirementSession(t *testing.T, m *Model, leaf string) (*Session, *v41RMSRetirementRecorder) {
	t.Helper()
	b := &v41RMSRetirementRecorder{v41DenseTestBackend: newV41DenseTestBackend(), leaf: leaf, outputs: map[compute.Buffer]bool{}}
	s := v41DenseTestSession(t, m, b)
	b.owner = s
	t.Cleanup(func() { b.primary, b.badRead, b.closeFault = nil, false, false })
	return s, b
}

func v41RMSRetirementFinite(t *testing.T, row []float32, width int) {
	t.Helper()
	if len(row) != width || width == 0 {
		t.Fatal("healthy RMSNorm control has incorrect width")
	}
	for _, value := range row {
		if !finite32(value) {
			t.Fatal("healthy RMSNorm control is nonfinite")
		}
	}
}

// All five existing callback constructors share the selected generic helper.
// An incoming typed error names another operation but must retire this owner.
// fak-test:runtime medium est=3s lane=default
func TestV41RMSNormRetiresIncomingForwardError(t *testing.T) {
	if b := compute.Default(); b == nil || b.Name() != "cpu-ref" || b.Caps().DeviceMemory {
		t.Fatal("generic RMSNorm retirement witness requires cpu-ref")
	}
	for _, tc := range []struct {
		name, leaf string
		bind       func(*Session) func([]float32) ([]float32, error)
	}{
		{"attention", layerName(0, "attn_norm.weight"), func(s *Session) func([]float32) ([]float32, error) {
			fn := s.v41AttentionInputNormFunc()
			if fn == nil {
				return nil
			}
			return func(x []float32) ([]float32, error) { return fn(0, x) }
		}},
		{"query", layerName(0, "attn.wq_a_norm.weight"), func(s *Session) func([]float32) ([]float32, error) {
			fn := s.v41QueryNormFunc()
			if fn == nil {
				return nil
			}
			return func(x []float32) ([]float32, error) { return fn(0, x) }
		}},
		{"kv", layerName(0, "attn.kv_norm.weight"), func(s *Session) func([]float32) ([]float32, error) {
			fn := s.v41KVNormFunc()
			if fn == nil {
				return nil
			}
			return func(x []float32) ([]float32, error) { return fn(0, x) }
		}},
		{"ffn", layerName(0, "ffn_norm.weight"), func(s *Session) func([]float32) ([]float32, error) {
			fn := s.v41FFNNormFunc()
			if fn == nil {
				return nil
			}
			return func(x []float32) ([]float32, error) { return fn(0, x) }
		}},
		{"final", "model.norm.weight", func(s *Session) func([]float32) ([]float32, error) { return s.v41FinalNormFunc() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			s, b := v41RMSRetirementSession(t, m, tc.leaf)
			fn := tc.bind(s)
			if fn == nil {
				t.Fatal("available generic norm callback did not bind")
			}
			input := make([]float32, len(m.tensor(tc.leaf)))
			for i := range input {
				input[i] = float32(i%7+1) / 8
			}
			before := append([]float32(nil), input...)
			good, err := fn(input)
			if err != nil || b.norms != 1 || b.reads != 1 || len(b.outputs) != 0 {
				t.Fatal("healthy selected norm did not finish", err)
			}
			v41RMSRetirementFinite(t, good, len(input))
			primary := fmt.Errorf("original %s wrapper: %w", tc.name, &BackendForwardOperationError{Path: "another-owner", Layer: 9, Stage: "incoming", Cause: ErrV41ForwardStage})
			b.primary, b.closeFault = primary, true
			var output []float32
			got := recoverError(func() { output, _ = fn(input) })
			if got != primary || s.halFailure != primary || !s.BackendSessionClosed() || b.retireCalls != 1 {
				t.Fatal("generic callback lost incoming wrapper or current-owner retirement", got)
			}
			if output != nil || b.norms != 2 || b.reads != 1 || len(b.outputs) != 0 || !reflect.DeepEqual(input, before) {
				t.Fatal("failed generic callback replayed or published")
			}
			if retry := recoverError(func() { _, _ = fn(input) }); retry != primary || b.norms != 2 || b.retireCalls != 1 {
				t.Fatal("retired generic callback retried or changed its primary")
			}
		})
	}
}

// Real query-norm consumers distinguish returned malformed-readback failure
// from an incoming typed panic, with and without a secondary Close failure.
// fak-test:runtime medium est=4s lane=default
func TestV41RMSNormSessionRetirement(t *testing.T) {
	if b := compute.Default(); b == nil || b.Name() != "cpu-ref" || b.Caps().DeviceMemory {
		t.Fatal("generic RMSNorm retirement witness requires cpu-ref")
	}
	for _, suffix := range []bool{false, true} {
		for _, returned := range []bool{false, true} {
			for _, closeFault := range []bool{false, true} {
				t.Run(fmt.Sprintf("suffix=%t/returned=%t/close-fault=%t", suffix, returned, closeFault), func(t *testing.T) {
					m := v41FullStepPatchedNorms(t)
					s, b := v41RMSRetirementSession(t, m, layerName(0, "attn.wq_a_norm.weight"))
					peer := v41DenseTestSession(t, m, b)
					t.Cleanup(func() { b.primary, b.badRead, b.closeFault = nil, false, false })
					v41RMSRetirementFinite(t, s.Prefill([]int{1, 2}), m.Cfg.VocabSize)
					if !s.v41IncrementalEligible() || b.norms != 2 || b.reads != 2 {
						t.Fatal("real query norm did not seed the incremental path")
					}
					before := captureV41ForwardSnapshot(s.v41Forward)
					primary := fmt.Errorf("original query wrapper: %w", &BackendForwardOperationError{Path: "another-owner", Layer: 9, Stage: "incoming", Cause: ErrV41ForwardStage})
					if returned {
						b.badRead = true
					} else {
						b.primary = primary
					}
					b.closeFault = closeFault
					var output []float32
					got := recoverError(func() {
						if suffix {
							output = s.Prefill([]int{3})
						} else {
							output = s.Step(3)
						}
					})
					if returned {
						var selected *BackendForwardOperationError
						if !errors.As(got, &selected) || selected != s.halFailure || selected.Path != "v41-query-norm" || selected.Stage != "readback" || !errors.Is(got, errV41ProjectionResult) {
							t.Fatal("returned norm failure lost its original attributed cause", got)
						}
					} else if got != primary || s.halFailure != primary {
						t.Fatal("incoming norm panic lost wrapper identity", got)
					}
					wantReads := 2
					if returned {
						wantReads++
					}
					if !s.BackendSessionClosed() || b.retireCalls != 1 || output != nil || b.norms != 3 || b.reads != wantReads || len(b.outputs) != 0 || len(b.ops) != b.opsAtFault ||
						!reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) || peer.BackendSessionClosed() || peer.halFailure != nil {
						t.Fatal("failed query norm replayed, committed continuation or retired another owner")
					}
					if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.norms != 3 || b.retireCalls != 1 {
						t.Fatal("closed query norm owner retried or replaced its latch")
					}
					b.primary, b.badRead, b.closeFault = nil, false, false
					v41RMSRetirementFinite(t, peer.Prefill([]int{4}), m.Cfg.VocabSize)
					if peer.BackendSessionClosed() {
						t.Fatal("unrelated session could not continue")
					}
				})
			}
		}
	}
}
