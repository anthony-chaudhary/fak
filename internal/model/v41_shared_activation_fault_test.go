package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ActivationRetirementResource struct {
	output   bool
	attempts int
	released bool
}

type v41ActivationRetirementRecorder struct {
	*v41DenseTestBackend
	owner         compute.Backend
	resources     map[compute.Buffer]*v41ActivationRetirementResource
	primary       error
	faultSite     string
	cleanupSite   string
	armed         bool
	cleanupFaults int
	touches       int
	activations   int
}

type v41ActivationRetirementLimited struct {
	*v41ActivationRetirementRecorder
}

func newV41ActivationRetirementRecorder(limited bool) (*v41ActivationRetirementRecorder, compute.Backend) {
	b := &v41ActivationRetirementRecorder{v41DenseTestBackend: newV41DenseTestBackend(), resources: map[compute.Buffer]*v41ActivationRetirementResource{}}
	b.owner = b
	if limited {
		b.owner = &v41ActivationRetirementLimited{b}
	}
	return b, b.owner
}

func (b *v41ActivationRetirementRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.touches++
	if dt != compute.F32 {
		panic("activation recorder requires F32 operands")
	}
	out := compute.NewF32(b.owner, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.resources[out.Buf()] = &v41ActivationRetirementResource{}
	return out
}

func (b *v41ActivationRetirementRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, _ string) compute.Tensor {
	return b.Upload(x, dt)
}

func (b *v41ActivationRetirementRecorder) activate(gate, up compute.Tensor, limit float32) compute.Tensor {
	b.touches++
	b.activations++
	if b.primary != nil && b.faultSite == "operation" {
		b.armed = true
		panic(b.primary)
	}
	g, u := b.Backend.Read(gate), b.Backend.Read(up)
	values := make([]float32, len(g))
	for i := range values {
		x, y := g[i], u[i]
		if limit > 0 {
			x, y = min(x, limit), max(-limit, min(y, limit))
		}
		values[i] = x / (1 + float32(math.Exp(float64(-x)))) * y
	}
	out := compute.NewF32(b.owner, append([]int(nil), gate.Shape...), values)
	b.resources[out.Buf()] = &v41ActivationRetirementResource{output: true}
	return out
}

func (b *v41ActivationRetirementRecorder) SwiGLU(g, u compute.Tensor) compute.Tensor {
	return b.activate(g, u, 0)
}

func (b *v41ActivationRetirementLimited) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	return b.activate(g, u, limit)
}

func (b *v41ActivationRetirementRecorder) Read(x compute.Tensor) []float32 {
	b.touches++
	if resource := b.resources[x.Buf()]; resource != nil && resource.output && b.primary != nil && b.faultSite == "readback" {
		b.armed = true
		panic(b.primary)
	}
	return b.Backend.Read(x)
}

func (b *v41ActivationRetirementRecorder) BeginBatch() { b.touches++ }

func (b *v41ActivationRetirementRecorder) FlushBatch() {
	b.touches++
	if b.armed && b.cleanupSite == "flush" {
		b.cleanupFaults++
		panic(errors.New("secondary activation flush failure"))
	}
}

func (b *v41ActivationRetirementRecorder) Free(x compute.Tensor) {
	b.touches++
	resource := b.resources[x.Buf()]
	if resource == nil {
		panic("activation attempted to free foreign storage")
	}
	resource.attempts++
	if b.armed && b.cleanupSite == "free" && b.cleanupFaults == 0 {
		b.cleanupFaults++
		panic(errors.New("secondary activation free failure"))
	}
	for i := range b.Backend.Read(x) {
		b.Backend.Read(x)[i] = float32(math.NaN())
	}
	b.Backend.Free(x)
	resource.released = true
}

func v41ActivationRetirementCensus(t *testing.T, b *v41ActivationRetirementRecorder, failedFree bool) {
	t.Helper()
	if len(b.resources) == 0 {
		t.Fatal("activation ownership witness allocated no operands")
	}
	unreleased := 0
	for _, resource := range b.resources {
		if resource.attempts != 1 {
			t.Errorf("owned activation handle Free attempts=%d, want one", resource.attempts)
		}
		if !resource.released {
			unreleased++
		}
	}
	if (!failedFree && unreleased != 0) || (failedFree && unreleased != b.cleanupFaults) {
		t.Errorf("activation release census: unreleased=%d injected failed frees=%d", unreleased, b.cleanupFaults)
	}
}

// fak-test:runtime fast est=100ms lane=default
// fak-test:justify Pins supported arithmetic, owned publication and unsupported limited decline.
func TestV41SharedActivationRetirementBaseline(t *testing.T) {
	gate, up := []float32{4, -4, 1, -1}, []float32{-4, 4, 3, -3}
	for _, limited := range []bool{false, true} {
		for _, limit := range []float32{0, 2} {
			name := "ordinary"
			if limited {
				name = "limited"
			}
			if limit > 0 {
				name += "/positive-limit"
			} else {
				name += "/zero-limit"
			}
			t.Run(name, func(t *testing.T) {
				b, backend := newV41ActivationRetirementRecorder(limited)
				s := &Session{M: &Model{Cfg: Config{MoEIntermediateSize: len(gate)}}, Backend: backend, halW: map[string]compute.Tensor{}}
				t.Cleanup(s.Close)
				g, u := append([]float32(nil), gate...), append([]float32(nil), up...)
				got, outcome, err := s.v41SharedActivationFunc()(2, g, u, limit)
				if !limited && limit > 0 {
					if err != nil || outcome != v41ProjectionDeclined || got != nil || b.touches != 0 || s.BackendSessionClosed() {
						t.Fatalf("unsupported limit did not preserve decline: outcome=%v err=%v", outcome, err)
					}
					return
				}
				if err != nil || outcome != v41ProjectionHandled || len(got) != len(gate) || b.activations != 1 || s.BackendSessionClosed() {
					t.Fatalf("supported activation failed: outcome=%v err=%v", outcome, err)
				}
				for i := range gate {
					x, y := gate[i], up[i]
					if limit > 0 {
						x, y = min(x, limit), max(-limit, min(y, limit))
					}
					want := x / (1 + float32(math.Exp(float64(-x)))) * y
					if math.Float32bits(got[i]) != math.Float32bits(want) {
						t.Errorf("activation[%d] changed or aliased released storage", i)
					}
				}
				if !reflect.DeepEqual(g, gate) || !reflect.DeepEqual(u, up) {
					t.Error("activation mutated caller operands")
				}
				v41ActivationRetirementCensus(t, b, false)
			})
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
// fak-test:justify Reproduces selected primary identity/closure failures across secondary cleanup faults.
func TestV41SharedActivationFaultRetirement(t *testing.T) {
	for _, limited := range []bool{false, true} {
		for _, site := range []string{"operation", "readback"} {
			for _, kind := range []string{"selected-primary", "backend-primary"} {
				for _, cleanup := range []string{"flush", "free"} {
					name := "ordinary"
					if limited {
						name = "limited"
					}
					t.Run(name+"/"+site+"/"+kind+"/"+cleanup, func(t *testing.T) {
						b, backend := newV41ActivationRetirementRecorder(limited)
						s := &Session{M: &Model{Cfg: Config{MoEIntermediateSize: 4}}, Backend: backend, halW: map[string]compute.Tensor{}}
						defer func() { b.armed = false; s.Close() }()
						primary := error(&compute.BackendError{Backend: "activation-primary", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage})
						if kind == "selected-primary" {
							primary = &BackendForwardOperationError{Backend: "activation-primary", Path: "original-selected-operation", Stage: "execution", Cause: primary}
						}
						b.primary, b.faultSite, b.cleanupSite = primary, site, cleanup
						callback := s.v41SharedActivationFunc()
						gate, up := []float32{1, -2, 3, -4}, []float32{-4, 3, -2, 1}
						beforeG, beforeU := append([]float32(nil), gate...), append([]float32(nil), up...)
						limit := float32(0)
						if limited {
							limit = 2
						}
						var got []float32
						var outcome v41DenseProjectionOutcome
						var cause error
						caught := v41IndexerTestRecover(func() { got, outcome, cause = callback(2, gate, up, limit) })
						if kind == "selected-primary" {
							if caught != primary || s.halFailure != primary {
								t.Errorf("selected primary identity lost: panic=%v failure=%v", caught, s.halFailure)
							}
						} else if caught != nil || outcome != v41ProjectionError || !errors.Is(cause, primary) || !errors.Is(s.halFailure, primary) {
							t.Errorf("backend primary identity lost: panic=%v outcome=%v cause=%v failure=%v", caught, outcome, cause, s.halFailure)
						}
						if got != nil || !s.BackendSessionClosed() || b.cleanupFaults == 0 {
							t.Errorf("selected fault did not close and exercise cleanup: closed=%v cleanupFaults=%d", s.BackendSessionClosed(), b.cleanupFaults)
						}
						if !reflect.DeepEqual(gate, beforeG) || !reflect.DeepEqual(up, beforeU) {
							t.Error("failed activation mutated operands")
						}
						v41ActivationRetirementCensus(t, b, cleanup == "free")
						beforeTouches := b.touches
						retry := v41IndexerTestRecover(func() { _, _, _ = callback(2, gate, up, limit) })
						if retry != s.halFailure || b.touches != beforeTouches {
							t.Error("retired activation callback touched backend again")
						}
					})
				}
			}
		}
	}
}
