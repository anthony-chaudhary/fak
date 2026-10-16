package model

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41SharedCompletionBackend struct {
	*v41ActivationRetirementRecorder
	mu                  sync.Mutex
	session             *Session
	checkOwner          bool
	fault               string
	original, secondary error
	allocated, freed    []compute.Buffer
	gate, up, output    compute.Buffer
	tensors             map[compute.Buffer]compute.Tensor
	readsSeen           int
	readCompleted       bool
}

func (b *v41SharedCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *v41SharedCompletionBackend) assertOwner() {
	if !b.checkOwner {
		return
	}
	scope := b.session.backendExecutionScope
	if scope == nil || !scope.live || scope.owner != &b.mu {
		panic("shared activation lacks live owner")
	}
	member := false
	for _, s := range scope.sessions {
		member = member || s == b.session
	}
	if !member {
		panic("shared activation lacks session membership")
	}
	if b.mu.TryLock() {
		b.mu.Unlock()
		panic("shared activation owner is unlocked")
	}
}
func (b *v41SharedCompletionBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.assertOwner()
	out := b.v41ActivationRetirementRecorder.UploadClass(x, dt, class, site)
	b.allocated = append(b.allocated, out.Buf())
	b.tensors[out.Buf()] = out
	return out
}
func (b *v41SharedCompletionBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.assertOwner()
	if !reflect.DeepEqual(b.allocated, []compute.Buffer{g.Buf(), u.Buf()}) || g.Buf() == u.Buf() || g.Backend() != b || u.Backend() != b {
		panic("shared activation missed owned operands")
	}
	b.gate, b.up = g.Buf(), u.Buf()
	out := b.v41ActivationRetirementRecorder.SwiGLU(g, u)
	b.output = out.Buf()
	if b.output == b.gate || b.output == b.up {
		panic("shared activation did not create fresh output")
	}
	b.allocated = append(b.allocated, out.Buf())
	b.tensors[out.Buf()] = out
	return out
}
func (b *v41SharedCompletionBackend) Read(x compute.Tensor) []float32 {
	b.assertOwner()
	if b.output == nil || x.Buf() != b.output || x.Backend() != b {
		panic("shared activation read missed owned output")
	}
	b.readsSeen++
	values := b.v41ActivationRetirementRecorder.Read(x)
	if len(values) != b.session.M.Cfg.MoEIntermediateSize {
		panic("shared activation read has wrong geometry")
	}
	switch b.fault {
	case "read-nil":
		panic(nil)
	case "read-nonnil", "backend", "selected":
		panic(b.original)
	case "read-goexit":
		runtime.Goexit()
	case "returned-shape":
		return values[:len(values)-1]
	}
	b.readCompleted = true
	return values
}
func (b *v41SharedCompletionBackend) Free(x compute.Tensor) {
	b.assertOwner()
	b.freed = append(b.freed, x.Buf())
	b.v41ActivationRetirementRecorder.Free(x)
	if !b.checkOwner {
		return
	}
	if x.Buf() == b.output {
		switch b.fault {
		case "cleanup-nil", "cleanup-nil-secondary", "read-nonnil", "backend", "selected":
			panic(nil)
		case "read-goexit", "returned-shape":
			panic(b.secondary)
		case "cleanup-goexit":
			runtime.Goexit()
		}
	}
	if b.fault == "cleanup-nil-secondary" && x.Buf() == b.up {
		panic(b.secondary)
	}
}

// The real full shared-expert consumer surrounds the selected activation with
// scalar projection callbacks and BF16 publications. This is software ownership
// and completion coverage, not a full Session forward or physical backend test.
// fak-test:runtime fast est=100ms lane=default
func TestV41SharedActivationCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	t.Setenv("FAK_PAGED_KV", "0")
	allJoined := true
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"healthy", "decline", "read-nil", "read-nonnil", "read-goexit", "backend", "selected", "returned-shape", "cleanup-nil", "cleanup-nil-secondary", "cleanup-goexit"} {
				if !allJoined {
					return
				}
				t.Run(fault, func(t *testing.T) {
					m := &Model{Cfg: Config{HiddenSize: 4, MoEIntermediateSize: 4, NumHeads: 1, NumKVHeads: 1, HeadDim: 4}}
					if fault == "decline" {
						m.Cfg.SwigluLimit = .125
					}
					recorder, _ := newV41ActivationRetirementRecorder(false)
					b := &v41SharedCompletionBackend{v41ActivationRetirementRecorder: recorder, fault: fault, tensors: map[compute.Buffer]compute.Tensor{}, original: errors.New("shared activation original"), secondary: errors.New("shared activation secondary")}
					if fault == "backend" {
						b.original = &compute.BackendError{Backend: "shared-completion", Class: compute.VulkanClassExecutionFailed, Err: b.original}
					}
					if fault == "selected" {
						b.original = &BackendForwardOperationError{Backend: "original", Path: "original-shared", Layer: 7, Stage: "original", Cause: b.original}
					}
					recorder.owner = b
					s, err := m.NewBackendSessionChecked(b)
					if err != nil {
						t.Fatal(err)
					}
					b.session, b.checkOwner = s, true
					joined := true
					t.Cleanup(func() {
						if !joined {
							return
						}
						b.checkOwner = false
						for buf, resource := range b.resources {
							if !resource.released {
								b.Backend.Free(b.tensors[buf])
								resource.released = true
							}
						}
						s.Close()
						if err := m.CloseWeights(); err != nil {
							t.Errorf("shared fixture close: %v", err)
						}
					})
					input := []float32{1.00390625, -.625, .25, .5}
					beforeInput := append([]float32(nil), input...)
					w1 := []float32{1, 0, 0, 0, 0, .5, 0, 0, 0, 0, -1, 0, 0, 0, 0, 2}
					w3 := []float32{.5, 0, 0, 0, 0, -1, 0, 0, 0, 0, 2, 0, 0, 0, 0, .25}
					w2 := []float32{1, .25, 0, 0, 0, 1, -.5, 0, 0, 0, 1, .25, .5, 0, 0, 1}
					m.manifest = map[string]tensorMeta{}
					for _, item := range []struct {
						leaf   string
						values []float32
					}{
						{"ffn.shared_experts.w1.weight", w1}, {"ffn.shared_experts.w3.weight", w3}, {"ffn.shared_experts.w2.weight", w2},
					} {
						m.manifest[layerName(0, item.leaf)] = tensorMeta{Dtype: "F32", Shape: []int{m.Cfg.HiddenSize, m.Cfg.HiddenSize}, Offset: len(m.raw), Nbytes: 4 * len(item.values)}
						for _, v := range item.values {
							m.raw = binary.LittleEndian.AppendUint32(m.raw, math.Float32bits(v))
						}
					}
					want := v41SharedExpertBF16Oracle(w1, w3, w2, input, m.Cfg.MoEIntermediateSize, m.Cfg.HiddenSize, float32(m.Cfg.SwigluLimit))
					if len(want) != m.Cfg.HiddenSize || len(want) == 0 {
						t.Fatal("empty shared oracle")
					}
					projectCalls := 0
					project := func(layer int, leaf string, x []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
						b.assertOwner()
						projectCalls++
						var weight []float32
						switch leaf {
						case "ffn.shared_experts.w1.weight":
							weight = w1
						case "ffn.shared_experts.w3.weight":
							weight = w3
						case "ffn.shared_experts.w2.weight":
							weight = w2
						default:
							panic("unexpected shared projection")
						}
						if rows != 1 || layer != 0 {
							panic("unexpected shared projection panel")
						}
						return matRows(weight, x, out, in), v41ProjectionHandled, nil
					}
					callback := s.v41SharedActivationFunc()
					if callback == nil {
						t.Fatal("shared callback did not bind")
					}
					callbackReturned := false
					var callbackOutcome v41DenseProjectionOutcome
					activate := func(layer int, gate, up []float32, limit float32) ([]float32, v41DenseProjectionOutcome, error) {
						values, outcome, err := callback(layer, gate, up, limit)
						callbackReturned, callbackOutcome = true, outcome
						return values, outcome, err
					}
					var got []float32
					var cause error
					var caught any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					joined, allJoined = false, false
					go func() {
						defer close(done)
						func() {
							defer func() { caught = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, cause = m.v41FullSharedExpertSwiGLU(0, input, m.Cfg, project, activate, nil)
							returned, complete = true, true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						joined, allJoined = true, true
					case <-time.After(5 * time.Second):
						t.Fatal("shared activation worker did not join; cleanup withheld")
					}
					diagnostic := func() {
						t.Fatalf("shared activation fault=%q returned=%t boundaryReturned=%t callbackReturned=%t outcome=%v output=%d cause=%v panic=%v closed=%t projects=%d reads=%d readCompleted=%t freed=%d parity=%t", fault, returned, boundaryReturned, callbackReturned, callbackOutcome, len(got), cause, caught, s.halClosed, projectCalls, b.readsSeen, b.readCompleted, len(b.freed), reflect.DeepEqual(got, want))
					}
					if s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("shared activation stranded its owner")
					}
					b.mu.Unlock()
					if !reflect.DeepEqual(input, beforeInput) {
						t.Fatal("shared consumer mutated input")
					}
					if fault == "healthy" || fault == "decline" {
						if !returned || !boundaryReturned || !callbackReturned || cause != nil || caught != nil || !reflect.DeepEqual(got, want) || projectCalls != 3 || s.halClosed || s.halFailure != nil {
							diagnostic()
						}
						if fault == "decline" {
							if callbackOutcome != v41ProjectionDeclined || len(b.allocated) != 0 || b.readsSeen != 0 {
								diagnostic()
							}
							return
						}
					} else if got != nil || projectCalls != 2 {
						diagnostic()
					}
					roles := []compute.Buffer{b.gate, b.up, b.output}
					if !reflect.DeepEqual(b.allocated, roles) || len(b.resources) != len(roles) || b.activations != 1 || b.readsSeen != 1 {
						diagnostic()
					}
					wantFree := []compute.Buffer{b.output, b.up, b.gate}
					if fault == "cleanup-goexit" {
						wantFree = wantFree[:1]
					}
					if !reflect.DeepEqual(b.freed, wantFree) {
						diagnostic()
					}
					for i, buf := range b.allocated {
						wantAttempts := 1
						if fault == "cleanup-goexit" && i < len(b.allocated)-1 {
							wantAttempts = 0
						}
						if b.resources[buf].attempts != wantAttempts || b.resources[buf].released != (wantAttempts == 1) {
							diagnostic()
						}
					}
					if fault == "healthy" {
						return
					}
					if fault == "read-goexit" || fault == "cleanup-goexit" {
						if returned || boundaryReturned || callbackReturned || caught != nil || cause != nil {
							diagnostic()
						}
						return
					}
					if !s.halClosed || s.halFailure == nil {
						diagnostic()
					}
					var selected *BackendForwardOperationError
					if !errors.As(s.halFailure, &selected) {
						diagnostic()
					}
					if fault != "selected" && (selected.Path != "v41-shared-activation" || selected.Layer != 0 || selected.Stage != "readback") {
						diagnostic()
					}
					switch fault {
					case "backend", "returned-shape":
						wantCause := b.original
						if fault == "returned-shape" {
							wantCause = errV41ProjectionResult
						}
						if !returned || !boundaryReturned || !callbackReturned || callbackOutcome != v41ProjectionError || caught != nil || !errors.Is(cause, wantCause) || !errors.Is(s.halFailure, wantCause) {
							diagnostic()
						}
					default:
						if returned || !boundaryReturned || callbackReturned || cause != nil {
							diagnostic()
						}
						if fault == "read-nonnil" || fault == "selected" {
							if caught != b.original || !errors.Is(s.halFailure, b.original) {
								diagnostic()
							}
						} else if mode == "1" {
							if caught != nil {
								diagnostic()
							}
						} else if _, ok := caught.(*runtime.PanicNilError); !ok {
							diagnostic()
						}
					}
					beforeTouches, beforeFree := b.touches, len(b.freed)
					var retry any
					func() {
						defer func() { retry = recover() }()
						scope := s.beginBackendExecution()
						complete := false
						defer scope.finishCompleted(&complete)
						_, _, _ = callback(0, input, input, 0)
						complete = true
					}()
					if retry != s.halFailure || b.touches != beforeTouches || len(b.freed) != beforeFree || s.backendExecutionScope != nil {
						diagnostic()
					}
					if !b.mu.TryLock() {
						t.Fatal("shared retry stranded owner")
					}
					b.mu.Unlock()
				})
			}
		})
		if !allJoined {
			return
		}
	}
}
