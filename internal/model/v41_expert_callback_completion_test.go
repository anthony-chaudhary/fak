package model

import (
	"errors"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type expertCallbackCompletionBackend struct {
	*v41HalSeamBackend
	owner   sync.Mutex
	caps    func()
	decline bool
}

func (b *expertCallbackCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *expertCallbackCompletionBackend) Caps() compute.Caps {
	if b.caps != nil {
		b.caps()
	}
	c := b.v41HalSeamBackend.Caps()
	if b.decline {
		c.DeviceMemory = false
	}
	return c
}

// fak-test:runtime fast est=2s lane=default
func TestV41ExpertCallbacksPreserveIncompleteUnwind(t *testing.T) {
	priorSDOTForce := q4kSDOTForce
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { q4kSDOTForce = priorSDOTForce })
	m := v41MixedQuantExpertModel(t)
	if full, err := v41ForwardGeometry(m.Cfg); full || err != nil {
		t.Fatalf("reduced host-fallback fixture: full=%t err=%v", full, err)
	}
	input := make([]float32, m.Cfg.HiddenSize)
	for i := range input {
		// Bound mixed-quant reduction cancellation without relaxing parity.
		input[i] = float32((i%11)-5) / 1024
	}
	oracle, err := m.v41IncrementalExpert(0, "ffn.experts.0", input, 1, &v41ProjScratch{})
	if err != nil || len(oracle) != m.Cfg.HiddenSize {
		t.Fatalf("host oracle=%v err=%v", oracle, err)
	}
	primary := errors.New("expert callback primary")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, stage := range []string{"gate-up", "down"} {
				for _, fault := range []string{"nil", "nonnil", "goexit", "backend", "cuda", "closed-marker", "decline", "healthy"} {
					t.Run(stage+"/"+fault, func(t *testing.T) {
						b := &expertCallbackCompletionBackend{v41HalSeamBackend: &v41HalSeamBackend{Backend: compute.Default()}}
						s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
						t.Cleanup(s.Close)
						gate, down := s.v41ExpertGateUpFunc(), s.v41ExpertDownFunc()
						if gate == nil || down == nil {
							t.Fatal("callbacks not bound")
						}
						var typed error
						switch fault {
						case "backend":
							typed = &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: primary}
						case "cuda":
							typed = &compute.CUDAOpError{Op: "Expert", Err: primary}
						case "closed-marker":
							typed = &BackendForwardOperationError{Backend: "primary", Path: "already-closed", Stage: "expert", Cause: primary}
						}
						phase := ""
						calls, injections, gateDone := 0, 0, 0
						badOwned := false
						badIntermediate := false
						b.caps = func() {
							if phase != stage {
								return
							}
							calls++
							if s.backendExecutionScope == nil || !s.backendExecutionScope.live || s.backendExecutionScope.owner != &b.owner {
								badOwned = true
							}
							if b.owner.TryLock() {
								b.owner.Unlock()
								badOwned = true
							}
							if stage == "down" && (gateDone != 1 || b.matmuls != 2) {
								badOwned = true
							}
							if calls != 1 {
								return
							}
							switch fault {
							case "nil":
								injections++
								panic(nil)
							case "nonnil":
								injections++
								panic(primary)
							case "goexit":
								injections++
								runtime.Goexit()
							case "backend", "cuda", "closed-marker":
								injections++
								panic(typed)
							case "decline":
								b.decline = true
							}
						}
						scratch := &v41ProjScratch{}
						scratch.expertGateUp = func(l int, stem string, x []float32) ([]float32, v41ExpertGateUpOutcome, error) {
							phase = "gate-up"
							defer func() { phase = ""; b.decline = false }()
							y, outcome, cause := gate(l, stem, x)
							if outcome == v41GateUpHandled && cause == nil {
								gateDone++
								nonzero := false
								if len(y) != m.Cfg.MoEIntermediateSize {
									badIntermediate = true
								}
								for _, value := range y {
									if !(value >= -2 && value <= 2) {
										badIntermediate = true
									}
									nonzero = nonzero || value != 0
								}
								badIntermediate = badIntermediate || !nonzero
							}
							return y, outcome, cause
						}
						scratch.expertDown = func(l int, stem string, x []float32) ([]float32, v41ExpertDownOutcome, error) {
							phase = "down"
							defer func() { phase = ""; b.decline = false }()
							return down(l, stem, x)
						}
						var got []float32
						var cause error
						var payload any
						returned, boundaryReturned := false, false
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { payload = recover() }()
								func() {
									scope := s.beginBackendExecution()
									complete := false
									defer scope.finishCompleted(&complete)
									got, cause = m.v41IncrementalExpert(0, "ffn.experts.0", input, 1, scratch)
									complete = true
								}()
								returned = true
							}()
							boundaryReturned = true
						}()
						<-done
						b.caps = nil
						ordinary := fault == "backend" || fault == "cuda" || fault == "decline" || fault == "healthy"
						if returned != ordinary || boundaryReturned == (fault == "goexit") {
							t.Fatalf("expert stage=%q fault=%q returned=%t boundaryReturned=%t", stage, fault, returned, boundaryReturned)
						}
						wantInject := 1
						if fault == "decline" || fault == "healthy" {
							wantInject = 0
						}
						if badIntermediate {
							t.Fatal("handled gate intermediate must be nonempty, finite, bounded, and nonzero")
						}
						if calls < 1 || injections != wantInject || badOwned || (stage == "down" && gateDone != 1) {
							t.Fatalf("calls=%d injections=%d gateDone=%d badOwned=%t", calls, injections, gateDone, badOwned)
						}
						if s.backendExecutionScope != nil || !b.owner.TryLock() {
							t.Fatal("owner not released")
						}
						b.owner.Unlock()
						closed := fault != "healthy" && fault != "decline" && fault != "closed-marker" && fault != "goexit"
						if s.BackendSessionClosed() != closed {
							t.Fatalf("closed=%t want=%t", s.BackendSessionClosed(), closed)
						}
						switch fault {
						case "nil":
							if mode == "0" {
								if _, ok := payload.(*runtime.PanicNilError); !ok {
									t.Fatalf("nil panic=%T %v", payload, payload)
								}
							} else if payload != nil {
								t.Fatalf("nil panic=%v", payload)
							}
						case "nonnil":
							if payload != primary {
								t.Fatalf("primary=%v", payload)
							}
						case "goexit":
							if payload != nil {
								t.Fatalf("Goexit=%v", payload)
							}
						case "closed-marker":
							if payload != typed {
								t.Fatalf("closed marker=%v", payload)
							}
						case "backend", "cuda":
							var selected *V41ExpertOperationError
							var closed *BackendForwardOperationError
							if payload != nil || !errors.Is(cause, typed) || !errors.As(cause, &selected) || !errors.As(cause, &closed) || selected.Layer != 0 || closed.Path != "v41-routed-expert" {
								t.Fatalf("typed cause=%v panic=%v", cause, payload)
							}
							wantStage := "gate/up"
							if stage == "down" {
								wantStage = "down"
							}
							if selected.Stage != wantStage || closed.Stage != wantStage {
								t.Fatalf("stages=%q/%q", selected.Stage, closed.Stage)
							}
						case "healthy", "decline":
							if cause != nil || payload != nil || len(got) != m.Cfg.HiddenSize {
								t.Fatalf("healthy output=%v cause=%v panic=%v", got, cause, payload)
							}
							assertV41LogitsClose(t, got, oracle, "expert host parity")
							wantMatMul := 3
							if fault == "decline" {
								wantMatMul = 0
								if stage == "down" {
									wantMatMul = 2
								}
							}
							if b.matmuls != wantMatMul {
								t.Fatalf("MatMul=%d want=%d", b.matmuls, wantMatMul)
							}
						}
						if fault != "healthy" && fault != "decline" && got != nil {
							t.Fatal("fault returned output")
						}
					})
				}
			}
		})
	}
}
