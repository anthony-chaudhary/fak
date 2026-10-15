package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type rmsOrdinaryCompletionBackend struct {
	*v41RMSRetirementRecorder
	executionOwner sync.Mutex
	fault          string
	primary        error
	injected       int
	badOwner       bool
}

func (b *rmsOrdinaryCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.executionOwner }
func (b *rmsOrdinaryCompletionBackend) RMSNorm(x, gain compute.Tensor, eps float32) compute.Tensor {
	target := b.owner != nil && gain.Buf() == b.owner.halW[b.leaf].Buf()
	if target && b.fault != "" {
		scope := b.owner.backendExecutionScope
		if scope == nil || !scope.live || scope.owner != &b.executionOwner {
			b.badOwner = true
		}
		if b.executionOwner.TryLock() {
			b.executionOwner.Unlock()
			b.badOwner = true
		}
		switch b.fault {
		case "nil", "nonnil", "goexit", "backend", "cuda", "closed-marker":
			b.norms++
			b.opsAtFault = len(b.ops)
			b.injected++
			switch b.fault {
			case "nil":
				panic(nil)
			case "goexit":
				runtime.Goexit()
			default:
				panic(b.primary)
			}
		}
	}
	return b.v41RMSRetirementRecorder.RMSNorm(x, gain, eps)
}

// fak-test:runtime medium est=8s lane=default
func TestV41OrdinaryRMSNormPreservesIncompleteUnwind(t *testing.T) {
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, entry := range []string{"query", "step"} {
				faults := []string{"nil", "nonnil", "goexit", "healthy"}
				if entry == "query" {
					faults = append(faults, "backend", "cuda", "closed-marker", "bad-read")
				}
				for _, fault := range faults {
					t.Run(entry+"/"+fault, func(t *testing.T) {
						m := v41FullStepPatchedNorms(t)
						leaf := layerName(0, "attn.wq_a_norm.weight")
						rec := &v41RMSRetirementRecorder{v41DenseTestBackend: newV41DenseTestBackend(), leaf: leaf, outputs: map[compute.Buffer]bool{}}
						b := &rmsOrdinaryCompletionBackend{v41RMSRetirementRecorder: rec}
						s := v41DenseTestSession(t, m, b)
						rec.owner = s
						t.Cleanup(func() { b.fault = ""; rec.badRead = false; rec.closeFault = false })
						query := s.v41QueryNormFunc()
						if query == nil {
							t.Fatal("generic query norm not bound")
						}
						input := make([]float32, m.Cfg.QLoraRank)
						for i := range input {
							input[i] = float32(i%7+1) / 8
						}
						beforeInput := append([]float32(nil), input...)
						var want []float32
						if entry == "step" {
							v41RMSRetirementFinite(t, s.Prefill([]int{1, 2}), m.Cfg.VocabSize)
							if !s.v41IncrementalEligible() || b.norms != 2 {
								t.Fatal("real query norm did not seed incremental state")
							}
							if fault == "healthy" {
								control := v41DenseTestSession(t, m, newV41DenseTestBackend())
								control.Prefill([]int{1, 2})
								want = control.Step(3)
							}
						} else {
							want = rmsnormCfg(input, m.tensor(leaf), float32(m.Cfg.RMSNormEps), m.Cfg)
						}
						var history []int
						if entry == "step" {
							history = append([]int(nil), s.v41State().history...)
						}
						var before *v41ForwardSnapshot
						if entry == "step" {
							before = captureV41ForwardSnapshot(s.v41Forward)
						}
						primary := errors.New("ordinary norm primary")
						b.primary = primary
						switch fault {
						case "backend":
							b.primary = &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: primary}
						case "cuda":
							b.primary = &compute.CUDAOpError{Op: "RMSNorm", Err: primary}
						case "closed-marker":
							b.primary = &BackendForwardOperationError{Path: "another-owner", Layer: 9, Stage: "incoming", Cause: primary}
						}
						baseNorms, baseReads := b.norms, b.reads
						b.fault = fault
						rec.badRead = fault == "bad-read"
						rec.closeFault = fault != "healthy"
						returned, boundaryReturned := false, false
						var output []float32
						var cause error
						var payload any
						done := make(chan struct{})
						go func() {
							defer close(done)
							func() {
								defer func() { payload = recover() }()
								if entry == "step" {
									output = s.Step(3)
								} else {
									func() {
										scope := s.beginBackendExecution()
										complete := false
										defer func() { successful := complete && cause == nil; scope.finishCompleted(&successful) }()
										output, cause = query(0, input)
										complete = true
									}()
								}
								returned = true
							}()
							boundaryReturned = true
						}()
						<-done
						b.fault = ""
						rec.badRead = false
						rec.closeFault = false
						ordinary := fault == "healthy" || (entry == "query" && (fault == "backend" || fault == "cuda" || fault == "bad-read"))
						if returned != ordinary || boundaryReturned == (fault == "goexit") {
							t.Fatalf("ordinary norm entry=%q fault=%q returned=%t boundaryReturned=%t", entry, fault, returned, boundaryReturned)
						}
						wantInject := 1
						if fault == "healthy" || fault == "bad-read" {
							wantInject = 0
						}
						wantReads := 0
						if fault == "healthy" || fault == "bad-read" {
							wantReads = 1
						}
						if b.injected != wantInject || b.badOwner || b.norms != baseNorms+1 || b.reads != baseReads+wantReads || len(b.outputs) != 0 {
							t.Fatalf("injected=%d badOwner=%t norms=%d reads=%d live=%d", b.injected, b.badOwner, b.norms-baseNorms, b.reads-baseReads, len(b.outputs))
						}
						if s.backendExecutionScope != nil || !b.executionOwner.TryLock() {
							t.Fatal("owner not released")
						}
						b.executionOwner.Unlock()
						closed := fault != "healthy" && fault != "goexit"
						if s.BackendSessionClosed() != closed {
							t.Fatalf("closed=%t want=%t", s.BackendSessionClosed(), closed)
						}
						if !reflect.DeepEqual(input, beforeInput) {
							t.Fatal("input changed")
						}
						if fault != "healthy" {
							if output != nil {
								t.Fatal("failure returned output")
							}
							if entry == "step" && !reflect.DeepEqual(s.v41State().history, history) {
								t.Fatal("failure committed history")
							}
							if entry == "step" && !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
								t.Fatal("failed query norm changed state")
							}
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
						case "nonnil", "closed-marker":
							if payload != b.primary {
								t.Fatalf("primary=%v", payload)
							}
						case "goexit":
							if payload != nil || s.halFailure != nil {
								t.Fatalf("Goexit converted: %v / %v", payload, s.halFailure)
							}
						case "backend", "cuda", "bad-read":
							var selected *BackendForwardOperationError
							if payload != nil || !errors.As(cause, &selected) || selected.Path != "v41-query-norm" {
								t.Fatalf("selected cause=%v panic=%v", cause, payload)
							}
							wantStage := "rmsnorm"
							expected := b.primary
							if fault == "bad-read" {
								wantStage = "readback"
								expected = errV41ProjectionResult
							}
							if selected.Stage != wantStage || !errors.Is(cause, expected) {
								t.Fatalf("stage/cause=%v", cause)
							}
						case "healthy":
							if cause != nil || payload != nil || len(output) == 0 {
								t.Fatalf("healthy output=%v error=%v panic=%v", output, cause, payload)
							}
							assertV41LogitsClose(t, output, want, "ordinary RMSNorm parity")
						}
						if closed && b.retireCalls != 1 {
							t.Fatalf("retirement attempts=%d", b.retireCalls)
						}
					})
				}
			}
		})
	}
}
