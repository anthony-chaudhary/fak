package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This recorder injects retirement faults after an actual selected exact norm.
// CPU storage witnesses cause identity and session retirement, not hardware.
type v41ExactNormRetirementBackend struct {
	*v41ExactNormTestBackend
	primary       any
	operation     string
	retirement    string
	cleanup       any
	armed         bool
	cleanupFaults int
	retireCalls   [3]int
}

func (b *v41ExactNormRetirementBackend) V41ExactRMSNorm(x, gain compute.Tensor, eps float32) (compute.Tensor, error) {
	if b.primary == nil {
		return b.v41ExactNormTestBackend.V41ExactRMSNorm(x, gain, eps)
	}
	b.calls++
	b.armed = true
	if b.operation == "returned-error" {
		return compute.Tensor{}, b.primary.(error)
	}
	panic(b.primary)
}

func (b *v41ExactNormRetirementBackend) BeginBatch() {}
func (b *v41ExactNormRetirementBackend) FlushBatch() { b.retire("flush", 0) }
func (b *v41ExactNormRetirementBackend) Recycle()    { b.retire("recycle", 1) }
func (b *v41ExactNormRetirementBackend) Trim()       { b.retire("trim", 2) }

func (b *v41ExactNormRetirementBackend) retire(site string, index int) {
	b.retireCalls[index]++
	if b.armed && b.retirement == site {
		b.cleanupFaults++
		panic(b.cleanup)
	}
}

// Ref #13668. Session.Close may stop at a failing hook. This witness requires
// original fault identity and no replay, not successful completion of later
// teardown steps after a panic. Both real consumers reach the shared helper.
// fak-test:runtime fast est=100ms lane=default
func TestV41ExactNormPrimaryFailureSurvivesRetirement(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("exact norm retirement witness requires cpu-ref")
	}
	for _, consumer := range []string{"index-key", "compressor"} {
		for _, retirement := range []string{"flush", "recycle", "trim"} {
			for _, cleanupKind := range []string{"typed", "opaque"} {
				for _, operation := range []string{"returned-error", "typed-panic", "opaque-panic", "forward-error-panic", "success"} {
					t.Run(consumer+"/"+retirement+"/"+cleanupKind+"/"+operation, func(t *testing.T) {
						const eps = float32(1e-6)
						gain, input := v41ExactNormTestRow(128, 0.01), v41ExactNormTestRow(128, 2)
						path := "v41-index-key-norm"
						if consumer == "compressor" {
							path = "v41-compressor-norm"
						}
						primary := &compute.BackendError{Backend: "exact-norm-retirement", Class: compute.VulkanClassExecutionFailed, Err: errors.New("primary exact norm failure")}
						b := &v41ExactNormRetirementBackend{v41ExactNormTestBackend: newV41ExactNormTestBackend(),
							primary: primary, operation: operation, retirement: retirement,
							cleanup: &compute.BackendError{Backend: "retirement", Class: compute.VulkanClassExecutionFailed, Err: errors.New("secondary retirement failure")}}
						b.outputOwner = b
						if cleanupKind == "opaque" {
							b.cleanup = &struct{ retirement string }{retirement}
						}
						var wantPanic any
						if operation == "opaque-panic" {
							wantPanic = &struct{ marker int }{13668}
							b.primary = wantPanic
						} else if operation == "forward-error-panic" {
							wantPanic = &BackendForwardOperationError{Backend: b.Name(), Forward: ForwardPathKind("deepseek41"), Path: path, Layer: 0, Stage: "rmsnorm", Cause: primary}
							b.primary = wantPanic
						} else if operation == "success" {
							b.primary = nil
						}
						m := v41IndexKeyNormTinyModel(t, gain, eps)
						if consumer == "compressor" {
							m = v41CompressorNormTinyModel(t, gain, eps)
						}
						t.Cleanup(func() {
							if err := m.CloseWeights(); err != nil {
								t.Error(err)
							}
						})
						s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
						t.Cleanup(func() {
							// The parent may fail to retire an incoming forward error.
							// Keep that as an assertion failure, not a cleanup panic.
							b.armed = false
							s.Close()
						})
						var invoke func() ([]float32, error)
						if consumer == "index-key" {
							fn := s.v41IndexKeyNormFunc()
							if fn == nil {
								t.Fatal("selected index-key callback did not bind")
							}
							invoke = func() ([]float32, error) { return fn(0, input, gain, eps) }
						} else {
							fn := s.v41CompressorNormFunc()
							if fn == nil {
								t.Fatal("selected compressor callback did not bind")
							}
							invoke = func() ([]float32, error) { return fn(0, input, gain, eps) }
						}
						var values []float32
						var err error
						gotPanic := v41IndexerTestRecover(func() { values, err = invoke() })
						if operation == "success" {
							if gotPanic != nil || err != nil || len(values) != len(input) || s.BackendSessionClosed() || b.calls != 1 || b.retireCalls != [3]int{} {
								t.Fatalf("healthy exact norm unexpectedly retired: panic=%v error=%v calls=%d retirement=%v", gotPanic, err, b.calls, b.retireCalls)
							}
							return
						}
						if gotPanic != wantPanic {
							t.Errorf("retirement changed panic identity: got %v want %v", gotPanic, wantPanic)
						}
						var closed *BackendForwardOperationError
						if values != nil || !s.BackendSessionClosed() || !errors.As(s.halFailure, &closed) || closed.Path != path || closed.Layer != 0 || closed.Stage != "rmsnorm" {
							t.Fatalf("selected exact norm lost its closed-session latch: values=%v failure=%v closed=%t", values, s.halFailure, s.BackendSessionClosed())
						}
						if operation == "forward-error-panic" && s.halFailure != wantPanic {
							t.Error("incoming forward failure lost its original pointer latch")
						}
						if wantPanic == nil && (closed.Cause != primary || !errors.Is(err, primary) || !errors.Is(err, s.halFailure)) {
							t.Errorf("retirement replaced the selected primary cause: latch=%v error=%v", s.halFailure, err)
						}
						wantCalls := [3]int{1, 0, 0}
						if retirement != "flush" {
							wantCalls[1] = 1
						}
						if retirement == "trim" {
							wantCalls[2] = 1
						}
						if b.calls != 1 || b.cleanupFaults != 1 || b.retireCalls != wantCalls {
							t.Fatalf("retirement injection was not reached exactly once: operation=%d faults=%d calls=%v", b.calls, b.cleanupFaults, b.retireCalls)
						}
						retry := v41IndexerTestRecover(func() { _, _ = invoke() })
						if retry != s.halFailure || b.calls != 1 || b.cleanupFaults != 1 || b.retireCalls != wantCalls {
							t.Fatal("retired exact norm replayed work or changed the failure latch")
						}
					})
				}
			}
		}
	}
}
