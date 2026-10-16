package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type compressorNormCompletionBackend struct {
	*v41CompressorNormQualifiedRecorder
	mu         sync.Mutex
	armed      bool
	faultName  string
	injections int
	activation compute.Tensor
	output     compute.Tensor
}

func (b *compressorNormCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *compressorNormCompletionBackend) assertOwner() {
	if !b.armed {
		return
	}
	scope := b.owner.backendExecutionScope
	if scope == nil || !scope.live || scope.owner != &b.mu {
		panic("compressor normalization lacks its execution owner")
	}
	member := false
	for _, s := range scope.sessions {
		member = member || s == b.owner
	}
	if !member {
		panic("compressor normalization lacks its session binding")
	}
	if b.mu.TryLock() {
		b.mu.Unlock()
		panic("compressor normalization owner is unlocked")
	}
}
func (b *compressorNormCompletionBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.assertOwner()
	out := b.v41CompressorNormQualifiedRecorder.UploadClass(x, dt, class, site)
	if strings.HasPrefix(site, "V4.1 RMSNorm activation ") {
		b.activation = out
	}
	return out
}
func (b *compressorNormCompletionBackend) RMSNorm(x, gain compute.Tensor, eps float32) compute.Tensor {
	b.assertOwner()
	if b.armed && b.faultName != "healthy" && b.faultName != "cleanup-nil" {
		b.injections++
	}
	b.output = b.v41CompressorNormQualifiedRecorder.RMSNorm(x, gain, eps)
	return b.output
}
func (b *compressorNormCompletionBackend) Read(x compute.Tensor) []float32 {
	b.assertOwner()
	return b.v41CompressorNormQualifiedRecorder.Read(x)
}
func (b *compressorNormCompletionBackend) Free(x compute.Tensor) {
	b.assertOwner()
	_, output := b.outputs[x.Buf()]
	b.v41CompressorNormQualifiedRecorder.Free(x)
	if b.armed && output && b.faultName == "cleanup-nil" {
		b.injections++
		panic(nil)
	}
}

// The actual pool must not report a completed nil row after its selected
// normalization unwinds. This is a pool-completion witness, not a full forward.
// fak-test:runtime fast est=500ms lane=default
func TestV41CompressorNormCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	allJoined := true
	for _, mode := range []string{"0", "1"} {
		if !allJoined {
			return
		}
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"rmsnorm-nil", "nonnil", "backend", "goexit", "cleanup-nil", "healthy"} {
				if !allJoined {
					return
				}
				t.Run(fault, func(t *testing.T) {
					gain, eps := []float32{1, -1, 0.5, 2}, float32(1e-5)
					first, second, scores := []float32{1, 2, 3, 4}, []float32{3, 4, 5, 6}, []float32{0, 0, 0, 0}
					before := fmt.Sprint(first, second, scores, gain)
					m := v41CompressorNormTinyModel(t, gain, eps)
					b := &compressorNormCompletionBackend{v41CompressorNormQualifiedRecorder: &v41CompressorNormQualifiedRecorder{
						v41CompressorNormRecorder: newV41CompressorNormRecorder(), width: len(gain), epsBits: math.Float32bits(eps), allow: true,
					}, faultName: fault}
					s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
					b.owner = s
					t.Cleanup(func() {
						if allJoined {
							b.armed = false
							s.Close()
							if err := m.CloseWeights(); err != nil {
								t.Error(err)
							}
						}
					})
					if _, exact := any(b).(compute.V41ExactRMSNormBackend); exact || s.v41BF16PublishFunc() != nil {
						t.Fatal("fixture did not select the qualified generic normalization route")
					}
					normalize := s.v41CompressorNormFunc()
					if normalize == nil {
						t.Fatal("compressor normalization absent")
					}
					pool, err := NewV41CompressorPool(2, len(gain))
					if err != nil {
						t.Fatal(err)
					}
					callbackReturned := false
					pool.normalize = func(pooled, weight []float32, eps float32) ([]float32, error) {
						row, err := normalize(0, pooled, weight, eps)
						callbackReturned = true
						return row, err
					}
					if row, emitted, err := pool.PushNormalized(0, first, scores, gain, eps); err != nil || emitted || row != nil || callbackReturned || b.io != 0 {
						t.Fatal("incomplete pool dispatched normalization or published a row")
					}
					host, err := NewV41CompressorPool(2, len(gain))
					if err != nil {
						t.Fatal(err)
					}
					if _, emitted, err := host.PushNormalized(0, first, scores, gain, eps); err != nil || emitted {
						t.Fatal("host oracle did not retain the first row")
					}
					want, hostEmitted, err := host.PushNormalized(1, second, scores, gain, eps)
					if err != nil || !hostEmitted || len(want) != len(gain) || want[0] == 0 {
						t.Fatal("host pool oracle is empty")
					}
					primary := errors.New("compressor normalization primary")
					typed := &compute.BackendError{Backend: "compressor-recorder", Class: compute.VulkanClassExecutionFailed, Err: primary}
					if fault != "healthy" && fault != "cleanup-nil" {
						b.faultLayer, b.faultSite = 0, "rmsnorm"
						switch fault {
						case "nonnil":
							b.fault = primary
						case "backend":
							b.fault = typed
						case "goexit":
							b.beforeFault = runtime.Goexit
						}
					}
					var got []float32
					var cause error
					var payload any
					returned, boundaryReturned, emitted := false, false, false
					done := make(chan struct{})
					b.armed, allJoined = true, false
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, emitted, cause = pool.PushNormalized(1, second, scores, gain, eps)
							returned, complete = true, true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						allJoined = true
					case <-time.After(5 * time.Second):
						t.Fatal("compressor worker did not join; cleanup withheld")
					}
					diagnostic := func() {
						t.Fatalf("compressor norm fault=%q returned=%t boundaryReturned=%t callbackReturned=%t emitted=%t width=%d cause_nil=%t closed=%t panic=%v norms=%d reads=%d injections=%d", fault, returned, boundaryReturned, callbackReturned, emitted, len(got), cause == nil, s.halClosed, payload, len(b.records), b.readbacks, b.injections)
					}
					if s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("compressor normalization stranded its execution owner")
					}
					b.mu.Unlock()
					if before != fmt.Sprint(first, second, scores, gain) || len(b.records) != 1 || b.activation.Buf() == nil || b.frees[b.activation.Buf()] != 1 || pool.nextPos != 2 || pool.filled != 0 {
						diagnostic()
					}
					if !v41CompressorNormEqualBits(b.records[0].input, []float32{2, 3, 4, 5}) || !v41CompressorNormEqualBits(b.records[0].gain, gain) || math.Float32bits(b.records[0].eps) != math.Float32bits(eps) {
						t.Fatal("actual pooled operands did not reach the normalization backend")
					}
					v41CompressorNormAssertNoTransient(t, b.v41CompressorNormRecorder)
					for weight := range b.weights {
						if b.frees[weight] != 0 {
							t.Fatal("selected compressor released immutable model gain")
						}
					}
					wantReads, wantInjections := 0, 1
					if fault == "healthy" || fault == "cleanup-nil" {
						wantReads = 1
						if b.output.Buf() == nil || b.frees[b.output.Buf()] != 1 {
							diagnostic()
						}
					}
					if fault == "healthy" {
						wantInjections = 0
					}
					if b.readbacks != wantReads || b.injections != wantInjections {
						diagnostic()
					}
					if fault == "healthy" {
						if !returned || !boundaryReturned || !callbackReturned || !emitted || cause != nil || payload != nil || s.halClosed || s.halFailure != nil || !v41CompressorNormEqualBits(got, want) {
							diagnostic()
						}
						return
					}
					if fault == "goexit" {
						if returned || boundaryReturned || callbackReturned || emitted || got != nil || cause != nil || payload != nil {
							diagnostic()
						}
						return
					}
					var closed *BackendForwardOperationError
					if !s.halClosed || !errors.As(s.halFailure, &closed) || closed.Path != "v41-compressor-norm" || closed.Layer != 0 || got != nil {
						diagnostic()
					}
					stage := "rmsnorm"
					if fault == "cleanup-nil" {
						stage = "readback"
					}
					if closed.Stage != stage {
						diagnostic()
					}
					if fault == "backend" {
						if !returned || !boundaryReturned || !callbackReturned || !emitted || payload != nil || !errors.Is(cause, typed) || closed.Cause != typed {
							diagnostic()
						}
						v41CompressorNormAssertFailure(t, s, cause, 0, stage)
					} else {
						if returned || !boundaryReturned || callbackReturned || emitted || cause != nil {
							diagnostic()
						}
						if fault == "nonnil" {
							if payload != primary || closed.Cause != primary {
								diagnostic()
							}
						} else if mode == "1" {
							if payload != nil {
								diagnostic()
							}
						} else if _, ok := payload.(*runtime.PanicNilError); !ok {
							diagnostic()
						}
					}
					io := b.io
					var retry any
					func() {
						defer func() { retry = recover() }()
						scope := s.beginBackendExecution()
						complete := false
						defer scope.finishCompleted(&complete)
						_, _ = normalize(0, first, gain, eps)
						complete = true
					}()
					if retry != s.halFailure || b.io != io || len(b.records) != 1 {
						t.Fatal("closed compressor retried backend work or changed the primary latch")
					}
				})
			}
		})
	}
}
