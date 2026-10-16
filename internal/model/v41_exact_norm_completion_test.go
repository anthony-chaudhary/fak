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

type exactNormCompletionBackend struct {
	*v41ExactNormTestBackend
	mu                         sync.Mutex
	session                    *Session
	armed                      bool
	fault                      string
	primary                    error
	typed                      *compute.BackendError
	incoming                   *BackendForwardOperationError
	weight, activation, output compute.Tensor
	frees                      map[compute.Buffer]int
	order                      []string
	uploads, reads, injections int
	input, gain, scratch       []float32
}

func (b *exactNormCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *exactNormCompletionBackend) assertOwner() {
	if !b.armed {
		return
	}
	scope := b.session.backendExecutionScope
	if scope == nil || !scope.live || scope.owner != &b.mu {
		panic("exact normalization lacks its live owner")
	}
	member := false
	for _, s := range scope.sessions {
		member = member || s == b.session
	}
	if !member {
		panic("exact normalization lacks its session membership")
	}
	if b.mu.TryLock() {
		b.mu.Unlock()
		panic("exact normalization owner is unlocked")
	}
}
func (b *exactNormCompletionBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.assertOwner()
	b.uploads++
	y := b.v41DenseTestBackend.Upload(x, dt)
	name := layerName(0, "attn.compressor.norm.weight")
	switch {
	case dt == compute.F32 && class == compute.MemoryWeights && site == "hal-weight "+name:
		b.weight = y
	case dt == compute.F32 && class == compute.MemoryActivation && site == "V4.1 RMSNorm activation "+name:
		b.activation = y
	default:
		panic("unexpected exact normalization upload")
	}
	return y
}
func (b *exactNormCompletionBackend) V41ExactRMSNorm(x, weight compute.Tensor, eps float32) (compute.Tensor, error) {
	b.assertOwner()
	if x.Buf() != b.activation.Buf() || weight.Buf() != b.weight.Buf() || x.Buf() == weight.Buf() || b.frees[weight.Buf()] != 0 {
		panic("exact normalization changed activation or borrowed gain identity")
	}
	b.input = append([]float32(nil), b.Backend.Read(x)...)
	b.gain = append([]float32(nil), b.Backend.Read(weight)...)
	if b.fault == "backend-panic" || b.fault == "incoming-forward" {
		b.calls++
		b.injections++
		if b.fault == "incoming-forward" {
			panic(b.incoming)
		}
		panic(b.typed)
	}
	y, err := b.v41ExactNormTestBackend.V41ExactRMSNorm(x, weight, eps)
	if err != nil {
		return y, err
	}
	b.output = y
	if y.Backend() != b || y.Buf() == x.Buf() || y.Buf() == weight.Buf() {
		panic("exact normalization output is not freshly owned")
	}
	if b.fault == "returned-error" {
		b.injections++
		return y, b.typed
	}
	return y, nil
}
func (b *exactNormCompletionBackend) Read(x compute.Tensor) []float32 {
	b.assertOwner()
	if x.Buf() != b.output.Buf() {
		panic("unexpected exact normalization read")
	}
	b.reads++
	b.scratch = b.v41DenseTestBackend.Read(x)
	switch b.fault {
	case "read-nil":
		b.injections++
		panic(nil)
	case "read-nonnil":
		b.injections++
		panic(b.primary)
	case "read-goexit":
		b.injections++
		runtime.Goexit()
	case "short-read":
		b.injections++
		return b.scratch[:len(b.scratch)-1]
	}
	return b.scratch
}
func (b *exactNormCompletionBackend) Free(x compute.Tensor) {
	b.assertOwner()
	role := ""
	switch x.Buf() {
	case b.activation.Buf():
		role = "activation"
	case b.output.Buf():
		role = "output"
	case b.weight.Buf():
		role = "weight"
	}
	if role == "" || x.Buf() == nil {
		panic("unexpected exact normalization free")
	}
	b.frees[x.Buf()]++
	if b.frees[x.Buf()] != 1 {
		panic("exact normalization released a role twice")
	}
	if b.armed && role != "weight" && b.frees[b.weight.Buf()] != 0 {
		panic("borrowed gain retired before transient cleanup")
	}
	b.order = append(b.order, role)
	b.v41DenseTestBackend.Free(x)
	if !b.armed {
		return
	}
	if role == "output" {
		for i := range b.scratch {
			b.scratch[i] = float32(math.NaN())
		}
		switch b.fault {
		case "cleanup-nil", "cleanup-nil-secondary", "read-nonnil", "returned-error":
			b.injections++
			panic(nil)
		case "cleanup-goexit":
			b.injections++
			runtime.Goexit()
		case "read-goexit", "short-read":
			b.injections++
			panic(errors.New("secondary output retirement"))
		}
	}
	if role == "activation" {
		switch b.fault {
		case "cleanup-nil-secondary", "backend-panic":
			b.injections++
			panic(errors.New("secondary activation retirement"))
		case "incoming-forward":
			b.injections++
			panic(nil)
		}
	}
}

// The actual ratio-two pool must fail closed when its selected exact operation
// or transient retirement unwinds. The separately reviewed compressor outer
// completion repair is an explicit prerequisite for preserving unknown nil.
// fak-test:runtime fast est=500ms lane=default
func TestV41ExactNormCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	allJoined := true
	for _, mode := range []string{"0", "1"} {
		if !allJoined {
			return
		}
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"healthy", "read-nil", "read-nonnil", "read-goexit", "returned-error", "backend-panic", "incoming-forward", "short-read", "cleanup-nil", "cleanup-nil-secondary", "cleanup-goexit"} {
				if !allJoined {
					return
				}
				t.Run(fault, func(t *testing.T) {
					const width = 128
					eps := float32(1e-6)
					gain := v41ExactNormTestRow(width, 0.01)
					first, second := v41ExactNormTestRow(width, 1), v41ExactNormTestRow(width, 3)
					scores := make([]float32, width)
					before := fmt.Sprint(first, second, scores, gain)
					m := v41CompressorNormTinyModel(t, gain, eps)
					b := &exactNormCompletionBackend{v41ExactNormTestBackend: newV41ExactNormTestBackend(), fault: fault, frees: map[compute.Buffer]int{}, primary: errors.New("exact normalization read primary")}
					b.outputOwner = b
					b.typed = &compute.BackendError{Backend: "exact-norm-recorder", Class: compute.VulkanClassExecutionFailed, Err: b.primary}
					b.incoming = &BackendForwardOperationError{Backend: "incoming", Forward: ForwardPathKind("deepseek41"), Path: "original", Layer: 3, Stage: "original", Cause: b.primary}
					s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
					b.session = s
					t.Cleanup(func() {
						if !allJoined {
							return
						}
						b.armed = false
						s.Close()
						// A cleanup Goexit may interrupt the historical reverse loop.
						for _, x := range []compute.Tensor{b.output, b.activation} {
							if x.Buf() != nil && b.frees[x.Buf()] == 0 {
								b.Free(x)
							}
						}
						if err := m.CloseWeights(); err != nil {
							t.Error(err)
						}
						if b.weight.Buf() != nil && b.frees[b.weight.Buf()] != 1 {
							t.Error("joined model teardown did not retire its gain exactly once")
						}
					})
					if s.v41ExactRMSNormFunc(width, eps) == nil || s.v41BF16PublishFunc() != nil {
						t.Fatal("fixture did not select dedicated exact normalization with host BF16 boundaries")
					}
					normalize := s.v41CompressorNormFunc()
					if normalize == nil {
						t.Fatal("actual compressor callback absent")
					}
					pool, err := NewV41CompressorPool(2, width)
					if err != nil {
						t.Fatal(err)
					}
					callbackReturned := false
					pool.normalize = func(input, weight []float32, e float32) ([]float32, error) {
						row, err := normalize(0, input, weight, e)
						callbackReturned = true
						return row, err
					}
					if row, emitted, err := pool.PushNormalized(0, first, scores, gain, eps); row != nil || emitted || err != nil || callbackReturned || b.uploads != 0 {
						t.Fatal("incomplete pool performed selected work")
					}
					host, err := NewV41CompressorPool(2, width)
					if err != nil {
						t.Fatal(err)
					}
					if _, emitted, err := host.PushNormalized(0, first, scores, gain, eps); emitted || err != nil {
						t.Fatal("oracle did not retain first row")
					}
					want, hostEmitted, err := host.PushNormalized(1, second, scores, gain, eps)
					if err != nil || !hostEmitted || len(want) != width || want[0] == 0 {
						t.Fatal("empty host pool oracle")
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
						t.Fatal("exact normalization worker did not join; cleanup withheld")
					}
					diagnostic := func() {
						t.Fatalf("exact norm fault=%q returned=%t boundaryReturned=%t callbackReturned=%t emitted=%t width=%d host_match=%t cause_nil=%t closed=%t panic=%v norms=%d reads=%d frees=%v injections=%d", fault, returned, boundaryReturned, callbackReturned, emitted, len(got), v41CompressorNormEqualBits(got, want), cause == nil, s.halClosed, payload, b.calls, b.reads, b.order, b.injections)
					}
					if s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("exact normalization stranded its owner")
					}
					b.mu.Unlock()
					if before != fmt.Sprint(first, second, scores, gain) || b.calls != 1 || b.uploads != 2 || pool.nextPos != 2 || pool.filled != 0 || !v41CompressorNormEqualBits(b.gain, gain) || len(b.input) != width {
						diagnostic()
					}
					for i := range b.input {
						if math.Float32bits(b.input[i]) != math.Float32bits(v41RoundBF16((first[i]+second[i])*0.5)) {
							t.Fatal("actual pooled operands changed before exact normalization")
						}
					}
					wantOrder := []string{"output", "activation"}
					wantReads, wantInjections := 1, 1
					if fault == "backend-panic" || fault == "incoming-forward" {
						wantOrder, wantReads = []string{"activation"}, 0
					}
					if fault == "returned-error" {
						wantReads = 0
					}
					if fault == "cleanup-goexit" {
						wantOrder = []string{"output"}
					}
					if fault == "healthy" {
						wantInjections = 0
					}
					if fault == "read-nonnil" || fault == "read-goexit" || fault == "returned-error" || fault == "backend-panic" || fault == "incoming-forward" || fault == "short-read" || fault == "cleanup-nil-secondary" {
						wantInjections = 2
					}
					goexit := fault == "read-goexit" || fault == "cleanup-goexit"
					if strings.Join(b.order, ",") != strings.Join(wantOrder, ",") || b.reads != wantReads || b.injections != wantInjections || b.frees[b.weight.Buf()] != 0 {
						diagnostic()
					}
					if fault == "healthy" {
						if !returned || !boundaryReturned || !callbackReturned || !emitted || cause != nil || payload != nil || s.halClosed || s.halFailure != nil || !v41CompressorNormEqualBits(got, want) {
							diagnostic()
						}
						return
					}
					if goexit {
						if returned || boundaryReturned || callbackReturned || emitted || got != nil || cause != nil || payload != nil {
							diagnostic()
						}
						return
					}
					var closed *BackendForwardOperationError
					if !s.halClosed || !errors.As(s.halFailure, &closed) || got != nil {
						diagnostic()
					}
					if fault == "incoming-forward" {
						if s.halFailure != b.incoming || payload != b.incoming || returned || !boundaryReturned || callbackReturned || emitted || cause != nil {
							diagnostic()
						}
					} else {
						stage := "readback"
						if fault == "returned-error" || fault == "backend-panic" {
							stage = "rmsnorm"
						}
						if closed.Path != "v41-compressor-norm" || closed.Layer != 0 || closed.Stage != stage {
							diagnostic()
						}
						if fault == "returned-error" || fault == "backend-panic" || fault == "short-read" {
							wantCause := error(b.typed)
							if fault == "short-read" {
								wantCause = errV41ProjectionResult
							}
							if !returned || !boundaryReturned || !callbackReturned || !emitted || payload != nil || !errors.Is(cause, wantCause) || closed.Cause != wantCause {
								diagnostic()
							}
						} else {
							if returned || !boundaryReturned || callbackReturned || emitted || cause != nil {
								diagnostic()
							}
							if fault == "read-nonnil" {
								if payload != b.primary || closed.Cause != b.primary {
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
					}
					uploads, reads, calls, frees := b.uploads, b.reads, b.calls, len(b.order)
					var retry any
					func() {
						defer func() { retry = recover() }()
						scope := s.beginBackendExecution()
						complete := false
						defer scope.finishCompleted(&complete)
						_, _ = normalize(0, first, gain, eps)
						complete = true
					}()
					if retry != s.halFailure || b.uploads != uploads || b.reads != reads || b.calls != calls || len(b.order) != frees || s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("closed exact normalization replayed work or stranded its retry owner")
					}
					b.mu.Unlock()
				})
			}
		})
	}
}
