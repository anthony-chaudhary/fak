package model

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type indexSelectCompletionBackend struct {
	*v41DenseTestBackend
	owner                             sync.Mutex
	session                           *Session
	fault                             string
	primary, secondary                error
	input, output                     compute.Tensor
	order                             []string
	frees                             map[compute.Buffer]int
	uploads, calls, reads, injections int
}

func (b *indexSelectCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *indexSelectCompletionBackend) assertOwner() {
	scope := b.session.backendExecutionScope
	if scope == nil || !scope.live || scope.owner != &b.owner {
		panic("index selection lacks its execution owner")
	}
	member := false
	for _, session := range scope.sessions {
		member = member || session == b.session
	}
	if !member {
		panic("index selection lacks its owner binding")
	}
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("index selection owner is unlocked")
	}
}
func (b *indexSelectCompletionBackend) V41IndexSelectAvailable() bool { return true }
func (b *indexSelectCompletionBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.assertOwner()
	if dt != compute.F32 || class != compute.MemoryActivation || site != "V4.1 index-select scores" || !reflect.DeepEqual(x.Shape, []int{2}) {
		panic("index selection upload contract")
	}
	values := b.Backend.Read(x)
	if !reflect.DeepEqual(values, []float32{2, 1}) {
		panic("index selection actual score mismatch")
	}
	b.uploads++
	b.input = compute.NewF32(b, []int{len(values)}, append([]float32(nil), values...))
	return b.input
}
func (b *indexSelectCompletionBackend) V41IndexSelect(x compute.Tensor, rows, topK, blocks, size int) (compute.Tensor, error) {
	b.assertOwner()
	if x.Backend() != b || x.Buf() != b.input.Buf() || rows != 2 || topK != 1 || blocks != 0 || size != 1 {
		panic("index selection operand identity or geometry changed")
	}
	b.calls++
	b.output = compute.NewF32(b, []int{topK}, []float32{0})
	if b.fault == "returned-error-cleanup" {
		b.injections++
		return b.output, b.primary // Fresh output still belongs to the callback.
	}
	return b.output, nil
}
func (b *indexSelectCompletionBackend) Read(x compute.Tensor) []float32 {
	b.assertOwner()
	if x.Backend() != b || x.Buf() != b.output.Buf() {
		panic("index selection read is not its output")
	}
	b.reads++
	switch b.fault {
	case "read-nil":
		b.injections++
		panic(nil)
	case "primary-cleanup-nil":
		b.injections++
		panic(b.primary)
	case "read-goexit":
		b.injections++
		runtime.Goexit()
	}
	return b.Backend.Read(x)
}
func (b *indexSelectCompletionBackend) Free(x compute.Tensor) {
	b.assertOwner()
	kind := ""
	switch x.Buf() {
	case b.output.Buf():
		kind = "output"
	case b.input.Buf():
		kind = "input"
	default:
		panic("index selection released an unrelated tensor")
	}
	b.frees[x.Buf()]++
	b.order = append(b.order, kind)
	if b.frees[x.Buf()] != 1 {
		panic("index selection repeated a release")
	}
	b.Backend.Free(x)
	if kind == "output" {
		switch b.fault {
		case "cleanup-nil", "cleanup-nil-secondary", "primary-cleanup-nil", "returned-error-cleanup":
			b.injections++
			panic(nil)
		case "cleanup-goexit":
			b.injections++
			runtime.Goexit()
		}
	}
	if kind == "input" && b.fault == "cleanup-nil-secondary" {
		b.injections++
		panic(b.secondary)
	}
}

// Exercise actual score computation and the selected index consumer. A lost
// read panic becomes its width refusal, while a lost cleanup panic can leave a
// valid row. Neither observation is a successful full-forward claim.
// fak-test:runtime fast est=50ms lane=default
func TestV41IndexSelectCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	allJoined := true
	for _, mode := range []string{"0", "1"} {
		if !allJoined {
			return
		}
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"read-nil", "cleanup-nil", "primary-cleanup-nil", "returned-error-cleanup", "read-goexit", "cleanup-goexit", "cleanup-nil-secondary", "healthy"} {
				if !allJoined {
					return
				}
				t.Run(fault, func(t *testing.T) {
					b := &indexSelectCompletionBackend{v41DenseTestBackend: newV41DenseTestBackend(), fault: fault, primary: errors.New("index selection primary"), secondary: errors.New("later score cleanup"), frees: map[compute.Buffer]int{}}
					s := &Session{M: &Model{}, Backend: b, halW: map[string]compute.Tensor{}}
					b.session = s
					t.Cleanup(func() {
						if allJoined {
							b.fault = ""
							s.Close()
						}
					})
					selected := s.v41IndexSelectFunc()
					if selected == nil {
						t.Fatal("index selection callback absent")
					}
					q, keys, weights := []float32{1}, []float32{2, 1}, []float32{1}
					before := fmt.Sprint(q, keys, weights)
					want, err := v41IndexSelectHost([]float32{2, 1}, 1, 0, 1)
					if err != nil || len(want) != 1 || want[0] != 0 {
						t.Fatal("index selection oracle is not nonempty")
					}
					callbackReturned, callbackWidth, callbackCauseNil := false, -1, false
					wrapped := func(layer int, scores []float32, topK, blocks, size int) ([]int32, error) {
						row, cause := selected(layer, scores, topK, blocks, size)
						callbackReturned, callbackWidth, callbackCauseNil = true, len(row), cause == nil
						return row, cause
					}
					var got []int32
					var cause error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					allJoined = false
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, cause = v41IndexRowsSelected(4, q, keys, weights, 1, 1, len(keys), 0, 1, len(want), nil, nil, wrapped)
							returned, complete = true, true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						allJoined = true
					case <-time.After(5 * time.Second):
						t.Fatal("index selection worker did not join; cleanup withheld")
					}
					diagnostic := func() {
						t.Fatalf("index select fault=%q returned=%t boundaryReturned=%t callbackReturned=%t callbackWidth=%d callbackCauseNil=%t output=%v widthRefusal=%t closed=%t primary=%t cause=%v panic=%v uploads=%d calls=%d reads=%d frees=%v", fault, returned, boundaryReturned, callbackReturned, callbackWidth, callbackCauseNil, got, errors.Is(cause, errV41IndexSelectResult), s.halClosed, s.halFailure != nil, cause, payload, b.uploads, b.calls, b.reads, b.order)
					}
					if s.backendExecutionScope != nil || !b.owner.TryLock() {
						t.Fatal("index selection stranded its owner")
					}
					b.owner.Unlock()
					if before != fmt.Sprint(q, keys, weights) || b.uploads != 1 || b.calls != 1 || !reflect.DeepEqual(b.order, []string{"output", "input"}) || b.frees[b.input.Buf()] != 1 || b.frees[b.output.Buf()] != 1 {
						diagnostic()
					}
					wantReads, wantInjections := 1, 1
					switch fault {
					case "healthy":
						wantInjections = 0
					case "primary-cleanup-nil", "cleanup-nil-secondary":
						wantInjections = 2
					case "returned-error-cleanup":
						wantReads, wantInjections = 0, 2
					}
					if b.reads != wantReads || b.injections != wantInjections {
						diagnostic()
					}
					if fault == "healthy" {
						if !returned || !boundaryReturned || !callbackReturned || !callbackCauseNil || cause != nil || payload != nil || !reflect.DeepEqual(got, want) || s.halClosed || s.halFailure != nil {
							diagnostic()
						}
						return
					}
					if fault == "read-goexit" || fault == "cleanup-goexit" {
						if returned || boundaryReturned || callbackReturned || got != nil || cause != nil || payload != nil {
							diagnostic()
						}
						return
					}
					var closed *BackendForwardOperationError
					if !s.halClosed || !errors.As(s.halFailure, &closed) || closed.Path != "v41-index-select" || closed.Layer != 4 || got != nil {
						diagnostic()
					}
					stage := "selection readback"
					if fault == "returned-error-cleanup" {
						stage = "index-select"
						if !returned || !boundaryReturned || !callbackReturned || callbackCauseNil || payload != nil || !errors.Is(cause, b.primary) || closed.Cause != b.primary {
							diagnostic()
						}
					} else {
						if returned || !boundaryReturned || callbackReturned || cause != nil {
							diagnostic()
						}
						if fault == "primary-cleanup-nil" {
							if payload != b.primary || closed.Cause != b.primary {
								diagnostic()
							}
						} else if mode == "1" {
							if payload != nil || errors.Is(s.halFailure, b.secondary) {
								diagnostic()
							}
						} else if _, ok := payload.(*runtime.PanicNilError); !ok {
							diagnostic()
						}
					}
					if closed.Stage != stage {
						diagnostic()
					}
					counts := fmt.Sprint(b.uploads, b.calls, b.reads, b.order)
					var retry any
					func() {
						defer func() { retry = recover() }()
						scope := s.beginBackendExecution()
						complete := false
						defer scope.finishCompleted(&complete)
						_, _ = selected(4, []float32{2, 1}, 1, 0, 1)
						complete = true
					}()
					if retry != s.halFailure || counts != fmt.Sprint(b.uploads, b.calls, b.reads, b.order) {
						t.Fatal("closed index selector retried backend work or changed its latch")
					}
				})
			}
		})
	}
}
