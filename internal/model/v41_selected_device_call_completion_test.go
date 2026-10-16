package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type selectedCallCompletionBackend struct {
	*v41SumTestBackend
	owner    sync.Mutex
	before   func()
	calls    int
	recycles int
}

func (b *selectedCallCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *selectedCallCompletionBackend) AddInPlace(dst, src compute.Tensor) {
	b.calls++
	if b.before != nil {
		b.before()
	}
	b.v41SumTestBackend.AddInPlace(dst, src)
}
func (b *selectedCallCompletionBackend) Recycle() { b.recycles++ }

// The actual factories allocate and unwind their own tensors. These CPU
// recorders witness selected-call policy and owner release, not native safety
// or a complete drain after an interrupted cleanup callback.
// fak-test:runtime fast est=40ms lane=default
func TestV41SelectedDeviceCallCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	primary := errors.New("selected operation primary")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, operation := range []string{"routed-sum", "shared-add"} {
				t.Run(operation, func(t *testing.T) {
					for _, fault := range []string{"nil", "nonnil", "goexit", "backend", "cuda", "closed-marker", "invalid-read", "healthy"} {
						t.Run(fault, func(t *testing.T) {
							observation := &v41SumTestObservation{sizes: map[compute.Buffer]uint64{}}
							b := &selectedCallCompletionBackend{v41SumTestBackend: newV41SumTestBackend()}
							b.observation = observation
							s := &Session{M: &Model{}, Backend: b}
							t.Cleanup(s.Close)
							rows := [][]float32{{1e8, math.Float32frombits(0x80000000), 2, -4}, {-1e8, 0, 3, 5}, {1, 0, -7, 8}}
							picks := []routePick{{expert: 7}, {expert: 2}, {expert: 7}}
							original := make([][]float32, len(rows))
							for i := range rows {
								original[i] = slices.Clone(rows[i])
							}
							var invoke func() ([]float32, error)
							var want []float32
							var err error
							wantAdds, wantUploads := len(rows), len(rows)+1
							if operation == "routed-sum" {
								fn := s.v41RoutedSumFunc()
								if fn == nil {
									t.Fatal("routed-sum factory did not bind")
								}
								invoke = func() ([]float32, error) { return fn(3, picks, rows, 4) }
								want, err = v41FullRoutedSum(3, picks, rows, 4)
							} else {
								fn := s.v41SharedAddFunc()
								if fn == nil {
									t.Fatal("shared-add factory did not bind")
								}
								invoke = func() ([]float32, error) { return fn(3, rows[0], rows[1]) }
								want, err = v41SharedExpertAdd(rows[0], rows[1], v41RouterConfig{SharedCount: 1})
								wantAdds, wantUploads = 1, 2
							}
							if err != nil {
								t.Fatal(err)
							}
							var typed error
							switch fault {
							case "backend":
								typed = &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: primary}
							case "cuda":
								typed = &compute.CUDAOpError{Op: "AddInPlace", Err: primary}
							case "closed-marker":
								typed = fmt.Errorf("original wrapper: %w", &BackendForwardOperationError{Path: "another-owner", Stage: "incoming", Cause: primary})
							}
							badOwned, injections := false, 0
							b.before = func() {
								if s.backendExecutionScope == nil || !s.backendExecutionScope.live || s.backendExecutionScope.owner != &b.owner {
									badOwned = true
								}
								if b.owner.TryLock() {
									b.owner.Unlock()
									badOwned = true
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
								}
							}
							b.shortRead = fault == "invalid-read"
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
										got, cause = invoke()
										complete = true
									}()
									returned = true
								}()
								boundaryReturned = true
							}()
							<-done
							b.before = nil
							ordinary := fault == "backend" || fault == "cuda" || fault == "invalid-read" || fault == "healthy"
							if returned != ordinary || boundaryReturned == (fault == "goexit") {
								t.Fatalf("selected %s fault=%s returned=%t boundaryReturned=%t output=%v cause=%v", operation, fault, returned, boundaryReturned, got, cause)
							}
							wantCalls, wantReads, wantInject := 1, 0, 1
							if fault == "healthy" || fault == "invalid-read" {
								wantCalls, wantReads, wantInject = wantAdds, 1, 0
							}
							if badOwned || b.calls != wantCalls || b.uploads != wantUploads || observation.readAttempts != wantReads || injections != wantInject {
								t.Fatalf("owned=%t calls=%d uploads=%d reads=%d injections=%d", !badOwned, b.calls, b.uploads, observation.readAttempts, injections)
							}
							freed := slices.Clone(observation.uploaded)
							slices.Reverse(freed)
							if !slices.Equal(observation.freed, freed) || len(b.live) != 0 || observation.liveBytes != 0 {
								t.Fatal("actual operation did not release every successfully uploaded tensor in reverse order")
							}
							for i := range rows {
								if !v41SameBits(rows[i], original[i]) {
									t.Fatalf("selected call changed host row%d", i)
								}
							}
							if s.backendExecutionScope != nil || !b.owner.TryLock() {
								t.Fatal("selected call did not detach and release its execution owner")
							}
							b.owner.Unlock()
							closed := fault != "healthy" && fault != "goexit"
							wantRecycles := 0
							if closed {
								wantRecycles = 1
								if s.halFailure == nil {
									t.Fatal("selected fault did not latch a failure")
								}
							}
							if s.BackendSessionClosed() != closed || b.recycles != wantRecycles {
								t.Fatalf("closed=%t recycles=%d want=%t/%d", s.BackendSessionClosed(), b.recycles, closed, wantRecycles)
							}
							if fault != "healthy" && got != nil {
								t.Fatal("failed selected call returned output")
							}
							switch fault {
							case "nil":
								if mode == "0" {
									if _, ok := payload.(*runtime.PanicNilError); !ok {
										t.Fatalf("nil panic=%T %v", payload, payload)
									}
								} else if payload != nil {
									t.Fatalf("legacy nil panic=%v", payload)
								}
							case "nonnil":
								if payload != primary || !errors.Is(s.halFailure, primary) {
									t.Fatalf("primary panic changed: %v", payload)
								}
							case "goexit":
								if payload != nil || cause != nil || s.halFailure != nil {
									t.Fatal("Goexit was converted into a panic or selected result")
								}
							case "closed-marker":
								if payload != typed || s.halFailure != typed {
									t.Fatal("incoming marker wrapper identity changed")
								}
							case "backend", "cuda", "invalid-read":
								var selected *BackendForwardOperationError
								stage := operation
								if fault == "invalid-read" {
									stage = "readback"
								}
								if payload != nil || !errors.As(cause, &selected) || selected.Path != "v41-"+operation || selected.Layer != 3 || selected.Stage != stage || s.halFailure != cause {
									t.Fatalf("selected attribution cause=%v panic=%v", cause, payload)
								}
								if typed != nil && !errors.Is(cause, typed) {
									t.Fatal("typed backend cause changed")
								}
							case "healthy":
								if cause != nil || payload != nil || !v41SameBits(got, want) || b.adds != wantAdds || s.halFailure != nil {
									t.Fatalf("healthy parity result=%v want=%v cause=%v panic=%v", got, want, cause, payload)
								}
							}
						})
					}
				})
			}
		})
	}
}
