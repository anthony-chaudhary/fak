package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type indexerBindingCompletionBackend struct {
	*v41ScoreTestBackend
	owner             sync.Mutex
	before            func(string)
	afterFree         func(int)
	allocated, freed  []compute.Buffer
	dispatches, reads int
}

func (b *indexerBindingCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *indexerBindingCompletionBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if b.before != nil {
		b.before(strings.TrimPrefix(site, "V4.1 indexer score ") + " upload")
	}
	out := b.v41ScoreTestBackend.UploadClass(x, dt, class, site)
	if out.Buf() != nil {
		b.allocated = append(b.allocated, out.Buf())
	}
	return out
}
func (b *indexerBindingCompletionBackend) V41IndexerScore(q, keys, weights compute.Tensor, rows, heads, dim int) (compute.Tensor, error) {
	b.dispatches++
	if b.before != nil {
		b.before("score")
	}
	out, err := b.v41ScoreTestBackend.V41IndexerScore(q, keys, weights, rows, heads, dim)
	if out.Buf() != nil {
		b.allocated = append(b.allocated, out.Buf())
	}
	return out, err
}
func (b *indexerBindingCompletionBackend) Read(x compute.Tensor) []float32 {
	if x.Buf() == b.output {
		b.reads++
		if b.before != nil {
			b.before("readback")
		}
	}
	return b.v41ScoreTestBackend.Read(x)
}
func (b *indexerBindingCompletionBackend) Free(x compute.Tensor) {
	if !b.managed[x.Buf()] {
		b.v41ScoreTestBackend.Free(x)
		return
	}
	b.freed = append(b.freed, x.Buf())
	b.v41ScoreTestBackend.Free(x)
	if b.afterFree != nil {
		b.afterFree(len(b.freed))
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41IndexerBindingPreservesIncompleteUnwind(t *testing.T) {
	primary := errors.New("binding primary")
	secondary := &compute.BackendError{Backend: "cleanup", Class: compute.VulkanClassExecutionFailed, Err: errors.New("binding cleanup")}
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			rows := []struct{ site, fault string }{}
			for _, site := range []string{"keys upload", "score", "readback"} {
				for _, fault := range []string{"nil", "nonnil", "goexit"} {
					rows = append(rows, struct{ site, fault string }{site, fault})
				}
			}
			for _, fault := range []string{"backend", "cuda", "closed-marker", "partial-error", "cleanup-nil", "healthy"} {
				rows = append(rows, struct{ site, fault string }{"score", fault})
			}
			for _, tc := range rows {
				t.Run(strings.ReplaceAll(tc.site, " ", "-")+"/"+tc.fault, func(t *testing.T) {
					b := &indexerBindingCompletionBackend{v41ScoreTestBackend: newV41ScoreTestBackend()}
					b.identity = b
					s := &Session{M: &Model{}, Backend: b, halW: map[string]compute.Tensor{}}
					t.Cleanup(s.Close)
					op := s.v41IndexerScoreFunc()
					if op == nil {
						t.Fatal("operator not selected")
					}
					injections := 0
					badOwned := false
					checkOwned := func() {
						if s.backendExecutionScope == nil || !s.backendExecutionScope.live || s.backendExecutionScope.owner != &b.owner {
							badOwned = true
						}
						if b.owner.TryLock() {
							b.owner.Unlock()
							badOwned = true
						}
					}
					var typed error
					switch tc.fault {
					case "backend":
						typed = &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: primary}
					case "cuda":
						typed = &compute.CUDAOpError{Op: "IndexerScore", Err: primary}
					case "closed-marker":
						typed = &BackendForwardOperationError{Backend: "primary", Path: "already-closed", Stage: "score", Cause: primary}
					case "partial-error":
						b.fault, b.cause = "partial output", primary
					}
					b.before = func(site string) {
						checkOwned()
						if site != tc.site {
							return
						}
						switch tc.fault {
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
					b.afterFree = func(n int) {
						checkOwned()
						if tc.fault == "healthy" {
							return
						}
						if tc.fault == "cleanup-nil" {
							if n == 1 {
								injections++
								panic(nil)
							}
							panic(secondary)
						}
						if tc.fault == "nonnil" || tc.fault == "partial-error" || tc.fault == "backend" || tc.fault == "cuda" {
							panic(nil)
						}
						panic(secondary)
					}
					var result []float32
					var err error
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
								result, err = op(4, []float32{2}, []float32{3}, []float32{4}, 1, 1, 1)
								complete = true
							}()
							returned = true
						}()
						boundaryReturned = true
					}()
					<-done
					b.before, b.afterFree = nil, nil
					ordinary := tc.fault == "backend" || tc.fault == "cuda" || tc.fault == "partial-error" || tc.fault == "healthy"
					if returned != ordinary || boundaryReturned == (tc.fault == "goexit") {
						t.Fatalf("binding site=%q fault=%q returned=%t boundaryReturned=%t", tc.site, tc.fault, returned, boundaryReturned)
					}
					wantOwned := 3
					if tc.site == "keys upload" {
						wantOwned = 1
					} else if tc.site == "readback" || tc.fault == "partial-error" || tc.fault == "cleanup-nil" || tc.fault == "healthy" {
						wantOwned = 4
					}
					wantInjection := 1
					if tc.fault == "partial-error" || tc.fault == "healthy" {
						wantInjection = 0
					}
					if injections != wantInjection || badOwned || len(b.allocated) != wantOwned || len(b.freed) != wantOwned || b.freeAttempts != wantOwned || len(b.allocations) != 0 {
						t.Fatalf("injection=%d owned=%t allocated=%d freed=%d attempts=%d live=%d", injections, badOwned, len(b.allocated), len(b.freed), b.freeAttempts, len(b.allocations))
					}
					for i, buf := range b.freed {
						if buf != b.allocated[len(b.allocated)-1-i] {
							t.Fatal("Free order changed")
						}
					}
					wantDispatch := 1
					if tc.site == "keys upload" {
						wantDispatch = 0
					}
					wantRead := 0
					if tc.site == "readback" || tc.fault == "cleanup-nil" || tc.fault == "healthy" {
						wantRead = 1
					}
					if b.dispatches != wantDispatch || b.reads != wantRead {
						t.Fatalf("dispatch=%d read=%d", b.dispatches, b.reads)
					}
					if s.backendExecutionScope != nil || !b.owner.TryLock() {
						t.Fatal("owner not released")
					}
					b.owner.Unlock()
					closed := tc.fault != "goexit" && tc.fault != "closed-marker" && tc.fault != "healthy"
					if s.BackendSessionClosed() != closed {
						t.Fatalf("closed=%t want=%t", s.BackendSessionClosed(), closed)
					}
					if closed {
						marker, ok := s.halFailure.(*BackendForwardOperationError)
						stage := tc.site
						if tc.fault == "cleanup-nil" {
							stage = "readback"
						}
						if !ok || marker.Layer != 4 || marker.Path != "v41-indexer-score" || marker.Stage != stage {
							t.Fatalf("failure=%v", s.halFailure)
						}
						if tc.fault == "backend" || tc.fault == "cuda" {
							if marker.Cause != typed || !errors.Is(err, typed) {
								t.Fatalf("typed cause=%v err=%v", marker.Cause, err)
							}
						}
						if tc.fault == "nonnil" || tc.fault == "partial-error" {
							if marker.Cause != primary {
								t.Fatalf("original cause=%v", marker.Cause)
							}
						}
					} else if s.halFailure != nil {
						t.Fatalf("unexpected failure=%v", s.halFailure)
					}
					switch tc.fault {
					case "nil", "cleanup-nil":
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
					case "partial-error":
						if payload != nil || !errors.Is(err, primary) {
							t.Fatalf("returned primary=%v panic=%v", err, payload)
						}
					case "backend", "cuda":
						if payload != nil {
							t.Fatalf("typed panic escaped=%v", payload)
						}
					case "healthy":
						if payload != nil || err != nil || !reflect.DeepEqual(result, []float32{24}) {
							t.Fatalf("result=%v err=%v panic=%v", result, err, payload)
						}
					}
					if tc.fault != "healthy" && result != nil {
						t.Fatal("failed operation assigned scores")
					}
				})
			}
		})
	}
}
