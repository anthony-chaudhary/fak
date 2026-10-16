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
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type engramProjectionCompletionBackend struct {
	*v41DenseTestBackend
	mu                         sync.Mutex
	session                    *Session
	armed                      bool
	fault                      string
	primary                    error
	typed                      *compute.BackendError
	weight, activation, output compute.Tensor
	frees                      map[compute.Buffer]int
	order                      []string
	stages, uploads, calls     int
	readCount, injections      int
}

func (b *engramProjectionCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *engramProjectionCompletionBackend) assertOwner() {
	if !b.armed {
		return
	}
	scope := b.session.backendExecutionScope
	if scope == nil || !scope.live || scope.owner != &b.mu || !slices.Contains(scope.sessions, b.session) {
		panic("Engram projection lacks its live session owner")
	}
	if b.mu.TryLock() {
		b.mu.Unlock()
		panic("Engram projection owner is unlocked")
	}
}
func (b *engramProjectionCompletionBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.assertOwner()
	if len(x.Shape) == 2 {
		b.stages++
		if b.fault == "weight-nil" {
			b.injections++
			panic(nil)
		}
		y := b.v41DenseTestBackend.Upload(x, dt)
		b.weight = y
		return y
	}
	panic("unexpected Engram projection upload")
}
func (b *engramProjectionCompletionBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.assertOwner()
	if dt != compute.F32 || class != compute.MemoryActivation || site != "V4.1 Engram projection activation" || len(x.Shape) != 1 {
		panic("Engram projection changed its activation upload")
	}
	b.uploads++
	b.activation = b.v41DenseTestBackend.Upload(x, dt)
	return b.activation
}
func (b *engramProjectionCompletionBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.assertOwner()
	if w.Buf() != b.weight.Buf() || x.Buf() != b.activation.Buf() || w.Buf() == x.Buf() || b.frees[w.Buf()] != 0 {
		panic("Engram projection lost its borrowed weight or owned activation")
	}
	b.calls++
	if b.fault == "backend-panic" {
		b.injections++
		panic(b.typed)
	}
	b.output = b.v41DenseTestBackend.MatMul(w, x)
	if b.output.Buf() == nil || b.output.Buf() == w.Buf() || b.output.Buf() == x.Buf() {
		panic("Engram projection did not produce a fresh output")
	}
	return b.output
}
func (b *engramProjectionCompletionBackend) Read(x compute.Tensor) []float32 {
	b.assertOwner()
	if x.Buf() != b.output.Buf() {
		panic("Engram projection read the wrong owner")
	}
	b.readCount++
	values := b.v41DenseTestBackend.Read(x)
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
		return values[:len(values)-1]
	}
	return values
}
func (b *engramProjectionCompletionBackend) Free(x compute.Tensor) {
	b.assertOwner()
	role := ""
	switch x.Buf() {
	case b.output.Buf():
		role = "output"
	case b.activation.Buf():
		role = "activation"
	case b.weight.Buf():
		role = "weight"
	}
	if role == "" || x.Buf() == nil || b.frees[x.Buf()] != 0 {
		panic("Engram projection retired an unknown or already freed owner")
	}
	b.frees[x.Buf()]++
	b.order = append(b.order, role)
	b.v41DenseTestBackend.Free(x)
	if !b.armed {
		return
	}
	if role == "output" {
		switch b.fault {
		case "cleanup-nil", "cleanup-nil-secondary", "read-nonnil", "short-read":
			b.injections++
			panic(nil)
		case "read-nil", "read-goexit":
			b.injections++
			panic(errors.New("secondary Engram output retirement"))
		case "cleanup-goexit":
			b.injections++
			runtime.Goexit()
		}
	}
	if role == "activation" && (b.fault == "cleanup-nil-secondary" || b.fault == "backend-panic") {
		b.injections++
		panic(errors.New("secondary Engram activation retirement"))
	}
}

func engramProjectionCompletionBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// A real full-stream injection must not publish after selected projection
// unwinding. This is a reduced consumer witness, not a complete Session forward.
// fak-test:runtime fast est=500ms lane=default
func TestV41EngramProjectionCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	allJoined := true
	for _, mode := range []string{"0", "1"} {
		if !allJoined {
			return
		}
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"healthy", "decline", "weight-nil", "read-nil", "read-nonnil", "read-goexit", "backend-panic", "short-read", "cleanup-nil", "cleanup-nil-secondary", "cleanup-goexit"} {
				if !allJoined {
					return
				}
				t.Run(fault, func(t *testing.T) {
					m, _ := v41FullEngramModelWith(t, func(c *Config) {
						c.HiddenSize, c.NumExperts, c.NumExpertsPerTok = 8, 1, 1
					})
					b := &engramProjectionCompletionBackend{v41DenseTestBackend: newV41DenseTestBackend(), fault: fault, primary: errors.New("Engram selected read primary"), frees: map[compute.Buffer]int{}}
					b.deny = fault == "decline"
					b.typed = &compute.BackendError{Backend: "engram-projection-recorder", Class: compute.VulkanClassExecutionFailed, Err: b.primary}
					s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
					b.session = s
					t.Cleanup(func() {
						if !allJoined {
							return
						}
						b.armed = false
						s.Close()
						// Cleanup Goexit can interrupt the existing reverse-Free loop.
						for _, x := range []compute.Tensor{b.output, b.activation} {
							if x.Buf() != nil && b.frees[x.Buf()] == 0 {
								b.Free(x)
							}
						}
						if err := m.CloseWeights(); err != nil {
							t.Error(err)
						}
						if b.weight.Buf() != nil && b.frees[b.weight.Buf()] != 1 {
							t.Error("joined model close did not retire the borrowed weight exactly once")
						}
					})
					ids := []int{1}
					eps := float32(m.Cfg.RMSNormEps)
					x, streams := v41FullEngramAsymmetricStreams(ids, m.Cfg.HiddenSize)
					before := sysFlatten(streams[0])
					hostX, hostStreams := v41FullEngramAsymmetricStreams(ids, m.Cfg.HiddenSize)
					if err := m.v41EngramInjectWithProjection(0, hostX, hostStreams, true, ids, eps, nil); err != nil {
						t.Fatal(err)
					}
					want := sysFlatten(hostStreams[0])
					if len(want) != 4*m.Cfg.HiddenSize || engramProjectionCompletionBits(before, want) {
						t.Fatal("Engram host control did not change all requested stream storage")
					}
					project := s.v41EngramProjectionFunc()
					if project == nil {
						t.Fatal("actual selected Engram callback is absent")
					}
					callbackReturned, callbackWidth := false, 0
					callbackOutcome := v41ProjectionDeclined
					observed := func(l int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
						values, outcome, err := project(l, row, out, in)
						callbackReturned, callbackWidth, callbackOutcome = true, len(values), outcome
						return values, outcome, err
					}
					var cause error
					var payload any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					b.armed, allJoined = true, false
					go func() {
						defer close(done)
						func() {
							defer func() { payload = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							cause = m.v41EngramInjectWithProjection(0, x, streams, true, ids, eps, observed)
							returned, complete = true, true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						allJoined = true
					case <-time.After(5 * time.Second):
						t.Fatal("Engram projection worker did not join; cleanup withheld")
					}
					after := sysFlatten(streams[0])
					changed, hostMatch := !engramProjectionCompletionBits(after, before), engramProjectionCompletionBits(after, want)
					diagnostic := func() {
						t.Fatalf("Engram projection fault=%q returned=%t boundaryReturned=%t callbackReturned=%t outcome=%d width=%d changed=%t host_match=%t cause_nil=%t closed=%t panic=%v stages=%d uploads=%d matmuls=%d reads=%d frees=%v injections=%d", fault, returned, boundaryReturned, callbackReturned, callbackOutcome, callbackWidth, changed, hostMatch, cause == nil, s.halClosed, payload, b.stages, b.uploads, b.calls, b.readCount, b.order, b.injections)
					}
					if s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("Engram projection stranded its execution owner")
					}
					b.mu.Unlock()
					wantStages, wantUploads, wantCalls, wantReads, wantInjections := 1, 1, 1, 1, 1
					wantOrder := []string{"output", "activation"}
					switch fault {
					case "healthy":
						wantInjections = 0
					case "decline":
						wantStages, wantUploads, wantCalls, wantReads, wantInjections, wantOrder = 0, 0, 0, 0, 0, nil
					case "weight-nil":
						wantUploads, wantCalls, wantReads, wantOrder = 0, 0, 0, nil
					case "backend-panic":
						wantReads, wantInjections, wantOrder = 0, 2, []string{"activation"}
					case "read-nil", "read-nonnil", "read-goexit", "short-read", "cleanup-nil-secondary":
						wantInjections = 2
					case "cleanup-goexit":
						wantOrder = []string{"output"}
					}
					if b.stages != wantStages || b.uploads != wantUploads || b.calls != wantCalls || b.readCount != wantReads || b.injections != wantInjections || !slices.Equal(b.order, wantOrder) || b.frees[b.weight.Buf()] != 0 {
						diagnostic()
					}
					if fault == "healthy" || fault == "decline" {
						outcome, width := v41ProjectionHandled, 5*m.Cfg.HiddenSize
						if fault == "decline" {
							outcome, width = v41ProjectionDeclined, 0
						}
						if !returned || !boundaryReturned || !callbackReturned || callbackOutcome != outcome || callbackWidth != width || cause != nil || payload != nil || s.halClosed || s.halFailure != nil || !changed || !hostMatch {
							diagnostic()
						}
						return
					}
					if changed {
						diagnostic()
					}
					if fault == "read-goexit" || fault == "cleanup-goexit" {
						if returned || boundaryReturned || callbackReturned || cause != nil || payload != nil {
							diagnostic()
						}
						return
					}
					stage := "readback"
					if fault == "backend-panic" {
						stage = "matmul"
					} else if fault == "weight-nil" {
						stage = "weight upload"
					}
					var closed *BackendForwardOperationError
					if !s.halClosed || !errors.As(s.halFailure, &closed) || closed.Path != "v41-engram-projection" || closed.Layer != 0 || closed.Stage != stage {
						diagnostic()
					}
					if fault == "backend-panic" || fault == "short-read" {
						wantCause := error(b.typed)
						if fault == "short-read" {
							wantCause = errV41ProjectionResult
						}
						var selected *V41ProjectionOperationError
						if !returned || !boundaryReturned || !callbackReturned || callbackOutcome != v41ProjectionError || callbackWidth != 0 || payload != nil || !errors.As(cause, &selected) || selected.Leaf != "engram_kv.weight" || !errors.Is(cause, wantCause) || closed.Cause != wantCause {
							diagnostic()
						}
					} else {
						if returned || !boundaryReturned || callbackReturned || cause != nil {
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
					counts := fmt.Sprint(b.stages, b.uploads, b.calls, b.readCount, b.order)
					var retry any
					func() {
						defer func() { retry = recover() }()
						scope := s.beginBackendExecution()
						complete := false
						defer scope.finishCompleted(&complete)
						stage := m.v41EngramStageFor()
						_, _, _ = project(0, make([]float32, stage.columns*stage.headDim), (stage.hc+1)*m.Cfg.HiddenSize, stage.columns*stage.headDim)
						complete = true
					}()
					if retry != s.halFailure || fmt.Sprint(b.stages, b.uploads, b.calls, b.readCount, b.order) != counts || s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("closed Engram callback replayed work or stranded its retry owner")
					}
					b.mu.Unlock()
				})
			}
		})
	}
}
