package model

import (
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

type fusedRetirementRecorder struct {
	*v41FusedIndexScaleRecorder
	mu                    sync.Mutex
	session               *Session
	input, output, status compute.Tensor
	order                 []string
	row                   string
	primary, secondary    error
	readReturned, bound   bool
	readCalls, injections int
	readRow               []float32
	checkOwner            bool
	totalFrees            int
}

func (b *fusedRetirementRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *fusedRetirementRecorder) assertOwner() {
	scope := b.session.backendExecutionScope
	member := false
	if scope != nil {
		for _, session := range scope.sessions {
			if session == b.session {
				member = true
			}
		}
	}
	if scope == nil || !scope.live || scope.owner != &b.mu || !member {
		panic("fused retirement owner binding")
	}
	if b.mu.TryLock() {
		b.mu.Unlock()
		panic("fused retirement owner unlocked")
	}
}
func (b *fusedRetirementRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 index weight scale input" {
		return b.v41BF16PublishRecorder.UploadClass(x, dt, class, site)
	}
	b.input = b.track(compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...)))
	return b.input
}
func (b *fusedRetirementRecorder) SupportsV41IndexScaleBF16(x compute.Tensor, scale float32) bool {
	b.probes++
	if x.Backend() != b || x.Buf() != b.input.Buf() || len(x.Shape) != 1 || x.Shape[0] != 2 || scale != .5 {
		panic("fused retirement operands")
	}
	return true
}
func (b *fusedRetirementRecorder) V41IndexScaleBF16Deferred(x compute.Tensor, scale float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.fused++
	b.assertOwner()
	b.bound = b.session.backendExecutionScope != nil && b.session.backendExecutionScope.live
	if b.mu.TryLock() {
		b.mu.Unlock()
		b.bound = false
	}
	if !b.bound {
		panic("fused retirement missed owner")
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	for i := range values {
		values[i] = math.Float32frombits(v41BF16BoundaryOracle(math.Float32bits(values[i] * scale)))
	}
	b.output = b.track(compute.NewF32(b, []int{len(values)}, values))
	b.status = b.track(compute.NewF32(b, []int{4}, make([]float32, 4)))
	b.deferredPending[b.status.Buf()] = nil
	st := compute.V41DeferredStatus{Kind: compute.V41DeferredBF16Publish, Count: len(values), Site: compute.V41BF16PublishIndexerScaledWeight, Record: b.status}
	if b.row == "returned-nil" || b.row == "returned-nonnil" {
		return b.output, st, b.primary
	}
	return b.output, st, nil
}
func (b *fusedRetirementRecorder) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.readCalls++
	b.assertOwner()
	v, r, e := b.v41DeferredLedger.V41ReadDeferred(st, rows)
	b.readReturned = true
	if e == nil && len(v) == 1 && v[0] == nil && len(r) == 1 {
		b.readRow = append([]float32(nil), r[0]...)
	}
	switch b.row {
	case "read-nil":
		b.injections++
		panic(nil)
	case "read-nonnil":
		b.injections++
		panic(b.primary)
	case "read-goexit":
		b.injections++
		runtime.Goexit()
	}
	return v, r, e
}
func (b *fusedRetirementRecorder) Free(x compute.Tensor) {
	b.totalFrees++
	name := ""
	switch x.Buf() {
	case b.input.Buf():
		name = "input"
	case b.output.Buf():
		name = "output"
	case b.status.Buf():
		name = "status"
	}
	if name != "" && b.checkOwner {
		b.assertOwner()
	}
	b.v41FusedIndexScaleRecorder.Free(x)
	if name == "" {
		return
	}
	b.order = append(b.order, name)
	if name == "input" && b.row == "cleanup-nil-then-nonnil" {
		panic(b.secondary)
	}
	if name != "output" || b.row == "healthy" {
		return
	}
	if b.row != "returned-nil" && b.row != "returned-nonnil" && !b.readReturned {
		panic("output retirement preceded delegated read return")
	}
	b.injections++
	switch b.row {
	case "cleanup-goexit":
		runtime.Goexit()
	case "cleanup-nil", "cleanup-nil-then-nonnil", "returned-nil", "transport-nil":
		panic(nil)
	default:
		panic(b.secondary)
	}
}

// Actual selected fused callback and deferred read, with fixture-owned teardown.
// Free Goexit may interrupt the production drain; remaining cleanup is not credited.
// fak-test:runtime medium est=2s lane=default
func TestV41IndexWeightScaleRetirementCompletion(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, row := range []string{"healthy", "cleanup-nil", "cleanup-nonnil", "cleanup-goexit", "cleanup-nil-then-nonnil", "returned-nil", "returned-nonnil", "transport-nil", "read-nil", "read-nonnil", "read-goexit"} {
				t.Run(row, func(t *testing.T) {
					m := v41IndexerTestFixture(t)
					m.Cfg.IndexNHeads = 2
					b := &fusedRetirementRecorder{v41FusedIndexScaleRecorder: newV41FusedIndexScaleRecorder(), row: row, primary: errors.New("fused primary"), secondary: errors.New("fused secondary")}
					b.self = b
					s, err := m.NewBackendSessionChecked(b)
					if err != nil {
						t.Fatal(err)
					}
					b.session = s
					joined := false
					t.Cleanup(func() {
						if !joined {
							return
						}
						b.row = "healthy"
						b.checkOwner = false
						// The fixture owns any input abandoned by a Free Goexit.
						for _, x := range []compute.Tensor{b.status, b.output, b.input} {
							if x.Buf() != nil && b.frees[x.Buf()] == 0 {
								b.Free(x)
							}
						}
						s.Close()
						if err := m.CloseWeights(); err != nil {
							t.Error(err)
						}
					})
					fn := s.v41IndexWeightScaleFunc()
					if fn == nil {
						t.Fatal("missing fused callback")
					}
					if row == "transport-nil" {
						b.deferredReadErr = b.primary
					}
					var got []float32
					var cause error
					var recovered any
					returned, boundaryReturned := false, false
					b.checkOwner = true
					joinedCh := make(chan struct{})
					go func() {
						defer close(joinedCh)
						func() {
							defer func() { recovered = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, cause = fn(3, []float32{1, -2}, .5)
							returned = true
							complete = true
						}()
						boundaryReturned = true
					}()
					select {
					case <-joinedCh:
						joined = true
					case <-time.After(5 * time.Second):
						t.Fatal("fused retirement worker did not join; cleanup withheld")
					}
					wantReads := 1
					if row == "returned-nil" || row == "returned-nonnil" {
						wantReads = 0
					}
					if !b.bound || b.probes != 1 || b.fused != 1 || b.readCalls != wantReads || b.deferredFences != wantReads || len(b.scales) != 0 || b.deferredPublishes != 0 {
						t.Fatalf("selected fused path bound=%t probes=%d fused=%d reads=%d fences=%d scalar=%d publish=%d", b.bound, b.probes, b.fused, b.readCalls, b.deferredFences, len(b.scales), b.deferredPublishes)
					}
					wantInjections := 1
					if row == "healthy" {
						wantInjections = 0
					}
					if row == "read-nil" || row == "read-nonnil" || row == "read-goexit" {
						wantInjections = 2
					}
					if b.injections != wantInjections {
						t.Fatalf("fault injections=%d want=%d", b.injections, wantInjections)
					}
					if wantReads == 1 && row != "transport-nil" && !reflect.DeepEqual(b.readRow, []float32{.5, -1}) {
						t.Fatalf("delegated read did not complete exact nonempty row: %v", b.readRow)
					}
					wantOrder := []string{"status", "output", "input"}
					if row == "cleanup-goexit" {
						wantOrder = wantOrder[:2]
					}
					if !reflect.DeepEqual(b.order, wantOrder) || len(b.deferredPending) != 0 {
						t.Fatalf("retirement order=%v pending=%d", b.order, len(b.deferredPending))
					}
					for _, x := range []compute.Tensor{b.status, b.output, b.input} {
						want := 1
						if row == "cleanup-goexit" && x.Buf() == b.input.Buf() {
							want = 0
						}
						if b.frees[x.Buf()] != want {
							t.Fatalf("Free attempts=%d want=%d", b.frees[x.Buf()], want)
						}
					}
					if s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("owner scope not detached/unlocked")
					}
					b.mu.Unlock()
					switch row {
					case "healthy":
						if !returned || !boundaryReturned || cause != nil || !reflect.DeepEqual(got, []float32{.5, -1}) || s.halClosed || b.injections != 0 {
							t.Fatalf("healthy result=%v cause=%v", got, cause)
						}
					case "returned-nil", "returned-nonnil", "transport-nil":
						var closed *BackendForwardOperationError
						stage := "exact multiply and scaled weight publication"
						if row == "transport-nil" {
							stage = "scaled weight readback"
						}
						if !returned || !boundaryReturned || got != nil || !errors.Is(cause, b.primary) || !errors.Is(cause, ErrV41IndexWeightScaleOperation) || !errors.As(cause, &closed) || closed.Stage != stage || closed.Path != "v41-index-weight-scale" || closed.Layer != 3 || !s.halClosed || s.halFailure != cause || b.injections != 1 {
							t.Fatalf("selected primary lost: %v", cause)
						}
					case "cleanup-goexit", "read-goexit":
						if returned || boundaryReturned || recovered != nil || got != nil {
							t.Fatalf("Goexit became outcome returned=%t boundary=%t result=%v", returned, boundaryReturned, got)
						}
					default:
						if returned || !boundaryReturned || got != nil || cause != nil {
							t.Fatalf("fused retirement returned=%t boundaryReturned=%t result=%v cause=%v", returned, boundaryReturned, got, cause)
						}
						if row == "cleanup-nil" || row == "cleanup-nil-then-nonnil" || row == "read-nil" {
							if mode == "1" {
								if recovered != nil {
									t.Fatalf("nil primary replaced: %v", recovered)
								}
							} else if _, ok := recovered.(*runtime.PanicNilError); !ok {
								t.Fatalf("modern nil payload %T", recovered)
							}
						} else {
							want := b.secondary
							if row == "read-nonnil" {
								want = b.primary
							}
							if recovered != want {
								t.Fatalf("primary replaced: %v", recovered)
							}
						}
						if !s.halClosed || s.halFailure == nil {
							t.Fatal("panic did not fail-close session")
						}
					}
					if row != "healthy" && row != "cleanup-goexit" && row != "read-goexit" {
						before := []int{b.probes, b.fused, b.readCalls, b.totalFrees}
						var retry any
						retryReturned := false
						func() {
							defer func() { retry = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							_, _ = fn(3, []float32{1, -2}, .5)
							complete = true
							retryReturned = true
						}()
						if retryReturned || retry != s.halFailure || !reflect.DeepEqual(before, []int{b.probes, b.fused, b.readCalls, b.totalFrees}) {
							t.Fatalf("closed retry replayed work or lost stored cause: %v", retry)
						}
						if s.backendExecutionScope != nil || !b.mu.TryLock() {
							t.Fatal("retry stranded owner")
						}
						b.mu.Unlock()
					}

				})
			}
		})
	}
}
