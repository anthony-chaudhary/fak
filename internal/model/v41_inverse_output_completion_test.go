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

// The recorder owns actual cpu-ref tensors under an actual Session scope. Its
// public Read exercises the retained-read helper's wrapper/copy arm only.
type v41InverseCompletionRecorder struct {
	*v41DenseTestBackend
	mu                        sync.Mutex
	session                   *Session
	kinds                     map[compute.Buffer]string
	frees                     map[compute.Buffer]int
	input, output             compute.Tensor
	order                     []string
	row                       string
	primary, secondary        error
	uploads, calls, readCalls int
	injections                int
	readValues, readScratch   []float32
}

func (b *v41InverseCompletionRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *v41InverseCompletionRecorder) assertOwner() {
	scope := b.session.backendExecutionScope
	if scope == nil || !scope.live || scope.owner != &b.mu {
		panic("inverse completion lacks its Session owner")
	}
	member := false
	for _, session := range scope.sessions {
		member = member || session == b.session
	}
	if !member {
		panic("inverse completion owner lacks its Session binding")
	}
	if b.mu.TryLock() {
		b.mu.Unlock()
		panic("inverse completion owner is unlocked")
	}
}
func (b *v41InverseCompletionRecorder) V41TailRoPERowsAvailable() bool { return true }
func (b *v41InverseCompletionRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.assertOwner()
	kind := ""
	switch site {
	case "V4.1 inverse-RoPE output":
		kind = "input"
	case "V4.1 inverse-RoPE table":
		kind = "table"
	default:
		panic("inverse completion used an unrelated upload")
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("inverse completion upload contract")
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.kinds[out.Buf()] = kind
	b.uploads++
	if kind == "input" {
		b.input = out
	}
	return out
}
func (b *v41InverseCompletionRecorder) V41TailRoPERows(x, table compute.Tensor, rows, hd, rd int) (compute.Tensor, error) {
	b.assertOwner()
	if x.Backend() != b || table.Backend() != b || b.kinds[x.Buf()] != "input" || b.kinds[table.Buf()] != "table" ||
		rows != 1 || hd != 8 || rd != 4 || !reflect.DeepEqual(x.Shape, []int{rows, hd}) || !reflect.DeepEqual(table.Shape, []int{rd / 2, 2}) {
		panic("inverse completion received foreign or malformed operands")
	}
	b.calls++
	values := v41TailRoPEExactOracle(b.Backend.Read(x), b.Backend.Read(table), hd, rd)
	b.output = compute.NewF32(b, []int{rows, hd}, values)
	b.kinds[b.output.Buf()] = "output"
	if b.row == "returned-error-cleanup" {
		b.injections++
		// A fresh output accompanies the error so its actual Free is exercised.
		return b.output, b.primary
	}
	return b.output, nil
}
func (b *v41InverseCompletionRecorder) Read(x compute.Tensor) []float32 {
	b.assertOwner()
	if x.Backend() != b || b.kinds[x.Buf()] != "output" {
		panic("inverse completion read an unrelated or retired tensor")
	}
	b.readCalls++
	values := b.Backend.Read(x)
	b.readScratch = values
	b.readValues = append([]float32(nil), values...)
	switch b.row {
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
	return values
}
func (b *v41InverseCompletionRecorder) Free(x compute.Tensor) {
	b.assertOwner()
	kind := b.kinds[x.Buf()]
	if x.Backend() != b || kind == "" || b.frees[x.Buf()] != 0 {
		panic("inverse completion retried or freed a foreign tensor")
	}
	b.frees[x.Buf()]++
	delete(b.kinds, x.Buf())
	b.Backend.Free(x)
	if kind == "table" {
		return
	}
	b.order = append(b.order, kind)
	if kind == "input" && b.row == "cleanup-nil" {
		// A later nonnil cleanup must not replace the first nil cleanup.
		b.injections++
		panic(b.secondary)
	}
	if kind != "output" {
		return
	}
	switch b.row {
	case "cleanup-nil", "primary-cleanup-nil":
		b.injections++
		panic(nil)
	case "returned-error-cleanup":
		b.injections++
		panic(b.secondary)
	case "cleanup-goexit":
		b.injections++
		runtime.Goexit()
	}
}

// The selected callback must not report success after legacy nil unwind or lose
// a primary to transient cleanup. The public consumer's separate length check
// is not a substitute for this callback's retirement contract. Goexit controls
// establish only nonreturn, owner release and already-registered Free attempts.
// fak-test:runtime medium est=2s lane=default
func TestV41InverseOutputCompletion(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("inverse completion requires cpu-ref")
	}
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, row := range []string{"healthy", "read-nil", "cleanup-nil", "primary-cleanup-nil", "returned-error-cleanup", "read-goexit", "cleanup-goexit"} {
				t.Run(row, func(t *testing.T) {
					b := &v41InverseCompletionRecorder{
						v41DenseTestBackend: newV41DenseTestBackend(), kinds: map[compute.Buffer]string{}, frees: map[compute.Buffer]int{},
						primary:   fmt.Errorf("inverse original: %w", &BackendForwardOperationError{Backend: "incoming", Forward: ForwardPathKind("deepseek41"), Path: "original", Layer: 9, Stage: "original", Cause: ErrV41ForwardStage}),
						secondary: errors.New("inverse secondary cleanup"),
					}
					s := &Session{M: &Model{}, Backend: b}
					b.session = s
					joined := true
					t.Cleanup(func() {
						if !joined {
							return // A failed watchdog cannot race the still-owned fixture.
						}
						b.row = "healthy"
						s.Close()
						if len(b.kinds) != 0 {
							t.Error("inverse fixture retained owned tensors after cleanup")
						}
					})
					fn := s.v41InverseOutputRoPEFunc()
					if fn == nil {
						t.Fatal("inverse completion did not bind")
					}
					input := []float32{1, -2, 3, -4, 5, -6, 7, -8}
					beforeInput := append([]float32(nil), input...)
					cos, sin := []float32{1, .5}, []float32{0, -.25}
					want := v41TailRoPEExactOracle(input, []float32{0, 1, -.25, .5}, 8, 4)
					// Warm both exact cache entries, then select the first without a miss.
					for _, tableCos := range [][]float32{cos, {.75, .5}} {
						func() {
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, err := fn(3, input, tableCos, sin, 1, 8, 4)
							if err != nil || len(got) != len(input) {
								t.Fatalf("warm inverse did not complete: %v", err)
							}
							complete = true
						}()
					}
					if len(s.halW) != 2 || len(b.kinds) != 2 {
						t.Fatal("warm inverse did not retain exactly two tables")
					}
					first := s.halW["v41-inverse-rope-table:0"].Buf()
					second := s.halW["v41-inverse-rope-table:1"].Buf()
					if first == nil || second == nil || first == second || b.kinds[first] != "table" || b.kinds[second] != "table" {
						t.Fatal("warm inverse table identity is invalid")
					}
					b.uploads, b.calls, b.readCalls, b.injections = 0, 0, 0, 0
					b.order, b.readValues, b.readScratch, b.row = nil, nil, nil, row
					var got []float32
					var cause error
					var recovered any
					returned, boundaryReturned := false, false
					joinedCh := make(chan struct{})
					joined = false
					go func() {
						defer close(joinedCh)
						func() {
							defer func() { recovered = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							got, cause = fn(3, input, cos, sin, 1, 8, 4)
							returned, complete = true, true
						}()
						boundaryReturned = true
					}()
					select {
					case <-joinedCh:
						joined = true
					case <-time.After(5 * time.Second):
						t.Fatal("inverse completion worker did not join; cleanup withheld")
					}
					wantReads, wantInjections := 1, 1
					switch row {
					case "healthy":
						wantInjections = 0
					case "cleanup-nil", "primary-cleanup-nil":
						wantInjections = 2
					case "returned-error-cleanup":
						wantReads, wantInjections = 0, 2
					}
					if b.uploads != 1 || b.calls != 1 || b.readCalls != wantReads || b.injections != wantInjections {
						t.Fatalf("inverse selected calls uploads=%d dispatch=%d reads=%d injections=%d", b.uploads, b.calls, b.readCalls, b.injections)
					}
					if wantReads == 1 {
						v41TailRoPECheckBits(t, "delegated inverse read", b.readValues, want)
					}
					v41TailRoPECheckBits(t, "unchanged input", input, beforeInput)
					if !reflect.DeepEqual(b.order, []string{"output", "input"}) || b.frees[b.output.Buf()] != 1 || b.frees[b.input.Buf()] != 1 {
						t.Fatalf("inverse transient attempts=%v output=%d input=%d", b.order, b.frees[b.output.Buf()], b.frees[b.input.Buf()])
					}
					if s.backendExecutionScope != nil || !b.mu.TryLock() {
						t.Fatal("inverse completion stranded its owner")
					}
					b.mu.Unlock()
					switch row {
					case "healthy":
						if !returned || !boundaryReturned || recovered != nil || cause != nil || s.halClosed || s.halFailure != nil || len(got) != len(want) {
							t.Fatalf("healthy inverse returned=%t boundary=%t cause=%v", returned, boundaryReturned, cause)
						}
						v41TailRoPECheckBits(t, "healthy inverse", got, want)
						if &got[0] == &input[0] || &got[0] == &b.readScratch[0] {
							t.Fatal("healthy inverse returned borrowed storage")
						}
					case "read-goexit", "cleanup-goexit":
						if returned || boundaryReturned || recovered != nil || got != nil || cause != nil {
							t.Fatalf("inverse Goexit became an outcome: returned=%t boundary=%t", returned, boundaryReturned)
						}
					case "returned-error-cleanup":
						var closed *BackendForwardOperationError
						if !returned || !boundaryReturned || recovered != nil || got != nil || !errors.Is(cause, b.primary) || errors.Is(cause, b.secondary) ||
							!errors.As(cause, &closed) || closed.Path != "v41-inverse-output-rope" || closed.Stage != "inverse-rope" || closed.Layer != 3 || s.halFailure != cause {
							t.Fatalf("inverse returned primary replaced: %v", cause)
						}
					default:
						if returned || !boundaryReturned || got != nil || cause != nil {
							t.Fatalf("inverse returned=%t boundaryReturned=%t result=%v cause=%v closed=%t primary=%t", returned, boundaryReturned, got, cause, s.halClosed, s.halFailure != nil)
						}
						if row == "primary-cleanup-nil" {
							if recovered != b.primary || s.halFailure != b.primary {
								t.Fatalf("inverse original wrapper replaced: %v", recovered)
							}
						} else {
							if mode == "1" {
								if recovered != nil {
									t.Fatalf("legacy nil replaced: %v", recovered)
								}
							} else if _, ok := recovered.(*runtime.PanicNilError); !ok {
								t.Fatalf("modern nil replaced: %T", recovered)
							}
							var closed *BackendForwardOperationError
							if !errors.As(s.halFailure, &closed) || closed.Path != "v41-inverse-output-rope" || closed.Stage != "output readback" || closed.Layer != 3 {
								t.Fatalf("inverse nil lost selected identity: %v", s.halFailure)
							}
						}
					}
					if row == "healthy" || row == "read-goexit" || row == "cleanup-goexit" {
						if s.halW["v41-inverse-rope-table:0"].Buf() != first || s.halW["v41-inverse-rope-table:1"].Buf() != second {
							t.Fatal("transient cleanup changed cached table ownership")
						}
					} else {
						if !s.halClosed || s.halFailure == nil || len(b.kinds) != 0 || b.frees[first] != 1 || b.frees[second] != 1 {
							t.Fatal("inverse selected failure did not retire its session")
						}
						before := []int{b.uploads, b.calls, b.readCalls, len(b.order)}
						var retry any
						retryReturned := false
						func() {
							defer func() { retry = recover() }()
							scope := s.beginBackendExecution()
							complete := false
							defer scope.finishCompleted(&complete)
							_, _ = fn(3, input, cos, sin, 1, 8, 4)
							retryReturned, complete = true, true
						}()
						if retryReturned || retry != s.halFailure || !reflect.DeepEqual(before, []int{b.uploads, b.calls, b.readCalls, len(b.order)}) || s.backendExecutionScope != nil {
							t.Fatalf("closed inverse replayed work: %v", retry)
						}
						if !b.mu.TryLock() {
							t.Fatal("inverse retry stranded its owner")
						}
						b.mu.Unlock()
					}
				})
			}
		})
	}
}
