package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Actual CPU storage carries the outer selected operation owner. Faults affect
// only this boundary's transients and the session's existing retirement hooks.
type v41BF16RetirementRecorder struct {
	*v41DenseTestBackend
	live                  map[compute.Buffer]bool
	calls, uploads, frees int
	faultAt               int
	primary               error
	cleanup               any
	closeSite             string
	closeFault            any
	closeCalls            [3]int
	armed                 bool
}

func newV41BF16RetirementRecorder() *v41BF16RetirementRecorder {
	return &v41BF16RetirementRecorder{v41DenseTestBackend: newV41DenseTestBackend(), live: map[compute.Buffer]bool{}}
}
func (b *v41BF16RetirementRecorder) V41BF16BoundaryAvailable() bool { return true }
func (b *v41BF16RetirementRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 BF16 boundary input" {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	b.uploads++
	y := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.live[y.Buf()] = true
	return y
}
func (b *v41BF16RetirementRecorder) V41BF16Boundary(x compute.Tensor) (compute.Tensor, error) {
	b.calls++
	if x.Backend() != b {
		panic("boundary input belongs to another owner")
	}
	if b.primary != nil && b.calls == b.faultAt {
		b.armed = true
		panic(b.primary)
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	for i, v := range values {
		values[i] = math.Float32frombits(v41BF16BoundaryOracle(math.Float32bits(v)))
	}
	y := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.live[y.Buf()] = true
	return y, nil
}
func (b *v41BF16RetirementRecorder) Free(x compute.Tensor) {
	if b.live[x.Buf()] {
		b.frees++
		delete(b.live, x.Buf())
		b.Backend.Free(x)
		if b.cleanup != nil {
			panic(b.cleanup)
		}
		return
	}
	b.v41DenseTestBackend.Free(x)
}
func (b *v41BF16RetirementRecorder) BeginBatch() {}
func (b *v41BF16RetirementRecorder) FlushBatch() { b.retire("flush", 0) }
func (b *v41BF16RetirementRecorder) Recycle()    { b.retire("recycle", 1) }
func (b *v41BF16RetirementRecorder) Trim()       { b.retire("trim", 2) }
func (b *v41BF16RetirementRecorder) retire(site string, index int) {
	if !b.armed {
		return
	}
	b.closeCalls[index]++
	if b.closeSite == site && b.closeFault != nil {
		panic(b.closeFault)
	}
}

// Incoming typed errors may describe a different operation. They do not prove
// that the callback's captured owner has retired. Preserve the exact error while
// closing that owner, including secondary Free/Close panics and no-replay retry.
// Ref #13813. Software lifecycle evidence only; ordinary admission stays off.
// fak-test:runtime medium est=2s lane=default
func TestV41BF16BoundaryRetiresIncomingForwardError(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("BF16 retirement witness requires cpu-ref")
	}
	for _, site := range []string{"flush", "recycle", "trim"} {
		for _, wrapped := range []bool{false, true} {
			t.Run(fmt.Sprintf("callback/%s/wrapped=%t", site, wrapped), func(t *testing.T) {
				b := newV41BF16RetirementRecorder()
				s := &Session{M: &Model{}, Backend: b, halW: map[string]compute.Tensor{}}
				peer := &Session{M: &Model{}, Backend: b, halW: map[string]compute.Tensor{}}
				t.Cleanup(func() { b.cleanup, b.closeFault = nil, nil; s.Close(); peer.Close() })
				fn, other := s.v41BF16BoundaryFunc(), peer.v41BF16BoundaryFunc()
				if fn == nil || other == nil {
					t.Fatal("selected boundary did not bind")
				}
				values := []float32{1, math.Float32frombits(0x80000000), math.Float32frombits(0x00018000)}
				if got, err := other(0, "peer", values); err != nil || len(got) != len(values) || len(b.live) != 0 {
					t.Fatal("healthy peer control failed", err)
				}
				primary := error(&BackendForwardOperationError{Backend: "another-operation", Forward: ForwardPathKind("deepseek41"), Path: "incoming", Layer: 7, Stage: "round", Cause: ErrV41ForwardStage})
				if wrapped {
					primary = fmt.Errorf("original error wrapper: %w", primary)
				}
				b.primary, b.faultAt = primary, 2
				b.cleanup, b.closeSite, b.closeFault = "secondary transient Free", site, errors.New("secondary retirement fault")
				if got := recoverError(func() { _, _ = fn(2, "final-norm-output", values) }); got != primary || s.halFailure != primary || !s.BackendSessionClosed() {
					t.Fatalf("current owner not retired with original error: got=%v latch=%v closed=%t", got, s.halFailure, s.BackendSessionClosed())
				}
				wantClose := [3]int{1, 0, 0}
				if site != "flush" {
					wantClose[1] = 1
				}
				if site == "trim" {
					wantClose[2] = 1
				}
				if b.closeCalls != wantClose || b.calls != 2 || b.uploads != 2 || b.frees != 3 || len(b.live) != 0 || peer.BackendSessionClosed() || peer.halFailure != nil {
					t.Fatal("retirement lost ownership, transient release or peer isolation")
				}
				if got := recoverError(func() { _, _ = fn(2, "retry", values) }); got != primary || b.calls != 2 || b.uploads != 2 || b.closeCalls != wantClose {
					t.Fatal("closed callback replayed or changed original error")
				}
				b.primary, b.cleanup, b.closeFault = nil, nil, nil
				if _, err := other(0, "peer", values); err != nil || peer.BackendSessionClosed() || b.calls != 3 {
					t.Fatal("unrelated owner could not continue")
				}
			})
		}
	}
	for _, suffix := range []bool{false, true} {
		t.Run(fmt.Sprintf("session/suffix=%t", suffix), func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			b := newV41BF16RetirementRecorder()
			s := v41DenseTestSession(t, m, b)
			t.Cleanup(func() { b.cleanup, b.closeFault = nil, nil })
			s.Prefill([]int{1, 2})
			if b.calls != 4 {
				t.Fatal("fixture did not reach two final-norm boundaries per token")
			}
			before := captureV41ForwardSnapshot(s.v41Forward)
			primary := &BackendForwardOperationError{Backend: "bf16-retirement", Forward: ForwardPathKind("deepseek41"), Path: "incoming", Layer: -1, Stage: "round", Cause: ErrV41ForwardStage}
			b.primary, b.faultAt = primary, 5
			b.cleanup, b.closeSite, b.closeFault = "secondary transient Free", "recycle", errors.New("secondary Close")
			err := recoverError(func() {
				if suffix {
					s.Prefill([]int{3})
				} else {
					s.Step(3)
				}
			})
			if err != primary || s.halFailure != primary || !s.BackendSessionClosed() || b.calls != 5 || len(b.live) != 0 || b.closeCalls != [3]int{1, 1, 0} || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatalf("selected final norm replayed, changed fault or advanced state: err=%v calls=%d closed=%t", err, b.calls, s.BackendSessionClosed())
			}
			if retry := recoverError(func() { s.Step(3) }); retry != primary || b.calls != 5 {
				t.Fatal("closed final-norm owner retried")
			}
		})
	}
}
