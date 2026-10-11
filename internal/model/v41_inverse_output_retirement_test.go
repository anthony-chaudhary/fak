package model

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU storage with an explicit outer owner witnesses the existing inverse
// callback and Session lifecycle. It does not qualify device arithmetic.
type v41InverseRetirementRecorder struct {
	*v41DenseTestBackend
	inverseLive                  map[compute.Buffer]bool
	inverseCalls, inverseUploads int
	faultAt, retireCalls         int
	primary                      error
	closePanic                   any
}

func (b *v41InverseRetirementRecorder) V41TailRoPERowsAvailable() bool { return true }

func (b *v41InverseRetirementRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 inverse-RoPE ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("inverse retirement upload is not a classed F32 activation")
	}
	b.inverseUploads++
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.inverseLive[out.Buf()] = true
	return out
}

func (b *v41InverseRetirementRecorder) V41TailRoPERows(x, table compute.Tensor, rows, headDim, rotaryDim int) (compute.Tensor, error) {
	if x.Backend() != b || table.Backend() != b || !b.inverseLive[x.Buf()] || !b.inverseLive[table.Buf()] ||
		!reflect.DeepEqual(x.Shape, []int{rows, headDim}) || !reflect.DeepEqual(table.Shape, []int{rotaryDim / 2, 2}) {
		panic("inverse retirement received a foreign tensor or non-inverse upload")
	}
	b.inverseCalls++
	if b.inverseCalls == b.faultAt {
		panic(b.primary)
	}
	values := v41TailRoPEExactOracle(b.Backend.Read(x), b.Backend.Read(table), headDim, rotaryDim)
	out := compute.NewF32(b, []int{rows, headDim}, values)
	b.inverseLive[out.Buf()] = true
	return out, nil
}

func (b *v41InverseRetirementRecorder) Free(x compute.Tensor) {
	delete(b.inverseLive, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

func (b *v41InverseRetirementRecorder) Recycle() {
	b.retireCalls++
	if b.closePanic != nil {
		panic(b.closePanic)
	}
}

// The unchanged public consumers must latch the original wrapped typed panic,
// retire their captured owner and reject replay after rollback, even if Close
// faults. This test uses only APIs present on the parent revision.
// fak-test:runtime medium est=2s lane=default
func TestV41InverseOutputRetiresIncomingForwardError(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("inverse retirement witness requires cpu-ref")
	}
	for _, suffix := range []bool{false, true} {
		for _, closeFault := range []bool{false, true} {
			t.Run(fmt.Sprintf("suffix=%t/close-fault=%t", suffix, closeFault), func(t *testing.T) {
				m := v41IncrementalPlainModel(t, 2)
				b := &v41InverseRetirementRecorder{v41DenseTestBackend: newV41DenseTestBackend(), inverseLive: map[compute.Buffer]bool{}}
				s := v41DenseTestSession(t, m, b)
				peer := v41DenseTestSession(t, m, b)
				// LIFO: disarm before either registered Session.Close, including
				// a parent-arm assertion failure before retirement takes place.
				t.Cleanup(func() { b.closePanic = nil })
				for _, session := range []*Session{s, peer} {
					v41InverseOutputOnly(session)
					// The Rows interface also binds index-query rotation; isolate
					// the inverse consumer and require its classed input sites.
					session.v41State().gatherOps.indexRoPE = nil
				}
				v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
				if b.inverseCalls != 2*m.Cfg.NumLayers || len(b.inverseLive) != 0 {
					t.Fatal("healthy inverse consumer did not execute or release its outputs")
				}
				before := captureV41ForwardSnapshot(s.v41Forward)
				calls, uploads := b.inverseCalls, b.inverseUploads
				inner := &BackendForwardOperationError{Backend: "another-operation", Forward: ForwardPathKind("deepseek41"), Path: "incoming", Layer: 7, Stage: "rotation", Cause: ErrV41ForwardStage}
				primary := fmt.Errorf("original inverse error wrapper: %w", inner)
				b.primary, b.faultAt = primary, calls+1
				if closeFault {
					b.closePanic = errors.New("secondary inverse owner Close failure")
				}
				var output []float32
				got := recoverError(func() {
					if suffix {
						output = s.Prefill([]int{3})
					} else {
						output = s.Step(3)
					}
				})
				if got != primary || s.halFailure != primary || !s.BackendSessionClosed() {
					t.Fatalf("inverse owner lost original primary or retirement: got=%v latch=%v closed=%t", got, s.halFailure, s.BackendSessionClosed())
				}
				if output != nil || b.retireCalls != 1 || b.inverseCalls != calls+1 || b.inverseUploads != uploads+2 || len(b.inverseLive) != 0 ||
					!reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) || peer.BackendSessionClosed() || peer.halFailure != nil {
					t.Fatal("inverse failure replayed, published state, skipped retirement or retired another owner")
				}
				if retry := recoverError(func() { s.Step(3) }); retry != primary || b.inverseCalls != calls+1 || b.inverseUploads != uploads+2 || b.retireCalls != 1 {
					t.Fatal("closed inverse owner replayed or changed the original wrapper")
				}
				b.primary, b.closePanic, b.faultAt = nil, nil, 0
				v41GroupedParity(t, peer.Prefill([]int{4}), lastLogits(m.Forward([]int{4})), 1e-4)
				if peer.BackendSessionClosed() || b.inverseCalls != calls+1+m.Cfg.NumLayers || len(b.inverseLive) != 0 {
					t.Fatal("unrelated owner could not continue after selected retirement")
				}
			})
		}
	}
}
