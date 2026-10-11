package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This recorder gathers by copying cpu-ref table rows. Its DeviceMemory promise
// tests routing only, never physical qualification.
type v41EmbeddingGatherTestBackend struct {
	*v41DenseTestBackend
	calls    [][]int
	tables   int
	outputs  map[compute.Buffer]bool
	fault    string
	failCall int
	cause    error
}

func newV41EmbeddingGatherTestBackend() *v41EmbeddingGatherTestBackend {
	return &v41EmbeddingGatherTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
}

func (b *v41EmbeddingGatherTestBackend) V41EmbeddingGatherAvailable() bool { return true }

func (b *v41EmbeddingGatherTestBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if len(x.Shape) == 2 && dt == compute.F32 && x.Shape[0] > 1 {
		b.tables++
	}
	return b.v41DenseTestBackend.Upload(x, dt)
}

func (b *v41EmbeddingGatherTestBackend) V41EmbeddingGather(table compute.Tensor, ids []int, hidden int) (compute.Tensor, error) {
	if table.Backend() == nil || len(table.Shape) != 2 || table.Shape[1] != hidden {
		return compute.Tensor{}, errors.New("invalid embedding gather contract")
	}
	b.calls = append(b.calls, append([]int(nil), ids...))
	if len(b.calls) == b.failCall && b.fault == "dispatch" {
		return compute.Tensor{}, b.cause
	}
	rows := b.Backend.Read(table)
	panel := make([]float32, 0, len(ids)*hidden)
	for _, id := range ids {
		panel = append(panel, rows[id*hidden:(id+1)*hidden]...)
	}
	if len(b.calls) == b.failCall && b.fault == "short" {
		panel = panel[:len(panel)-1]
		out := compute.NewF32(b, []int{len(panel)}, panel)
		b.outputs[out.Buf()] = true
		return out, nil
	}
	out := compute.NewF32(b, []int{len(ids), hidden}, panel)
	b.outputs[out.Buf()] = true
	return out, nil
}

func (b *v41EmbeddingGatherTestBackend) Free(x compute.Tensor) {
	delete(b.outputs, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

func v41EmbeddingGatherOnly(s *Session) {
	v41ScoreOnly(s)
	st := s.v41State()
	st.indexerScore, st.indexScoreHealth = nil, nil
}

// Refs #13668. The token-embedding gather leaves the host: the panel is the
// host panel bit for bit, the table is staged once, a selected failure fails
// closed without host replay, and device-only refuses a table it cannot gather.
// fak-test:runtime medium est=3s lane=default
func TestV41EmbeddingGatherDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("embedding-gather software witness requires cpu-ref")
	}

	t.Run("forward-bit-identical", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		if !m.has(v41EmbeddingTableName) {
			t.Fatal("fixture lacks the F32 embedding table")
		}
		b := newV41EmbeddingGatherTestBackend()
		s := v41DenseTestSession(t, m, b)
		v41EmbeddingGatherOnly(s)
		host := m.NewSession()
		t.Cleanup(host.Close)
		v41ScoreTestBits(t, s.Prefill([]int{1, 2, 1}), host.Prefill([]int{1, 2, 1}))
		v41ScoreTestBits(t, s.Step(3), host.Step(3))
		if !reflect.DeepEqual(b.calls, [][]int{{1, 2, 1}, {3}}) {
			t.Fatalf("gather calls=%v", b.calls)
		}
		want, err := m.v41EmbeddingPanel([]int{2, 2, 0})
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.v41EmbeddingPanelOn([]int{2, 2, 0}, s.v41State().gatherOps)
		if err != nil {
			t.Fatal(err)
		}
		v41TailRoPECheckBits(t, "device panel", got, want)
		if len(b.outputs) != 0 {
			t.Fatal("gathered panel leaked")
		}
	})

	for _, fault := range []string{"dispatch", "short"} {
		t.Run("selected-failure-"+fault, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			b := newV41EmbeddingGatherTestBackend()
			s := v41DenseTestSession(t, m, b)
			v41EmbeddingGatherOnly(s)
			s.Prefill([]int{1, 2})
			b.fault, b.failCall = fault, len(b.calls)+1
			b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			err := recoverError(func() { s.Step(3) })
			var closed *BackendForwardOperationError
			if !errors.As(err, &closed) || closed.Path != "v41-embedding-gather" || !s.BackendSessionClosed() {
				t.Fatalf("selected gather failure lost identity: %v", err)
			}
			if fault == "dispatch" && !errors.Is(err, b.cause) {
				t.Fatalf("selected gather failure lost its cause: %v", err)
			}
			if len(b.calls) != b.failCall || len(b.outputs) != 0 {
				t.Fatal("selected gather failure replayed or leaked")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || len(b.calls) != b.failCall {
				t.Fatalf("closed session retried the gather: %v", retry)
			}
		})
	}

	t.Run("device-only-without-op-refuses", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		s := v41DenseTestSession(t, m, newV41DenseTestBackend())
		ops := s.v41State().gatherOps
		if ops.embeddingGatherFunc() != nil {
			t.Fatal("a backend without the operation selected a device gather")
		}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		got, err := m.v41EmbeddingPanelOn([]int{1}, ops)
		var closed *BackendForwardOperationError
		if got != nil || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &closed) || closed.Path != "device-only" || !s.BackendSessionClosed() {
			t.Fatalf("device-only missing gather did not fail closed: %v", err)
		}
	})
}
