package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Keep the recorder's public backend identity while failing the inner CPU
// implementation at the panel's third, down-projection recording call.
type v41SharedPanelDownRecordingFault struct {
	compute.Backend
	calls int
	fault error
}

func (b *v41SharedPanelDownRecordingFault) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	b.calls++
	if b.calls == 3 {
		panic(b.fault)
	}
	return b.Backend.BatchedMatMul(w, x, rows)
}

// An observed earlier arithmetic verdict must survive a later recording
// failure even when there is no final panel output to read. The scalar arm
// provides the existing first-cause contract on the same publication prefix.
// fak-test:runtime fast est=100ms lane=default
func TestV41ColdSharedResidentPanelPartialNumericPrimary(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, panel := range []bool{false, true} {
		name := "scalar-control"
		if panel {
			name = "panel-candidate"
		}
		t.Run(name, func(t *testing.T) {
			m, _ := v41ColdSharedFixture(t, compute.F32)
			b := newV41ColdSharedRecorder()
			b.panel = true
			s := v41ColdSharedSession(m, b)
			t.Cleanup(func() {
				s.Close()
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
			up := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishSharedExpertProjection, Index: 0}
			later := &compute.BackendError{Backend: "test", Site: "down recording", Err: errors.New("later down recording failure")}
			b.deferredFaultAt, b.deferredFault = 2, up
			b.failMatMul, b.fault = 3, later
			inner := &v41SharedPanelDownRecordingFault{Backend: b.Backend, fault: later}
			b.Backend = inner
			input := v41ColdSharedPanelInput(2, m.Cfg.HiddenSize)
			var out []float32
			var selected bool
			var err error
			if panel {
				out, selected, err = s.v41ColdSharedExpertPanel(0)(input, 2)
			} else {
				out, selected, err = s.v41ColdSharedExpertChain(0)(input[:m.Cfg.HiddenSize])
			}
			orderedSites := []uint32{compute.V41BF16PublishSharedExpertProjection, compute.V41BF16PublishSharedExpertProjection, compute.V41BF16PublishSharedExpertDownInput}
			if b.deferredPublishes != 3 || b.deferredFences != 1 || len(b.deferredPending) != 0 ||
				!reflect.DeepEqual(b.deferredReadSites, [][]uint32{orderedSites}) || !reflect.DeepEqual(b.deferredReadRows, []int{0}) {
				t.Fatalf("wrong partial-recording witness: publishes=%d fences=%d pending=%d sites=%v rows=%v", b.deferredPublishes, b.deferredFences, len(b.deferredPending), b.deferredReadSites, b.deferredReadRows)
			}
			if panel && inner.calls != 3 || !panel && b.matmuls != 3 {
				t.Fatalf("down recording did not fail at the third projection: panel=%t batched=%d scalar=%d", panel, inner.calls, b.matmuls)
			}
			if !selected || out != nil || !s.BackendSessionClosed() || err == nil || err != s.halFailure {
				t.Fatalf("wrong selected-failure lifecycle: panel=%t selected=%t out=%v closed=%t err=%v", panel, selected, out, s.BackendSessionClosed(), err)
			}
			v41ColdSharedAssertReleased(t, b, s)
			var boundary *V41BF16BoundaryOperationError
			var operation *BackendForwardOperationError
			if !errors.Is(err, up) || errors.Is(err, later) || errors.Is(err, errV41BF16BoundaryResult) ||
				!errors.As(err, &boundary) || boundary.Layer != 0 || boundary.Site != "ffn.shared_experts.w3.weight projection" ||
				!errors.As(err, &operation) || operation.Path != "v41-bf16-publish" || operation.Stage != "checked publication" {
				t.Fatalf("PANEL_PARTIAL_NUMERIC_PRIMARY_LOST: panel=%t selected=%t out=%v closed=%t err=%v", panel, selected, out, s.BackendSessionClosed(), err)
			}
		})
	}
}
