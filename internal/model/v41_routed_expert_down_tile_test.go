package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU orchestration witness. Quantized math is the scalar reference inside
// the fake tile; no shader or physical throughput result is implied.
type v41ExpertDownTileWitness struct {
	*v41ExpertTileWitness
	downTiles         bool
	downRows          []int
	activationCalls   int
	failActivationAt  int
	activationFailure error
	failDownAt        int
	downFailure       error
}

func (b *v41ExpertDownTileWitness) SupportsV41Q3KTokenPanel(w compute.Tensor, rows int) bool {
	return b.downTiles && w.Dtype == compute.Q3_K && (rows == 2 || rows == 4 || rows == 8)
}
func (b *v41ExpertDownTileWitness) V41Q3KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.downRows = append(b.downRows, rows)
	if len(b.downRows) == b.failDownAt {
		return compute.Tensor{}, b.downFailure
	}
	in, out := w.Shape[1], w.Shape[0]
	values := b.Backend.Read(x)
	panel := make([]float32, rows*out)
	for row := 0; row < rows; row++ {
		xr := b.Backend.Upload(compute.NewF32(compute.Default(), []int{in}, values[row*in:(row+1)*in]), compute.F32)
		y := b.Backend.MatMul(w, xr)
		copy(panel[row*out:(row+1)*out], b.Backend.Read(y))
		b.Backend.Free(y)
		b.Backend.Free(xr)
	}
	y := b.tensor([]int{rows * out}, panel)
	b.panels = append(b.panels, y.Buf())
	return y, nil
}
func (b *v41ExpertDownTileWitness) V41RoutedSwiGLUDeferred(g, u compute.Tensor, limit, weight float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.activationCalls++
	if b.activationCalls == b.failActivationAt {
		return compute.Tensor{}, compute.V41DeferredStatus{}, b.activationFailure
	}
	a, st, err := b.v41RoutedLayerTestBackend.V41RoutedSwiGLUDeferred(g, u, limit, weight)
	if a.Buf() != nil {
		b.ownedTiles[a.Buf()] = true
	}
	return a, st, err
}

// fak-test:runtime medium est=5s lane=default
func TestV41ExpertDownTokenTileConsumer(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	inputs, weights := make([][]float32, 15), make([]float32, 15)
	for row := range inputs {
		inputs[row] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[row] {
			inputs[row][i] = float32((row*3+i)%13-6) / 16
		}
		weights[row] = float32(row+1) / 32
	}
	newSession := func(downTiles bool) (*Session, *v41ExpertDownTileWitness) {
		b := &v41ExpertDownTileWitness{v41ExpertTileWitness: &v41ExpertTileWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, tiles: true, ownedTiles: map[compute.Buffer]bool{}}, downTiles: downTiles}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	for _, count := range []int{8, 15} {
		control, cb := newSession(false)
		candidate, b := newSession(true)
		want, oldOutcome, oldErr := control.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:count], weights[:count])
		got, outcome, err := candidate.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:count], weights[:count])
		if oldOutcome != v41RoutedExpertHandled || oldErr != nil || outcome != v41RoutedExpertHandled || err != nil || len(got) != count {
			t.Fatalf("count=%d control=%v/%v candidate=%v/%v", count, oldOutcome, oldErr, outcome, err)
		}
		for row := range got {
			if !v41SameBits(got[row], want[row]) {
				t.Fatalf("count=%d row=%d output differs", count, row)
			}
		}
		wantDown, singleton := []int{8}, 0
		if count == 15 {
			wantDown, singleton = []int{8, 4, 2}, 1
		}
		if !reflect.DeepEqual(b.downRows, wantDown) || len(b.matmulInputs) != 3*singleton || len(cb.matmulInputs) != count+2*singleton || b.gathers-cb.gathers != count-singleton+len(wantDown) {
			t.Fatalf("count=%d down tiles=%v scalar matmuls=%d/%d extra gathers=%d", count, b.downRows, len(b.matmulInputs), len(cb.matmulInputs), b.gathers-cb.gathers)
		}
		if b.fences != 1 || cb.fences != 1 || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
			t.Fatal("tile introduced extra fence or retained activation/panel ownership")
		}
	}
	t.Run("unissued_down_is_not_fabricated_after_activation_failure", func(t *testing.T) {
		s, b := newSession(true)
		failure := &compute.BackendError{Backend: "test", Site: "second activation", Err: errors.New("recording failed")}
		b.failActivationAt, b.activationFailure = 2, failure
		b.publishFaultAt = map[int]bool{1: true} // Hypothetical scalar down: never issued here.
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:2], weights[:2])
		if out != nil || outcome != v41RoutedExpertError || !errors.Is(err, failure) || !s.BackendSessionClosed() || len(b.downRows) != 0 || b.publishes != 0 || b.activationCalls != 2 || b.fences != 1 || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
			t.Fatalf("unissued down invented or selected work replayed: %v", err)
		}
	})
	t.Run("earlier_issued_activation_verdict_survives_first_down_record_failure", func(t *testing.T) {
		s, b := newSession(true)
		b.activationFaults = map[float32]*compute.V41RoutedSwiGLUFault{weights[1]: {Stage: compute.V41RoutedSwiGLUStageDownOperand}}
		b.publicationErrorAt = 1
		b.failure = &compute.BackendError{Backend: "test", Site: "first down publication", Err: errors.New("later recording failure")}
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:2], weights[:2])
		var numeric *V41ExpertOperationError
		if out != nil || outcome != v41RoutedExpertError || !errors.As(err, &numeric) || numeric.Stage != "down" || errors.Is(err, b.failure) || !s.BackendSessionClosed() || b.fences != 1 || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
			t.Fatalf("earlier actually issued activation verdict lost: %v", err)
		}
	})
	t.Run("earlier_issued_down_verdict_survives_next_tile_activation_failure", func(t *testing.T) {
		s, b := newSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.failActivationAt, b.activationFailure = 9, &compute.BackendError{Backend: "test", Site: "next tile activation", Err: errors.New("later recording failure")}
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:10], weights[:10])
		var numeric *V41ExpertOperationError
		if out != nil || outcome != v41RoutedExpertError || !errors.As(err, &numeric) || numeric.Stage != "down" || !s.BackendSessionClosed() || !reflect.DeepEqual(b.downRows, []int{8}) || b.fences != 1 || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
			t.Fatalf("earlier issued down verdict lost: %v", err)
		}
	})
}
