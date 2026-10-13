package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The tile and byte-copy implementations below are CPU contract recorders.
// They witness the production cohort consumer, not native shader parity.
type v41ExpertTileWitness struct {
	*v41ScalarCohortWitness
	tiles       bool
	tileRows    []int
	gathers     int
	ownedTiles  map[compute.Buffer]bool
	panels      []compute.Buffer
	failTileAt  int
	tileFailure error
}

func (b *v41ExpertTileWitness) SupportsV41Q2KTokenPanel(w compute.Tensor, rows int) bool {
	return b.tiles && w.Dtype == compute.Q2_K && (rows == 2 || rows == 4 || rows == 8)
}

func (b *v41ExpertTileWitness) tensor(shape []int, values []float32) compute.Tensor {
	out := b.Backend.Upload(compute.NewF32(compute.Default(), shape, values), compute.F32)
	b.ownedTiles[out.Buf()] = true
	return out
}

func (b *v41ExpertTileWitness) V41Q2KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.tileRows = append(b.tileRows, rows)
	if len(b.tileRows) == b.failTileAt {
		return compute.Tensor{}, b.tileFailure
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

func (b *v41ExpertTileWitness) V41RowLogAvailable() bool { return true }
func (b *v41ExpertTileWitness) V41RowLogAlloc(int, int) (compute.Tensor, error) {
	panic("expert token tile must use fresh row gathers only")
}
func (b *v41ExpertTileWitness) V41RowLogCopy(compute.Tensor, int, compute.Tensor, int, int) error {
	panic("expert token tile must use fresh row gathers only")
}
func (b *v41ExpertTileWitness) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	rows, err := compute.ValidateV41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	b.gathers++
	var values []float32
	for _, span := range spans {
		if !b.ownedTiles[span.Src.Buf()] {
			return compute.Tensor{}, errors.New("tile panel released before row copy")
		}
		x := b.Backend.Read(span.Src)
		values = append(values, x[span.Row*width:(span.Row+span.Rows)*width]...)
	}
	return b.tensor([]int{rows, width}, values), nil
}
func (b *v41ExpertTileWitness) Free(x compute.Tensor) {
	delete(b.ownedTiles, x.Buf())
	b.v41ScalarCohortWitness.Free(x)
}
func (b *v41ExpertTileWitness) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	for _, panel := range b.panels {
		if !b.ownedTiles[panel] {
			return nil, nil, errors.New("tile panel released before final status read")
		}
	}
	return b.v41ScalarCohortWitness.V41ReadDeferred(statuses, rows)
}

// fak-test:runtime medium est=5s lane=default
func TestV41ExpertTokenTileConsumer(t *testing.T) {
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
	newSession := func(tiles bool) (*Session, *v41ExpertTileWitness) {
		b := &v41ExpertTileWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, tiles: tiles, ownedTiles: map[compute.Buffer]bool{}}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	for _, count := range []int{8, 15} {
		control, cb := newSession(false)
		candidate, tb := newSession(true)
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
		wantTiles, singleton := []int{8, 8}, 0
		if count == 15 {
			wantTiles, singleton = []int{8, 8, 4, 4, 2, 2}, 1
		}
		if !reflect.DeepEqual(tb.tileRows, wantTiles) || len(tb.matmulInputs) != count+2*singleton || len(cb.matmulInputs) != 3*count || len(cb.tileRows) != 0 || tb.gathers != 2*(count-singleton) {
			t.Fatalf("count=%d tiles=%v scalar matmuls=%d control=%d gathers=%d", count, tb.tileRows, len(tb.matmulInputs), len(cb.matmulInputs), tb.gathers)
		}
		if cb.fences != 1 || tb.fences != 1 || len(tb.pending) != 0 || len(tb.ownedTiles) != 0 {
			t.Fatalf("count=%d fences=%d/%d pending=%d tile owners=%d", count, cb.fences, tb.fences, len(tb.pending), len(tb.ownedTiles))
		}
	}
	// A later tile's selected failure must not hide an earlier row's numeric
	// verdict, and must retire the session without scalar replay or leaked panels.
	s, b := newSession(true)
	b.publishFaultAt = map[int]bool{1: true}
	b.failTileAt = 3 // First projection of the two-row tail after eight rows.
	b.tileFailure = &compute.BackendError{Backend: "test", Site: "token tile", Err: errors.New("selected tile failure")}
	out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:10], weights[:10])
	var numeric *V41ExpertOperationError
	if out != nil || outcome != v41RoutedExpertError || !errors.As(err, &numeric) || numeric.Stage != "down" || !s.BackendSessionClosed() || len(b.matmulInputs) != 8 || len(b.pending) != 0 || len(b.ownedTiles) != 0 || b.fences != 1 {
		t.Fatalf("selected failure replay/precedence/cleanup: outcome=%v err=%v matmuls=%d pending=%d owners=%d fences=%d", outcome, err, len(b.matmulInputs), len(b.pending), len(b.ownedTiles), b.fences)
	}
}
