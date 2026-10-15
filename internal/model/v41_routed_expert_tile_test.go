package model

import (
	"errors"
	"fmt"
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
	const completedTileRows, tailRows = 8, 2
	s, b := newSession(true)
	b.publishFaultAt = map[int]bool{1: true}
	b.failTileAt = 3 // First projection of the two-row tail after eight rows.
	b.tileFailure = &compute.BackendError{Backend: "test", Site: "token tile", Err: errors.New("selected tile failure")}
	out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:completedTileRows+tailRows], weights[:completedTileRows+tailRows])
	var numeric *V41ExpertOperationError
	if out != nil || outcome != v41RoutedExpertError || !errors.As(err, &numeric) || numeric.Stage != "down" || !s.BackendSessionClosed() || len(b.matmulInputs) != completedTileRows || len(b.pending) != 0 || len(b.ownedTiles) != 0 || b.fences != 1 {
		t.Fatalf("selected failure replay/precedence/cleanup: outcome=%v err=%v matmuls=%d pending=%d owners=%d fences=%d", outcome, err, len(b.matmulInputs), len(b.pending), len(b.ownedTiles), b.fences)
	}
}

// The wide consumer witness exercises the real cold cohort with CPU arithmetic.
// It measures orchestration counts only, never native parity or elapsed speed.
type v41ExpertGateUp32Witness struct {
	*v41ExpertDownTileWitness
	grid                               bool
	gridChecks, declineGridAt, uploads int
	scalarUploads                      int
	gridProbeRows                      []int
	partialTileFailure                 error
}

func (b *v41ExpertGateUp32Witness) SupportsV41Q2KTokenPanel(w compute.Tensor, rows int) bool {
	if rows > 2 && rows != 4 && rows != 8 && rows <= 32 {
		b.gridProbeRows = append(b.gridProbeRows, rows)
		b.gridChecks++
		return b.grid && b.gridChecks != b.declineGridAt && w.Dtype == compute.Q2_K
	}
	return b.v41ExpertTileWitness.SupportsV41Q2KTokenPanel(w, rows)
}

func (b *v41ExpertGateUp32Witness) V41Q2KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	out, err := b.v41ExpertTileWitness.V41Q2KTokenPanel(w, x, rows)
	if err == nil && len(b.tileRows) == 2 && b.partialTileFailure != nil {
		return out, b.partialTileFailure
	}
	return out, err
}

func (b *v41ExpertGateUp32Witness) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 same-expert token tile" {
		b.uploads++
	}
	if site == "moe expert gate/up activation" {
		b.scalarUploads++
	}
	return b.Backend.Upload(x, dt)
}

// fak-test:runtime medium est=10s lane=default
func TestV41ExpertGateUp32Consumer(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	inputs, weights := make([][]float32, 32), make([]float32, 32)
	for row := range inputs {
		inputs[row] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[row] {
			inputs[row][i] = float32((row*3+i)%13-6) / 16
		}
		weights[row] = float32(row+1) / 64
	}
	newSession := func(grid, down bool, decline int) (*Session, *v41ExpertGateUp32Witness) {
		b := &v41ExpertGateUp32Witness{
			v41ExpertDownTileWitness: &v41ExpertDownTileWitness{
				v41ExpertTileWitness: &v41ExpertTileWitness{
					v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true},
					tiles:                  true, ownedTiles: map[compute.Buffer]bool{},
				}, downTiles: down,
			}, grid: grid, declineGridAt: decline,
		}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	control, cb := newSession(false, true, 0)
	want, oldOutcome, oldErr := control.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
	if oldOutcome != v41RoutedExpertHandled || oldErr != nil {
		t.Fatalf("control: %v/%v", oldOutcome, oldErr)
	}
	for _, tc := range []struct {
		name       string
		grid, down bool
		decline    int
		wide       bool
	}{
		{"qualified", true, true, 0, true},
		{"legacy", false, true, 0, false},
		{"gate_refuses", true, true, 1, false},
		{"up_refuses", true, true, 2, false},
		{"down_unavailable", true, false, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, b := newSession(tc.grid, tc.down, tc.decline)
			got, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
			if outcome != v41RoutedExpertHandled || err != nil || len(got) != 32 {
				t.Fatalf("%v/%v rows=%d", outcome, err, len(got))
			}
			for row := range got {
				if !v41SameBits(got[row], want[row]) {
					t.Fatalf("row%d differs", row)
				}
			}
			wantTiles, wantUploads := []int{8, 8, 8, 8, 8, 8, 8, 8}, 4
			if tc.wide {
				wantTiles, wantUploads = []int{32, 32}, 1
			}
			if !reflect.DeepEqual(b.tileRows, wantTiles) || b.uploads != wantUploads {
				t.Fatalf("tiles=%v uploads=%d", b.tileRows, b.uploads)
			}
			if tc.down {
				if !reflect.DeepEqual(b.downRows, []int{8, 8, 8, 8}) || b.publishes != 32 || len(b.matmulInputs) != 0 || b.gathers != cb.gathers {
					t.Fatalf("down schedule changed: rows=%v publishes=%d scalar=%d gathers=%d/%d", b.downRows, b.publishes, len(b.matmulInputs), b.gathers, cb.gathers)
				}
			} else if len(b.downRows) != 0 || len(b.matmulInputs) != 32 || b.gridChecks != 0 {
				t.Fatal("missing Q3 capability enabled wide route")
			}
			if b.fences != 1 || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
				t.Fatal("wide cohort fence/ownership leak")
			}
		})
	}
	t.Run("selected_up_failure_is_fatal", func(t *testing.T) {
		s, b := newSession(true, true, 0)
		failure := &compute.BackendError{Backend: "test", Site: "wide up", Err: errors.New("selected failure")}
		b.failTileAt, b.tileFailure = 2, failure
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
		if out != nil || outcome != v41RoutedExpertError || !errors.Is(err, failure) || !s.BackendSessionClosed() || !reflect.DeepEqual(b.tileRows, []int{32, 32}) || len(b.downRows) != 0 || len(b.matmulInputs) != 0 || len(b.pending) != 0 || len(b.ownedTiles) != 0 || b.fences != 0 {
			t.Fatalf("selected wide failure replay/cleanup: %v", err)
		}
	})
	t.Run("earlier_down_verdict_survives_later_down_failure", func(t *testing.T) {
		s, b := newSession(true, true, 0)
		b.publishFaultAt = map[int]bool{1: true}
		b.failDownAt = 2
		b.downFailure = &compute.BackendError{Backend: "test", Site: "second down tile", Err: errors.New("later failure")}
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
		var numeric *V41ExpertOperationError
		if out != nil || outcome != v41RoutedExpertError || !errors.As(err, &numeric) || numeric.Stage != "down" || !s.BackendSessionClosed() || !reflect.DeepEqual(b.downRows, []int{8, 8}) || b.publishes != 8 || len(b.matmulInputs) != 0 || len(b.pending) != 0 || len(b.ownedTiles) != 0 || b.fences != 1 {
			t.Fatalf("wide status priority changed: %v", err)
		}
	})
}

// fak-test:runtime medium est=20s lane=default
func TestV41ExpertGateUpTailConsumer(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	inputs, weights := make([][]float32, 33), make([]float32, 33)
	for row := range inputs {
		inputs[row] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[row] {
			inputs[row][i] = float32((row*3+i)%13-6) / 16
		}
		weights[row] = float32(row+1) / 64
	}
	newSession := func(grid, down bool, decline int) (*Session, *v41ExpertGateUp32Witness) {
		b := &v41ExpertGateUp32Witness{
			v41ExpertDownTileWitness: &v41ExpertDownTileWitness{
				v41ExpertTileWitness: &v41ExpertTileWitness{
					v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true},
					tiles:                  true, ownedTiles: map[compute.Buffer]bool{},
				}, downTiles: down,
			}, grid: grid, declineGridAt: decline,
		}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	for _, count := range []int{2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 24, 31} {
		t.Run(fmt.Sprintf("rows%d", count), func(t *testing.T) {
			control, cb := newSession(false, true, 0)
			want, outcome, err := control.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:count], weights[:count])
			if outcome != v41RoutedExpertHandled || err != nil {
				t.Fatalf("control: %v/%v", outcome, err)
			}
			for _, tc := range []struct {
				name       string
				grid, down bool
				decline    int
			}{
				{"qualified", true, true, 0}, {"legacy", false, true, 0},
				{"gate_refuses", true, true, 1}, {"up_refuses", true, true, 2}, {"down_unavailable", true, false, 0},
			} {
				t.Run(tc.name, func(t *testing.T) {
					s, b := newSession(tc.grid, tc.down, tc.decline)
					got, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:count], weights[:count])
					if outcome != v41RoutedExpertHandled || err != nil || len(got) != count {
						t.Fatalf("%v/%v rows=%d", outcome, err, len(got))
					}
					for row := range got {
						if !v41SameBits(got[row], want[row]) {
							t.Fatalf("row%d differs", row)
						}
					}
					gridCount := count > 2 && count != 4 && count != 8
					wide := gridCount && tc.grid && tc.down && tc.decline == 0
					wantProbes := []int(nil)
					if gridCount && tc.down {
						wantProbes = []int{count}
						if tc.grid && tc.decline != 1 {
							wantProbes = append(wantProbes, count)
						}
					}
					if !reflect.DeepEqual(b.gridProbeRows, wantProbes) {
						t.Fatalf("grid probes=%v want=%v", b.gridProbeRows, wantProbes)
					}
					wantTiles, wantUploads, wantScalarUploads := cb.tileRows, cb.uploads, cb.scalarUploads
					if wide {
						wantTiles, wantUploads, wantScalarUploads = []int{count, count}, 1, 0
					}
					if !reflect.DeepEqual(b.tileRows, wantTiles) || b.uploads != wantUploads || b.scalarUploads != wantScalarUploads {
						t.Fatalf("tiles=%v uploads=%d scalarUploads=%d", b.tileRows, b.uploads, b.scalarUploads)
					}
					if tc.down {
						wantScalar := len(cb.matmulInputs)
						wantGathers := cb.gathers
						if wide && count%2 == 1 {
							wantScalar -= 2
							wantGathers += 2
						}
						if !reflect.DeepEqual(b.downRows, cb.downRows) || b.publishes != count || len(b.matmulInputs) != wantScalar || b.gathers != wantGathers {
							t.Fatalf("down/singleton schedule changed: down=%v scalar=%d gathers=%d", b.downRows, len(b.matmulInputs), b.gathers)
						}
					} else if len(b.downRows) != 0 || len(b.matmulInputs) != count+2*(count%2) || b.gridChecks != 0 {
						t.Fatal("missing Q3 capability changed legacy route")
					}
					if b.fences != 1 || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
						t.Fatal("tail ownership/fence leak")
					}
				})
			}
		})
	}
	t.Run("bounded_admission", func(t *testing.T) {
		s, b := newSession(true, true, 0)
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
		if out != nil || outcome != v41RoutedExpertDeclined || err != nil || len(b.tileRows) != 0 || b.uploads != 0 || b.gridChecks != 0 {
			t.Fatal("33-row cohort admitted or performed work")
		}
	})
	t.Run("partial_up_error_is_owned_and_fatal", func(t *testing.T) {
		s, b := newSession(true, true, 0)
		failure := &compute.BackendError{Backend: "test", Site: "tail up", Err: errors.New("partial selected failure")}
		b.partialTileFailure = failure
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:31], weights[:31])
		if out != nil || outcome != v41RoutedExpertError || !errors.Is(err, failure) || !s.BackendSessionClosed() || !reflect.DeepEqual(b.tileRows, []int{31, 31}) || len(b.panels) != 2 || len(b.downRows) != 0 || len(b.matmulInputs) != 0 || len(b.pending) != 0 || len(b.ownedTiles) != 0 || b.fences != 0 {
			t.Fatalf("partial error ownership/replay: %v", err)
		}
	})
	t.Run("earlier_down_verdict_survives_tail_failure", func(t *testing.T) {
		s, b := newSession(true, true, 0)
		b.publishFaultAt = map[int]bool{1: true}
		b.failDownAt = 4 // The four-row tail follows three unchanged eight-row tiles.
		b.downFailure = &compute.BackendError{Backend: "test", Site: "tail down", Err: errors.New("later failure")}
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:31], weights[:31])
		var numeric *V41ExpertOperationError
		if out != nil || outcome != v41RoutedExpertError || !errors.As(err, &numeric) || numeric.Stage != "down" || !s.BackendSessionClosed() || !reflect.DeepEqual(b.downRows, []int{8, 8, 8, 4}) || b.publishes != 24 || len(b.matmulInputs) != 0 || len(b.pending) != 0 || len(b.ownedTiles) != 0 || b.fences != 1 {
			t.Fatalf("tail status priority changed: %v", err)
		}
	})
}
