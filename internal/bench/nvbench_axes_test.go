package bench

import (
	"bytes"
	"strings"
	"testing"
)

// testSweep is the shared fixture: two axes, 2x3 = 6 points.
func testSweep() Sweep {
	s, err := NewSweep([]Axis{
		{Name: "m", Values: []string{"128", "256"}},
		{Name: "n", Values: []string{"64", "128", "256"}},
	})
	if err != nil {
		panic(err)
	}
	return s
}

// Subtest 1: axis/cartesian sweep enumeration — the sweep is the ordered
// cartesian product, last axis fastest, with stable point IDs.
func TestNVBenchAxes(t *testing.T) {
	t.Run("CartesianEnumeration", func(t *testing.T) {
		s := testSweep()
		if got, want := s.PointCount(), 6; got != want {
			t.Fatalf("PointCount = %d, want %d", got, want)
		}
		pts := s.Points()
		if len(pts) != 6 {
			t.Fatalf("Points() = %d points, want 6", len(pts))
		}
		wantIDs := []string{
			"128|64", "128|128", "128|256",
			"256|64", "256|128", "256|256",
		}
		for i, p := range pts {
			if p.ID != wantIDs[i] {
				t.Errorf("point %d ID = %q, want %q", i, p.ID, wantIDs[i])
			}
		}
		// PointAt agrees with the enumeration.
		p, err := s.PointAt(4)
		if err != nil {
			t.Fatalf("PointAt(4): %v", err)
		}
		if p.ID != "256|128" {
			t.Errorf("PointAt(4) = %q, want %q", p.ID, "256|128")
		}
		if _, err := s.PointAt(6); err == nil {
			t.Error("PointAt(6) should fail closed out of range")
		}
		// ParsePointID inverts the ID and fails closed on a foreign value.
		vals, err := ParsePointID("128|256", s.Axes)
		if err != nil {
			t.Fatalf("ParsePointID: %v", err)
		}
		if vals[0] != "128" || vals[1] != "256" {
			t.Errorf("ParsePointID = %v, want [128 256]", vals)
		}
		if _, err := ParsePointID("128|512", s.Axes); err == nil {
			t.Error("ParsePointID should refuse a value not on the axis")
		}
		if _, err := ParsePointID("128", s.Axes); err == nil {
			t.Error("ParsePointID should refuse a wrong arity")
		}
	})

	t.Run("WarmupStreamStates", func(t *testing.T) {
		plan := MeasurementPlan{Warmup: 2, Stream: 3}
		st, err := NewMeasurementState(plan)
		if err != nil {
			t.Fatalf("NewMeasurementState: %v", err)
		}
		if st.Iteration != 0 || st.Phase != PhaseWarmup {
			t.Fatalf("initial state = %+v, want iteration 0 in warmup", st)
		}
		wantPhases := []Phase{PhaseWarmup, PhaseWarmup, PhaseStream, PhaseStream, PhaseStream}
		for i, want := range wantPhases {
			var phase Phase
			st, phase, err = st.Advance()
			if err != nil {
				t.Fatalf("Advance %d: %v", i, err)
			}
			if phase != want {
				t.Errorf("iteration %d phase = %s, want %s", i, phase, want)
			}
			if st.Iteration != i+1 {
				t.Errorf("iteration %d state.Iteration = %d", i, st.Iteration)
			}
		}
		if !st.Done() {
			t.Error("state should be Done after the full plan")
		}
		if _, _, err := st.Advance(); err == nil {
			t.Error("Advance past an exhausted plan must fail closed")
		}

		// A zero-warmup plan starts in stream.
		zs, err := NewMeasurementState(MeasurementPlan{Warmup: 0, Stream: 1})
		if err != nil {
			t.Fatalf("zero-warmup plan: %v", err)
		}
		if zs.Phase != PhaseStream {
			t.Errorf("zero-warmup initial phase = %s, want stream", zs.Phase)
		}

		// Negative counts are refused.
		if _, err := NewMeasurementState(MeasurementPlan{Warmup: -1, Stream: 1}); err == nil {
			t.Error("negative warmup must be refused")
		}
		if _, err := NewMeasurementState(MeasurementPlan{Warmup: 1, Stream: -1}); err == nil {
			t.Error("negative stream must be refused")
		}
	})

	t.Run("CSVContract", func(t *testing.T) {
		s := testSweep()
		plan := MeasurementPlan{Warmup: 2, Stream: 3}
		// Deterministic fake measurement: warmup reads 0.5ms, stream reads 1ms.
		results, err := RunSweep(s, plan, func(p Point, iteration int, phase Phase) float64 {
			if phase == PhaseWarmup {
				return 0.5
			}
			return 1.0
		})
		if err != nil {
			t.Fatalf("RunSweep: %v", err)
		}
		if len(results) != 6 {
			t.Fatalf("RunSweep = %d results, want 6", len(results))
		}
		for i, r := range results {
			if r.PointID != s.Points()[i].ID {
				t.Errorf("result %d point id = %q, want %q", i, r.PointID, s.Points()[i].ID)
			}
			if r.WarmupMS != 1.0 { // 2 warmup x 0.5
				t.Errorf("result %d WarmupMS = %v, want 1.0", i, r.WarmupMS)
			}
			if r.StreamMS != 3.0 { // 3 stream x 1.0
				t.Errorf("result %d StreamMS = %v, want 3.0", i, r.StreamMS)
			}
			if r.StreamN != 3 {
				t.Errorf("result %d StreamN = %d, want 3", i, r.StreamN)
			}
		}

		var buf bytes.Buffer
		if err := s.WriteCSV(&buf, results); err != nil {
			t.Fatalf("WriteCSV: %v", err)
		}
		lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
		if len(lines) != 7 { // header + 6 points
			t.Fatalf("CSV = %d lines, want 7", len(lines))
		}
		wantHeader := "point_id,m,n,warmup_ms,stream_ms,stream_n"
		if lines[0] != wantHeader {
			t.Errorf("CSV header = %q, want %q", lines[0], wantHeader)
		}
		wantRow0 := "128|64,128,64,1,3,3"
		if lines[1] != wantRow0 {
			t.Errorf("CSV row 0 = %q, want %q", lines[1], wantRow0)
		}
		wantRow5 := "256|256,256,256,1,3,3"
		if lines[6] != wantRow5 {
			t.Errorf("CSV row 5 = %q, want %q", lines[6], wantRow5)
		}

		// The contract fails closed on a truncated or reordered result set.
		if err := s.WriteCSV(&bytes.Buffer{}, results[:5]); err == nil {
			t.Error("WriteCSV should refuse a truncated result set")
		}
		shuffled := append([]PointResult(nil), results...)
		shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
		if err := s.WriteCSV(&bytes.Buffer{}, shuffled); err == nil {
			t.Error("WriteCSV should refuse a reordered result set")
		}
	})

	t.Run("EmptyAxisSetRefused", func(t *testing.T) {
		if _, err := NewSweep(nil); err == nil {
			t.Error("NewSweep(nil) must fail closed on an empty axis set")
		}
		if _, err := NewSweep([]Axis{}); err == nil {
			t.Error("NewSweep([]) must fail closed on an empty axis set")
		}
		if _, err := NewSweep([]Axis{{Name: "m", Values: nil}}); err == nil {
			t.Error("NewSweep must refuse an axis with no values")
		}
		if _, err := NewSweep([]Axis{{Name: "", Values: []string{"1"}}}); err == nil {
			t.Error("NewSweep must refuse an unnamed axis")
		}
		if _, err := NewSweep([]Axis{{Name: "m", Values: []string{"1"}}, {Name: "m", Values: []string{"2"}}}); err == nil {
			t.Error("NewSweep must refuse a duplicate axis name")
		}
	})
}
