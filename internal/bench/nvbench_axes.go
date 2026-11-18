// Package bench — nvbench_axes.go
//
// INSPIRE-ONLY port of the NVIDIA/nvbench benchmark-harness shape (Apache-2.0,
// pinned at NVIDIA/nvbench@41fc6a46358c4d2b202b7e714da7333d3b4cc498) into the
// native kernel bench runner. Borrowed: the axis/cartesian sweep model (a
// benchmark is a named set of axes, each a list of values; the sweep is the
// ordered cartesian product), the warmup/stream measurement state machine
// (warmup iterations are discarded, stream iterations are timed), and the CSV
// output contract (one row per sweep point, axis columns first, then the
// measurement columns). Harness shape only: no CUDA runtime port, no kernel,
// no kernel-number claim, no GPU, no network.
package bench

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// NVBenchSweepSchema is the stable schema tag stamped on sweep receipts.
const NVBenchSweepSchema = "fak-nvbench-sweep/1"

// NVBenchCSVSchema is the stable schema tag for the CSV output contract.
const NVBenchCSVSchema = "fak-nvbench-csv/1"

// Axis is one named sweep dimension: a list of values the runner visits in
// declaration order. Mirrors nvbench's axis<T> (name + value list).
type Axis struct {
	Name   string
	Values []string
}

// Sweep is a validated, ordered cartesian product of axes.
type Sweep struct {
	Schema string
	Axes   []Axis
}

// NewSweep validates the axis set and builds the sweep. It refuses an empty
// axis set, an unnamed axis, a duplicate axis name, and an axis with no
// values — a sweep with no coordinates is a misconfiguration, not a no-op.
func NewSweep(axes []Axis) (Sweep, error) {
	if len(axes) == 0 {
		return Sweep{}, errors.New("nvbench sweep: empty axis set")
	}
	seen := make(map[string]bool, len(axes))
	for i, a := range axes {
		if a.Name == "" {
			return Sweep{}, fmt.Errorf("nvbench sweep: axis %d has an empty name", i)
		}
		if seen[a.Name] {
			return Sweep{}, fmt.Errorf("nvbench sweep: duplicate axis name %q", a.Name)
		}
		seen[a.Name] = true
		if len(a.Values) == 0 {
			return Sweep{}, fmt.Errorf("nvbench sweep: axis %q has no values", a.Name)
		}
	}
	return Sweep{Schema: NVBenchSweepSchema, Axes: axes}, nil
}

// PointCount is the size of the cartesian product.
func (s Sweep) PointCount() int {
	n := 1
	for _, a := range s.Axes {
		n *= len(a.Values)
	}
	return n
}

// Point is one coordinate of the sweep: its stable ID and the per-axis values.
type Point struct {
	ID     string
	Values []string
}

// PointID joins the per-axis values with '|' in axis order — the stable
// coordinate key used by the CSV contract and by result matching.
func PointID(values []string) string {
	return strings.Join(values, "|")
}

// ParsePointID inverts PointID. It fails closed on a wrong arity or a value
// that itself contains the separator.
func ParsePointID(id string, axes []Axis) ([]string, error) {
	parts := strings.Split(id, "|")
	if len(parts) != len(axes) {
		return nil, fmt.Errorf("nvbench sweep: point id %q has %d fields, want %d", id, len(parts), len(axes))
	}
	for i, p := range parts {
		found := false
		for _, v := range axes[i].Values {
			if v == p {
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("nvbench sweep: point id %q field %d (%q) is not a value of axis %q", id, i, p, axes[i].Name)
		}
	}
	return parts, nil
}

// Points enumerates the cartesian product in lexicographic axis order: the
// last axis varies fastest, matching nvbench's nested-loop enumeration.
func (s Sweep) Points() []Point {
	n := s.PointCount()
	pts := make([]Point, 0, n)
	idx := make([]int, len(s.Axes))
	for i := 0; i < n; i++ {
		vals := make([]string, len(s.Axes))
		for a := range s.Axes {
			vals[a] = s.Axes[a].Values[idx[a]]
		}
		pts = append(pts, Point{ID: PointID(vals), Values: vals})
		for a := len(s.Axes) - 1; a >= 0; a-- {
			idx[a]++
			if idx[a] < len(s.Axes[a].Values) {
				break
			}
			idx[a] = 0
		}
	}
	return pts
}

// PointAt returns the i-th sweep point in enumeration order.
func (s Sweep) PointAt(i int) (Point, error) {
	if i < 0 || i >= s.PointCount() {
		return Point{}, fmt.Errorf("nvbench sweep: point index %d out of range [0,%d)", i, s.PointCount())
	}
	return s.Points()[i], nil
}

// MeasurementPlan is the warmup/stream split for one sweep point: run
// Warmup untimed iterations to warm caches and clocks, then run Stream
// timed iterations. Mirrors nvbench's warmup_count / stream_count.
type MeasurementPlan struct {
	Warmup int
	Stream int
}

// Phase is the measurement state of the runner at one iteration.
type Phase int

const (
	PhaseWarmup Phase = iota
	PhaseStream
)

func (p Phase) String() string {
	if p == PhaseWarmup {
		return "warmup"
	}
	return "stream"
}

// MeasurementState is the per-point state machine the runner walks:
// Warmup iterations first (discarded), then Stream iterations (timed).
type MeasurementState struct {
	Plan      MeasurementPlan
	Iteration int
	Phase     Phase
}

// NewMeasurementState builds the state machine at iteration 0. It refuses a
// plan with a negative count: a measurement plan must be finite and non-negative.
func NewMeasurementState(plan MeasurementPlan) (MeasurementState, error) {
	if plan.Warmup < 0 || plan.Stream < 0 {
		return MeasurementState{}, errors.New("nvbench measurement: warmup and stream counts must be non-negative")
	}
	phase := PhaseWarmup
	if plan.Warmup == 0 {
		phase = PhaseStream
	}
	return MeasurementState{Plan: plan, Iteration: 0, Phase: phase}, nil
}

// Done reports whether every planned iteration has been consumed.
func (m MeasurementState) Done() bool {
	return m.Iteration >= m.Plan.Warmup+m.Plan.Stream
}

// Advance moves the state machine one iteration and reports the phase the
// caller should run in. It fails closed once the plan is exhausted: a runner
// that keeps sampling past the plan is a bug, not a longer measurement.
func (m MeasurementState) Advance() (MeasurementState, Phase, error) {
	if m.Done() {
		return m, m.Phase, errors.New("nvbench measurement: plan exhausted")
	}
	next := m
	next.Iteration++
	if next.Iteration > m.Plan.Warmup {
		next.Phase = PhaseStream
	}
	return next, next.Phase, nil
}

// PointResult is the measured record for one sweep point.
type PointResult struct {
	PointID  string
	WarmupMS float64
	StreamMS float64
	StreamN  int
}

// CSVColumns is the CSV output contract: the point-id column, then one column
// per axis in declaration order, then the measurement columns. Stable across
// runs for a stable axis set.
func (s Sweep) CSVColumns() []string {
	cols := []string{"point_id"}
	for _, a := range s.Axes {
		cols = append(cols, a.Name)
	}
	cols = append(cols, "warmup_ms", "stream_ms", "stream_n")
	return cols
}

// WriteCSV writes the sweep results in the CSV output contract: a header row
// (CSVColumns) then one row per sweep point in enumeration order. It fails
// closed if the results are not exactly one complete, in-order set — a
// truncated or reordered receipt is a misconfiguration, not a shorter file.
func (s Sweep) WriteCSV(w io.Writer, results []PointResult) error {
	pts := s.Points()
	if len(results) != len(pts) {
		return fmt.Errorf("nvbench csv: %d results for %d sweep points", len(results), len(pts))
	}
	cw := csv.NewWriter(w)
	if err := cw.Write(s.CSVColumns()); err != nil {
		return err
	}
	for i, r := range results {
		if r.PointID != pts[i].ID {
			return fmt.Errorf("nvbench csv: result %d has point id %q, want %q (results must follow sweep order)", i, r.PointID, pts[i].ID)
		}
		row := append([]string{r.PointID}, pts[i].Values...)
		row = append(row, csvFloat(r.WarmupMS), csvFloat(r.StreamMS), strconv.Itoa(r.StreamN))
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}

// csvFloat renders a measurement with a stable, compact format ('g', shortest
// round-trip) so the CSV contract is byte-stable for a given result set.
func csvFloat(f float64) string {
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// RunSweep drives the measurement state machine over every sweep point with
// the caller's per-iteration function and collects one PointResult per point.
// The fn receives the point, the iteration index, and the phase; it returns
// the iteration's measured milliseconds (0 is a valid warmup reading).
func RunSweep(s Sweep, plan MeasurementPlan, fn func(p Point, iteration int, phase Phase) float64) ([]PointResult, error) {
	if _, err := NewMeasurementState(plan); err != nil {
		return nil, err
	}
	results := make([]PointResult, 0, s.PointCount())
	for _, p := range s.Points() {
		st, err := NewMeasurementState(plan)
		if err != nil {
			return nil, err
		}
		var res PointResult
		for !st.Done() {
			var phase Phase
			var err error
			st, phase, err = st.Advance()
			if err != nil {
				return nil, err
			}
			ms := fn(p, st.Iteration-1, phase)
			if phase == PhaseWarmup {
				res.WarmupMS += ms
			} else {
				res.StreamMS += ms
				res.StreamN++
			}
		}
		res.PointID = p.ID
		results = append(results, res)
	}
	return results, nil
}
