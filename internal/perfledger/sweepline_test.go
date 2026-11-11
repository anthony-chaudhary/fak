// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"math"
	"reflect"
	"testing"
)

// Expectations in this file are hand-derived from the upstream Python
// (aiperf src/aiperf/analysis/sweepline.py and sweepline_kv_cache.py), not
// from the Go port: _sweep_line_cumsum sorts events by timestamp with ends
// (delta <= 0) before starts (delta > 0) at ties, cumsums, and snaps
// |v| < 1e-9*max|v| to exactly 0.0; _step_lookup holds the last event's value
// and returns 0 before the first event. Timestamps are epoch nanoseconds;
// small integer-friendly values (1e9 = 1s) keep the arithmetic exact.

// sweepCheck asserts a (timestamps, values) curve: timestamps must match
// exactly, values within tol. Skips the value check when wantVals is nil.
func sweepCheck(t *testing.T, name string, gotTS, gotVals, wantTS, wantVals []float64, tol float64) {
	t.Helper()
	if len(gotTS) != len(wantTS) {
		t.Fatalf("%s: timestamp length got %d want %d\ngot ts  %v\nwant ts %v", name, len(gotTS), len(wantTS), gotTS, wantTS)
	}
	if len(gotVals) != len(wantTS) {
		t.Fatalf("%s: value length got %d want %d", name, len(gotVals), len(wantTS))
	}
	for i := range wantTS {
		if gotTS[i] != wantTS[i] {
			t.Errorf("%s: ts[%d] got %v want %v (exact)", name, i, gotTS[i], wantTS[i])
		}
		if wantVals != nil && math.Abs(gotVals[i]-wantVals[i]) > tol {
			t.Errorf("%s: val[%d] got %v want %v (tol %g)", name, i, gotVals[i], wantVals[i], tol)
		}
	}
}

// Semantics 1: at a shared timestamp an END event sorts BEFORE a START event,
// so concurrency momentarily dips to 0 at the handoff (Python lexsort on
// (event_type, timestamp), event_type 0=end). A zero-duration record
// (start == end) likewise emits its -1 before its +1, producing a transient -1.
func TestSweepLineConcurrencyEndsBeforeStartsAtTies(t *testing.T) {
	// A: [1e9, 3e9); B: [3e9, 5e9); C: zero-duration at 6e9.
	starts := []float64{1e9, 3e9, 6e9}
	ends := []float64{3e9, 5e9, 6e9}
	gotTS, gotVals := ConcurrencySweepLine(starts, ends)
	// Events sorted: +1@1e9, -1@3e9 (A end), +1@3e9 (B start), -1@5e9,
	// -1@6e9 (C end), +1@6e9 (C start).
	wantTS := []float64{1e9, 3e9, 3e9, 5e9, 6e9, 6e9}
	wantVals := []float64{1, 0, 1, 0, -1, 0}
	sweepCheck(t, "ConcurrencySweepLine tie order", gotTS, gotVals, wantTS, wantVals, 1e-7)
}

// Semantics 2+4: AddStepFunctions merges unique timestamps; each side holds
// its last value forward and contributes 0 before its first event, so a value
// present in only one input still appears on the union grid. Empty inputs
// return a copy of the other side (no aliasing).
func TestSweepLineAddStepFunctionsMergedGridAndEmptyInputs(t *testing.T) {
	aTS, aVals := []float64{1e9, 3e9}, []float64{10, 20}
	bTS, bVals := []float64{2e9}, []float64{5}
	gotTS, gotVals := AddStepFunctions(aTS, aVals, bTS, bVals)
	// a at merged: 10, 10 (held), 20; b at merged: 0 (before first), 5, 5 (held).
	wantTS := []float64{1e9, 2e9, 3e9}
	wantVals := []float64{10, 15, 25}
	sweepCheck(t, "AddStepFunctions union grid", gotTS, gotVals, wantTS, wantVals, 1e-7)

	// Empty a returns a copy of b: equal content, mutating the result must
	// not touch the caller's slice.
	cTS, cVals := AddStepFunctions(nil, nil, bTS, bVals)
	if !reflect.DeepEqual(cTS, bTS) || !reflect.DeepEqual(cVals, bVals) {
		t.Fatalf("AddStepFunctions empty a: got (%v,%v) want copy of (%v,%v)", cTS, cVals, bTS, bVals)
	}
	cVals[0] = 99
	if bVals[0] != 5 {
		t.Fatalf("AddStepFunctions empty a aliases caller memory: bVals mutated to %v", bVals)
	}
	// Empty b returns a copy of a; both empty returns empty.
	dTS, dVals := AddStepFunctions(aTS, aVals, nil, nil)
	if !reflect.DeepEqual(dTS, aTS) || !reflect.DeepEqual(dVals, aVals) {
		t.Fatalf("AddStepFunctions empty b: got (%v,%v) want copy of (%v,%v)", dTS, dVals, aTS, aVals)
	}
	eTS, eVals := AddStepFunctions(nil, nil, nil, nil)
	if len(eTS) != 0 || len(eVals) != 0 {
		t.Fatalf("AddStepFunctions both empty: got (%v,%v) want empty", eTS, eVals)
	}
}

// Semantics 3: cumsum values with |v| < 1e-9*max|v| snap to exactly 0.
// Weights 0.1/0.2/-0.3 make the running sum pass through ~2.8e-17 residues at
// the cancellation points; those must come back as exact 0.0, while the
// genuine -0.2/0.1 plateaus survive unsnapped.
func TestSweepLineWeightedConcurrencySnapsResidueToExactZero(t *testing.T) {
	// A: [1e9,4e9) w=0.1; B: [2e9,3e9) w=0.2; C: [2e9,3e9) w=-0.3.
	starts := []float64{1e9, 2e9, 2e9}
	ends := []float64{4e9, 3e9, 3e9}
	weights := []float64{0.1, 0.2, -0.3}
	gotTS, gotVals := WeightedConcurrencySweepLine(starts, ends, weights)
	// Sorted events: +0.1@1e9, -0.3@2e9 (end-type before start-type), +0.2@2e9,
	// -0.2@3e9, +0.3@3e9, -0.1@4e9. Cumsum: 0.1, -0.2, ~2.8e-17, -0.2, ~0.1, ~1e-17.
	wantTS := []float64{1e9, 2e9, 2e9, 3e9, 3e9, 4e9}
	wantVals := []float64{0.1, -0.2, 0, -0.2, 0.1, 0}
	sweepCheck(t, "WeightedConcurrency snap", gotTS, gotVals, wantTS, wantVals, 1e-7)
	// The snapped positions must be exactly zero, not merely tiny: maxAbs is
	// 0.2 so the snap threshold is 2e-10, far above the ~1e-17 residue.
	if gotVals[2] != 0 || gotVals[5] != 0 {
		t.Errorf("snap did not fire exactly: vals[2]=%v vals[5]=%v want exact 0", gotVals[2], gotVals[5])
	}
}

// Semantics 5: DivideStepFunctions is safe — denominator <= 0 or NaN yields 0
// (never Inf/NaN); either input empty yields empty.
func TestSweepLineDivideStepFunctionsSafeDivision(t *testing.T) {
	numTS, numVals := []float64{1e9, 2e9, 3e9}, []float64{10, 7, 0}
	denTS, denVals := []float64{1.5e9, 3e9}, []float64{2, 0}
	gotTS, gotVals := DivideStepFunctions(numTS, numVals, denTS, denVals)
	// num at merged [1,1.5,2,3]e9: 10, 10, 7, 0.
	// den at merged: 0 (before first), 2, 2, 0 -> quotients 0, 5, 3.5, 0.
	wantTS := []float64{1e9, 1.5e9, 2e9, 3e9}
	wantVals := []float64{0, 5, 3.5, 0}
	sweepCheck(t, "DivideStepFunctions safe division", gotTS, gotVals, wantTS, wantVals, 1e-7)

	// NaN and negative denominators both yield 0, not Inf.
	nTS, nVals := DivideStepFunctions([]float64{1e9}, []float64{9}, []float64{1e9, 2e9}, []float64{math.NaN(), -5})
	wantN := []float64{0, 0}
	sweepCheck(t, "DivideStepFunctions NaN/neg denominator", nTS, nVals, []float64{1e9, 2e9}, wantN, 1e-7)
	for i, v := range nVals {
		if math.IsInf(v, 0) || math.IsNaN(v) {
			t.Errorf("unsafe division leaked at [%d]: %v", i, v)
		}
	}

	// Empty numerator or denominator -> empty result.
	eTS, eVals := DivideStepFunctions(nil, nil, denTS, denVals)
	if len(eTS) != 0 || len(eVals) != 0 {
		t.Fatalf("DivideStepFunctions empty numerator: got (%v,%v) want empty", eTS, eVals)
	}
	eTS, eVals = DivideStepFunctions(numTS, numVals, nil, nil)
	if len(eTS) != 0 || len(eVals) != 0 {
		t.Fatalf("DivideStepFunctions empty denominator: got (%v,%v) want empty", eTS, eVals)
	}
}

// __CHUNK2__
