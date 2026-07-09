package dsparity

import (
	"encoding/json"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// TestRequiredFieldsLocked pins the row schema: a marshaled row must carry exactly
// the keys RequiredFields() declares — no more, no fewer. Adding or renaming a
// ParityRow field without updating RequiredFields fails here on purpose (this is the
// "expected FIELDS" half of the issue's acceptance criterion 1).
func TestRequiredFieldsLocked(t *testing.T) {
	b, err := json.Marshal(Rows()[0])
	if err != nil {
		t.Fatalf("marshal row: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal row: %v", err)
	}
	want := RequiredFields()
	if len(m) != len(want) {
		t.Fatalf("row has %d JSON keys, want %d (%v vs %v)", len(m), len(want), keysOf(m), want)
	}
	for _, k := range want {
		if _, ok := m[k]; !ok {
			t.Errorf("required field %q missing from marshaled row", k)
		}
	}
}

// TestEveryRowValidates asserts every row in the table satisfies the closed
// vocabularies and the consistency rules (bitwise<=>zero tol, bounded<=>positive
// tol, >=2 variants).
func TestEveryRowValidates(t *testing.T) {
	for _, r := range Rows() {
		if err := r.Validate(); err != nil {
			t.Errorf("invalid row: %v", err)
		}
	}
}

// TestNoDuplicateIDs — ids are the stable handle, so they must be unique.
func TestNoDuplicateIDs(t *testing.T) {
	seen := make(map[string]bool)
	for _, r := range Rows() {
		if seen[r.ID] {
			t.Errorf("duplicate row id %q", r.ID)
		}
		seen[r.ID] = true
	}
}

// TestAllRequiredAxesCovered — the issue enumerates four invariance axes; the table
// must cover all four (acceptance criterion 1: the rows define the parity axes).
func TestAllRequiredAxesCovered(t *testing.T) {
	covered := CoveredAxes()
	for _, a := range RequiredAxes() {
		if !covered[a] {
			t.Errorf("required invariance axis %q has no parity row", a)
		}
	}
}

// TestEveryAxisHasOfflineWitness — the first witness must need NO weights. Every
// axis therefore needs at least one offline-synthetic row, so `go test` alone can
// exercise each axis (acceptance criterion 2: offline-runnable).
func TestEveryAxisHasOfflineWitness(t *testing.T) {
	offlineByAxis := make(map[InvarianceAxis]bool)
	for _, r := range Rows() {
		if r.Witness == WitnessOfflineSynthetic {
			offlineByAxis[r.Axis] = true
		}
	}
	for _, a := range RequiredAxes() {
		if !offlineByAxis[a] {
			t.Errorf("axis %q has no offline-synthetic parity row; the first witness would need weights", a)
		}
	}
}

// TestBitwisePreferred — the harness prefers bitwise; a bounded row is the
// exception, so bitwise rows must be a strict majority of the table. This keeps the
// FP4/FP8 escape hatch from quietly becoming the default.
func TestBitwisePreferred(t *testing.T) {
	var bitwise, total int
	for _, r := range Rows() {
		total++
		if r.Tolerance == ToleranceBitwise {
			bitwise++
		}
	}
	if bitwise*2 <= total {
		t.Errorf("bitwise rows (%d) must be a strict majority of %d; bounded tolerance is the exception, not the default", bitwise, total)
	}
}

// TestToleranceConsistency — restates the invariant at table scope so a future edit
// that sets a nonzero tol on a bitwise row (or forgets the bound on an FP4/FP8 row)
// fails independently of Validate.
func TestToleranceConsistency(t *testing.T) {
	for _, r := range Rows() {
		switch r.Tolerance {
		case ToleranceBitwise:
			if r.MaxAbsTol != 0 || r.MaxRelTol != 0 {
				t.Errorf("row %q bitwise but has tolerance abs=%g rel=%g", r.ID, r.MaxAbsTol, r.MaxRelTol)
			}
		case ToleranceFP4Bounded, ToleranceFP8Bounded:
			if r.MaxAbsTol <= 0 && r.MaxRelTol <= 0 {
				t.Errorf("row %q is %s but documents no positive tolerance", r.ID, r.Tolerance)
			}
		default:
			t.Errorf("row %q has off-vocabulary tolerance %q", r.ID, r.Tolerance)
		}
	}
}

// TestSyntheticRequestOrderInvariance is the offline, weight-free EXECUTED witness
// for the request-order and seed axes: the top-k selector's total tie-break order
// makes its output invariant to the order candidates are supplied in. We select
// over synthetic scores, then again over a shuffled permutation of the same
// (position, score) pairs, and require an identical result — the concrete property
// the "expert routing under different request ordering" and "top-k under a fixed
// seed" rows assert, demonstrated with no model and no GPU.
func TestSyntheticRequestOrderInvariance(t *testing.T) {
	const n, k = 32, 8
	positions := make([]int, n)
	scores := make([]float64, n)
	// deterministic synthetic scores with intentional ties (i%5) to stress the tie-break.
	for i := 0; i < n; i++ {
		positions[i] = i
		scores[i] = float64(i%5) + 0.1*float64(i%3)
	}
	want := stableTopK(positions, scores, k)

	rng := rand.New(rand.NewSource(1)) // fixed seed => deterministic permutation
	for trial := 0; trial < 16; trial++ {
		perm := rng.Perm(n)
		pp := make([]int, n)
		ss := make([]float64, n)
		for dst, src := range perm {
			pp[dst] = positions[src]
			ss[dst] = scores[src]
		}
		got := stableTopK(pp, ss, k)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("top-k not order-invariant: trial %d got %v want %v", trial, got, want)
		}
	}
}

// TestSyntheticTopKIsCausalPrefixStable is a second offline witness for the seed
// axis: re-running the identical selection under the "same seed" yields the identical
// index set (idempotent), and the selected positions are exactly the k highest under
// the documented order — the bit-identical reproducibility the seed rows require.
func TestSyntheticTopKIsCausalPrefixStable(t *testing.T) {
	positions := []int{0, 1, 2, 3, 4, 5}
	scores := []float64{0.5, 0.9, 0.9, 0.1, 0.7, 0.7}
	a := stableTopK(positions, scores, 3)
	b := stableTopK(positions, scores, 3)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("same-seed selection not reproducible: %v vs %v", a, b)
	}
	// Expected under (score desc, then position asc): 0.9@1, 0.9@2, 0.7@4.
	want := []int{1, 2, 4}
	if !reflect.DeepEqual(a, want) {
		t.Fatalf("top-k tie-break wrong: got %v want %v", a, want)
	}
}

func keysOf(m map[string]json.RawMessage) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
