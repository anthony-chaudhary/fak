package fleetmetrics

import (
	"strings"
	"testing"
)

// Percentile method under test: NEAREST-RANK (the C = 1 variant), 1-indexed:
//
//	rank  = ceil( (p/100) * N )   clamped to [1, N]
//	value = sorted[rank-1]
//
// Every expected number below is hand-computed against that formula so the
// fixture PROVES the method, not just self-agreement with the code.

// ledgerFixture is a deterministic worker-session ledger: 20 sessions whose
// durations are 10,20,...,200 seconds, offered to the code UNSORTED (reversed)
// so the test also proves Percentiles sorts a copy before ranking.
func ledgerFixture() []Session {
	// durations 200,190,...,10 — reverse order on purpose.
	ss := make([]Session, 0, 20)
	for i := 20; i >= 1; i-- {
		ss = append(ss, Session{
			WorkerID:    "w" + string(rune('A'+(i%5))),
			DurationSec: float64(i * 10),
		})
	}
	return ss
}

// TestP95FixtureNearestRank is the load-bearing witness: over the 20-element
// ledger (sorted values 10..200 step 10), nearest-rank gives
//
//	p95: rank = ceil(0.95 * 20) = ceil(19)   = 19 -> sorted[18] = 190
//	p50: rank = ceil(0.50 * 20) = ceil(10)   = 10 -> sorted[ 9] = 100
func TestP95FixtureNearestRank(t *testing.T) {
	sessions := ledgerFixture()
	p50, p95 := P50P95(sessions)

	if p50 != 100 {
		t.Errorf("p50: got %v, want 100 (nearest-rank rank 10 of 20)", p50)
	}
	if p95 != 190 {
		t.Errorf("p95: got %v, want 190 (nearest-rank rank 19 of 20)", p95)
	}
}

// TestPercentilesExactRanks pins several ranks over the same fixture, including
// the clamped endpoints p0 (rank clamps to 1 -> the min) and p100 (rank = N ->
// the max).
func TestPercentilesExactRanks(t *testing.T) {
	ds := durations(ledgerFixture())
	m := Percentiles(ds, 0, 25, 50, 90, 95, 99, 100)

	// rank = ceil((p/100)*20), clamped to [1,20]; value = sorted[rank-1].
	// p0:   ceil(0)     = 0 -> clamp 1  -> sorted[0]  = 10
	// p25:  ceil(5)     = 5             -> sorted[4]  = 50
	// p50:  ceil(10)    = 10           -> sorted[9]  = 100
	// p90:  ceil(18)    = 18           -> sorted[17] = 180
	// p95:  ceil(19)    = 19           -> sorted[18] = 190
	// p99:  ceil(19.8)  = 20           -> sorted[19] = 200
	// p100: ceil(20)    = 20           -> sorted[19] = 200
	want := map[float64]float64{
		0:   10,
		25:  50,
		50:  100,
		90:  180,
		95:  190,
		99:  200,
		100: 200,
	}
	for p, w := range want {
		if got := m[p]; got != w {
			t.Errorf("p%v: got %v, want %v", p, got, w)
		}
	}
}

// TestSmallOddFixture is a second hand-verified fixture with an odd count and
// non-uniform spacing, offered unsorted.
//
//	durations (unsorted): 5, 3, 9, 1, 7    sorted: 1,3,5,7,9   N=5
//	p50: rank = ceil(0.50*5) = ceil(2.5) = 3 -> sorted[2] = 5
//	p95: rank = ceil(0.95*5) = ceil(4.75)= 5 -> sorted[4] = 9
func TestSmallOddFixture(t *testing.T) {
	sessions := []Session{
		{WorkerID: "w1", DurationSec: 5},
		{WorkerID: "w2", DurationSec: 3},
		{WorkerID: "w3", DurationSec: 9},
		{WorkerID: "w4", DurationSec: 1},
		{WorkerID: "w5", DurationSec: 7},
	}
	p50, p95 := P50P95(sessions)
	if p50 != 5 {
		t.Errorf("p50: got %v, want 5", p50)
	}
	if p95 != 9 {
		t.Errorf("p95: got %v, want 9", p95)
	}
}

// TestEmpty: empty input -> every percentile is 0, and both convenience values
// are 0.
func TestEmpty(t *testing.T) {
	m := Percentiles(nil, 50, 95)
	if m[50] != 0 || m[95] != 0 {
		t.Errorf("empty Percentiles: got p50=%v p95=%v, want 0/0", m[50], m[95])
	}
	p50, p95 := P50P95(nil)
	if p50 != 0 || p95 != 0 {
		t.Errorf("empty P50P95: got %v/%v, want 0/0", p50, p95)
	}
}

// TestSingle: a single element is the value for every percentile.
func TestSingle(t *testing.T) {
	sessions := []Session{{WorkerID: "w1", DurationSec: 42.5}}
	p50, p95 := P50P95(sessions)
	if p50 != 42.5 || p95 != 42.5 {
		t.Errorf("single P50P95: got %v/%v, want 42.5/42.5", p50, p95)
	}
	m := Percentiles([]float64{42.5}, 0, 50, 95, 100)
	for _, p := range []float64{0, 50, 95, 100} {
		if m[p] != 42.5 {
			t.Errorf("single Percentiles p%v: got %v, want 42.5", p, m[p])
		}
	}
}

// TestPercentilesClamp: percentiles outside [0,100] are clamped, not indexed
// out of bounds.
func TestPercentilesClamp(t *testing.T) {
	ds := durations(ledgerFixture())
	m := Percentiles(ds, -10, 150)
	if m[-10] != 10 { // clamps to p0 -> min
		t.Errorf("p-10: got %v, want 10 (clamped to min)", m[-10])
	}
	if m[150] != 200 { // clamps to p100 -> max
		t.Errorf("p150: got %v, want 200 (clamped to max)", m[150])
	}
}

// TestPercentilesDoesNotMutate: the caller's slice order is preserved.
func TestPercentilesDoesNotMutate(t *testing.T) {
	ds := []float64{5, 3, 9, 1, 7}
	orig := append([]float64(nil), ds...)
	_ = Percentiles(ds, 50, 95)
	for i := range ds {
		if ds[i] != orig[i] {
			t.Fatalf("Percentiles mutated caller slice at %d: got %v, want %v", i, ds[i], orig[i])
		}
	}
}

// TestRender surfaces p50 AND p95 over a window, and reports an empty fleet
// explicitly.
func TestRender(t *testing.T) {
	out := Render("last 20 sessions", ledgerFixture())
	if !strings.Contains(out, "p50=100.0s") {
		t.Errorf("Render missing p50: %q", out)
	}
	if !strings.Contains(out, "p95=190.0s") {
		t.Errorf("Render missing p95: %q", out)
	}
	if !strings.Contains(out, "n=20") {
		t.Errorf("Render missing n: %q", out)
	}

	empty := Render("24h", nil)
	if !strings.Contains(empty, "n=0") || !strings.Contains(empty, "no sessions") {
		t.Errorf("Render empty fleet: %q", empty)
	}
}
