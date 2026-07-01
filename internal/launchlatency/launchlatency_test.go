package launchlatency

import (
	"strings"
	"testing"
)

// Percentile method under test (inherited from internal/fleetmetrics):
// NEAREST-RANK (the C = 1 variant), 1-indexed:
//
//	rank  = ceil( (p/100) * N )   clamped to [1, N]
//	value = sorted[rank-1]
//
// Histogram buckets are half-open, lower-closed: edges e0<e1<...<e_{k-1} yield
//
//	[0,e0) [e0,e1) ... [e_{k-2},e_{k-1}) [e_{k-1},+Inf)
//
// Every expected number below is hand-computed against those two rules so the
// fixture PROVES the fold, not just self-agreement with the code.

// fixtureBuckets are the operator-facing edges used by the ledger fixture.
func fixtureBuckets() []float64 { return []float64{1, 2, 5, 10, 30} }

// ledgerFixture is a deterministic worker-launch ledger of 8 records with
// hand-chosen dispatch/heartbeat pairs. The clamped latencies (heartbeat -
// dispatch, negatives clamped to 0) are, in ledger order:
//
//	w0: 100.0 -> 100.5 =  0.5
//	w1: 200.0 -> 201.5 =  1.5
//	w2: 300.0 -> 303.0 =  3.0
//	w3: 300.0 -> 303.0 =  3.0
//	w4: 400.0 -> 404.0 =  4.0
//	w5: 500.0 -> 507.0 =  7.0
//	w6: 600.0 -> 625.0 = 25.0
//	w7: 700.0 -> 740.0 = 40.0
//
// sorted latencies: 0.5, 1.5, 3, 3, 4, 7, 25, 40   (N=8)
func ledgerFixture() []Launch {
	return []Launch{
		{WorkerID: "w0", DispatchSec: 100, HeartbeatSec: 100.5},
		{WorkerID: "w1", DispatchSec: 200, HeartbeatSec: 201.5},
		{WorkerID: "w2", DispatchSec: 300, HeartbeatSec: 303},
		{WorkerID: "w3", DispatchSec: 300, HeartbeatSec: 303},
		{WorkerID: "w4", DispatchSec: 400, HeartbeatSec: 404},
		{WorkerID: "w5", DispatchSec: 500, HeartbeatSec: 507},
		{WorkerID: "w6", DispatchSec: 600, HeartbeatSec: 625},
		{WorkerID: "w7", DispatchSec: 700, HeartbeatSec: 740},
	}
}

// TestLatencySecClampsNegative pins the per-record latency, including the clamp.
func TestLatencySecClampsNegative(t *testing.T) {
	if got := (Launch{DispatchSec: 100, HeartbeatSec: 103}).LatencySec(); got != 3 {
		t.Errorf("positive latency: got %v, want 3", got)
	}
	skew := Launch{DispatchSec: 100, HeartbeatSec: 98}
	if got := skew.LatencySec(); got != 0 {
		t.Errorf("negative latency should clamp to 0, got %v", got)
	}
	if !skew.IsNegative() {
		t.Errorf("IsNegative should be true for heartbeat before dispatch")
	}
	if (Launch{DispatchSec: 100, HeartbeatSec: 100}).IsNegative() {
		t.Errorf("equal stamps must not count as negative")
	}
}

// TestHistogramFixtureBuckets is the load-bearing witness for the bucket fold.
// With edges [1,2,5,10,30] the six buckets and their hand-counted membership:
//
//	[0,1):   0.5                 -> 1
//	[1,2):   1.5                 -> 1
//	[2,5):   3, 3, 4             -> 3
//	[5,10):  7                   -> 1
//	[10,30): 25                  -> 1
//	[30,+Inf): 40                -> 1
func TestHistogramFixtureBuckets(t *testing.T) {
	got := Histogram(ledgerFixture(), fixtureBuckets())

	want := []BucketCount{
		{Lo: 0, Hi: 1, Count: 1},
		{Lo: 1, Hi: 2, Count: 1},
		{Lo: 2, Hi: 5, Count: 3},
		{Lo: 5, Hi: 10, Count: 1},
		{Lo: 10, Hi: 30, Count: 1},
		{Lo: 30, Unbounded: true, Count: 1},
	}
	if len(got) != len(want) {
		t.Fatalf("bucket count: got %d buckets, want %d", len(got), len(want))
	}
	for i := range want {
		w, g := want[i], got[i]
		if g.Lo != w.Lo || g.Unbounded != w.Unbounded || g.Count != w.Count ||
			(!w.Unbounded && g.Hi != w.Hi) {
			t.Errorf("bucket %d: got %+v, want %+v", i, g, w)
		}
	}

	// Total must conserve every record.
	total := 0
	for _, b := range got {
		total += b.Count
	}
	if total != len(ledgerFixture()) {
		t.Errorf("bucket counts sum to %d, want %d (every record folded exactly once)", total, len(ledgerFixture()))
	}
}

// TestHistogramEdgeIsLowerClosed proves a latency exactly on an edge lands in the
// bucket that STARTS at the edge. A single launch of latency 5.0 over edges
// [1,2,5,10,30] must land in [5,10), not [2,5).
func TestHistogramEdgeIsLowerClosed(t *testing.T) {
	got := Histogram([]Launch{{DispatchSec: 0, HeartbeatSec: 5}}, fixtureBuckets())
	// buckets: [0,1)[1,2)[2,5)[5,10)[10,30)[30,+Inf)  -> index 3 is [5,10)
	for i, b := range got {
		wantCount := 0
		if i == 3 {
			wantCount = 1
		}
		if b.Count != wantCount {
			t.Errorf("edge=5 launch: bucket %d %+v got count %d, want %d", i, b, b.Count, wantCount)
		}
	}
}

// TestHistogramUnsortedEdges proves the edges are sorted on a copy: passing
// [30,5,1,10,2] yields the same buckets as the sorted fixture, and the caller's
// slice is left untouched.
func TestHistogramUnsortedEdges(t *testing.T) {
	edges := []float64{30, 5, 1, 10, 2}
	orig := append([]float64(nil), edges...)
	got := Histogram(ledgerFixture(), edges)

	wantCounts := []int{1, 1, 3, 1, 1, 1}
	if len(got) != len(wantCounts) {
		t.Fatalf("got %d buckets, want %d", len(got), len(wantCounts))
	}
	for i, wc := range wantCounts {
		if got[i].Count != wc {
			t.Errorf("bucket %d count: got %d, want %d", i, got[i].Count, wc)
		}
	}
	for i := range edges {
		if edges[i] != orig[i] {
			t.Fatalf("Histogram mutated caller edges at %d: got %v, want %v", i, edges[i], orig[i])
		}
	}
}

// TestHistogramEmptyEdges: no edges -> a single [0,+Inf) bucket holding all.
func TestHistogramEmptyEdges(t *testing.T) {
	got := Histogram(ledgerFixture(), nil)
	if len(got) != 1 {
		t.Fatalf("empty edges: got %d buckets, want 1", len(got))
	}
	if !got[0].Unbounded || got[0].Lo != 0 || got[0].Count != 8 {
		t.Errorf("empty edges bucket: got %+v, want [0,+Inf) count 8", got[0])
	}
}

// TestP50P95Fixture is the load-bearing percentile witness. Over sorted
// latencies 0.5,1.5,3,3,4,7,25,40 (N=8):
//
//	p50: rank = ceil(0.50*8) = ceil(4.0) = 4 -> sorted[3] = 3
//	p95: rank = ceil(0.95*8) = ceil(7.6) = 8 -> sorted[7] = 40
func TestP50P95Fixture(t *testing.T) {
	p50, p95 := P50P95(ledgerFixture())
	if p50 != 3 {
		t.Errorf("p50: got %v, want 3 (nearest-rank rank 4 of 8)", p50)
	}
	if p95 != 40 {
		t.Errorf("p95: got %v, want 40 (nearest-rank rank 8 of 8)", p95)
	}
}

// TestNegativeLatencyFolds proves a heartbeat-before-dispatch record clamps to 0
// (folds into the first bucket) AND is counted by Negatives so the skew stays
// visible. Fixture: one skewed record (98 before 100) plus one clean 3s launch.
//
//	clamped latencies: 0, 3   sorted: 0,3   N=2
//	p50: rank = ceil(0.50*2) = 1 -> sorted[0] = 0
//	p95: rank = ceil(0.95*2) = ceil(1.9) = 2 -> sorted[1] = 3
func TestNegativeLatencyFolds(t *testing.T) {
	ls := []Launch{
		{WorkerID: "skew", DispatchSec: 100, HeartbeatSec: 98},
		{WorkerID: "clean", DispatchSec: 200, HeartbeatSec: 203},
	}
	if n := Negatives(ls); n != 1 {
		t.Errorf("Negatives: got %d, want 1", n)
	}
	hist := Histogram(ls, fixtureBuckets())
	if hist[0].Count != 1 { // clamped 0 lands in [0,1)
		t.Errorf("clamped negative should land in first bucket, got %+v", hist[0])
	}
	if hist[2].Count != 1 { // the clean 3s lands in [2,5)
		t.Errorf("clean 3s should land in [2,5), got %+v", hist[2])
	}
	p50, p95 := P50P95(ls)
	if p50 != 0 || p95 != 3 {
		t.Errorf("skew p50/p95: got %v/%v, want 0/3", p50, p95)
	}
}

// TestEmpty: empty input -> zero percentiles, all-zero bucket counts, no negatives.
func TestEmpty(t *testing.T) {
	p50, p95 := P50P95(nil)
	if p50 != 0 || p95 != 0 {
		t.Errorf("empty P50P95: got %v/%v, want 0/0", p50, p95)
	}
	if n := Negatives(nil); n != 0 {
		t.Errorf("empty Negatives: got %d, want 0", n)
	}
	for i, b := range Histogram(nil, fixtureBuckets()) {
		if b.Count != 0 {
			t.Errorf("empty Histogram bucket %d: got count %d, want 0", i, b.Count)
		}
	}
}

// TestSingle: one record is that record's latency for every percentile and folds
// into exactly one bucket.
func TestSingle(t *testing.T) {
	one := []Launch{{WorkerID: "w1", DispatchSec: 10, HeartbeatSec: 16.5}} // latency 6.5
	p50, p95 := P50P95(one)
	if p50 != 6.5 || p95 != 6.5 {
		t.Errorf("single P50P95: got %v/%v, want 6.5/6.5", p50, p95)
	}
	hist := Histogram(one, fixtureBuckets())
	total := 0
	for _, b := range hist {
		total += b.Count
	}
	if total != 1 {
		t.Errorf("single Histogram total: got %d, want 1", total)
	}
	if hist[3].Count != 1 { // 6.5 lands in [5,10)
		t.Errorf("6.5s launch should land in [5,10), got %+v", hist[3])
	}
}

// TestRender surfaces p50 AND p95 and one histogram line per bucket, and reports
// an empty fleet explicitly.
func TestRender(t *testing.T) {
	out := Render(ledgerFixture(), fixtureBuckets())
	if !strings.Contains(out, "p50=3.0s") {
		t.Errorf("Render missing p50: %q", out)
	}
	if !strings.Contains(out, "p95=40.0s") {
		t.Errorf("Render missing p95: %q", out)
	}
	if !strings.Contains(out, "n=8") {
		t.Errorf("Render missing n: %q", out)
	}
	if !strings.Contains(out, "[30.0,+Inf)s 1") {
		t.Errorf("Render missing open-topped bucket: %q", out)
	}

	empty := Render(nil, fixtureBuckets())
	if !strings.Contains(empty, "n=0") || !strings.Contains(empty, "no launches") {
		t.Errorf("Render empty fleet: %q", empty)
	}
}

// TestRenderNegativeNote surfaces the clock-skew count in the header when present.
func TestRenderNegativeNote(t *testing.T) {
	ls := []Launch{{WorkerID: "skew", DispatchSec: 100, HeartbeatSec: 90}}
	out := Render(ls, fixtureBuckets())
	if !strings.Contains(out, "negative-latency records clamped to 0: 1") {
		t.Errorf("Render should note the skewed record: %q", out)
	}
}
