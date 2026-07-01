package issuecost

import (
	"strings"
	"testing"
)

// Percentile method under test: NEAREST-RANK (the C = 1 variant), 1-indexed,
// inherited from internal/fleetmetrics:
//
//	rank  = ceil( (p/100) * N )   clamped to [1, N]
//	value = sorted[rank-1]
//
// Every expected elapsed number below is hand-computed against that formula so
// the fixture PROVES the method (and its reuse), not just self-agreement.

// ledgerFixture is a deterministic per-issue ledger: 20 rows whose ELAPSED
// seconds are 10,20,...,200, offered UNSORTED (reversed by issue number) so the
// test also proves the fold sorts a copy before ranking. Attempts and outcomes
// are fixed so the attempts total and per-outcome counts are hand-verifiable:
//
//	attempts: each row's attempts = (i mod 3) + 1 for i=1..20
//	          sum over i=1..20 of ((i mod 3)+1) = 20 + sum(i mod 3)
//	          i mod 3 over 1..20: (1,2,0) x6 = 18 then i=19,20 -> 1,2
//	          sum(i mod 3) = 6*3 + 3 = 21
//	          => total attempts = 20 + 21 = 41
//	outcomes: i mod 5 -> 0..4; map {0,1,2}->shipped, 3->blocked, 4->abandoned
//	          over i=1..20 each residue class 0..4 appears exactly 4 times
//	          => shipped = 12 (classes 0,1,2), blocked = 4 (class 3),
//	             abandoned = 4 (class 4)
func ledgerFixture() []IssueCost {
	outcomeFor := func(i int) Outcome {
		switch i % 5 {
		case 3:
			return Blocked
		case 4:
			return Abandoned
		default:
			return Shipped
		}
	}
	rows := make([]IssueCost, 0, 20)
	for i := 20; i >= 1; i-- { // reverse issue order on purpose
		rows = append(rows, IssueCost{
			Issue:      i,
			ElapsedSec: float64(i * 10),
			Attempts:   (i % 3) + 1,
			Outcome:    outcomeFor(i),
		})
	}
	return rows
}

// TestMedianP95FixtureNearestRank is the load-bearing witness: over the
// 20-element ledger (sorted elapsed 10..200 step 10), nearest-rank gives
//
//	median: rank = ceil(0.50 * 20) = 10 -> sorted[ 9] = 100
//	p95:    rank = ceil(0.95 * 20) = 19 -> sorted[18] = 190
func TestMedianP95FixtureNearestRank(t *testing.T) {
	rows := ledgerFixture()
	if got := Median(rows); got != 100 {
		t.Errorf("Median: got %v, want 100 (nearest-rank rank 10 of 20)", got)
	}
	if got := P95(rows); got != 190 {
		t.Errorf("P95: got %v, want 190 (nearest-rank rank 19 of 20)", got)
	}
}

// TestSummaryFixture pins the whole Report over the fixture: median, p95, the
// hand-verified attempts total (40), and the per-outcome counts (12/4/4).
func TestSummaryFixture(t *testing.T) {
	rep := Summary(ledgerFixture())
	if rep.N != 20 {
		t.Errorf("N: got %d, want 20", rep.N)
	}
	if rep.MedianSec != 100 {
		t.Errorf("MedianSec: got %v, want 100", rep.MedianSec)
	}
	if rep.P95Sec != 190 {
		t.Errorf("P95Sec: got %v, want 190", rep.P95Sec)
	}
	if rep.TotalAttempts != 41 {
		t.Errorf("TotalAttempts: got %d, want 41", rep.TotalAttempts)
	}
	if got := rep.OutcomeCounts[Shipped]; got != 12 {
		t.Errorf("shipped: got %d, want 12", got)
	}
	if got := rep.OutcomeCounts[Blocked]; got != 4 {
		t.Errorf("blocked: got %d, want 4", got)
	}
	if got := rep.OutcomeCounts[Abandoned]; got != 4 {
		t.Errorf("abandoned: got %d, want 4", got)
	}
	// counts must sum back to N.
	if sum := rep.OutcomeCounts[Shipped] + rep.OutcomeCounts[Blocked] + rep.OutcomeCounts[Abandoned]; sum != rep.N {
		t.Errorf("outcome counts sum %d != N %d", sum, rep.N)
	}
}

// TestSmallOddFixture is a second hand-verified fixture with an odd count and
// non-uniform elapsed spacing, offered unsorted:
//
//	elapsed (unsorted): 5, 3, 9, 1, 7    sorted: 1,3,5,7,9   N=5
//	median: rank = ceil(0.50*5) = ceil(2.5) = 3 -> sorted[2] = 5
//	p95:    rank = ceil(0.95*5) = ceil(4.75)= 5 -> sorted[4] = 9
//	attempts total: 1+2+3+4+5 = 15
func TestSmallOddFixture(t *testing.T) {
	rows := []IssueCost{
		{Issue: 101, ElapsedSec: 5, Attempts: 1, Outcome: Shipped},
		{Issue: 102, ElapsedSec: 3, Attempts: 2, Outcome: Shipped},
		{Issue: 103, ElapsedSec: 9, Attempts: 3, Outcome: Blocked},
		{Issue: 104, ElapsedSec: 1, Attempts: 4, Outcome: Shipped},
		{Issue: 105, ElapsedSec: 7, Attempts: 5, Outcome: Abandoned},
	}
	rep := Summary(rows)
	if rep.MedianSec != 5 {
		t.Errorf("MedianSec: got %v, want 5", rep.MedianSec)
	}
	if rep.P95Sec != 9 {
		t.Errorf("P95Sec: got %v, want 9", rep.P95Sec)
	}
	if rep.TotalAttempts != 15 {
		t.Errorf("TotalAttempts: got %d, want 15", rep.TotalAttempts)
	}
	if rep.OutcomeCounts[Shipped] != 3 || rep.OutcomeCounts[Blocked] != 1 || rep.OutcomeCounts[Abandoned] != 1 {
		t.Errorf("outcome counts: got %v, want shipped=3 blocked=1 abandoned=1", rep.OutcomeCounts)
	}
}

// TestEmpty: an empty ledger folds to a zero Report with a non-nil (empty)
// OutcomeCounts, and the bare Median/P95 helpers return 0.
func TestEmpty(t *testing.T) {
	rep := Summary(nil)
	if rep.N != 0 || rep.MedianSec != 0 || rep.P95Sec != 0 || rep.TotalAttempts != 0 {
		t.Errorf("empty Summary: got %+v, want all zero", rep)
	}
	if rep.OutcomeCounts == nil {
		t.Error("empty Summary: OutcomeCounts is nil, want non-nil empty map")
	}
	if len(rep.OutcomeCounts) != 0 {
		t.Errorf("empty Summary: OutcomeCounts = %v, want empty", rep.OutcomeCounts)
	}
	if Median(nil) != 0 || P95(nil) != 0 {
		t.Errorf("empty Median/P95: got %v/%v, want 0/0", Median(nil), P95(nil))
	}
}

// TestSingle: a single row is the value for every percentile and its own
// attempts/outcome.
func TestSingle(t *testing.T) {
	rows := []IssueCost{{Issue: 7, ElapsedSec: 42.5, Attempts: 3, Outcome: Blocked}}
	rep := Summary(rows)
	if rep.MedianSec != 42.5 || rep.P95Sec != 42.5 {
		t.Errorf("single median/p95: got %v/%v, want 42.5/42.5", rep.MedianSec, rep.P95Sec)
	}
	if rep.TotalAttempts != 3 {
		t.Errorf("single attempts: got %d, want 3", rep.TotalAttempts)
	}
	if rep.OutcomeCounts[Blocked] != 1 {
		t.Errorf("single outcome: got %v, want blocked=1", rep.OutcomeCounts)
	}
	if Median(rows) != 42.5 || P95(rows) != 42.5 {
		t.Errorf("single Median/P95: got %v/%v, want 42.5/42.5", Median(rows), P95(rows))
	}
}

// TestRender surfaces median AND p95 plus attempts and per-outcome counts, and
// reports an empty ledger explicitly.
func TestRender(t *testing.T) {
	out := Summary(ledgerFixture()).Render()
	for _, want := range []string{
		"n=20", "median=100.0s", "p95=190.0s",
		"attempts=41", "shipped=12", "blocked=4", "abandoned=4",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("Render missing %q in %q", want, out)
		}
	}
	empty := Summary(nil).Render()
	if !strings.Contains(empty, "n=0") || !strings.Contains(empty, "no rows") {
		t.Errorf("Render empty ledger: %q", empty)
	}
}

// TestParseLedgerRoundTrip proves the JSONL persistence path: AppendRow then
// ParseLedger recovers the same rows a fixture folds, and Summary over the
// parsed rows matches Summary over the in-memory fixture.
func TestParseLedgerRoundTrip(t *testing.T) {
	rows := ledgerFixture()
	var buf []byte
	for _, r := range rows {
		var err error
		buf, err = AppendRow(buf, r)
		if err != nil {
			t.Fatalf("AppendRow issue %d: %v", r.Issue, err)
		}
	}
	parsed, err := ParseLedger(buf)
	if err != nil {
		t.Fatalf("ParseLedger: %v", err)
	}
	if len(parsed) != len(rows) {
		t.Fatalf("parsed %d rows, want %d", len(parsed), len(rows))
	}
	got := Summary(parsed)
	want := Summary(rows)
	// Report carries a map, so compare the scalar fields plus each outcome
	// count rather than using == on the struct.
	if got.N != want.N || got.MedianSec != want.MedianSec || got.P95Sec != want.P95Sec || got.TotalAttempts != want.TotalAttempts {
		t.Errorf("Summary(parsed) scalars = %+v, want %+v", got, want)
	}
	for _, o := range []Outcome{Shipped, Blocked, Abandoned} {
		if got.OutcomeCounts[o] != want.OutcomeCounts[o] {
			t.Errorf("Summary(parsed) %s = %d, want %d", o, got.OutcomeCounts[o], want.OutcomeCounts[o])
		}
	}
}

// TestParseLedgerSkipsBlankLines: blank lines between rows are ignored.
func TestParseLedgerSkipsBlankLines(t *testing.T) {
	data := []byte(`{"issue":1,"elapsed_sec":10,"attempts":1,"outcome":"shipped"}

{"issue":2,"elapsed_sec":20,"attempts":2,"outcome":"blocked"}
`)
	rows, err := ParseLedger(data)
	if err != nil {
		t.Fatalf("ParseLedger: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2 (blank line skipped)", len(rows))
	}
}

// TestParseLedgerRejectsBadOutcome: an unrecognized outcome is a located error,
// not a silently-folded row.
func TestParseLedgerRejectsBadOutcome(t *testing.T) {
	data := []byte(`{"issue":1,"elapsed_sec":10,"attempts":1,"outcome":"shipped"}
{"issue":2,"elapsed_sec":20,"attempts":1,"outcome":"exploded"}
`)
	if _, err := ParseLedger(data); err == nil {
		t.Fatal("ParseLedger accepted an unknown outcome, want error")
	} else if !strings.Contains(err.Error(), "line 2") {
		t.Errorf("error should name line 2: %v", err)
	}
}

// TestParseLedgerRejectsBadJSON: malformed JSON is a located error.
func TestParseLedgerRejectsBadJSON(t *testing.T) {
	data := []byte("{not json}\n")
	if _, err := ParseLedger(data); err == nil {
		t.Fatal("ParseLedger accepted malformed JSON, want error")
	}
}

// TestAppendRowRefusesBadOutcome: a corrupt outcome never reaches the ledger.
func TestAppendRowRefusesBadOutcome(t *testing.T) {
	if _, err := AppendRow(nil, IssueCost{Issue: 1, Outcome: Outcome("nope")}); err == nil {
		t.Fatal("AppendRow accepted an unknown outcome, want error")
	}
}

// TestSortedByIssue sorts a copy ascending by issue and leaves the caller slice
// untouched (the fixture is reverse-ordered, so this is a real reordering).
func TestSortedByIssue(t *testing.T) {
	rows := ledgerFixture()
	first := rows[0].Issue // fixture starts at issue 20
	sorted := SortedByIssue(rows)
	for i := 1; i < len(sorted); i++ {
		if sorted[i-1].Issue > sorted[i].Issue {
			t.Fatalf("not ascending at %d: %d > %d", i, sorted[i-1].Issue, sorted[i].Issue)
		}
	}
	if rows[0].Issue != first {
		t.Errorf("SortedByIssue mutated caller slice: rows[0].Issue = %d, want %d", rows[0].Issue, first)
	}
}
