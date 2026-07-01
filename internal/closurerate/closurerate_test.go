package closurerate

import (
	"math"
	"strings"
	"testing"
)

// fixtureLedger is a fixed, hand-verified ledger.
//
// 10 records total:
//   - 8 closed, 2 open (issues 900, 901 open)
//   - of the 8 closed, 6 carry a witness and 2 do NOT (issues 803, 807)
//
// Hand-verified counters over a 4-hour window:
//
//	Total                 = 10
//	Closed                = 8
//	ClosureRate           = 8/10 = 0.8
//	ClosesPerHour         = 8/4  = 2.0
//	Witnessed             = 6
//	WitnessedCloseRate    = 6/8  = 0.75
//	ClaimedWithoutWitness = 8-6  = 2   (~25% of closes, mirroring the ~16% drift)
var fixtureLedger = []CloseRecord{
	{Issue: 800, Closed: true, HasWitness: true, Note: "commit abc123"},
	{Issue: 801, Closed: true, HasWitness: true, Note: "diff #801"},
	{Issue: 802, Closed: true, HasWitness: true, Note: "test TestFoo"},
	{Issue: 803, Closed: true, HasWitness: false, Note: "closed, nothing to show"},
	{Issue: 804, Closed: true, HasWitness: true, Note: "commit def456"},
	{Issue: 805, Closed: true, HasWitness: true, Note: "diff #805"},
	{Issue: 806, Closed: true, HasWitness: true, Note: "test TestBar"},
	{Issue: 807, Closed: true, HasWitness: false, Note: "claimed without witness"},
	{Issue: 900, Closed: false, HasWitness: false, Note: "still open"},
	{Issue: 901, Closed: false, HasWitness: true, Note: "open; witness ignored"},
}

const eps = 1e-9

func approx(got, want float64) bool { return math.Abs(got-want) < eps }

func TestFold_FixtureLedgerAllThreeCounters(t *testing.T) {
	r := Fold(fixtureLedger, 4.0)

	// Throughput.
	if r.Total != 10 {
		t.Errorf("Total = %d, want 10", r.Total)
	}
	if r.Closed != 8 {
		t.Errorf("Closed = %d, want 8", r.Closed)
	}
	if !approx(r.ClosureRate, 0.8) {
		t.Errorf("ClosureRate = %v, want 0.8", r.ClosureRate)
	}
	if !approx(r.ClosesPerHour, 2.0) {
		t.Errorf("ClosesPerHour = %v, want 2.0", r.ClosesPerHour)
	}

	// Honesty — the three counters the issue names.
	if r.Witnessed != 6 {
		t.Errorf("Witnessed = %d, want 6", r.Witnessed)
	}
	if !approx(r.WitnessedCloseRate, 0.75) {
		t.Errorf("WitnessedCloseRate = %v, want 0.75", r.WitnessedCloseRate)
	}
	if r.ClaimedWithoutWitness != 2 {
		t.Errorf("ClaimedWithoutWitness = %d, want 2", r.ClaimedWithoutWitness)
	}

	// Invariant: claimed-without-witness is exactly the un-witnessed closes.
	if r.ClaimedWithoutWitness != r.Closed-r.Witnessed {
		t.Errorf("ClaimedWithoutWitness (%d) != Closed-Witnessed (%d)",
			r.ClaimedWithoutWitness, r.Closed-r.Witnessed)
	}
}

func TestFold_Empty(t *testing.T) {
	r := Fold(nil, 4.0)
	if r.Total != 0 || r.Closed != 0 || r.Witnessed != 0 || r.ClaimedWithoutWitness != 0 {
		t.Errorf("empty ledger should zero all counts, got %+v", r)
	}
	// No divide-by-zero: all rates are a clean zero.
	if r.ClosureRate != 0 || r.WitnessedCloseRate != 0 || r.ClosesPerHour != 0 {
		t.Errorf("empty ledger should zero all rates, got %+v", r)
	}
	if math.IsNaN(r.ClosureRate) || math.IsNaN(r.WitnessedCloseRate) || math.IsNaN(r.ClosesPerHour) {
		t.Errorf("empty ledger produced NaN rate, got %+v", r)
	}
}

func TestFold_AllWitnessed(t *testing.T) {
	ledger := []CloseRecord{
		{Issue: 1, Closed: true, HasWitness: true},
		{Issue: 2, Closed: true, HasWitness: true},
		{Issue: 3, Closed: true, HasWitness: true},
	}
	r := Fold(ledger, 1.0)
	if !approx(r.WitnessedCloseRate, 1.0) {
		t.Errorf("WitnessedCloseRate = %v, want 1.0", r.WitnessedCloseRate)
	}
	if r.ClaimedWithoutWitness != 0 {
		t.Errorf("ClaimedWithoutWitness = %d, want 0", r.ClaimedWithoutWitness)
	}
	if !approx(r.ClosureRate, 1.0) {
		t.Errorf("ClosureRate = %v, want 1.0", r.ClosureRate)
	}
}

func TestFold_NoneWitnessed(t *testing.T) {
	ledger := []CloseRecord{
		{Issue: 1, Closed: true, HasWitness: false},
		{Issue: 2, Closed: true, HasWitness: false},
	}
	r := Fold(ledger, 2.0)
	if !approx(r.WitnessedCloseRate, 0.0) {
		t.Errorf("WitnessedCloseRate = %v, want 0.0", r.WitnessedCloseRate)
	}
	if r.ClaimedWithoutWitness != 2 {
		t.Errorf("ClaimedWithoutWitness = %d, want 2", r.ClaimedWithoutWitness)
	}
	// A perfect ClosureRate with a zero WitnessedCloseRate is exactly the
	// case the honesty split exists to expose.
	if !approx(r.ClosureRate, 1.0) {
		t.Errorf("ClosureRate = %v, want 1.0", r.ClosureRate)
	}
}

func TestFold_OpenRecordWitnessIgnored(t *testing.T) {
	// An open record must never count as witnessed, even if HasWitness is set.
	ledger := []CloseRecord{
		{Issue: 1, Closed: false, HasWitness: true},
	}
	r := Fold(ledger, 1.0)
	if r.Closed != 0 || r.Witnessed != 0 {
		t.Errorf("open record leaked into close/witness counts: %+v", r)
	}
	if r.WitnessedCloseRate != 0 {
		t.Errorf("WitnessedCloseRate = %v, want 0 (no closes)", r.WitnessedCloseRate)
	}
}

func TestFold_ZeroWindowNoPerHour(t *testing.T) {
	r := Fold(fixtureLedger, 0)
	if r.ClosesPerHour != 0 {
		t.Errorf("ClosesPerHour = %v, want 0 with no window", r.ClosesPerHour)
	}
	// Ratio counters are unaffected by the missing window.
	if !approx(r.ClosureRate, 0.8) || !approx(r.WitnessedCloseRate, 0.75) {
		t.Errorf("ratio counters should ignore the window, got %+v", r)
	}
}

func TestReport_StringSeparatesThroughputFromHonesty(t *testing.T) {
	r := Fold(fixtureLedger, 4.0)
	s := r.String()
	for _, want := range []string{"throughput:", "honesty:", "witnessed:", "claimed w/o witness: 2"} {
		if !strings.Contains(s, want) {
			t.Errorf("String() missing %q\n%s", want, s)
		}
	}
	// Throughput must render before honesty so a big close count is read
	// alongside — not instead of — the witness split.
	if strings.Index(s, "throughput:") >= strings.Index(s, "honesty:") {
		t.Errorf("throughput block should precede honesty block:\n%s", s)
	}
}

func TestReport_LineIsCompactAndComplete(t *testing.T) {
	r := Fold(fixtureLedger, 4.0)
	line := r.Line()
	if strings.Contains(line, "\n") {
		t.Errorf("Line() should be one line, got:\n%s", line)
	}
	for _, want := range []string{"closure=", "witnessed=", "claimed-no-witness=2"} {
		if !strings.Contains(line, want) {
			t.Errorf("Line() missing %q: %s", want, line)
		}
	}
}

func TestSortedByIssue_DeterministicAndNonMutating(t *testing.T) {
	in := []CloseRecord{
		{Issue: 3}, {Issue: 1}, {Issue: 2},
	}
	out := SortedByIssue(in)
	for i, want := range []int{1, 2, 3} {
		if out[i].Issue != want {
			t.Errorf("SortedByIssue[%d].Issue = %d, want %d", i, out[i].Issue, want)
		}
	}
	// Input untouched.
	if in[0].Issue != 3 {
		t.Errorf("SortedByIssue mutated its input: in[0]=%d", in[0].Issue)
	}
}
