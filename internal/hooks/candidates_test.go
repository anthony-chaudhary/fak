package hooks

import (
	"fmt"
	"sync"
	"testing"
)

// TestCandidatesUnreportedIsNotZero pins the claim the whole file is built around, and the one
// that is easiest to regress away: UNREPORTED and ZERO are DIFFERENT ANSWERS.
//
// A gate that never recorded returns ok=false — the CLI renders that as JSON null. A gate that
// recorded zero returns n=0 with ok=TRUE, which is a real and load-bearing measurement: "this
// gate ran and its filter admitted nothing". Collapsing the two would rebuild, one level up,
// exactly the ambiguity #5602 exists to retire — because {"findings":[],"count":0} already fails
// to separate "judged forty files, found nothing" from "judged nothing at all".
func TestCandidatesUnreportedIsNotZero(t *testing.T) {
	d := &StagedDiff{}

	// Never recorded: UNREPORTED.
	if n, unit, ok := d.Candidates("GOFMT"); ok {
		t.Errorf("a gate that recorded nothing reported ok=true (n=%d unit=%q); UNREPORTED must be distinguishable from zero", n, unit)
	}

	// Recorded an honest zero: REPORTED, and the value is zero.
	d.NoteCandidates("GOFMT", 0, "staged .go file(s)")
	n, unit, ok := d.Candidates("GOFMT")
	if !ok {
		t.Fatal("a gate that recorded 0 reported ok=false; zero is an answer, not the absence of one")
	}
	if n != 0 {
		t.Errorf("n = %d, want 0", n)
	}
	if unit != "staged .go file(s)" {
		t.Errorf("unit = %q, want the gate's own words back verbatim", unit)
	}

	// A DIFFERENT gate is still unreported — recording for one gate must not imply an answer
	// for its neighbours. The ledger is per gate, not a single run-wide count.
	if _, _, ok := d.Candidates("INDEX_SYNC"); ok {
		t.Error("recording for GOFMT made INDEX_SYNC look reported")
	}
}

// TestCandidatesUnitIsPerGate pins that the unit travels with the count. The unit is deliberately
// per gate — a file-scoped gate counts files and a line-scoped gate counts added lines — so a
// denominator rendered without its unit is uninterpretable, and two gates must never be forced
// to share one vocabulary.
func TestCandidatesUnitIsPerGate(t *testing.T) {
	d := &StagedDiff{}
	d.NoteCandidates("GOFMT", 4, "staged .go file(s)")
	d.NoteCandidates("DUPLICATION", 91, "staged .go block(s) above the clone window")

	for _, c := range []struct {
		gate string
		n    int
		unit string
	}{
		{"GOFMT", 4, "staged .go file(s)"},
		{"DUPLICATION", 91, "staged .go block(s) above the clone window"},
	} {
		n, unit, ok := d.Candidates(c.gate)
		if !ok {
			t.Fatalf("%s unreported", c.gate)
		}
		if n != c.n || unit != c.unit {
			t.Errorf("%s = (%d, %q), want (%d, %q)", c.gate, n, unit, c.n, c.unit)
		}
	}
}

// TestNoteCandidatesLastCallWins pins the documented override: a gate that narrows its filter in
// stages records the FINAL domain it actually judged, not an intermediate one. The denominator
// must describe the set the verdict came from — an early, wider count would overstate what was
// checked, which is the specific dishonesty this ledger exists to prevent.
func TestNoteCandidatesLastCallWins(t *testing.T) {
	d := &StagedDiff{}
	d.NoteCandidates("PRIOR_ART", 40, "staged path(s)")     // everything touched
	d.NoteCandidates("PRIOR_ART", 12, "staged .go file(s)") // after the extension filter
	d.NoteCandidates("PRIOR_ART", 3, "newly added symbol(s)")

	n, unit, ok := d.Candidates("PRIOR_ART")
	if !ok {
		t.Fatal("PRIOR_ART unreported after three notes")
	}
	if n != 3 || unit != "newly added symbol(s)" {
		t.Errorf("got (%d, %q), want the LAST note (3, \"newly added symbol(s)\")", n, unit)
	}

	// Narrowing all the way to zero is a legitimate final answer, not a reset to UNREPORTED.
	d.NoteCandidates("PRIOR_ART", 0, "newly added symbol(s)")
	if n, _, ok := d.Candidates("PRIOR_ART"); !ok || n != 0 {
		t.Errorf("after narrowing to zero: (n=%d ok=%v), want (0, true)", n, ok)
	}
}

// TestReportedGatesSorted pins the ordering the CLI depends on to tell "no gate reports a count"
// from "every gate reported zero". Map iteration order in Go is randomized, so an unsorted list
// would make the pre-commit summary reorder itself between identical runs — noise that reads as
// a real change in a diffed or logged payload.
func TestReportedGatesSorted(t *testing.T) {
	d := &StagedDiff{}
	// Recorded deliberately out of order.
	for _, g := range []string{"PRIOR_ART", "GOFMT", "DUPLICATION", "INDEX_SYNC"} {
		d.NoteCandidates(g, 1, "thing(s)")
	}
	got := d.ReportedGates()
	want := []string{"DUPLICATION", "GOFMT", "INDEX_SYNC", "PRIOR_ART"}
	if len(got) != len(want) {
		t.Fatalf("ReportedGates() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ReportedGates() = %v, want %v (sorted)", got, want)
		}
	}

	// A run where no gate recorded anything reports an EMPTY list, not a phantom entry.
	if got := (&StagedDiff{}).ReportedGates(); len(got) != 0 {
		t.Errorf("fresh diff reported gates %v, want none", got)
	}
}

// TestNoteCandidatesConcurrentWithAbandonedGate is the reason candMu exists, stated as a test.
//
// The pre-commit CLI bounds each gate with a wall clock and ABANDONS one that overruns (#5335)
// WITHOUT cancelling it — Gate.Check takes no context — so the abandoned Check keeps running
// against this same StagedDiff while the loop hands it to the next gate. Two gates therefore
// reach this ledger at once. An unsynchronized map write is a Go RUNTIME FATAL: not a failed
// gate, not a recoverable panic, but a process kill that would take down the very commit the
// timeout bound exists to let through.
//
// Run under -race this fails loudly on an unguarded map; even without -race the concurrent
// write/read mix reliably trips the runtime's own map detector.
func TestNoteCandidatesConcurrentWithAbandonedGate(t *testing.T) {
	d := &StagedDiff{}
	const gates, notes = 8, 200

	var wg sync.WaitGroup
	for g := 0; g < gates; g++ {
		name := fmt.Sprintf("GATE_%d", g)
		wg.Add(2)
		// The over-budget gate, still writing after it was abandoned.
		go func() {
			defer wg.Done()
			for i := 0; i < notes; i++ {
				d.NoteCandidates(name, i, "staged item(s)")
			}
		}()
		// The next gate, reading the ledger the runner already moved on to.
		go func() {
			defer wg.Done()
			for i := 0; i < notes; i++ {
				d.Candidates(name)
				d.ReportedGates()
			}
		}()
	}
	wg.Wait()

	if got := len(d.ReportedGates()); got != gates {
		t.Errorf("ReportedGates() has %d entries, want %d — a note was lost under concurrency", got, gates)
	}
}

// TestNoteCandidatesDegradesRatherThanPanics pins that the ledger never becomes the reason a
// commit dies. It is observability wiring called from inside every gate, so a nil receiver or an
// unnamed gate must be a silent no-op — the same best-effort contract the rest of the hook
// surface holds itself to.
func TestNoteCandidatesDegradesRatherThanPanics(t *testing.T) {
	var nilDiff *StagedDiff
	nilDiff.NoteCandidates("GOFMT", 1, "file(s)") // must not panic
	if n, unit, ok := nilDiff.Candidates("GOFMT"); ok || n != 0 || unit != "" {
		t.Errorf("nil diff reported (%d, %q, %v), want the zero answer with ok=false", n, unit, ok)
	}
	if got := nilDiff.ReportedGates(); got != nil {
		t.Errorf("nil diff reported gates %v, want nil", got)
	}

	// An unnamed gate has no key to file the count under, so the note is dropped rather than
	// stored under "" where no caller could ever ask for it.
	d := &StagedDiff{}
	d.NoteCandidates("", 5, "file(s)")
	if got := d.ReportedGates(); len(got) != 0 {
		t.Errorf("an empty gate name was recorded as %v, want dropped", got)
	}
}
