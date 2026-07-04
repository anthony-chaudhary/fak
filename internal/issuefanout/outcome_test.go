package issuefanout

import (
	"errors"
	"strings"
	"testing"
)

// The observability follow-on (#2519): invocation outcomes of the planner are
// bucketed into success/refusal/error counts, queryable from the package's report
// surface, so a regression in the refusal contract shows up in a fold instead of a
// bug report. Drives real Build invocations plus a synthetic internal error so all
// three buckets are exercised, then asserts the captured readout shows real counts.
func TestOutcomeCountsFold(t *testing.T) {
	var counts OutcomeCounts

	// success: a valid spine builds a plan.
	if _, err := BuildInto(spineInput(), &counts); err != nil {
		t.Fatalf("BuildInto(valid): unexpected error %v", err)
	}
	// refusal: an empty spine_ref is a deliberate contract refusal.
	badSpine := spineInput()
	badSpine.SpineRef = " "
	if _, err := BuildInto(badSpine, &counts); err == nil {
		t.Fatal("BuildInto(empty spine) should refuse")
	}
	// refusal: a cap below the fan-out floor is also a contract refusal.
	lowCap := spineInput()
	lowCap.Max = MinFanout - 1
	if _, err := BuildInto(lowCap, &counts); err == nil {
		t.Fatal("BuildInto(below-floor cap) should refuse")
	}
	// error: an unexpected (non-Refusal) failure from the invocation path.
	counts.Observe(errors.New("boom: unexpected internal failure"))

	if counts.Success != 1 || counts.Refused != 2 || counts.Error != 1 {
		t.Fatalf("outcome buckets wrong: %+v", counts)
	}
	if counts.Total() != 4 {
		t.Fatalf("Total() = %d, want 4", counts.Total())
	}

	// Classification of each result kind is principled, not string-sniffed.
	if got := ClassifyOutcome(nil); got != OutcomeSuccess {
		t.Fatalf("ClassifyOutcome(nil) = %q, want %q", got, OutcomeSuccess)
	}
	if _, err := Build(badSpine); ClassifyOutcome(err) != OutcomeRefused {
		t.Fatalf("a contract refusal must classify as %q", OutcomeRefused)
	}
	if got := ClassifyOutcome(errors.New("boom")); got != OutcomeError {
		t.Fatalf("a non-Refusal error must classify as %q, got %q", OutcomeError, got)
	}

	// The readout is the witness: it shows the real per-bucket counts on one line.
	readout := RenderOutcomes(counts)
	for _, want := range []string{"4", "1 success", "2 refused", "1 error"} {
		if !strings.Contains(readout, want) {
			t.Fatalf("readout %q missing %q", readout, want)
		}
	}
	t.Logf("outcome readout: %s", readout)
}
