package issuefanout

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func spineInput() Input {
	return Input{
		Title:    "issue fanout planner",
		Leaf:     "issuefanout",
		SpineRef: "fak issue fanout --title ... --json (internal/issuefanout Build)",
	}
}

// The point of the leaf: every generated follow-on is dispatchable under the
// full issuecontract scope rules the moment it is planned.
func TestBuildEveryCandidateDispatchable(t *testing.T) {
	plan, err := Build(spineInput())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if len(plan.Candidates) < MinFanout {
		t.Fatalf("fan-out floor: got %d candidates, want >= %d", len(plan.Candidates), MinFanout)
	}
	for _, c := range plan.Candidates {
		r := issuecontract.ReviewCandidate(c, issuecontract.Options{})
		if r.Dispatchability != issuecontract.Dispatchable {
			t.Errorf("candidate %s not dispatchable: verdict=%s reasons=%v missing=%v",
				c.Key, r.Verdict, r.Reasons, r.MissingFields)
		}
	}
}

// No fan-out without a spine witness: the refusal must name the recovery.
func TestBuildRequiresSpineWitness(t *testing.T) {
	in := spineInput()
	in.SpineRef = " "
	_, err := Build(in)
	if err == nil {
		t.Fatal("Build accepted an empty spine_ref")
	}
	if !strings.Contains(err.Error(), "working spine first") {
		t.Fatalf("refusal does not name the recovery: %v", err)
	}
}

func TestBuildAreaFilterAndCap(t *testing.T) {
	in := spineInput()
	in.Areas = []string{"qa"}
	plan, err := Build(in)
	if err != nil {
		t.Fatalf("Build(areas=qa): %v", err)
	}
	for _, c := range plan.Candidates {
		if len(c.Labels) < 2 || c.Labels[1] != "qa" {
			t.Fatalf("area filter leaked non-qa candidate %s (labels %v)", c.Key, c.Labels)
		}
	}
	if plan.AreaCounts["qa"] != len(plan.Candidates) {
		t.Fatalf("area counts disagree: %v vs %d candidates", plan.AreaCounts, len(plan.Candidates))
	}

	in = spineInput()
	in.Max = MinFanout
	plan, err = Build(in)
	if err != nil {
		t.Fatalf("Build(max=%d): %v", MinFanout, err)
	}
	if len(plan.Candidates) != MinFanout {
		t.Fatalf("cap ignored: got %d, want %d", len(plan.Candidates), MinFanout)
	}

	in.Max = MinFanout - 1
	if _, err := Build(in); err == nil {
		t.Fatal("Build accepted a cap below the fan-out floor")
	}

	in = spineInput()
	in.Areas = []string{"nonsense"}
	if _, err := Build(in); err == nil || !strings.Contains(err.Error(), "known:") {
		t.Fatalf("unknown area not refused with the known list: %v", err)
	}
}

// The planner is pure: same input, byte-identical plan.
func TestBuildDeterministic(t *testing.T) {
	a, err := Build(spineInput())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	b, err := Build(spineInput())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("two Builds over the same input differ")
	}
}
