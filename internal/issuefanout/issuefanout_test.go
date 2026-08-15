package issuefanout

import (
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
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
		r := issuepolicy.ReviewCandidate(c, issuepolicy.Options{})
		if r.Dispatchability != issuepolicy.Dispatchable {
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

// TestBuildConcurrentDeterminism is the #2513 race witness. The sequential
// TestBuildDeterministic proves same-input-same-output on one goroutine, but the
// planner's promise ("cohort/replay layers can trust it") is a CONCURRENT one:
// many callers fan the same spine out at once. This test runs a pool of
// goroutines over three distinct Build paths (default spine, area-filtered,
// capped) and asserts every concurrent plan deep-equals a sequential reference
// computed before any goroutine starts. Under `go test -race` it exercises the
// shared read of the package-level taxonomy and proves that path carries no data
// race — the meaningful `-race` witness the sequential test cannot give.
func TestBuildConcurrentDeterminism(t *testing.T) {
	qaDocs := spineInput()
	qaDocs.Areas = []string{"qa", "docs"}
	capped := spineInput()
	capped.Max = MinFanout
	inputs := []Input{spineInput(), qaDocs, capped}

	// Sequential references, fixed before the goroutines read them.
	refs := make([]Plan, len(inputs))
	for i, in := range inputs {
		p, err := Build(in)
		if err != nil {
			t.Fatalf("reference Build[%d]: %v", i, err)
		}
		refs[i] = p
	}

	const workers = 64
	got := make([]Plan, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		// Each goroutine owns its own got[w]/errs[w] slot: the only shared reads
		// are the immutable inputs/refs and the package-level taxonomy.
		go func(w int) {
			defer wg.Done()
			got[w], errs[w] = Build(inputs[w%len(inputs)])
		}(w)
	}
	wg.Wait()

	for w := 0; w < workers; w++ {
		if errs[w] != nil {
			t.Fatalf("concurrent Build[%d]: %v", w, errs[w])
		}
		if ref := refs[w%len(inputs)]; !reflect.DeepEqual(got[w], ref) {
			t.Fatalf("concurrent Build[%d] diverged from its sequential reference", w)
		}
	}
}

func TestBuildGivesEveryChildCanonicalChildSpecificProblemFrame(t *testing.T) {
	plan, err := Build(Input{Title: "Cache spine", Leaf: "cache", SpineRef: "abc123", Max: 15})
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range plan.Candidates {
		frame := candidate.ProblemFrame
		if frame.Schema != issuepolicy.ProblemFrameSchema || !frame.Enforced || !frame.Ready {
			t.Fatalf("%s frame = %+v", candidate.Key, frame)
		}
		if frame.Centrality == issuepolicy.CentralityEnabling && frame.CentralityTarget == "" {
			t.Fatalf("%s enabling frame lost Core target", candidate.Key)
		}
		if frame.Centrality == issuepolicy.CentralityStewardship && frame.CentralityTarget == "" {
			t.Fatalf("%s stewardship frame lost obligation", candidate.Key)
		}
		for _, id := range []string{"p1", "p2", "p3", "p4"} {
			if check := frame.Checks[id]; !check.Valid || check.Evidence == "" {
				t.Fatalf("%s %s = %+v", candidate.Key, id, check)
			}
		}
	}
}

func TestBuildDoesNotInheritProblemFrameFromParentMetadata(t *testing.T) {
	left, err := Build(Input{Title: "Core parent", Leaf: "core", SpineRef: "abc", ParentRef: "Core epic", Max: 3})
	if err != nil {
		t.Fatal(err)
	}
	right, err := Build(Input{Title: "Peripheral parent", Leaf: "peripheral", SpineRef: "abc", ParentRef: "Peripheral epic", Max: 3})
	if err != nil {
		t.Fatal(err)
	}
	for i := range left.Candidates {
		if !reflect.DeepEqual(left.Candidates[i].ProblemFrame, right.Candidates[i].ProblemFrame) {
			t.Fatalf("parent metadata changed child frame:\nleft=%+v\nright=%+v", left.Candidates[i].ProblemFrame, right.Candidates[i].ProblemFrame)
		}
	}
}
