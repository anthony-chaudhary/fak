package issuefanout

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func liveTestPlan(t *testing.T) Plan {
	t.Helper()
	plan, err := Build(Input{
		Title:             "fanout live test",
		Leaf:              "fanoutlivetest",
		SpineRef:          "deadbeef",
		ParentIssue:       36,
		ParentBaseline:    100,
		TargetEnvelope:    "- concurrent users: 10 users\n- sustained duration: 60 minutes",
		WitnessedEnvelope: "- concurrent users: 10 users\n- sustained duration: 60 minutes",
		Areas:             []string{"qa"},
	})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if len(plan.Candidates) != 3 {
		t.Fatalf("qa area = %d candidates, want 3", len(plan.Candidates))
	}
	return plan
}

// fakeGh records every create argv and mints sequential issue URLs.
type fakeGh struct {
	calls [][]string
	next  int
}

func (f *fakeGh) run(args []string) (string, string, bool) {
	f.calls = append(f.calls, args)
	f.next++
	return fmt.Sprintf("https://github.com/o/r/issues/%d", 900+f.next), "", true
}

func TestFileLiveFilesUnseenSkipsSeenAndRerunsClean(t *testing.T) {
	plan := liveTestPlan(t)
	seenKey := plan.Candidates[0].Key // fanout-fanoutlivetest-qa-edge-sweep
	existing := []Issue{
		{Number: 42, Body: "prose that mentions " + seenKey + " without a marker"},
	}
	gh := &fakeGh{}
	res, err := FileLive(plan, existing, LiveOptions{DedupeCap: 50, Runner: gh.run})
	if err != nil {
		t.Fatalf("FileLive: %v", err)
	}
	if res.Filed != 2 || res.Skipped != 1 || res.Failed != 0 {
		t.Fatalf("first run filed/skipped/failed = %d/%d/%d, want 2/1/0\n%s",
			res.Filed, res.Skipped, res.Failed, RenderLive(res))
	}
	if res.Rows[0].Action != "skipped" || res.Rows[0].SeenIn == nil || *res.Rows[0].SeenIn != 42 {
		t.Fatalf("seen candidate row = %+v, want skipped via issue #42", res.Rows[0])
	}
	if len(gh.calls) != 2 {
		t.Fatalf("gh create calls = %d, want 2", len(gh.calls))
	}

	// The filed create carries the marker body, the mapped labels, and the
	// generation milestone.
	call := strings.Join(gh.calls[0], "\x00")
	c := plan.Candidates[1]
	for _, want := range []string{
		"<!-- fak-issuefanout-key: " + c.Key + " -->",
		"\x00--label\x00fanout",
		"\x00--label\x00qa",
		"\x00--label\x00" + c.Generation,
		"\x00--label\x00" + c.Priority,
		"\x00--milestone\x00" + MilestoneForGeneration(c.Generation),
	} {
		if !strings.Contains(call, want) {
			t.Fatalf("create argv missing %q:\n%v", want, gh.calls[0])
		}
	}

	// Rerun against a tracker that now carries the filed bodies: files zero,
	// skips everything, calls gh create not once.
	rerunExisting := append([]Issue{}, existing...)
	for i, row := range res.Rows {
		if row.Action == "filed" {
			rerunExisting = append(rerunExisting, Issue{Number: *row.Number, Body: LiveBody(plan.Candidates[i])})
		}
	}
	gh2 := &fakeGh{}
	res2, err := FileLive(plan, rerunExisting, LiveOptions{DedupeCap: 50, Runner: gh2.run})
	if err != nil {
		t.Fatalf("FileLive rerun: %v", err)
	}
	if res2.Filed != 0 || res2.Skipped != 3 || res2.Failed != 0 || len(gh2.calls) != 0 {
		t.Fatalf("rerun filed/skipped/failed/calls = %d/%d/%d/%d, want 0/3/0/0\n%s",
			res2.Filed, res2.Skipped, res2.Failed, len(gh2.calls), RenderLive(res2))
	}
	if !strings.Contains(RenderLive(res2), "rerun clean") {
		t.Fatalf("rerun render missing the clean line:\n%s", RenderLive(res2))
	}
}

// The nil-runner refusal is covered by the refusalContract table in
// failure_paths_test.go, which pins every refusef site exhaustively.

func TestFileLiveReportsGhFailure(t *testing.T) {
	plan := liveTestPlan(t)
	run := func(args []string) (string, string, bool) { return "", "gh: boom", false }
	res, err := FileLive(plan, nil, LiveOptions{Runner: run})
	if err != nil {
		t.Fatalf("FileLive: %v", err)
	}
	if res.Failed != 3 || res.Filed != 0 {
		t.Fatalf("failed/filed = %d/%d, want 3/0", res.Failed, res.Filed)
	}
	if res.Rows[0].Reason != "gh: boom" {
		t.Fatalf("failure reason = %q, want gh stderr", res.Rows[0].Reason)
	}
}

func TestFileLiveReportsUnparsableCreateOutput(t *testing.T) {
	plan := liveTestPlan(t)
	run := func(args []string) (string, string, bool) { return "created something", "", true }
	res, err := FileLive(plan, nil, LiveOptions{Runner: run})
	if err != nil {
		t.Fatalf("FileLive: %v", err)
	}
	if res.Failed != 3 {
		t.Fatalf("failed = %d, want 3 (no issue URL in stdout)", res.Failed)
	}
}

func TestMilestoneForGeneration(t *testing.T) {
	for gen, want := range map[string]string{
		"gen/now":         "Generation G0 - Now / Immediate",
		"gen/next":        "Generation G1 - Next Gen",
		"gen/second-next": "Generation G2 - Second Next Gen",
		"gen/future":      "Generation G3 - Future",
		"next":            "Generation G1 - Next Gen",
		"gen/unknown":     "",
		"":                "",
	} {
		if got := MilestoneForGeneration(gen); got != want {
			t.Fatalf("MilestoneForGeneration(%q) = %q, want %q", gen, got, want)
		}
	}
}

func TestListExistingArgsBoundsTheScan(t *testing.T) {
	got := strings.Join(ListExistingArgs("o/r", 0), " ")
	want := fmt.Sprintf("issue list --state all --limit %d --json number,body --repo o/r", DefaultDedupeCap)
	if got != want {
		t.Fatalf("ListExistingArgs = %q, want %q", got, want)
	}
}

func TestLiveBodyRoundTripsCanonicalProblemFrame(t *testing.T) {
	plan, err := Build(Input{Title: "frame spine", Leaf: "framespine", SpineRef: "abc", Max: 3})
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range plan.Candidates {
		body := LiveBody(candidate)
		review := issuepolicy.ReviewIssueDraft(issuepolicy.IssueDraft{Title: candidate.Title, Body: body}, issuepolicy.Options{})
		if !review.ProblemFrame.Ready || review.ProblemFrame.Centrality != candidate.ProblemFrame.Centrality || review.ProblemFrame.CentralityTarget != candidate.ProblemFrame.CentralityTarget || len(review.ProblemFrame.Checks) != 4 {
			t.Fatalf("%s body lost frame: %+v\n%s", candidate.Key, review.ProblemFrame, body)
		}
	}
}
