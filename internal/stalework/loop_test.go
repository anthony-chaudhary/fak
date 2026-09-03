package stalework

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func loopCandidate(path string) Candidate {
	return Candidate{
		Path: path, Batch: path, Score: 50, Status: "candidate",
		Components:         []Component{{Name: "dependency_drift", Points: 50, Provenance: "git", Evidence: "one dependency commit"}},
		LastSemanticCommit: "old", DependencyCommits: []Commit{{SHA: "new", Subject: "changed", Dependency: "cmd/fak"}},
		Excerpt: "current operator text", ExcerptSHA256: "excerpt",
		DedupeKey:      "stale-work:" + path,
		ProposedDoD:    []string{"adjudicate current truth", "record retain/update/delete"},
		VerifyWith:     "fak stale-work --path " + path + " --json",
		Recommendation: "adjudicate only",
	}
}

func validLoopIssue(number int, candidate Candidate, state string) IssueSnapshot {
	digest := EvidenceDigest(candidate)
	contract := contractCandidate(candidate, digest)
	title, body := renderIssue(contract, candidate, digest)
	labels := make([]issuepolicy.IssueLabel, len(contract.Labels))
	for i, l := range contract.Labels {
		labels[i] = issuepolicy.IssueLabel{Name: l}
	}
	return IssueSnapshot{Number: number, Title: title, Body: body, State: state, Labels: labels, URL: "https://example.invalid/issues/" + contract.Key}
}

func TestBuildLoopRefusesDispatchUntilDedicatedIssueExists(t *testing.T) {
	c := loopCandidate("docs/operator.md")
	plan := BuildLoop(Packet{Schema: Schema, Head: "head", Candidates: []Candidate{c}}, LoopOptions{})
	if len(plan.Units) != 1 {
		t.Fatalf("units=%d, want 1", len(plan.Units))
	}
	u := plan.Units[0]
	frame := u.Issue.Review.ProblemFrame
	if !frame.Enforced || !frame.Ready || frame.Centrality != issuepolicy.CentralityStewardship || frame.CentralityTarget == "" || len(frame.Checks) != 4 {
		t.Fatalf("problem frame = %+v", frame)
	}
	if u.Issue.Action != "create" || !u.Issue.Review.OK {
		t.Fatalf("issue plan=%+v, want contract-valid create", u.Issue)
	}
	if u.Dispatch.Status != DispatchRefuse || u.Dispatch.Reason != ReasonIssueRequired {
		t.Fatalf("dispatch=%+v, want issue-required refusal", u.Dispatch)
	}
	for _, want := range []string{"retained, updated, or deleted", "fresh worker identity", "dos commit-audit"} {
		if !strings.Contains(u.Issue.Body, want) {
			t.Fatalf("generated issue missing %q", want)
		}
	}
	if plan.Counts.Launches != 0 || plan.Counts.CreatePlanned != 1 {
		t.Fatalf("counts=%+v, want dry-run create and zero launches", plan.Counts)
	}
	if len(plan.Batches) != 1 || plan.Batches[0].TruthSource != c.Path || plan.Batches[0].AcceptanceWitness != c.VerifyWith {
		t.Fatalf("batches=%+v, want artifact+witness bounded batch", plan.Batches)
	}
}

func TestBuildLoopRefusesIssueWithoutContractDoD(t *testing.T) {
	c := loopCandidate("docs/operator.md")
	issue := IssueSnapshot{
		Number: 66180, State: "OPEN", Title: "stale-work: operator",
		Body: "<!-- fak-stale-work-key: stale-work:docs/operator.md -->\n\n## Current state\nOld.\n\n## Likely files\n- `docs/operator.md`\n",
	}
	plan := BuildLoop(Packet{Head: "head", Candidates: []Candidate{c}}, LoopOptions{Issues: []IssueSnapshot{issue}})
	u := plan.Units[0]
	if u.Issue.Action != "repair" || u.Dispatch.Reason != ReasonIssueContractInvalid {
		t.Fatalf("unit=%+v, want contract repair refusal", u)
	}
}

func TestBuildLoopUsesDistinctWorkerIdentityAndSerializesCollision(t *testing.T) {
	a := loopCandidate("docs/shared")
	b := loopCandidate("docs/shared/a.md")
	b.ExcerptSHA256 = "different"
	issues := []IssueSnapshot{validLoopIssue(7001, a, "OPEN"), validLoopIssue(7002, b, "OPEN")}
	plan := BuildLoop(Packet{Head: "head", Candidates: []Candidate{a, b}}, LoopOptions{Issues: issues, MaxWave: 5})
	if plan.Counts.DispatchReady != 2 || plan.Counts.CollisionPairs != 1 || plan.Counts.Waves != 2 {
		t.Fatalf("counts=%+v, want two ready workers serialized over one collision", plan.Counts)
	}
	if plan.Units[0].Dispatch.WorkerID == plan.Units[1].Dispatch.WorkerID {
		t.Fatalf("worker id reused: %q", plan.Units[0].Dispatch.WorkerID)
	}
	if plan.Units[0].Dispatch.Wave == plan.Units[1].Dispatch.Wave {
		t.Fatalf("waves=(%d,%d), want overlap serialized", plan.Units[0].Dispatch.Wave, plan.Units[1].Dispatch.Wave)
	}
	if plan.Counts.Launches != 0 {
		t.Fatalf("dry-run launches=%d, want 0", plan.Counts.Launches)
	}
}

func TestBuildLoopReusesAndInvalidatesAdjudicationByEvidenceDigest(t *testing.T) {
	c := loopCandidate("docs/cache.md")
	digest := EvidenceDigest(c)
	state := LoopState{Schema: StateSchema, Adjudications: []AdjudicationRecord{{
		DedupeKey: c.DedupeKey, EvidenceDigest: digest, Decision: "retained", Witness: "issue #1 + git read-back",
	}}}
	plan := BuildLoop(Packet{Candidates: []Candidate{c}}, LoopOptions{State: state})
	if plan.Counts.Cached != 1 || plan.Units[0].Cache != "reused" {
		t.Fatalf("plan=%+v, want cache reuse", plan)
	}
	c.ExcerptSHA256 = "changed"
	plan = BuildLoop(Packet{Candidates: []Candidate{c}}, LoopOptions{State: state})
	if plan.Counts.Cached != 0 || plan.Units[0].Cache != "invalidated" || plan.Units[0].Issue.Action != "create" {
		t.Fatalf("unit=%+v, want evidence invalidation and re-adjudication", plan.Units[0])
	}
}

func TestBuildLoopReconcilesOnlyIndependentIssueGitAndTestWitnesses(t *testing.T) {
	c := loopCandidate("docs/witness.md")
	closed := validLoopIssue(8001, c, "CLOSED")
	base := Packet{Candidates: []Candidate{c}}

	plan := BuildLoop(base, LoopOptions{Issues: []IssueSnapshot{closed}})
	if got := plan.Units[0].Reconciliation; got.Status != ReconcileAbstain || got.Reason != ReasonCommitWitnessMissing {
		t.Fatalf("without independent witnesses got %+v", got)
	}

	w := WitnessRecord{
		Issue: 8001, SHA: "abc", Verdict: "OK", Witness: dispatchtick.WitnessOK,
		TestClaim: dispatchtick.ClaimTestGreen, Decision: "updated", Source: "git+issue+test read-back",
	}
	plan = BuildLoop(base, LoopOptions{Issues: []IssueSnapshot{closed}, Witnesses: []WitnessRecord{w}})
	got := plan.Units[0].Reconciliation
	if got.Status != ReconcileShipped || got.Decision != "updated" || got.SHA != "abc" {
		t.Fatalf("witnessed reconcile=%+v, want SHIPPED", got)
	}
	if plan.Units[0].Dispatch.Status != DispatchRefuse || plan.Units[0].Dispatch.Reason != ReasonAlreadyShipped {
		t.Fatalf("dispatch=%+v, want no launch for closed witnessed issue", plan.Units[0].Dispatch)
	}
}

func TestSeedIssuesRenderThreeBoundUnitsAndZeroLaunches(t *testing.T) {
	candidates := []Candidate{
		loopCandidate("docs/cli-reference.md"),
		loopCandidate("docs/native-device-mesh.md"),
		loopCandidate("docs/scorecards.md"),
	}
	issues := []IssueSnapshot{
		validLoopIssue(3869, candidates[0], "OPEN"),
		validLoopIssue(5305, candidates[1], "OPEN"),
		validLoopIssue(5631, candidates[2], "OPEN"),
	}
	plan := BuildLoop(Packet{Head: "seed", Candidates: candidates}, LoopOptions{Issues: issues})
	if plan.Counts.IssueBound != 3 || plan.Counts.DispatchReady != 3 || plan.Counts.Launches != 0 {
		t.Fatalf("seed counts=%+v, want 3 issue-bound dry-run units and zero launches", plan.Counts)
	}
}
