package blockerpost

import "testing"

// TestTriageSelfcheck runs the packaged no-I/O proof.
func TestTriageSelfcheck(t *testing.T) {
	if err := TriageSelfcheck(); err != nil {
		t.Fatalf("TriageSelfcheck: %v", err)
	}
}

// TestFoldIssuesTriagedDecentersRoutable proves the core behavior change: an
// unowned-but-fleet-routable backlog pages under the ownership-only fold but is
// recorded as background status once triaged.
func TestFoldIssuesTriagedDecentersRoutable(t *testing.T) {
	issues := []Issue{
		{Number: 10, Title: "regenerate the scorecard and rerun `fak cadence`"},
		{Number: 11, Title: "investigate the flaky worktree reap test"},
	}

	base := FoldIssues(issues, "blocked", "https://github.com/o/r")
	if base.Severity != SeverityOperator {
		t.Fatalf("baseline severity = %q, want %q (unowned pages)", base.Severity, SeverityOperator)
	}

	tri := FoldIssuesTriaged(issues, "blocked", "https://github.com/o/r")
	if tri.Severity != SeverityStatus {
		t.Fatalf("triaged severity = %q, want %q (all fleet-routable, no page)", tri.Severity, SeverityStatus)
	}
	// The backlog rows are preserved — an operator who looks anyway still sees them.
	if len(tri.Lines) != 2 {
		t.Fatalf("triaged roll-up dropped the backlog rows, got %d lines: %+v", len(tri.Lines), tri.Lines)
	}
}

// TestFoldIssuesTriagedKeepsAuthorityPage proves a genuine authority decision
// still pages, and the detail names how many of the unowned issues need a person.
func TestFoldIssuesTriagedKeepsAuthorityPage(t *testing.T) {
	issues := []Issue{
		{Number: 20, Title: "regenerate the report"},                        // routable
		{Number: 21, Title: "approve the production release", Labels: []Label{{Name: "release"}}}, // authority
	}
	tri := FoldIssuesTriaged(issues, "blocked", "")
	if tri.Severity != SeverityOperator {
		t.Fatalf("triaged severity = %q, want %q (an authority decision remains)", tri.Severity, SeverityOperator)
	}
}

// TestFoldIssuesTriagedPassThrough proves a clear or all-owned backlog is
// returned unchanged — there is no operator page to decenter.
func TestFoldIssuesTriagedPassThrough(t *testing.T) {
	if got := FoldIssuesTriaged(nil, "blocked", "").Severity; got != SeverityClear {
		t.Fatalf("empty backlog triaged = %q, want %q", got, SeverityClear)
	}
	owned := []Issue{{Number: 30, Title: "in flight", Assignees: []Assignee{{Login: "dev"}}}}
	if got := FoldIssuesTriaged(owned, "blocked", "").Severity; got != SeverityStatus {
		t.Fatalf("all-owned backlog triaged = %q, want %q", got, SeverityStatus)
	}
}
