package blockerpost

import (
	"fmt"
	"testing"
)

// TestBenchmarkSanity ensures benchmark fixtures are valid and exercisable
// across all production severity paths and folding operations.
func TestBenchmarkSanity(t *testing.T) {
	op := sampleBlockerOperator()
	if op.Text() == "" || len(op.Blocks()) == 0 {
		t.Fatal("expected non-empty operator text and blocks")
	}

	st := sampleBlockerStatus()
	if st.Text() == "" || len(st.Blocks()) == 0 {
		t.Fatal("expected non-empty status text and blocks")
	}

	cl := sampleBlockerClear()
	if cl.Text() == "" || len(cl.Blocks()) == 0 {
		t.Fatal("expected non-empty clear text and blocks")
	}

	issues := sampleIssues(15, 5)
	folded := FoldIssues(issues, "blocked", "https://github.com/anthony-chaudhary/fak")
	if folded.Severity != SeverityOperator {
		t.Fatalf("expected operator severity for unowned issues, got %s", folded.Severity)
	}

	triaged := FoldIssuesTriaged(sampleTriagedIssuesRoutable(), "blocked", "https://github.com/anthony-chaudhary/fak")
	if triaged.Severity != SeverityStatus {
		t.Fatalf("expected status severity for routable triaged issues, got %s", triaged.Severity)
	}

	sigs := sampleSignatures(10, 2, 6)
	blast := FoldBlast(sigs, "https://github.com/anthony-chaudhary/fak")
	if blast.Severity != SeverityOperator {
		t.Fatalf("expected operator severity for overdue signatures, got %s", blast.Severity)
	}
}

func sampleBlockerOperator() Blocker {
	return Blocker{
		Severity:  SeverityOperator,
		Title:     "CPU host unreachable",
		Detail:    "CPU GLM-5.2 node is not responding to health checks.",
		Lines:     []string{"heartbeat missed at 04:12:00 UTC", "queue backpressure rising on lane L3", "3 subagents stranded"},
		Owner:     "<!here>",
		Action:    "restart the CPU-host serve",
		ActionURL: "https://example.invalid/runbook/cpu-restart",
		Ref:       "host:cpu-server-a",
		Source:    "ci",
	}
}

func sampleBlockerStatus() Blocker {
	return Blocker{
		Severity: SeverityStatus,
		Title:    "GPU-gated, waiting on private GPU-server hours",
		Detail:   "Rungs 1/2/3/5 need the private GPU server.",
		Lines:    []string{"lane L1 queued behind lock", "lane L2 holding lease"},
		Ref:      "gpu-gate",
		Source:   "agent",
	}
}

func sampleBlockerClear() Blocker {
	return Blocker{
		Severity: SeverityClear,
		Title:    "no standing blockers",
		Detail:   "0 open blocked issues — the board is clear.",
		Ref:      "label:blocked",
	}
}

func sampleIssues(total, unowned int) []Issue {
	issues := make([]Issue, total)
	for i := 0; i < total; i++ {
		iss := Issue{
			Number: 100 + i,
			Title:  fmt.Sprintf("investigate worker stall on partition %d", i),
			URL:    fmt.Sprintf("https://github.com/anthony-chaudhary/fak/issues/%d", 100+i),
			Labels: []Label{{Name: "blocked"}},
		}
		if i >= unowned {
			iss.Assignees = []Assignee{{Login: fmt.Sprintf("dev-%d", i)}}
		}
		issues[i] = iss
	}
	return issues
}

func sampleTriagedIssuesRoutable() []Issue {
	return []Issue{
		{Number: 1, Title: "regenerate the scorecard and rerun `fak cadence`"},
		{Number: 2, Title: "investigate the flaky worktree reap test"},
		{Number: 3, Title: "run go test ./internal/blockerpost and update docs"},
	}
}

func sampleTriagedIssuesAuthority() []Issue {
	return []Issue{
		{Number: 1, Title: "regenerate the scorecard and rerun `fak cadence`"},
		{Number: 2, Title: "approve the production release before publish", Labels: []Label{{Name: "release"}}},
		{Number: 3, Title: "grant billing authorization for lab GPU cluster", Labels: []Label{{Name: "policy"}}},
	}
}

func sampleSignatures(total, overdue, claimed int) []Signature {
	sigs := make([]Signature, total)
	for i := 0; i < total; i++ {
		sig := Signature{
			ID:             fmt.Sprintf("sha256:abcd%08x12345678", i),
			Reason:         "build-regression",
			Trees:          []string{fmt.Sprintf("internal/subsystem%d/**", i)},
			Affected:       3 + (i % 5),
			WitnessPending: true,
		}
		if i < overdue {
			sig.NoFixerOverdue = true
		} else if i < overdue+claimed {
			sig.Fixer = fmt.Sprintf("fixer-agent-%d", i)
		}
		sigs[i] = sig
	}
	return sigs
}

// Global sinks to prevent compiler dead-code elimination.
var (
	sinkText   string
	sinkBlocks []any
	sinkBlock  Blocker
	sinkSev    Severity
	sinkOK     bool
)

// BenchmarkBlockerText measures plain-text rendering for Slack fallback / notifications
// across all three severity tiers.
func BenchmarkBlockerText(b *testing.B) {
	cases := []struct {
		name    string
		blocker Blocker
	}{
		{name: "Operator", blocker: sampleBlockerOperator()},
		{name: "Status", blocker: sampleBlockerStatus()},
		{name: "Clear", blocker: sampleBlockerClear()},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkText = tc.blocker.Text()
			}
		})
	}
}

// BenchmarkBlockerBlocks measures Slack Block Kit generation across all severity tiers.
func BenchmarkBlockerBlocks(b *testing.B) {
	cases := []struct {
		name    string
		blocker Blocker
	}{
		{name: "Operator", blocker: sampleBlockerOperator()},
		{name: "Status", blocker: sampleBlockerStatus()},
		{name: "Clear", blocker: sampleBlockerClear()},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBlocks = tc.blocker.Blocks()
			}
		})
	}
}

// BenchmarkFoldIssues measures folding GitHub issue lists into a single Blocker.
func BenchmarkFoldIssues(b *testing.B) {
	repoURL := "https://github.com/anthony-chaudhary/fak"
	cases := []struct {
		name   string
		issues []Issue
	}{
		{name: "Empty", issues: nil},
		{name: "Mixed_15", issues: sampleIssues(15, 5)},
		{name: "AllOwned_15", issues: sampleIssues(15, 0)},
		{name: "LargeTruncated_50", issues: sampleIssues(50, 25)},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBlock = FoldIssues(tc.issues, "blocked", repoURL)
			}
		})
	}
}

// BenchmarkFoldIssuesTriaged measures the decenter-the-human choice triage evaluation
// during blocker backlog folding.
func BenchmarkFoldIssuesTriaged(b *testing.B) {
	repoURL := "https://github.com/anthony-chaudhary/fak"
	cases := []struct {
		name   string
		issues []Issue
	}{
		{name: "Empty", issues: nil},
		{name: "Routable_Decentered", issues: sampleTriagedIssuesRoutable()},
		{name: "Authority_Surfaced", issues: sampleTriagedIssuesAuthority()},
		{name: "MixedBatch_15", issues: sampleIssues(15, 5)},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBlock = FoldIssuesTriaged(tc.issues, "blocked", repoURL)
			}
		})
	}
}

// BenchmarkFoldBlast measures folding live known-bad signatures into blast-framed cards.
func BenchmarkFoldBlast(b *testing.B) {
	repoURL := "https://github.com/anthony-chaudhary/fak"
	cases := []struct {
		name string
		sigs []Signature
	}{
		{name: "Empty", sigs: nil},
		{name: "Overdue_Surfaced", sigs: sampleSignatures(5, 2, 3)},
		{name: "Claimed_Contained", sigs: sampleSignatures(5, 0, 5)},
		{name: "LargeTruncated_30", sigs: sampleSignatures(30, 5, 20)},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBlock = FoldBlast(tc.sigs, repoURL)
			}
		})
	}
}

// BenchmarkParseSeverity measures string severity parsing.
func BenchmarkParseSeverity(b *testing.B) {
	cases := []string{"", "status", "operator", "clear", " OPERATOR "}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range cases {
			sinkSev, sinkOK = ParseSeverity(s)
		}
	}
}
