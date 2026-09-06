package blockerpost

import (
	"fmt"
	"testing"
)

// TestBenchmarkSanity ensures benchmark fixtures are valid and exercisable
// across all production severity paths and folding operations before
// benchmark iterations run. It verifies that sample data sets (operator, status,
// clear blockers, issue lists, triaged sets, and blast signatures) produce
// valid Slack text and block payloads without runtime panic, ensuring all
// benchmark loops run against vetted non-nil structures.
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

// sampleBlockerOperator constructs an operator-tier Blocker fixture containing
// actionable routing instructions, an urgent callout owner, and reference links
// representative of high-priority infrastructure outages. This ensures that
// benchmarks evaluating operator-level escalation paths measure realistic
// payloads including multi-line diagnostics and runbook URLs.
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

// sampleBlockerStatus constructs an advisory status-tier Blocker fixture
// indicating deferred or gated operations that do not require immediate manual
// human intervention. Used to benchmark muted status cards that report queue
// backpressure or waiting states without paging on-call operators.
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

// sampleBlockerClear constructs a clear-tier Blocker fixture representing an
// empty blocker backlog with all operational conditions satisfied. Used to
// measure the minimal rendering path when no blocking issues or failing
// signatures require attention.
func sampleBlockerClear() Blocker {
	return Blocker{
		Severity: SeverityClear,
		Title:    "no standing blockers",
		Detail:   "0 open blocked issues — the board is clear.",
		Ref:      "label:blocked",
	}
}

// sampleIssues generates a synthetic slice of GitHub issues configured with
// labels and assignees to simulate realistic triage ratios between claimed
// work and unassigned blocking items. The total parameter sets the issue count,
// while unowned specifies how many issues lack assignees to trigger operator
// severity escalation during folding.
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

// sampleTriagedIssuesRoutable produces an issue slice where titles and labels
// qualify for autonomous worker routing without requiring human escalation.
// This benchmark fixture exercises the decenter-the-human choice path where
// routine maintenance tasks are filtered to background status cards.
func sampleTriagedIssuesRoutable() []Issue {
	return []Issue{
		{Number: 1, Title: "regenerate the scorecard and rerun `fak cadence`"},
		{Number: 2, Title: "investigate the flaky worktree reap test"},
		{Number: 3, Title: "run go test ./internal/blockerpost and update docs"},
	}
}

// sampleTriagedIssuesAuthority produces an issue slice containing items tagged
// with release and policy markers that require elevated human authorization.
// This fixture verifies that the triage engine correctly surfaces authority-gated
// decisions to operator severity despite de-centering heuristics.
func sampleTriagedIssuesAuthority() []Issue {
	return []Issue{
		{Number: 1, Title: "regenerate the scorecard and rerun `fak cadence`"},
		{Number: 2, Title: "approve the production release before publish", Labels: []Label{{Name: "release"}}},
		{Number: 3, Title: "grant billing authorization for lab GPU cluster", Labels: []Label{{Name: "policy"}}},
	}
}

// sampleSignatures generates synthetic known-bad regression signatures with
// parameterized overdue thresholds and fixer claim statuses for blast card folding.
// Overdue signatures trigger operator severity, claimed signatures test fixer
// attribution display, and excess entries verify truncation bounds.
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

// Global sinks to prevent compiler dead-code elimination across benchmark runs.
// Storing results into package-level sinks guarantees that the Go compiler
// does not optimize away pure function calls or intermediate data structures.
var (
	sinkText   string
	sinkBlocks []any
	sinkBlock  Blocker
	sinkSev    Severity
	sinkOK     bool
)

// BenchmarkSanity exercises baseline fixture creation, card folding, and payload
// generation in an end-to-end composite loop to catch holistic allocation regressions.
// It runs the full lifecycle from raw issue data to final Slack block structures
// to establish an integrated performance baseline for the blockerpost package.
func BenchmarkSanity(b *testing.B) {
	repoURL := "https://github.com/anthony-chaudhary/fak"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBlock = FoldIssues(sampleIssues(15, 5), "blocked", repoURL)
		sinkText = sinkBlock.Text()
		sinkBlocks = sinkBlock.Blocks()
	}
}

// BenchmarkBlockerText measures plain-text rendering throughput and allocation
// counts for Slack fallback notifications across all three severity tiers
// (Operator, Status, and Clear). This path is critical for terminal feeds,
// log scrapers, and notification digests.
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

// BenchmarkBlockerBlocks measures Slack Block Kit JSON-compatible payload
// generation across operator, status, and clear severity tiers. This benchmarks
// header construction, section division, and contextual markdown formatting
// sent to the Slack Webhook API.
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

// BenchmarkFoldIssues measures folding varying quantities of GitHub issues
// into a unified Blocker card, evaluating empty, mixed unowned/owned, fully owned,
// and large truncated issue sets to profile memory allocations and slice growth.
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
// during blocker backlog folding. It benchmarks classification throughput across
// routable tasks, human-authority gates, and heterogeneous issue batches.
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
// Evaluates empty sets, overdue uncontained regressions requiring operator action,
// active fixer containment, and signature list truncation under large loads.
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

// BenchmarkParseSeverity measures string severity parsing throughput across
// canonical, empty, unknown, and whitespace-padded severity tokens, ensuring
// lookup and normalization overhead remains minimal.
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
