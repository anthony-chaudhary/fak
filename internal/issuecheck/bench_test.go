package issuecheck

import (
	"fmt"
	"testing"
)

func TestBenchmarkIssueCheckSanity(t *testing.T) {
	issue := benchIssue()
	review := benchReview(t, issue)
	if err := ValidateReview(issue, review); err != nil {
		t.Fatalf("sanity review validation failed: %v", err)
	}
}

func BenchmarkIssueCheck(b *testing.B) {
	issue := benchIssue()
	review := benchReview(b, issue)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := ValidateReview(issue, review); err != nil {
			b.Fatalf("ValidateReview failed: %v", err)
		}
	}
}

func benchIssue() Issue {
	return Issue{
		Number: 9568,
		Title:  "Require a Top-5 issue review",
		Body:   "## Scope\nPost one durable review before editing.\n\n## Witness\nRead the comment back.",
		Labels: []string{"gen/now", "dev-ex"},
	}
}

func benchReview(tb testing.TB, issue Issue) Review {
	tb.Helper()
	digest, err := IssueDigest(issue)
	if err != nil {
		tb.Fatalf("digest issue: %v", err)
	}
	rows := make([]ReviewRow, RequiredReviewRows)
	for i := range rows {
		rows[i] = ReviewRow{
			ID:         fmt.Sprintf("TC-%02d", i+1),
			Relevance:  fmt.Sprintf("Issue #%d changes the worker admission path at row %d", issue.Number, i+1),
			Assessment: fmt.Sprintf("The issue names a bounded behavior and witness for risk %d", i+1),
			Evidence:   Evidence{Status: EvidenceSupported, Refs: []string{"body:## Scope", "body:## Witness"}},
			Action:     fmt.Sprintf("Keep acceptance criterion %d in the focused test", i+1),
		}
	}
	return Review{
		Schema:          ReviewSchema,
		IssueNumber:     issue.Number,
		IssueDigest:     digest,
		IssueBinding:    CanonicalIssueBinding(issue),
		CatalogVersion:  CatalogVersion,
		ReviewerVersion: "codex/gpt-5.6@2026-08-27",
		Rows:            rows,
	}
}
