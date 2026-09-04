package issuesmallness

import (
	"fmt"
	"testing"
)

func BenchmarkIssueSmallness(b *testing.B) {
	issues := benchmarkIssueCorpus()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := ReportOpen(issues)
		if report.Scanned != len(issues) {
			b.Fatalf("expected %d scanned issues, got %d", len(issues), report.Scanned)
		}
	}
}

func BenchmarkLintBody(b *testing.B) {
	cases := []struct {
		name string
		body string
	}{
		{"SingleDeliverable", singleDeliverableBody},
		{"TwoDeliverables", twoDeliverableBody},
		{"ThreeUnrelated", threeUnrelatedTasksBody},
		{"ProseClauses", proseThreeTasksBody},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = LintBody(tc.body)
			}
		})
	}
}

func BenchmarkFindDeliverables(b *testing.B) {
	bulletSection := "- Fix login redirect bug\n- Add throughput dashboard\n- Rewrite onboarding docs"
	proseSection := "Fix login redirect bug; add throughput dashboard; and then rewrite onboarding docs"

	b.Run("Bullets", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = FindDeliverables(bulletSection)
		}
	})

	b.Run("ProseClauses", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = FindDeliverables(proseSection)
		}
	})
}

func TestBenchmarkIssueSmallnessCorpus(t *testing.T) {
	issues := benchmarkIssueCorpus()
	report := ReportOpen(issues)
	if report.Scanned != len(issues) {
		t.Fatalf("Scanned = %d, want %d", report.Scanned, len(issues))
	}
	if !HasFailReport(report) {
		t.Fatal("expected benchmark corpus to have failing reports")
	}
	if report.Counts[Pass] == 0 || report.Counts[Warn] == 0 || report.Counts[Fail] == 0 {
		t.Fatalf("expected all verdicts represented, got counts: %+v", report.Counts)
	}
}

func benchmarkIssueCorpus() []Issue {
	corpus := []Issue{
		{Number: 1, Title: "Single Deliverable", Body: singleDeliverableBody},
		{Number: 2, Title: "Two Deliverables", Body: twoDeliverableBody},
		{Number: 3, Title: "Three Tasks", Body: threeUnrelatedTasksBody},
		{Number: 4, Title: "Prose Tasks", Body: proseThreeTasksBody},
		{Number: 5, Title: "Missing Witness", Body: "## Goal\nAdd a retry counter.\n"},
	}

	var expanded []Issue
	for i := 0; i < 5; i++ {
		for _, issue := range corpus {
			num := len(expanded) + 1
			expanded = append(expanded, Issue{
				Number: num,
				Title:  fmt.Sprintf("%s #%d", issue.Title, num),
				Body:   issue.Body,
			})
		}
	}
	return expanded
}
