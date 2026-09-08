package issuesolved

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"testing"
	"time"
)

func BenchmarkExtractIssueNumbers_Public(b *testing.B) {
	cases := []struct {
		subject string
		body    string
	}{
		{
			subject: "fix(core): resolve race in buffer pool (#1234)",
			body:    "Closes #1234.\nFixes #1235.\nTested on Windows and Linux.",
		},
		{
			subject: "feat(agent): support autonomous execution (fak #5678)",
			body:    "Resolves #5678\nSigned-off-by: dev",
		},
		{
			subject: "docs: update architecture guide and claims",
			body:    "No issue reference here.",
		},
		{
			subject: "refactor(gateway): split session handlers #9012 #9013",
			body:    "Resolves anthony-chaudhary/fak-private#9999\nFixes #9014",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		total := 0
		for _, c := range cases {
			nums := extractIssueNumbers(c.subject, c.body, false)
			total += len(nums)
		}
		if total == 0 {
			b.Fatal("unexpected zero extracted issues")
		}
	}
}

func BenchmarkExtractIssueNumbers_Private(b *testing.B) {
	cases := []struct {
		subject string
		body    string
	}{
		{
			subject: "fix(private): rotate credentials for cluster (#301)",
			body:    "Closes #301\nFixes fak-private#302",
		},
		{
			subject: "feat(security): boundary check anthony-chaudhary/fak-private#401",
			body:    "Resolves anthony-chaudhary/fak-private#402",
		},
		{
			subject: "chore: update private test fixtures",
			body:    "Trivial test fixture updates.",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		total := 0
		for _, c := range cases {
			nums := extractIssueNumbers(c.subject, c.body, true)
			total += len(nums)
		}
		if total == 0 {
			b.Fatal("unexpected zero extracted issues")
		}
	}
}

func BenchmarkParseTime(b *testing.B) {
	timestamps := []string{
		"2026-09-08T12:25:00.123456789Z",
		"2026-09-08T12:25:00Z",
		"2026-09-08T12:25:00+00:00",
		"2026-09-08 12:25:00",
		"2026-09-08",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range timestamps {
			t, err := parseTime(s)
			if err != nil || t.IsZero() {
				b.Fatalf("failed parsing %s: %v", s, err)
			}
		}
	}
}

func BenchmarkCollectIssuesFromGit(b *testing.B) {
	ctx := context.Background()
	since := time.Date(2026, 9, 8, 0, 0, 0, 0, time.UTC)
	until := time.Date(2026, 9, 8, 23, 59, 59, 0, time.UTC)

	var buf bytes.Buffer
	for j := 1; j <= 50; j++ {
		sha := fmt.Sprintf("sha%04d", j)
		ts := time.Date(2026, 9, 8, 12, j%60, 0, 0, time.UTC).Format(time.RFC3339)
		subj := fmt.Sprintf("fix(subsys): address problem (#%d)", 1000+j)
		body := fmt.Sprintf("Commit body with trailer\n\nFixes #%d\n(fak cmd)", 2000+j)
		buf.WriteString(fmt.Sprintf("%s\x1f%s\x1f%s\x1f%s\x1e", sha, ts, subj, body))
	}
	gitOutput := buf.Bytes()

	mockGit := func(ctx context.Context, dir string, args ...string) ([]byte, error) {
		return gitOutput, nil
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		items, err := collectIssuesFromGit(ctx, "anthony-chaudhary/fak", ".", since, until, mockGit)
		if err != nil {
			b.Fatalf("collectIssuesFromGit failed: %v", err)
		}
		if len(items) == 0 {
			b.Fatal("expected non-empty items")
		}
	}
}

func BenchmarkCollectIssuesFromGh(b *testing.B) {
	ctx := context.Background()
	since := time.Date(2026, 9, 8, 0, 0, 0, 0, time.UTC)
	until := time.Date(2026, 9, 8, 23, 59, 59, 0, time.UTC)

	type ghWire struct {
		Number      int    `json:"number"`
		Title       string `json:"title"`
		ClosedAt    string `json:"closedAt"`
		StateReason string `json:"stateReason"`
		URL         string `json:"url"`
	}
	wires := make([]ghWire, 50)
	for j := 0; j < 50; j++ {
		reason := "COMPLETED"
		if j%5 == 0 {
			reason = "NOT_PLANNED"
		}
		wires[j] = ghWire{
			Number:      100 + j,
			Title:       fmt.Sprintf("Issue title for benchmark %d", j),
			ClosedAt:    time.Date(2026, 9, 8, 10, j%60, 0, 0, time.UTC).Format(time.RFC3339),
			StateReason: reason,
			URL:         fmt.Sprintf("https://github.com/anthony-chaudhary/fak/issues/%d", 100+j),
		}
	}
	ghJSON, err := json.Marshal(wires)
	if err != nil {
		b.Fatalf("marshal wire json: %v", err)
	}

	mockGh := func(ctx context.Context, args ...string) ([]byte, error) {
		return ghJSON, nil
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		items, err := collectIssuesFromGh(ctx, "anthony-chaudhary/fak", since, until, mockGh)
		if err != nil {
			b.Fatalf("collectIssuesFromGh failed: %v", err)
		}
		if len(items) != 50 {
			b.Fatalf("expected 50 items, got %d", len(items))
		}
	}
}

func makeSampleReport() *Report {
	since := time.Date(2026, 9, 8, 0, 0, 0, 0, time.UTC)
	until := time.Date(2026, 9, 8, 23, 59, 59, 0, time.UTC)

	issuesPub := make([]IssueItem, 30)
	for j := 0; j < 30; j++ {
		reason := "COMPLETED"
		if j%6 == 0 {
			reason = "NOT_PLANNED"
		}
		issuesPub[j] = IssueItem{
			Number:      1000 + j,
			Title:       fmt.Sprintf("feat(engine): benchmark item %d", j),
			ClosedAt:    since.Add(time.Duration(j) * 30 * time.Minute),
			StateReason: reason,
			URL:         fmt.Sprintf("https://github.com/anthony-chaudhary/fak/issues/%d", 1000+j),
		}
	}

	issuesPriv := make([]IssueItem, 15)
	for j := 0; j < 15; j++ {
		reason := "COMPLETED"
		if j%4 == 0 {
			reason = "NOT_PLANNED"
		}
		issuesPriv[j] = IssueItem{
			Number:      2000 + j,
			Title:       fmt.Sprintf("fix(private): cluster security item %d", j),
			ClosedAt:    since.Add(time.Duration(j) * 45 * time.Minute),
			StateReason: reason,
			URL:         fmt.Sprintf("https://github.com/anthony-chaudhary/fak-private/issues/%d", 2000+j),
		}
	}

	return &Report{
		Schema:          Schema,
		GeneratedAt:     until,
		Since:           since,
		Until:           until,
		Hours:           24,
		TotalSolved:     38,
		TotalClosed:     45,
		TotalNotPlanned: 7,
		TotalCommits:    120,
		Repos: []RepoReport{
			{
				Repo:        "anthony-chaudhary/fak",
				Path:        ".",
				Source:      "github",
				Solved:      25,
				Closed:      30,
				NotPlanned:  5,
				CommitCount: 95,
				Issues:      issuesPub,
			},
			{
				Repo:        "anthony-chaudhary/fak-private",
				Path:        "../fak-private",
				Source:      "github",
				Solved:      13,
				Closed:      15,
				NotPlanned:  2,
				CommitCount: 25,
				Issues:      issuesPriv,
			},
		},
	}
}

func BenchmarkRenderSummary(b *testing.B) {
	rep := makeSampleReport()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := RenderSummary(io.Discard, rep); err != nil {
			b.Fatalf("RenderSummary failed: %v", err)
		}
	}
}

func BenchmarkRenderDetailed(b *testing.B) {
	rep := makeSampleReport()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := RenderDetailed(io.Discard, rep); err != nil {
			b.Fatalf("RenderDetailed failed: %v", err)
		}
	}
}

func BenchmarkRenderJSON(b *testing.B) {
	rep := makeSampleReport()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := RenderJSON(io.Discard, rep); err != nil {
			b.Fatalf("RenderJSON failed: %v", err)
		}
	}
}

func BenchmarkCollectPipeline(b *testing.B) {
	fixedNow := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	ctx := context.Background()

	mockGh := func(ctx context.Context, args ...string) ([]byte, error) {
		issues := []map[string]any{
			{
				"number":      101,
				"title":       "fix(agent): benchmark test issue 1",
				"closedAt":    "2026-09-08T10:00:00Z",
				"stateReason": "COMPLETED",
				"url":         "https://github.com/anthony-chaudhary/fak/issues/101",
			},
			{
				"number":      102,
				"title":       "chore: benchmark test issue 2",
				"closedAt":    "2026-09-08T09:00:00Z",
				"stateReason": "NOT_PLANNED",
				"url":         "https://github.com/anthony-chaudhary/fak/issues/102",
			},
		}
		return json.Marshal(issues)
	}

	mockGit := func(ctx context.Context, dir string, args ...string) ([]byte, error) {
		return []byte("42\n"), nil
	}

	opts := Options{
		Hours:          24,
		Now:            fixedNow,
		IncludePrivate: true,
		Source:         "github",
		GhExecutor:     mockGh,
		GitExecutor:    mockGit,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep, err := Collect(ctx, opts)
		if err != nil {
			b.Fatalf("Collect failed: %v", err)
		}
		if rep.TotalClosed == 0 {
			b.Fatal("expected closed issues in report")
		}
	}
}
