package issuecatalog

import (
	"encoding/json"
	"fmt"
	"testing"
)

func benchRows(n int) []Row {
	rows := make([]Row, n)
	for i := 0; i < n; i++ {
		r := completeRow()
		r.Key = fmt.Sprintf("perf/cache/bench-%d", i)
		r.Title = fmt.Sprintf("bench issue %d", i)
		r.Paths = []string{fmt.Sprintf("internal/pkg%d/file.go", i)}
		r.Lane = fmt.Sprintf("lane-%d", i%3)
		rows[i] = r
	}
	return rows
}

func TestBenchmarkIntegrity(t *testing.T) {
	rows := benchRows(10)
	plan := CohortPlan(rows, Options{})
	if len(plan.Waves) == 0 {
		t.Fatalf("expected non-empty cohort plan waves")
	}

	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	parsed, err := ParseCatalog(raw)
	if err != nil {
		t.Fatalf("ParseCatalog failed: %v", err)
	}
	if len(parsed) != len(rows) {
		t.Fatalf("expected %d rows, got %d", len(rows), len(parsed))
	}
}

func BenchmarkIssueCatalog(b *testing.B) {
	rows := benchRows(20)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		plan := CohortPlan(rows, Options{})
		if len(plan.Waves) == 0 {
			b.Fatalf("CohortPlan produced empty waves")
		}
	}
}

func BenchmarkParseCatalog(b *testing.B) {
	rows := benchRows(20)
	raw, err := json.Marshal(rows)
	if err != nil {
		b.Fatalf("marshal failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		parsed, err := ParseCatalog(raw)
		if err != nil || len(parsed) != len(rows) {
			b.Fatalf("ParseCatalog failed: %v", err)
		}
	}
}
