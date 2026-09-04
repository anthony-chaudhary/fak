package learningdebt

import (
	"fmt"
	"testing"
)

func makeBenchmarkPayload(docCount, defectsPerDoc int) map[string]any {
	priorities := make([]any, 0, docCount)
	docs := make([]any, 0, docCount)
	classes := []string{"orientation", "runnable", "worked", "clarity", "orphan"}
	for i := 0; i < docCount; i++ {
		p := fmt.Sprintf("docs/learning/module_%d.md", i)
		priorities = append(priorities, map[string]any{
			"path":     p,
			"priority": float64(docCount-i) * 0.25,
		})
		defects := make([]any, 0, defectsPerDoc)
		for j := 0; j < defectsPerDoc; j++ {
			cls := classes[j%len(classes)]
			defects = append(defects, fmt.Sprintf("%s: finding in module_%d defect_%d", cls, i, j))
		}
		docs = append(docs, map[string]any{
			"path":    p,
			"score":   float64(50 + (i % 50)),
			"grade":   "C",
			"defects": defects,
		})
	}
	coverageDefects := []any{
		"orphan lesson (unreachable from any front door): docs/learning/hidden.md",
		"uncovered learning topic: distributed-cache",
		"uncovered learning topic: zero-copy-io",
	}
	return map[string]any{
		"schema":  "fleet-learning-scorecard/1",
		"verdict": "ACTION",
		"corpus": map[string]any{
			"priorities": priorities,
		},
		"docs": docs,
		"coverage": map[string]any{
			"defects": coverageDefects,
		},
		"stamp_freshness": map[string]any{
			"stale_stamp": true,
			"flag":        "stale-stamp",
			"doc":         "docs/LEARNING-SCORECARD.md",
			"reason":      "stale-stamp: docs/LEARNING-SCORECARD.md stamp is 35d old",
		},
	}
}

// BenchmarkLearningDebt measures the end-to-end throughput of defect extraction,
// plan construction, issue body generation, and marker key parsing in a b.N loop.
func BenchmarkLearningDebt(b *testing.B) {
	payload := makeBenchmarkPayload(20, 4)
	seen := SeenCache{
		Schema: SeenSchema,
		Seen: map[string]SeenRecord{
			"learning-debt/orientation/cached1": {
				FiledAt: "2026-09-01T00:00:00Z",
				Doc:     "docs/learning/module_0.md",
				Class:   "orientation",
			},
		},
	}
	existing := []Issue{
		{Number: 101, Body: "<!-- fak-learning-debt-key: learning-debt/runnable/existing1 -->"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		defects := ExtractDefects(payload)
		plan, stats := BuildPlan(defects, seen, existing, 10, "scorecard.json")
		if stats.TotalDefects == 0 || len(plan) == 0 {
			b.Fatal("expected defects and plan items")
		}
		for _, row := range plan {
			if k := MarkerKey(row.Body); k == "" {
				b.Fatal("missing marker key in planned body")
			}
		}
	}
}

// BenchmarkExtractDefects measures payload parsing, classification, and priority-sorted defect ranking.
func BenchmarkExtractDefects(b *testing.B) {
	for _, size := range []int{5, 25, 100} {
		b.Run(fmt.Sprintf("Docs_%d", size), func(b *testing.B) {
			payload := makeBenchmarkPayload(size, 4)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				defects := ExtractDefects(payload)
				if len(defects) == 0 {
					b.Fatal("expected non-empty defects")
				}
			}
		})
	}
}

// BenchmarkBuildPlan measures plan computation across seen-cache and issue-marker deduplication structures.
func BenchmarkBuildPlan(b *testing.B) {
	payload := makeBenchmarkPayload(30, 4)
	defects := ExtractDefects(payload)
	seen := SeenCache{
		Schema: SeenSchema,
		Seen:   make(map[string]SeenRecord, len(defects)/3),
	}
	for i := 0; i < len(defects)/3; i++ {
		seen.Seen[defects[i].Key] = SeenRecord{
			FiledAt: "2026-09-01T00:00:00Z",
			Doc:     defects[i].Doc,
			Class:   defects[i].Class,
		}
	}
	existing := make([]Issue, 0, len(defects)/4)
	for i := len(defects) / 3; i < len(defects)/3+len(defects)/4; i++ {
		existing = append(existing, Issue{
			Number: 100 + i,
			Body:   fmt.Sprintf("<!-- fak-learning-debt-key: %s -->\n", defects[i].Key),
		})
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan, stats := BuildPlan(defects, seen, existing, 15, "scorecard.json")
		if stats.Planned != len(plan) {
			b.Fatalf("plan count mismatch: %d != %d", stats.Planned, len(plan))
		}
	}
}

// BenchmarkIssueBody measures markdown formatting and HTML comment marker generation.
func BenchmarkIssueBody(b *testing.B) {
	defect := Defect{
		Key:    "learning-debt/orientation/abcdef0123456789",
		Doc:    "docs/learning/module_1.md",
		Class:  "orientation",
		Exact:  "orientation: missing prerequisites and learning goals",
		Source: "docs",
		Score:  "42.0",
		Grade:  "F",
		Rank:   1,
		Prio:   3.5,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		body := IssueBody(defect, "scorecard.json")
		if len(body) == 0 {
			b.Fatal("unexpected empty body")
		}
	}
}

// BenchmarkMarkerKey measures key extraction using compiled regular expressions.
func BenchmarkMarkerKey(b *testing.B) {
	body := "<!-- fak-learning-debt-key: learning-debt/orientation/abcdef0123456789 -->\n# Learning-debt triage\n"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := MarkerKey(body)
		if key != "learning-debt/orientation/abcdef0123456789" {
			b.Fatalf("unexpected key: %s", key)
		}
	}
}

// TestBenchmarkLearningDebtExecution verifies that BenchmarkLearningDebt executes cleanly.
func TestBenchmarkLearningDebtExecution(t *testing.T) {
	res := testing.Benchmark(BenchmarkLearningDebt)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
