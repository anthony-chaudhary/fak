package programreport

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/worktype"
)

// BenchmarkReportGeneration measures end-to-end report generation: signal interpretation,
// folding into envelope, attaching historical trend, and rendering formatted output.
func BenchmarkReportGeneration(b *testing.B) {
	signals := []Signal{
		{Class: worktype.KernelOptimization, Label: "kernel-optimization", Frontier: "perf work landing", Metric: 5, Direction: "advancing", Activity: 5, Window: "7d", OK: true},
		{Class: worktype.CacheOptimization, Label: "cache-optimization", Frontier: "realized reuse 0.650 -> 0.680", Metric: 0.68, Direction: "advancing", OK: true, Note: "marginal-over-tuned-warm-KV"},
		{Class: worktype.HumanOperatorEffectiveness, Label: "human-operator-effectiveness", Frontier: "operator-heaviness pressure 10; lightness 0.900", Metric: 0.90, Direction: "holding", OK: true},
	}
	opts := FoldOpts{
		Workspace:   "/fak",
		Commit:      "c0ffee123456",
		GeneratedAt: "2026-09-05T12:00:00Z",
		Date:        "2026-09-05",
	}
	prior := []LedgerRow{
		{
			Date:         "2026-09-04",
			Commit:       "deadbeef0000",
			GeneratedAt:  "2026-09-04T12:00:00Z",
			Verdict:      "OK",
			Tracked:      3,
			Measured:     3,
			Advancing:    2,
			KernelMetric: 4,
			CacheMetric:  0.65,
			HumanMetric:  0.90,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := InterpretPrograms(signals)
		r := Fold(p, opts)
		trend := TrendVsLast(RowFromReport(r), prior)
		r = r.WithTrend(trend)
		out := Render(r)
		if len(out) == 0 || !r.OK {
			b.Fatal("unexpected empty render or failed report")
		}
	}
}

// BenchmarkStatusAggregation measures parsing, projection, and serialization of
// durable JSONL ledger history across a multi-tick window.
func BenchmarkStatusAggregation(b *testing.B) {
	var builder strings.Builder
	for i := 0; i < 50; i++ {
		row := LedgerRow{
			Schema:       LedgerSchema,
			Date:         fmt.Sprintf("2026-08-%02d", (i%30)+1),
			Commit:       fmt.Sprintf("sha%04d", i),
			GeneratedAt:  fmt.Sprintf("2026-08-%02dT12:00:00Z", (i%30)+1),
			Verdict:      "OK",
			Tracked:      3,
			Measured:     3,
			Advancing:    1 + (i % 2),
			KernelMetric: float64(2 + (i % 5)),
			CacheMetric:  0.50 + float64(i%20)*0.01,
			HumanMetric:  0.80 + float64(i%10)*0.01,
		}
		line, err := AppendLedgerLine(row)
		if err != nil {
			b.Fatalf("failed to build seed ledger: %v", err)
		}
		builder.WriteString(line)
		builder.WriteByte('\n')
	}
	rawLedger := builder.String()

	p := InterpretPrograms([]Signal{
		kernelAdvancing(),
		cacheHolding(),
		humanHolding(),
	})
	report := Fold(p, FoldOpts{Date: "2026-09-05", Commit: "head", GeneratedAt: "2026-09-05T00:00:00Z"})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows := ParseLedger(rawLedger)
		if len(rows) != 50 {
			b.Fatalf("expected 50 rows, got %d", len(rows))
		}
		currentRow := RowFromReport(report)
		line, err := AppendLedgerLine(currentRow)
		if err != nil || len(line) == 0 {
			b.Fatalf("append line failed: %v", err)
		}
	}
}

// BenchmarkCadenceEvaluation measures historical trend analysis and gate validation
// across prior ledger ticks.
func BenchmarkCadenceEvaluation(b *testing.B) {
	prior := make([]LedgerRow, 100)
	for i := 0; i < 100; i++ {
		prior[i] = LedgerRow{
			Schema:       LedgerSchema,
			Date:         fmt.Sprintf("2026-05-%02d", (i%28)+1),
			Commit:       fmt.Sprintf("commit%04d", i),
			GeneratedAt:  fmt.Sprintf("2026-05-%02dT10:00:00Z", (i%28)+1),
			Verdict:      "OK",
			Tracked:      3,
			Measured:     3,
			Advancing:    2,
			KernelMetric: float64(i % 10),
			CacheMetric:  0.60 + float64(i%20)*0.01,
			HumanMetric:  0.85,
		}
	}
	current := LedgerRow{
		Schema:       LedgerSchema,
		Date:         "2026-06-01",
		Commit:       "headcommit",
		GeneratedAt:  "2026-06-01T10:00:00Z",
		Verdict:      "OK",
		Tracked:      3,
		Measured:     3,
		Advancing:    3,
		KernelMetric: 12,
		CacheMetric:  0.82,
		HumanMetric:  0.95,
	}
	p := InterpretPrograms([]Signal{
		kernelAdvancing(),
		cacheHolding(),
		humanHolding(),
	})
	r := Fold(p, FoldOpts{Date: "2026-06-01", Commit: "headcommit"})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tr := TrendVsLast(current, prior)
		if tr.Direction == "" {
			b.Fatal("unexpected empty trend direction")
		}
		code, _ := CheckGate(r)
		if code != 0 {
			b.Fatalf("check gate failed: %d", code)
		}
		tCode, _ := CheckGateTriaged(r, false)
		if tCode != 0 {
			b.Fatalf("check gate triaged failed: %d", tCode)
		}
	}
}

// BenchmarkReviewRubricEvaluation measures anchored expert-review rubric grading,
// rater consensus computation, and review gate decisions.
func BenchmarkReviewRubricEvaluation(b *testing.B) {
	c := ReviewCase{
		Schema:     ReviewSchema,
		ID:         "REV-bench",
		Subject:    "Kernel throughput rose +12%; cache-value ledger shows +20% reuse; human-operator lightness holding at 0.900.",
		Provenance: validProvenance(),
		Tier:       TierNightly,
		CostNote:   "two raters, ~5 min benchmark load",
		Raters: []RaterScores{
			scores("rater-1", 4, 4, 5, 4, 4, 4),
			scores("rater-2", 5, 4, 4, 4, 5, 4),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Review(c)
		if !res.Pass {
			b.Fatalf("expected review pass, got: %+v", res)
		}
		code, _ := CheckReviewGate(res)
		if code != 0 {
			b.Fatalf("expected gate exit 0, got %d", code)
		}
	}
}

// BenchmarkJudgeValidation measures evaluation of LLM judge candidates across
// position, repeatability, verbosity, bias, correlation, and escalation probes.
func BenchmarkJudgeValidation(b *testing.B) {
	jc := validJudgeCase()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := ReviewJudge(jc)
		if !res.Pass {
			b.Fatalf("expected judge validation pass, got: %+v", res)
		}
		code, _ := CheckJudgeGate(res)
		if code != 0 {
			b.Fatalf("expected judge gate exit 0, got %d", code)
		}
	}
}

// BenchmarkDogfoodCorpusGrading measures batch grading and verification of
// representative dogfood cases across all operational tiers.
func BenchmarkDogfoodCorpusGrading(b *testing.B) {
	corpus := SeedDogfoodCorpus()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := GradeDogfoodCorpus(corpus)
		if !res.Pass {
			b.Fatalf("expected dogfood corpus grade pass, got: %+v", res)
		}
		code, _ := CheckDogfoodGate(res)
		if code != 0 {
			b.Fatalf("expected dogfood gate exit 0, got %d", code)
		}
	}
}
