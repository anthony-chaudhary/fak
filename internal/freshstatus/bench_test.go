package freshstatus

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

var (
	benchSinkString  string
	benchSinkBool    bool
	benchSinkPayload Payload
	benchSinkPane    map[string]any
	benchSinkCounts  map[string]int
	benchSinkStrings []string
	benchSinkTime    time.Time
)

// BenchmarkClassify evaluates provenance classification across the complete 6-rule ladder.
func BenchmarkClassify(b *testing.B) {
	runs := []map[string]any{
		{"provenance": "measured"},
		{"artifact_engines": []string{"fak-in-kernel"}},
		{"peak_tok_per_sec": 142.5},
		{"tags": []any{"radix-benchmark"}},
		{"tags": []any{"turn-tax"}},
		{"tags": []any{"agent-live"}},
		{"run_id": "cpu-q8-parity-test"},
		{"run_id": "fanbench-run-01"},
		{"run_id": "parity-arm64"},
		{"run_id": "unclassified-run-random"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, r := range runs {
			benchSinkString = Classify(r)
		}
	}
}

// BenchmarkClassifyAll evaluates batch provenance classification across 100 diverse runs.
func BenchmarkClassifyAll(b *testing.B) {
	runs := make([]map[string]any, 100)
	for j := 0; j < 100; j++ {
		switch j % 4 {
		case 0:
			runs[j] = map[string]any{"tags": []any{"cuda", "decode"}, "peak_tok_per_sec": 88.0}
		case 1:
			runs[j] = map[string]any{"tags": []any{"fanout"}, "run_id": fmt.Sprintf("fleet-writeheavy-%d", j)}
		case 2:
			runs[j] = map[string]any{"tags": []any{"subsystem-checks"}, "run_id": fmt.Sprintf("subsystem-checks-%d", j)}
		default:
			runs[j] = map[string]any{"run_id": fmt.Sprintf("custom-eval-%d", j)}
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkCounts = ClassifyAll(runs)
	}
}

// BenchmarkParseRunTS evaluates timestamp parsing across compact, RFC3339, RFC3339Nano, and ISO formats.
func BenchmarkParseRunTS(b *testing.B) {
	samples := []string{
		"20260625T050017Z",
		"2026-06-25T05:00:17Z",
		"2026-06-25T05:00:17.123456789Z",
		"2026-06-25 05:00:17",
		"invalid-timestamp-format",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			t, ok := ParseRunTS(s)
			benchSinkTime = t
			benchSinkBool = ok
		}
	}
}

// BenchmarkFoldBenchmarks measures folding a benchmark catalog with 50 runs and 3 machines into a pane.
func BenchmarkFoldBenchmarks(b *testing.B) {
	now := time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC)
	runs := make([]any, 50)
	for j := 0; j < 50; j++ {
		runs[j] = map[string]any{
			"timestamp": "20260625T050017Z",
			"model":     "qwen",
			"tags":      []any{"radix-benchmark", "cuda"},
			"run_id":    fmt.Sprintf("radix-run-%d", j),
		}
	}
	cat := map[string]any{
		"runs": runs,
		"machines": map[string]any{
			"m0": map[string]any{},
			"m1": map[string]any{},
			"m2": map[string]any{},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPane = FoldBenchmarks(cat, now, StaleDays)
	}
}

// BenchmarkFoldWork measures folding a plan_audit payload into the work pane.
func BenchmarkFoldWork(b *testing.B) {
	payload := map[string]any{
		"counts": map[string]any{
			"total_plans": 12,
			"shipped":     8,
			"remaining":   4,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPane = FoldWork(payload, "")
	}
}

// BenchmarkFoldIndustry measures folding an industry_scorecard payload into the industry pane.
func BenchmarkFoldIndustry(b *testing.B) {
	payload := map[string]any{
		"corpus": map[string]any{
			"parity_debt": 2,
			"grade":       "A-",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPane = FoldIndustry(payload, "")
	}
}

// BenchmarkDirtyPathsFromPorcelain evaluates porcelain parsing performance on dirty paths.
func BenchmarkDirtyPathsFromPorcelain(b *testing.B) {
	porcelain := strings.Join([]string{
		" M internal/freshstatus/freshstatus.go",
		"M  cmd/fak/main.go",
		"?? internal/freshstatus/bench_test.go",
		"?? \"path with spaces/file.txt\"",
		"R  old/path.go -> new/path.go",
		"D  deleted/file.go",
		"A  added/feature.go",
		"?? .dos/scratch/temp.json",
	}, "\n")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStrings = DirtyPathsFromPorcelain(porcelain)
	}
}

// BenchmarkGitPaneWithRunner evaluates Git pane assembly and lag calculations using a mock runner.
func BenchmarkGitPaneWithRunner(b *testing.B) {
	now := time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC)
	mockRunner := func(root string, args ...string) (string, error) {
		if len(args) == 0 {
			return "", nil
		}
		switch args[0] {
		case "rev-parse":
			if len(args) > 1 && args[1] == "--short" {
				return "c0ffee1", nil
			}
			return "main", nil
		case "status":
			return " M file1.go\n?? file2.go\n", nil
		case "rev-list":
			return "0 2", nil
		case "log":
			return fmt.Sprintf("%d\n%d", now.Unix()-300, now.Unix()-150), nil
		}
		return "", nil
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPane = GitPaneWithRunner("dummy-root", mockRunner, now, DefaultPushLagActionSeconds, DefaultDirtyLagActionSeconds)
	}
}

// BenchmarkFold evaluates aggregation of domain panes into the top-level Payload envelope.
func BenchmarkFold(b *testing.B) {
	panes := []map[string]any{
		{
			"key":     "git",
			"label":   "git",
			"ok":      true,
			"verdict": "OK",
			"reason":  "c0ffee1 (main), clean tree, +0/-0 vs upstream",
		},
		{
			"key":        "benchmarks",
			"label":      "benchmarks",
			"ok":         true,
			"verdict":    "OK",
			"reason":     "42 runs / 3 machines; newest 20260625T050017Z (0.3d ago)",
			"provenance": map[string]int{"measured": 30, "modeled": 5, "functional": 7, "unknown": 0},
		},
		{
			"key":         "work",
			"label":       "work",
			"ok":          true,
			"verdict":     "OK",
			"reason":      "10 plans, 8 shipped, 2 remaining",
			"total_plans": 10,
		},
		{
			"key":         "industry",
			"label":       "industry",
			"ok":          true,
			"verdict":     "OK",
			"reason":      "parity-debt 0 vs SOTA, grade A+",
			"parity_debt": 0,
		},
	}
	workspace := "/workspace/fak"
	commit := "c0ffee1"
	genAt := "2026-06-25T12:00:00Z"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPayload = Fold(panes, workspace, commit, genAt)
	}
}

// BenchmarkRender measures formatting a Payload into human CLI rollup text.
func BenchmarkRender(b *testing.B) {
	payload := Payload{
		Schema:      Schema,
		OK:          true,
		Verdict:     "OK",
		Finding:     "all_green",
		Reason:      "git: clean; benchmarks: fresh; work: 10 plans; industry: grade A",
		NextAction:  "rollup is green; nothing required",
		Workspace:   "/workspace/fak",
		Commit:      "c0ffee1",
		GeneratedAt: "2026-06-25T12:00:00Z",
		Panes: map[string]map[string]any{
			"git": {
				"label":   "git",
				"verdict": "OK",
				"reason":  "c0ffee1 (main), clean tree",
			},
			"benchmarks": {
				"label":   "benchmarks",
				"verdict": "OK",
				"reason":  "42 runs / 3 machines",
			},
			"work": {
				"label":   "work",
				"verdict": "OK",
				"reason":  "10 plans, 8 shipped, 2 remaining",
			},
			"industry": {
				"label":   "industry",
				"verdict": "OK",
				"reason":  "parity-debt 0 vs SOTA, grade A",
			},
		},
		PaneOrder: []string{"git", "benchmarks", "work", "industry"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = Render(payload)
	}
}

// BenchmarkRenderDoc measures Markdown documentation snapshot rendering from Payload.
func BenchmarkRenderDoc(b *testing.B) {
	payload := Payload{
		Schema:      Schema,
		OK:          true,
		Verdict:     "OK",
		Finding:     "all_green",
		Reason:      "all systems green",
		NextAction:  "rollup is green; nothing required",
		Workspace:   "/workspace/fak",
		Commit:      "c0ffee1",
		GeneratedAt: "2026-06-25T12:00:00Z",
		Panes: map[string]map[string]any{
			"git": {
				"label":   "git",
				"verdict": "OK",
				"reason":  "c0ffee1 (main), clean tree",
				"sha":     "c0ffee1",
				"branch":  "main",
				"dirty":   0,
				"ahead":   0,
				"behind":  0,
			},
			"benchmarks": {
				"label":      "benchmarks",
				"verdict":    "OK",
				"reason":     "42 runs / 3 machines",
				"runs":       42,
				"machines":   3,
				"newest":     "20260625T050017Z",
				"age_days":   0.3,
				"provenance": map[string]int{"measured": 30, "modeled": 5, "functional": 7, "unknown": 0},
			},
			"work": {
				"label":   "work",
				"verdict": "OK",
				"reason":  "10 plans, 8 shipped, 2 remaining",
			},
			"industry": {
				"label":   "industry",
				"verdict": "OK",
				"reason":  "parity-debt 0 vs SOTA, grade A",
			},
		},
		PaneOrder: []string{"git", "benchmarks", "work", "industry"},
	}
	date := "2026-06-25"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = RenderDoc(payload, date)
	}
}

// BenchmarkSummaryLine evaluates provenance counts summary formatting.
func BenchmarkSummaryLine(b *testing.B) {
	counts := map[string]int{
		"measured":   42,
		"modeled":    12,
		"functional": 18,
		"unknown":    1,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = SummaryLine(counts)
	}
}

// BenchmarkEnrichCatalogEngines evaluates catalog run traversal for artifact engine enrichment.
func BenchmarkEnrichCatalogEngines(b *testing.B) {
	runs := make([]any, 20)
	for j := 0; j < 20; j++ {
		runs[j] = map[string]any{
			"run_id": fmt.Sprintf("run-%d", j),
			"path":   "experiments/benchmark/runs/by-machine/m0/run-0",
		}
	}
	cat := map[string]any{
		"runs": runs,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPane = EnrichCatalogEngines("/nonexistent-root", cat)
	}
}

// TestBenchmarkSanity ensures all benchmark functions execute validly under testing.Benchmark.
func TestBenchmarkSanity(t *testing.T) {
	benchmarks := []struct {
		name string
		fn   func(b *testing.B)
	}{
		{"BenchmarkClassify", BenchmarkClassify},
		{"BenchmarkClassifyAll", BenchmarkClassifyAll},
		{"BenchmarkParseRunTS", BenchmarkParseRunTS},
		{"BenchmarkFoldBenchmarks", BenchmarkFoldBenchmarks},
		{"BenchmarkFoldWork", BenchmarkFoldWork},
		{"BenchmarkFoldIndustry", BenchmarkFoldIndustry},
		{"BenchmarkDirtyPathsFromPorcelain", BenchmarkDirtyPathsFromPorcelain},
		{"BenchmarkGitPaneWithRunner", BenchmarkGitPaneWithRunner},
		{"BenchmarkFold", BenchmarkFold},
		{"BenchmarkRender", BenchmarkRender},
		{"BenchmarkRenderDoc", BenchmarkRenderDoc},
		{"BenchmarkSummaryLine", BenchmarkSummaryLine},
		{"BenchmarkEnrichCatalogEngines", BenchmarkEnrichCatalogEngines},
	}

	for _, bm := range benchmarks {
		t.Run(bm.name, func(t *testing.T) {
			res := testing.Benchmark(bm.fn)
			if res.N <= 0 {
				t.Fatalf("%s executed 0 iterations", bm.name)
			}
		})
	}
}
