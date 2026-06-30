package benchruns

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadFilterAndRenderCatalog(t *testing.T) {
	root := seedCatalog(t)
	cat, err := LoadCatalog(root)
	if err != nil {
		t.Fatalf("LoadCatalog: %v", err)
	}
	if got := len(cat.Runs); got != 2 {
		t.Fatalf("runs = %d, want 2", got)
	}
	filtered := FilterRuns(cat.Runs, Filter{Machine: "box-a", Model: "smollm", Precision: "q8"})
	if len(filtered) != 1 || stringField(filtered[0], "run_id") != "run-a" {
		t.Fatalf("filtered = %#v, want run-a", filtered)
	}
	table := RenderList(filtered)
	if !strings.Contains(table, "run-a") || !strings.Contains(table, "123.4") {
		t.Fatalf("RenderList missing run/peak:\n%s", table)
	}
}

func TestLoadRunShowCompareBestSummary(t *testing.T) {
	root := seedCatalog(t)
	cat, err := LoadCatalog(root)
	if err != nil {
		t.Fatalf("LoadCatalog: %v", err)
	}
	a, err := LoadRun(root, cat, "run-a")
	if err != nil {
		t.Fatalf("LoadRun a: %v", err)
	}
	b, err := LoadRun(root, cat, "run-b")
	if err != nil {
		t.Fatalf("LoadRun b: %v", err)
	}
	show := RenderShow(a)
	for _, want := range []string{"Run: run-a", "Baseline (B=1): 10.0 tok/s", "Speedup: 12.30x"} {
		if !strings.Contains(show, want) {
			t.Fatalf("RenderShow missing %q:\n%s", want, show)
		}
	}
	compare := RenderCompare(a, b)
	if !strings.Contains(compare, "Benchmark Run Comparison") || !strings.Contains(compare, "Peak tok/s") {
		t.Fatalf("RenderCompare missing content:\n%s", compare)
	}
	best, err := Best(cat.Runs, "smollm", "peak_tok_per_sec")
	if err != nil {
		t.Fatalf("Best: %v", err)
	}
	if stringField(best, "run_id") != "run-b" {
		t.Fatalf("best = %s, want run-b", stringField(best, "run_id"))
	}
	md := RenderMarkdownTable(cat.Runs)
	if !strings.Contains(md, "| Machine") || !strings.Contains(md, "box-b") {
		t.Fatalf("RenderMarkdownTable:\n%s", md)
	}
	summary := RenderSummary(cat.Runs, "machine")
	if !strings.Contains(summary, "box-a:") || !strings.Contains(summary, "Avg peak") {
		t.Fatalf("RenderSummary:\n%s", summary)
	}
}

func seedCatalog(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	benchDir := filepath.Join(root, "experiments", "benchmark")
	if err := os.MkdirAll(benchDir, 0o755); err != nil {
		t.Fatal(err)
	}
	catalog := map[string]any{"runs": []map[string]any{
		{
			"run_id": "run-a", "machine_id": "box-a", "model": "smollm2-135m",
			"precision": "q8", "timestamp": "2026-01-02T00:00:00Z",
			"peak_tok_per_sec": 123.4, "baseline_tok_per_sec": 10.0,
			"speedup": 12.34, "path": "experiments/benchmark/run-a",
		},
		{
			"run_id": "run-b", "machine_id": "box-b", "model": "smollm2-135m",
			"precision": "q8", "timestamp": "2026-01-03T00:00:00Z",
			"peak_tok_per_sec": 150.0, "baseline_tok_per_sec": 12.0,
			"speedup": 12.5, "path": "experiments/benchmark/run-b",
		},
	}}
	writeJSON(t, filepath.Join(benchDir, "catalog.json"), catalog)
	for _, id := range []string{"run-a", "run-b"} {
		dir := filepath.Join(benchDir, id)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		writeJSON(t, filepath.Join(dir, "manifest.json"), map[string]any{
			"config": map[string]any{"batch_sizes": []int{1, 2}, "workers": 4, "decode_steps": 16},
		})
		peak := 123.0
		if id == "run-b" {
			peak = 150.0
		}
		writeJSON(t, filepath.Join(dir, "batch.json"), map[string]any{
			"baseline": map[string]any{"tok_per_sec": 10.0},
			"peak": map[string]any{
				"batch": 8, "agg_tok_per_sec": peak, "speedup_vs_baseline": 12.3,
			},
		})
	}
	return root
}

func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}
