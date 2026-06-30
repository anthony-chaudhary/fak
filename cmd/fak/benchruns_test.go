package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunBenchRunsListAndShow(t *testing.T) {
	root := seedBenchRunsWorkspace(t)
	var out, err bytes.Buffer
	if code := runBenchRuns(&out, &err, []string{"list", "--workspace", root, "--model", "smollm"}); code != 0 {
		t.Fatalf("list code=%d stderr=%s", code, err.String())
	}
	if !strings.Contains(out.String(), "run-a") || !strings.Contains(out.String(), "box-a") {
		t.Fatalf("list output:\n%s", out.String())
	}

	out.Reset()
	err.Reset()
	if code := runBenchRuns(&out, &err, []string{"show", "--workspace", root, "run-a"}); code != 0 {
		t.Fatalf("show code=%d stderr=%s", code, err.String())
	}
	if !strings.Contains(out.String(), "Run: run-a") || !strings.Contains(out.String(), "Path: experiments/benchmark/run-a") {
		t.Fatalf("show output:\n%s", out.String())
	}
}

func seedBenchRunsWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	benchDir := filepath.Join(root, "experiments", "benchmark")
	if err := os.MkdirAll(filepath.Join(benchDir, "run-a"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeBenchRunsJSON(t, filepath.Join(benchDir, "catalog.json"), map[string]any{"runs": []map[string]any{{
		"run_id": "run-a", "machine_id": "box-a", "model": "smollm2",
		"precision": "q8", "timestamp": "2026-01-02", "peak_tok_per_sec": 42.0,
		"path": "experiments/benchmark/run-a",
	}}})
	writeBenchRunsJSON(t, filepath.Join(benchDir, "run-a", "batch.json"), map[string]any{
		"baseline": map[string]any{"tok_per_sec": 10.0},
		"peak":     map[string]any{"batch": 4, "agg_tok_per_sec": 42.0, "speedup_vs_baseline": 4.2},
	})
	return root
}

func writeBenchRunsJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}
