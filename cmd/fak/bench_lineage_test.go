package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBuiltInBenchmarkVerbsStampBenchmarkArtifact(t *testing.T) {
	t.Setenv("FAK_BENCH_COMMIT", "4164164164164164164164164164164164164164")
	t.Setenv("FAK_BENCH_UTC", "2026-06-30T00:00:00Z")
	t.Setenv("FAK_BENCH_NODE", "bench-lineage-test")

	dir := t.TempDir()

	benchOut := filepath.Join(dir, "bench-report.json")
	cmdBench([]string{"--trace", "../../testdata/tau2/tau2-smoke.json", "--out", benchOut, "--no-baseline"})
	assertBenchmarkArtifact(t, benchOut)

	turntaxOut := filepath.Join(dir, "turntax-report.json")
	cmdTurnTax([]string{"--trace", "../../testdata/turntax/turntax-happy.json", "--out", turntaxOut})
	assertBenchmarkArtifact(t, turntaxOut)
}

func assertBenchmarkArtifact(t *testing.T, path string) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var root struct {
		BenchmarkArtifact struct {
			Schema  string `json:"schema"`
			Commit  string `json:"fak_commit"`
			Witness struct {
				ReproductionCommand string `json:"reproduction_command"`
			} `json:"witness"`
		} `json:"benchmark_artifact"`
	}
	if err := json.Unmarshal(raw, &root); err != nil {
		t.Fatalf("decode %s: %v\n%s", path, err, raw)
	}
	if root.BenchmarkArtifact.Schema != "fak-benchmark-artifact/1" {
		t.Fatalf("%s benchmark_artifact schema = %q", path, root.BenchmarkArtifact.Schema)
	}
	if root.BenchmarkArtifact.Commit != "4164164164164164164164164164164164164164" {
		t.Fatalf("%s fak_commit = %q", path, root.BenchmarkArtifact.Commit)
	}
	if root.BenchmarkArtifact.Witness.ReproductionCommand == "" {
		t.Fatalf("%s reproduction command was not stamped", path)
	}
}
