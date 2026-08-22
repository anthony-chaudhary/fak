package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

func TestModelObserveReportSpine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "observations.jsonl")
	data := strings.Join([]string{
		`{"schema":"fak-model-perf/1","timestamp":"2026-08-21T00:00:00Z","request_id":"r1","backend":"http://qwen","status":200,"streaming":true,"duration_ms":2500,"ttft_ms":1800,"tpot_ms":36,"output_tokens_per_second":27}`,
		`{"schema":"fak-model-perf/1","timestamp":"2026-08-21T00:00:01Z","request_id":"r2","backend":"http://qwen","status":200,"streaming":true,"duration_ms":2700,"ttft_ms":2000,"tpot_ms":37,"output_tokens_per_second":26}`,
	}, "\n")
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := modelperfobs.ReadObservations(f)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := modelperfobs.WriteMarkdown(&out, modelperfobs.Summarize(rows)); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "Likely bottleneck: **prefill-or-queue**") {
		t.Fatal(out.String())
	}
}

func TestModelObserveStateBenchSpine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache-state.json")
	if err := runModelObserveStateBench([]string{"--output", path}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	report, err := modelperfobs.ReadStateReport(f)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "admitted" || len(report.Arms) != 4 {
		t.Fatalf("cache-state report = verdict %q, arms %d", report.Verdict, len(report.Arms))
	}
}
