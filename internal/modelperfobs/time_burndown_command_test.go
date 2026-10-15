package modelperfobs

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=20ms
func TestTimeBurndownCommand(t *testing.T) {
	var help bytes.Buffer
	if err := RunTimeBurndown([]string{"--help"}, &help); err != nil || !strings.Contains(help.String(), "--phases") {
		t.Fatalf("help contract: err=%v output=%s", err, &help)
	}
	input := filepath.Join(t.TempDir(), "observations.jsonl")
	data := strings.Join([]string{
		`{"schema":"fak-model-perf/1","request_id":"cold","backend":"https://engine.example","model":"example","status":200,"streaming":true,"cache_state":"cold","duration_ms":1000,"ttft_ms":700}`,
		`{"schema":"fak-model-perf/1","request_id":"warm","backend":"https://engine.example","model":"example","status":200,"streaming":true,"cache_state":"warm","duration_ms":100,"ttft_ms":20}`,
		`{"schema":"fak-model-perf/1","request_id":"legacy","backend":"https://engine.example","model":"example","status":200,"duration_ms":200}`,
		`{"schema":"fak-model-perf/1","request_id":"prime","backend":"https://engine.example","model":"example","status":200,"cache_state":"warm","warmup":true,"duration_ms":9000}`,
	}, "\n")
	if err := os.WriteFile(input, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := RunTimeBurndown([]string{"--input", input, "--format", "json"}, &out); err != nil {
		t.Fatal(err)
	}
	var report TimeBurndown
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Requests != 4 || report.WarmupExcluded != 1 || report.IncludedRequests != 3 || len(report.Groups) != 3 {
		t.Fatalf("incorrect cohort/exclusion contract: %+v", report)
	}
	for _, format := range []string{"md", "prom"} {
		out.Reset()
		if err := RunTimeBurndown([]string{"--input", input, "--format", format}, &out); err != nil {
			t.Fatal(err)
		}
		if out.Len() == 0 {
			t.Fatalf("%s output is empty", format)
		}
	}
	for _, args := range [][]string{
		{"--input", input, "--format", "invalid"},
		{"--input", input, "unexpected"},
		{"--input", input, "--phases", filepath.Join(t.TempDir(), "absent")},
	} {
		out.Reset()
		if err := RunTimeBurndown(args, &out); err == nil || out.Len() != 0 {
			t.Fatalf("invalid invocation emitted report: args=%v err=%v out=%s", args, err, &out)
		}
	}
}
