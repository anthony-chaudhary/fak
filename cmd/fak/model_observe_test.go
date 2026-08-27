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

func TestModelObserveBandwidthJSONSpine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bandwidth.json")
	output := filepath.Join(t.TempDir(), "report.json")
	data := `{"schema":"fak-model-bandwidth/1","engine":"fak-native","trigger":{"symptom_window":2,"resource_window":2,"latency_threshold_ms":100,"resource_utilization":0.8},"samples":[{"phase":"decode","shape":"small","provenance":{"source":"synthetic-test","machine":"fixture-host","device":"cpu-ddr5","collector":"fixture"},"rooflines":{"theoretical_gb_s":100,"measured_sustainable_gb_s":80},"live":{"read_gb_s":70,"write_gb_s":2},"request":{"latency_ms":120,"completion_tokens":1},"device":{},"capacity":{},"transfer":{},"software":{}},{"phase":"decode","shape":"small","provenance":{"source":"synthetic-test","machine":"fixture-host","device":"cpu-ddr5","collector":"fixture"},"rooflines":{"theoretical_gb_s":100,"measured_sustainable_gb_s":80},"live":{"read_gb_s":72,"write_gb_s":2},"request":{"latency_ms":125,"completion_tokens":1},"device":{},"capacity":{},"transfer":{},"software":{}}]}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := runModelObserveBandwidth([]string{"--input", path, "--output", output, "--pretty=false"}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(got)
	for _, want := range []string{`"schema":"fak-model-bandwidth/1"`, `"engine":"fak-native"`, `"selected_source":"measured-sustainable"`, `"write_gb_s":2`, `"transfer":{}`, `"triggered":true`} {
		if !strings.Contains(text, want) {
			t.Fatalf("report missing %s: %s", want, text)
		}
	}
	if strings.Contains(text, `"host_to_device_gb_s":0`) {
		t.Fatalf("unavailable transfer counter serialized as zero: %s", text)
	}
}

func TestModelObserveBandwidthCollectSpine(t *testing.T) {
	output := filepath.Join(t.TempDir(), "collection.json")
	if err := runModelObserveBandwidth([]string{"collect", "--count", "1", "--interval", "10ms", "--phase", "decode", "--shape", "small", "--theoretical-gb-s", "100", "--measured-gb-s", "80", "--output", output, "--pretty=false"}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(got)
	for _, want := range []string{`"schema":"fak-model-bandwidth-collection/1"`, `"engine":"fak-native"`, `"machine_class":"`, `"capture":{`, `"report":{`, `"selected_source":"measured-sustainable"`, `"dram_counters":false`} {
		if !strings.Contains(text, want) {
			t.Fatalf("collection missing %s: %s", want, text)
		}
	}
	for _, forbidden := range []string{`"total_gb_s":0`, `"read_gb_s":0`, `"write_gb_s":0`} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("unavailable DRAM counter serialized as zero: %s", text)
		}
	}
	if strings.Contains(text, `"live":{"process_`) {
		t.Fatalf("process signal mislabeled as live DRAM bandwidth: %s", text)
	}
}
