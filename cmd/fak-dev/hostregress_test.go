package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeHostRegressSeries lays one receipt JSON per name into a fresh dir,
// returning the dir path. Bodies carry both hostshare evidence
// (device_busy fraction) and regime evidence (device_busy_fraction / gap
// sums) because the two wave-1 readers key their signals differently.
func writeHostRegressSeries(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// computeBoundNote is a healthy small-batch reading: host_share 0.13, device
// busy 87% -> COMPUTE_BOUND.
const computeBoundNote = `{"model":"Qwen3.8-27B","machine":{"platform":"linux/amd64","backend":"vulkan"},"device_busy":0.87,"device_busy_fraction":0.87,"observed_utc":"%s"}`

// TestHostRegressVerbFlatSeriesPasses witnesses the pass case: three oldest
// readings at host_share 0.13 (baseline median 0.13) and a latest 0.15 is a
// +15% delta, below the +50% line, with no regime flip.
func TestHostRegressVerbFlatSeriesPasses(t *testing.T) {
	files := map[string]string{
		"r1.json": sprintfNote(computeBoundNote, "2026-09-01T00:00:00Z"),
		"r2.json": sprintfNote(computeBoundNote, "2026-09-15T00:00:00Z"),
		"r3.json": sprintfNote(computeBoundNote, "2026-10-01T00:00:00Z"),
		"r4.json": sprintfNote(`{"model":"Qwen3.8-27B","machine":{"platform":"linux/amd64","backend":"vulkan"},"device_busy":0.85,"device_busy_fraction":0.85,"observed_utc":"%s"}`, "2026-10-15T00:00:00Z"),
	}
	var out, errOut bytes.Buffer
	code := runHostRegress(&out, &errOut, []string{"--receipts", writeHostRegressSeries(t, files)})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	s := out.String()
	if !strings.Contains(s, "gate: PASS") || !strings.Contains(s, "+15.4%") {
		t.Errorf("missing PASS verdict with +15.4%% delta:\n%s", s)
	}
	if !strings.Contains(s, "COMPUTE_BOUND") {
		t.Errorf("missing regime column:\n%s", s)
	}
}

// TestHostRegressVerbDeltaRegressionFails witnesses gate (a): 0.13 -> 0.20 is
// +53.8%, at/above the +50% relative line, so the verb exits non-zero naming
// the delta reason (the latest stays COMPUTE_BOUND, so no flip is claimed).
func TestHostRegressVerbDeltaRegressionFails(t *testing.T) {
	files := map[string]string{
		"r1.json": sprintfNote(computeBoundNote, "2026-09-01T00:00:00Z"),
		"r2.json": sprintfNote(computeBoundNote, "2026-09-15T00:00:00Z"),
		"r3.json": sprintfNote(computeBoundNote, "2026-10-01T00:00:00Z"),
		"r4.json": sprintfNote(`{"model":"Qwen3.8-27B","machine":{"platform":"linux/amd64","backend":"vulkan"},"device_busy":0.80,"device_busy_fraction":0.80,"observed_utc":"%s"}`, "2026-10-15T00:00:00Z"),
	}
	var out, errOut bytes.Buffer
	code := runHostRegress(&out, &errOut, []string{"--receipts", writeHostRegressSeries(t, files)})
	if code != hostShareGateExit {
		t.Fatalf("exit = %d, want %d\nstdout: %s\nstderr: %s", code, hostShareGateExit, out.String(), errOut.String())
	}
	s := out.String()
	if !strings.Contains(s, "gate: FAIL") || !strings.Contains(s, "delta-regression") {
		t.Errorf("missing FAIL delta-regression verdict:\n%s", s)
	}
	if strings.Contains(s, "regime-flip") {
		t.Errorf("no flip expected when the latest stays COMPUTE_BOUND:\n%s", s)
	}
}

// TestHostRegressVerbRegimeFlipFails witnesses gate (b) in isolation: the
// latest reading keeps host_share under the +50% line (+15.4%) but its
// inter-kernel gap share (0.6 >= 0.5) classifies it LAUNCH_BOUND against a
// COMPUTE_BOUND baseline majority, so the flip alone fails the gate.
func TestHostRegressVerbRegimeFlipFails(t *testing.T) {
	files := map[string]string{
		"r1.json": sprintfNote(computeBoundNote, "2026-09-01T00:00:00Z"),
		"r2.json": sprintfNote(computeBoundNote, "2026-09-15T00:00:00Z"),
		"r3.json": sprintfNote(computeBoundNote, "2026-10-01T00:00:00Z"),
		"r4.json": sprintfNote(`{"model":"Qwen3.8-27B","machine":{"platform":"linux/amd64","backend":"vulkan"},"device_busy":0.85,"device_busy_fraction":0.85,"inter_kernel_gap_sum_us":600,"kernel_execution_sum_us":400,"observed_utc":"%s"}`, "2026-10-15T00:00:00Z"),
	}
	var out, errOut bytes.Buffer
	code := runHostRegress(&out, &errOut, []string{"--receipts", writeHostRegressSeries(t, files)})
	if code != hostShareGateExit {
		t.Fatalf("exit = %d, want %d\nstdout: %s\nstderr: %s", code, hostShareGateExit, out.String(), errOut.String())
	}
	s := out.String()
	if !strings.Contains(s, "regime-flip") || !strings.Contains(s, "LAUNCH_BOUND") {
		t.Errorf("missing regime-flip verdict:\n%s", s)
	}
	if strings.Contains(s, "delta-regression") {
		t.Errorf("+15.4%% delta must not be claimed as a regression:\n%s", s)
	}
}

// TestHostRegressVerbUnknownLatestSkips witnesses the honesty invariant: a
// perf-receipt-v2 without device-busy data as the most recent reading prints
// UNKNOWN and exits zero, never failing on absent evidence.
func TestHostRegressVerbUnknownLatestSkips(t *testing.T) {
	v2 := `{
	  "schema": "fak-native-performance-receipt/v2",
	  "role": "candidate",
	  "envelope_id": "qwen38-cuda-fixture",
	  "machine": {"scrubbed_id": "lab", "platform": "linux/amd64", "backend": "cuda"},
	  "controls": {"prompt_tokens": 64, "decode_tokens": 32, "batch": 1, "context_tokens": 4096,
	    "temperature": 0, "sampling": "greedy", "cache_state": "warm", "warmups": 1, "repetitions": 1},
	  "repetitions": [{"end_to_end_milliseconds": 100, "tokens_per_second": 10}],
	  "execution": {"engine": "fak-native", "forward_path": "device/cuda"},
	  "generated_at": "2026-10-20T00:00:00Z"
	}`
	files := map[string]string{
		"r1.json": sprintfNote(computeBoundNote, "2026-09-01T00:00:00Z"),
		"r2.json": sprintfNote(computeBoundNote, "2026-09-15T00:00:00Z"),
		"r3.json": sprintfNote(computeBoundNote, "2026-10-01T00:00:00Z"),
		"r4.json": sprintfNote(computeBoundNote, "2026-10-10T00:00:00Z"),
		"r5.json": v2,
	}
	var out, errOut bytes.Buffer
	code := runHostRegress(&out, &errOut, []string{"--receipts", writeHostRegressSeries(t, files)})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "gate: UNKNOWN") {
		t.Errorf("missing UNKNOWN gate line:\n%s", out.String())
	}
}

// TestHostRegressVerbInsufficientDataExitsZero witnesses the sparse-history
// case: two known readings cannot fill K+1=4, so the verb warns
// INSUFFICIENT_DATA and exits zero instead of guessing a trend.
func TestHostRegressVerbInsufficientDataExitsZero(t *testing.T) {
	files := map[string]string{
		"r1.json": sprintfNote(computeBoundNote, "2026-09-01T00:00:00Z"),
		"r2.json": sprintfNote(computeBoundNote, "2026-09-15T00:00:00Z"),
	}
	var out, errOut bytes.Buffer
	code := runHostRegress(&out, &errOut, []string{"--receipts", writeHostRegressSeries(t, files)})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "INSUFFICIENT_DATA") {
		t.Errorf("missing INSUFFICIENT_DATA warning:\n%s", out.String())
	}
}

// TestHostRegressVerbPinnedBaselineBoundary witnesses --baseline-file: a bare
// number pin lowers the requirement to one known reading, and exactly +50%
// (0.10 -> 0.15) fails the gate because the line is inclusive.
func TestHostRegressVerbPinnedBaselineBoundary(t *testing.T) {
	dir := writeHostRegressSeries(t, map[string]string{
		"latest.json": sprintfNote(`{"device_busy":0.85,"device_busy_fraction":0.85,"observed_utc":"%s"}`, "2026-10-15T00:00:00Z"),
	})
	pin := filepath.Join(t.TempDir(), "pin.txt")
	if err := os.WriteFile(pin, []byte("0.10\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runHostRegress(&out, &errOut, []string{"--receipts", dir, "--baseline-file", pin})
	if code != hostShareGateExit {
		t.Fatalf("boundary exit = %d, want %d (+50%% is inclusive)\nstdout: %s\nstderr: %s", code, hostShareGateExit, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "pinned-file") {
		t.Errorf("baseline source should name the pin:\n%s", out.String())
	}

	if err := os.WriteFile(pin, []byte(`{"host_share": 0.13}`), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errOut.Reset()
	code = runHostRegress(&out, &errOut, []string{"--receipts", dir, "--baseline-file", pin})
	if code != 0 {
		t.Fatalf("json-object pin exit = %d, want 0 (+15%% passes)\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "gate: PASS") {
		t.Errorf("missing PASS for pinned 0.13 baseline:\n%s", out.String())
	}
}

// TestHostRegressVerbJSONShape witnesses the machine-readable contract:
// schema version, verdict, reasons, and the numeric delta over the same
// failing series.
func TestHostRegressVerbJSONShape(t *testing.T) {
	files := map[string]string{
		"r1.json": sprintfNote(computeBoundNote, "2026-09-01T00:00:00Z"),
		"r2.json": sprintfNote(computeBoundNote, "2026-09-15T00:00:00Z"),
		"r3.json": sprintfNote(computeBoundNote, "2026-10-01T00:00:00Z"),
		"r4.json": sprintfNote(`{"device_busy":0.80,"device_busy_fraction":0.80,"observed_utc":"%s"}`, "2026-10-15T00:00:00Z"),
	}
	var out, errOut bytes.Buffer
	code := runHostRegress(&out, &errOut, []string{"--receipts", writeHostRegressSeries(t, files), "--json"})
	if code != hostShareGateExit {
		t.Fatalf("exit = %d, want %d\nstdout: %s\nstderr: %s", code, hostShareGateExit, out.String(), errOut.String())
	}
	var doc hostRegressDoc
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out.String())
	}
	if doc.Schema != hostRegressSchema || doc.Verdict != "FAIL" {
		t.Errorf("schema/verdict = %q/%q, want %q/FAIL", doc.Schema, doc.Verdict, hostRegressSchema)
	}
	if len(doc.Reasons) != 1 || !strings.Contains(doc.Reasons[0], "delta-regression") {
		t.Errorf("reasons = %v, want exactly delta-regression", doc.Reasons)
	}
	if doc.Baseline == nil || doc.Baseline.Value < 0.129 || doc.Baseline.Value > 0.131 {
		t.Errorf("baseline = %+v, want ~0.13 median of oldest 3", doc.Baseline)
	}
	if doc.Delta == nil || *doc.Delta < 0.53 || *doc.Delta > 0.55 {
		t.Errorf("delta = %v, want ~+0.538", doc.Delta)
	}
	if doc.Latest == nil || doc.Latest.Role != "latest" {
		t.Errorf("latest row = %+v, want role latest", doc.Latest)
	}
}

// TestHostRegressVerbUsageErrors witnesses flag validation: a zero baseline
// window and a missing receipts dir are distinct failures (usage vs IO).
func TestHostRegressVerbUsageErrors(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runHostRegress(&out, &errOut, []string{"--receipts", t.TempDir(), "--baseline-k", "0"}); code != 2 {
		t.Errorf("--baseline-k 0 exit = %d, want 2\nstderr: %s", code, errOut.String())
	}
	errOut.Reset()
	if code := runHostRegress(&out, &errOut, []string{"--receipts", filepath.Join(t.TempDir(), "nope")}); code != 1 {
		t.Errorf("missing dir exit = %d, want 1\nstderr: %s", code, errOut.String())
	}
}

func sprintfNote(format, ts string) string { return strings.Replace(format, "%s", ts, 1) }
