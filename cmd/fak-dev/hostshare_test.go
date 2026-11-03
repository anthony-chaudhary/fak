package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeHostShareFixture lays one receipt/profile-note JSON into a fresh dir.
func writeHostShareFixture(t *testing.T, name, body string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestHostShareVerbPHL051ShapeExitsZero witnesses the healthy case: a
// PHL-051-shaped device-busy note (host_share ~0.13) passes the default gate.
func TestHostShareVerbPHL051ShapeExitsZero(t *testing.T) {
	dir := writeHostShareFixture(t, "bringup.json",
		`{"model":"Qwen3.8-27B","machine":{"platform":"linux/amd64","backend":"vulkan"},"device_busy":0.87,"observed_utc":"2026-10-20T08:00:00Z"}`)
	var out, errOut bytes.Buffer
	code := runHostShare(&out, &errOut, []string{"--receipts", dir})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "0.130") || !strings.Contains(out.String(), "OK") {
		t.Errorf("table missing healthy reading:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "gate: PASS") {
		t.Errorf("missing PASS gate line:\n%s", out.String())
	}
}

// TestHostShareVerbLaunchBoundExitsNonZero witnesses the regression case: a
// synthetic launch-bound window (host_share ~0.45) as the most recent reading
// must fail the gate with the documented non-zero exit.
func TestHostShareVerbLaunchBoundExitsNonZero(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"old-healthy.json":      `{"device_busy":0.87,"decode_steps":64,"observed_utc":"2026-10-01T00:00:00Z"}`,
		"new-launch-bound.json": `{"device_busy":0.55,"decode_steps":128,"gpu_ms_per_token":4.1,"wall_ms_per_token":7.5,"observed_utc":"2026-11-01T00:00:00Z"}`,
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var out, errOut bytes.Buffer
	code := runHostShare(&out, &errOut, []string{"--receipts", dir})
	if code != hostShareGateExit {
		t.Fatalf("exit = %d, want %d\nstdout: %s\nstderr: %s", code, hostShareGateExit, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "LAUNCH-BOUND") || !strings.Contains(out.String(), "gate: FAIL") {
		t.Errorf("missing LAUNCH-BOUND gate line:\n%s", out.String())
	}
}

// TestHostShareVerbUnknownDoesNotFail witnesses the honesty invariant at the
// verb level: a perf-receipt-v2 without device-busy data prints UNKNOWN and
// exits zero.
func TestHostShareVerbUnknownDoesNotFail(t *testing.T) {
	body := `{
	  "schema": "fak-native-performance-receipt/v2",
	  "role": "baseline",
	  "envelope_id": "qwen38-cuda-fixture",
	  "machine": {"scrubbed_id": "lab", "platform": "linux/amd64", "backend": "cuda"},
	  "controls": {"prompt_tokens": 64, "decode_tokens": 32, "batch": 1, "context_tokens": 4096,
	    "temperature": 0, "sampling": "greedy", "cache_state": "warm", "warmups": 1, "repetitions": 1},
	  "repetitions": [{"end_to_end_milliseconds": 100, "tokens_per_second": 10}],
	  "execution": {"engine": "fak-native", "forward_path": "device/cuda"}
	}`
	dir := writeHostShareFixture(t, "receipt.json", body)
	var out, errOut bytes.Buffer
	code := runHostShare(&out, &errOut, []string{"--receipts", dir})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "UNKNOWN") {
		t.Errorf("missing UNKNOWN row:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "NO-SIGNAL") {
		t.Errorf("missing NO-SIGNAL gate line:\n%s", out.String())
	}
}

// TestHostShareVerbUsageAndJSON pins flag validation and the machine form.
func TestHostShareVerbUsageAndJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runHostShare(&out, &errOut, []string{"--threshold", "2"}); code != 2 {
		t.Errorf("bad threshold: exit = %d, want 2", code)
	}

	dir := writeHostShareFixture(t, "ok.json", `{"device_busy":0.87,"decode_steps":1}`)
	out.Reset()
	errOut.Reset()
	if code := runHostShare(&out, &errOut, []string{"--receipts", dir, "--json"}); code != 0 {
		t.Fatalf("json exit = %d, want 0\n%s", code, out.String())
	}
	if !strings.Contains(out.String(), `"schema": "fak.dev.hostshare.v1"`) || !strings.Contains(out.String(), `"verdict": "OK"`) {
		t.Errorf("json doc incomplete:\n%s", out.String())
	}

	// A missing receipts dir is an operational failure, not a gate pass.
	out.Reset()
	errOut.Reset()
	if code := runHostShare(&out, &errOut, []string{"--receipts", filepath.Join(dir, "nope")}); code != 1 {
		t.Errorf("missing dir: exit = %d, want 1", code)
	}
}
