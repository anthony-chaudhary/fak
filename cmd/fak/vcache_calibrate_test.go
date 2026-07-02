package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func TestRunVCacheCalibrateFitsProbeSamples(t *testing.T) {
	samples := writeLines(t, "probe.jsonl",
		`{"provider":"anthropic","model_id":"opus-4.8","endpoint":"messages","delay_millis":30000,"prefix_tokens":4096,"cached_tokens":10000,"read_cost_equiv":2000}`,
		`{"provider":"anthropic","model_id":"opus-4.8","endpoint":"messages","delay_millis":120000,"prefix_tokens":4096,"cached_tokens":10000}`,
		`{"provider":"anthropic","model_id":"opus-4.8","endpoint":"messages","delay_millis":1200000,"prefix_tokens":2048,"cached_tokens":8000}`,
		`{"provider":"anthropic","model_id":"opus-4.8","endpoint":"messages","delay_millis":1500000,"prefix_tokens":4096,"cached_tokens":0}`,
		`{"provider":"anthropic","model_id":"opus-4.8","endpoint":"messages","delay_millis":30000,"prefix_tokens":512,"cached_tokens":0}`,
	)
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"calibrate", "--samples", samples, "--json"}); code != 0 {
		t.Fatalf("calibrate --json exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var cal vcachecal.Calibration
	if err := json.Unmarshal(out.Bytes(), &cal); err != nil {
		t.Fatalf("stdout is invalid calibration JSON: %v\n%s", err, out.String())
	}
	if cal.Provider != "anthropic" || cal.ModelID != "opus-4.8" || cal.Endpoint != "messages" {
		t.Fatalf("calibration identity = %+v", cal)
	}
	if cal.TTLMillis != 1_200_000 || !cal.TTLMeasured {
		t.Fatalf("ttl = %d measured=%v, want 1200000/true", cal.TTLMillis, cal.TTLMeasured)
	}
	if cal.MinPrefixTokens != 2048 || !cal.MinPrefixMeasured {
		t.Fatalf("min prefix = %d measured=%v, want 2048/true", cal.MinPrefixTokens, cal.MinPrefixMeasured)
	}
	if cal.ReadMult != 0.2 || !cal.ReadMultMeasured {
		t.Fatalf("read mult = %g measured=%v, want 0.2/true", cal.ReadMult, cal.ReadMultMeasured)
	}
}

func TestRunVCacheCalibrateWritesObserveCalibration(t *testing.T) {
	dir := t.TempDir()
	samples := filepath.Join(dir, "probe.json")
	raw := `{"samples":[` +
		`{"provider":"anthropic","model":"opus-4.8","endpoint":"messages","delay_millis":30000,"prefix_tokens":4096,"cached_tokens":10000,"read_cost_equiv":2500},` +
		`{"provider":"anthropic","model":"opus-4.8","endpoint":"messages","delay_millis":1200000,"prefix_tokens":4096,"cached_tokens":10000},` +
		`{"provider":"anthropic","model":"opus-4.8","endpoint":"messages","delay_millis":1500000,"prefix_tokens":4096,"cached_tokens":0}` +
		`]}`
	if err := os.WriteFile(samples, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	calPath := filepath.Join(dir, "calibration.json")
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"calibrate", "--samples", samples, "--out", calPath}); code != 0 {
		t.Fatalf("calibrate --out exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "ttl: 1200000 ms (measured)") || !strings.Contains(out.String(), "calibration: "+calPath) {
		t.Fatalf("human calibration output missing measured TTL/path:\n%s", out.String())
	}

	telemetry := writeLines(t, "telemetry.jsonl",
		`{"session_id":"cal","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_read_input_tokens":40000}`,
		`{"session_id":"cal","captured_utc":"2026-06-26T00:10:00Z","input_tokens":100,"cache_creation_input_tokens":1}`,
	)
	out.Reset()
	errb.Reset()
	if code := runVCacheObserve(&out, &errb, []string{"--telemetry", telemetry, "--calibration", calPath, "--json"}); code != 0 {
		t.Fatalf("observe with calibration exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcacheobserve.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("observe output is invalid JSON: %v\n%s", err, out.String())
	}
	if rep.Prediction.FalseWarm != 1 {
		t.Fatalf("observe did not consume fitted 20m TTL calibration, prediction=%+v", rep.Prediction)
	}
}

func TestRunVCacheCalibrateRejectsMissingRequiredFields(t *testing.T) {
	samples := writeLines(t, "bad.jsonl", `{"provider":"anthropic","delay_millis":30000,"cached_tokens":0}`)
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"calibrate", "--samples", samples}); code != 2 {
		t.Fatalf("bad calibrate exit=%d, want 2; stderr=%s output=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(errb.String(), "prefix_tokens is required") {
		t.Fatalf("missing field error not actionable:\n%s", errb.String())
	}
}
