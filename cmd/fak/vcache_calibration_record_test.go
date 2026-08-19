package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
)

func TestVCacheCalibrationRecordPersistsProviderTelemetry(t *testing.T) {
	dir := t.TempDir()
	telemetry := filepath.Join(dir, "usage.jsonl")
	output := filepath.Join(dir, "calibration.jsonl")
	raw := "{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":1200,\"cached_input_tokens\":0}}\n" +
		"{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":1200,\"cached_input_tokens\":900}}\n"
	if err := os.WriteFile(telemetry, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	rc := runVCacheCalibrationRecord(&stdout, &stderr, []string{"--provider", "openai", "--model", "gpt-test", "--source", "test-probe", "--telemetry", telemetry, "--output", output})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	latest, err := vcachecalibration.ReadLatestCalibrations(output)
	if err != nil {
		t.Fatal(err)
	}
	row := latest["openai"]
	if row.Model != "gpt-test" || row.Source != "test-probe" || row.Predictions != 2 || row.FalseCold != 1 {
		t.Fatalf("row=%+v", row)
	}
	if !strings.Contains(stdout.String(), "RECORDED provider=openai model=gpt-test") {
		t.Fatalf("stdout=%s", stdout.String())
	}
}
