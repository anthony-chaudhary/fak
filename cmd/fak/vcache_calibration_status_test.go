package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
)

func TestVCacheCalibrationStatusFlagsMissingAndFresh(t *testing.T) {
	path := filepath.Join(t.TempDir(), "calibration.jsonl")
	var out, errOut bytes.Buffer
	if rc := runVCacheCalibrationStatus(&out, &errOut, []string{"--file", path, "--providers", "anthropic", "--json"}); rc != 1 || !strings.Contains(out.String(), `"state": "missing"`) {
		t.Fatalf("missing rc/output = %d %s stderr=%s", rc, out.String(), errOut.String())
	}
	row := vcachecalibration.ProviderCalibration{Schema: vcachecalibration.CalibrationSchema, TS: time.Now().UTC().Format(time.RFC3339Nano), Provider: "anthropic", Source: "guard:claude", Turns: 2, Predictions: 1, TrueWarm: 1, StaleAfterDays: 7}
	if err := vcachecalibration.AppendCalibration(path, row); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errOut.Reset()
	if rc := runVCacheCalibrationStatus(&out, &errOut, []string{"--file", path, "--providers", "anthropic", "--json"}); rc != 0 || !strings.Contains(out.String(), `"state": "fresh"`) {
		t.Fatalf("fresh rc/output = %d %s stderr=%s", rc, out.String(), errOut.String())
	}
}
