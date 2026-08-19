package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
)

func TestLoadVCacheRuntimeCalibrationFreshOnly(t *testing.T) {
	root := t.TempDir()
	t.Setenv(ledgerRootEnv, root)
	path := filepath.Join(root, filepath.FromSlash(vcachecalibration.DefaultCalibrationRel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	row := vcachecalibration.ProviderCalibration{
		Schema: vcachecalibration.CalibrationSchema, TS: time.Now().UTC().Format(time.RFC3339Nano),
		Provider: "anthropic", Model: "claude-sonnet", Source: "probe:test", Turns: 2, Predictions: 2, TrueCold: 2,
		MinPrefixTokens: 2048, MinPrefixMeasured: true,
	}
	if err := vcachecalibration.AppendCalibration(path, row); err != nil {
		t.Fatal(err)
	}
	got := loadVCacheRuntimeCalibration("anthropic", "claude-sonnet")
	if got == nil || got.MinPrefixTokens != 2048 || !got.MinPrefixMeasured {
		t.Fatalf("got=%+v", got)
	}
}

func TestLoadVCacheRuntimeCalibrationStaleFallsBack(t *testing.T) {
	root := t.TempDir()
	t.Setenv(ledgerRootEnv, root)
	path := filepath.Join(root, filepath.FromSlash(vcachecalibration.DefaultCalibrationRel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	row := vcachecalibration.ProviderCalibration{
		Schema: vcachecalibration.CalibrationSchema, TS: time.Now().Add(-8 * 24 * time.Hour).UTC().Format(time.RFC3339Nano),
		Provider: "anthropic", Model: "claude-sonnet", Source: "probe:test", Turns: 2, Predictions: 2, TrueCold: 2,
		MinPrefixTokens: 2048, MinPrefixMeasured: true,
	}
	if err := vcachecalibration.AppendCalibration(path, row); err != nil {
		t.Fatal(err)
	}
	if got := loadVCacheRuntimeCalibration("anthropic", "claude-sonnet"); got != nil {
		t.Fatalf("stale calibration wired into runtime: %+v", got)
	}
}
