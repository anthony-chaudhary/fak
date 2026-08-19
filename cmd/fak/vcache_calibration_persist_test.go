package main

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func TestPersistVCacheCalibrationWritesRealTurnWindow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "calibration.jsonl")
	old := appendVCacheCalibration
	appendVCacheCalibration = func(_ string, row vcachecalibration.ProviderCalibration) error {
		return vcachecalibration.AppendCalibration(path, row)
	}
	t.Cleanup(func() { appendVCacheCalibration = old })
	now := time.Now()
	persistVCacheCalibration("anthropic", "guard:claude", []vcacheobserve.Turn{
		{UnixMillis: now.Add(-time.Minute).UnixMilli(), Family: "session", InputTokens: 1000, CacheCreation: 800},
		{UnixMillis: now.UnixMilli(), Family: "session", InputTokens: 1000, CacheRead: 800},
	})
	latest, err := vcachecalibration.ReadLatestCalibrations(path)
	if err != nil || latest["anthropic"].Source != "guard:claude" {
		t.Fatalf("latest=%+v err=%v", latest, err)
	}
}
