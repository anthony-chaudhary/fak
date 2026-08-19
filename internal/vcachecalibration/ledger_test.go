package vcachecalibration

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func TestCalibrationFromTurnsAppendsAndReadsPerProvider(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	turns := []vcacheobserve.Turn{
		{UnixMillis: now.Add(-2 * time.Minute).UnixMilli(), Family: "prefix", InputTokens: 1000, CacheCreation: 800},
		{UnixMillis: now.Add(-time.Minute).UnixMilli(), Family: "prefix", InputTokens: 1000, CacheRead: 800},
	}
	row, ok := CalibrationFromTurns("Anthropic", "guard:claude", turns, now)
	if !ok || row.Provider != "anthropic" || row.Predictions == 0 || row.Turns != 2 {
		t.Fatalf("row=%+v ok=%v", row, ok)
	}
	path := filepath.Join(t.TempDir(), "calibration.jsonl")
	if err := AppendCalibration(path, row); err != nil {
		t.Fatal(err)
	}
	latest, err := ReadLatestCalibrations(path)
	if err != nil || latest["anthropic"].TS != row.TS {
		t.Fatalf("latest=%+v err=%v", latest, err)
	}
}

func TestCalibrationStatusesFlagMissingAndStale(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "calibration.jsonl")
	row := ProviderCalibration{Schema: CalibrationSchema, TS: now.Add(-8 * 24 * time.Hour).Format(time.RFC3339Nano), Provider: "anthropic", Source: "guard", Turns: 2, Predictions: 1, TrueWarm: 1, StaleAfterDays: 7}
	if err := AppendCalibration(path, row); err != nil {
		t.Fatal(err)
	}
	statuses, err := CalibrationStatuses(path, []string{"openai", "anthropic"}, now, 7*24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if len(statuses) != 2 || statuses[0].Provider != "anthropic" || statuses[0].State != "stale" || statuses[0].Action == "" || statuses[1].State != "missing" {
		t.Fatalf("statuses=%+v", statuses)
	}
}

func TestCalibrationFromTurnsRequiresProviderFeedback(t *testing.T) {
	_, ok := CalibrationFromTurns("anthropic", "guard", []vcacheobserve.Turn{{UnixMillis: time.Now().UnixMilli(), Family: "x", InputTokens: 100}}, time.Now())
	if ok {
		t.Fatal("calibration created without cache activity")
	}
}
