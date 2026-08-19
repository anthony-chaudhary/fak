package vcachecalibration

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
)

func TestCalibrationFromProbePersistsMeasuredConstants(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	row, err := CalibrationFromProbe(vcachecal.Calibration{
		Provider: "Anthropic", ModelID: "claude-sonnet", TTLMillis: 600_000, TTLMeasured: true,
		MinPrefixTokens: 2048, MinPrefixMeasured: true, ReadMult: 0.12, ReadMultMeasured: true,
	}, "probe:test", 4, now)
	if err != nil {
		t.Fatal(err)
	}
	if row.Provider != "anthropic" || row.MinPrefixTokens != 2048 || !row.MinPrefixMeasured || row.Predictions != 4 {
		t.Fatalf("row=%+v", row)
	}
}

func TestFreshRuntimeConstantsSelectsMatchingFreshMeasuredRow(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "calibration.jsonl")
	rows := []ProviderCalibration{
		probeRow(now.Add(-time.Hour), "anthropic", "", 1024),
		probeRow(now.Add(-30*time.Minute), "anthropic", "claude-sonnet", 2048),
		probeRow(now.Add(-10*time.Minute), "anthropic", "claude-opus", 4096),
	}
	for _, row := range rows {
		if err := AppendCalibration(path, row); err != nil {
			t.Fatal(err)
		}
	}
	got, ok, reason := FreshRuntimeConstants(path, "Anthropic", "claude-sonnet", now, DefaultCalibrationTTL)
	if !ok || got.MinPrefixTokens != 2048 || got.Model != "claude-sonnet" || reason != "fresh measured calibration" {
		t.Fatalf("got=%+v ok=%v reason=%q", got, ok, reason)
	}
}

func TestFreshRuntimeConstantsRejectsStaleAndObservationOnlyRows(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		name string
		row  ProviderCalibration
		want string
	}{
		{name: "stale", row: probeRow(now.Add(-8*24*time.Hour), "anthropic", "claude", 2048), want: "matching calibration is stale"},
		{name: "observation only", row: ProviderCalibration{Schema: CalibrationSchema, TS: now.Format(time.RFC3339Nano), Provider: "anthropic", Model: "claude", Source: "guard", Turns: 2, Predictions: 1, TrueWarm: 1}, want: "fresh row has no measured runtime constants"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "calibration.jsonl")
			if err := AppendCalibration(path, tc.row); err != nil {
				t.Fatal(err)
			}
			_, ok, reason := FreshRuntimeConstants(path, "anthropic", "claude", now, DefaultCalibrationTTL)
			if ok || reason != tc.want {
				t.Fatalf("ok=%v reason=%q, want false/%q", ok, reason, tc.want)
			}
		})
	}
}

func probeRow(ts time.Time, provider, model string, minPrefix int64) ProviderCalibration {
	return ProviderCalibration{
		Schema: CalibrationSchema, TS: ts.Format(time.RFC3339Nano), Provider: provider, Model: model,
		Source: "probe", Turns: 2, Predictions: 2, TrueCold: 2,
		MinPrefixTokens: minPrefix, MinPrefixMeasured: true,
	}
}

func TestFreshRuntimeConstantsCarriesNewestModelWhenWireModelIsDynamic(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "calibration.jsonl")
	for _, row := range []ProviderCalibration{
		probeRow(now.Add(-time.Hour), "anthropic", "claude-opus", 4096),
		probeRow(now.Add(-time.Minute), "anthropic", "claude-sonnet", 2048),
	} {
		if err := AppendCalibration(path, row); err != nil {
			t.Fatal(err)
		}
	}
	got, ok, _ := FreshRuntimeConstants(path, "anthropic", "", now, DefaultCalibrationTTL)
	if !ok || got.Model != "claude-sonnet" || got.MinPrefixTokens != 2048 {
		t.Fatalf("got=%+v ok=%v", got, ok)
	}
}

func TestFreshRuntimeConstantsCarriesMeasuredWritePricing(t *testing.T) {
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "calibration.jsonl")
	row := ProviderCalibration{
		Schema: CalibrationSchema, TS: now.Format(time.RFC3339Nano), Provider: "anthropic", Model: "claude",
		Source: "probe", Turns: 1, Predictions: 1, TrueCold: 1,
		Write5mMult: 1.4, Write5mMeasured: true, Write1hMult: 2.2, Write1hMeasured: true,
	}
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	got, ok, reason := FreshRuntimeConstants(path, "anthropic", "claude", now, DefaultCalibrationTTL)
	if !ok {
		t.Fatalf("FreshRuntimeConstants: %s", reason)
	}
	if got.Write5mMult != 1.4 || !got.Write5mMeasured || got.Write1hMult != 2.2 || !got.Write1hMeasured {
		t.Fatalf("write constants = %+v", got)
	}
}
