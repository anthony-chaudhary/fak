package modelperfobs

import (
	"reflect"
	"strings"
	"testing"
)

func TestDatedLongContextPresetsUseReleasedIdentitiesAndSeparateEvidence(t *testing.T) {
	presets := DatedLongContextPresets()
	if got, want := len(presets), 2; got != want {
		t.Fatalf("preset count = %d, want %d", got, want)
	}
	want := map[string]struct {
		total, active, ngram float64
		max                  uint64
	}{
		"Qwen3.8-Flash-Next": {125e9, 6e9, 51e9, 262_144},
		"GLM-5.3-Flash":      {320e9, 18e9, 0, 1_048_576},
	}
	for _, preset := range presets {
		expect, ok := want[preset.Identity]
		if !ok {
			t.Fatalf("invented or unsupported identity %q", preset.Identity)
		}
		if strings.Contains(preset.Identity, "3.8-Next") || strings.Contains(preset.Identity, "5.3-Next") {
			t.Fatalf("generic Next identity leaked into preset: %q", preset.Identity)
		}
		if preset.Schema != LongContextPresetSchema || preset.AsOfDate != "2026-08-28" {
			t.Fatalf("undated/unversioned preset: %+v", preset)
		}
		if preset.Facts.TotalParameters != expect.total || preset.Facts.ActiveParameters != expect.active || preset.Facts.NGramEmbeddingParameters != expect.ngram || preset.Facts.MaxPositionEmbeddings != expect.max {
			t.Fatalf("%s facts = %+v", preset.Identity, preset.Facts)
		}
		if len(preset.Sources) < 2 || len(preset.Unknowns) == 0 {
			t.Fatalf("%s must retain sources and unknowns", preset.Identity)
		}
		if preset.Assumptions.KVBytesPerToken.Min >= preset.Assumptions.KVBytesPerToken.Max {
			t.Fatalf("%s KV uncertainty collapsed: %v", preset.Identity, preset.Assumptions.KVBytesPerToken)
		}
		if !strings.Contains(preset.Assumptions.KVTrafficBounds, "analytical lower") || !strings.Contains(preset.Assumptions.KVTrafficBounds, "analytical upper") || !strings.Contains(preset.Assumptions.KVTrafficBounds, "not measured fak-native") {
			t.Fatalf("%s traffic label is not bounded: %q", preset.Identity, preset.Assumptions.KVTrafficBounds)
		}
	}
}

func TestLongContextPresetScenarioMatrixRunsThroughEstimator(t *testing.T) {
	runtime := longContextPresetRuntime()
	for _, preset := range DatedLongContextPresets() {
		first, err := LongContextScenarios(preset, runtime)
		if err != nil {
			t.Fatal(err)
		}
		second, err := LongContextScenarios(preset, runtime)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, second) {
			t.Fatalf("%s scenarios are not deterministic", preset.Identity)
		}
		if got, want := len(first), 8; got != want {
			t.Fatalf("%s scenario count = %d, want %d", preset.Identity, got, want)
		}
		seen := map[[2]uint64]bool{}
		for _, scenario := range first {
			seen[[2]uint64{scenario.ContextTokens, scenario.PrefillDecodeRatio}] = true
			wantTotal := preset.Facts.TotalParameters + preset.Facts.NGramEmbeddingParameters
			if scenario.Input.TotalParameters != wantTotal || scenario.Input.ActiveParameters != preset.Facts.ActiveParameters {
				t.Fatalf("%s estimator parameters = total %g active %g, want %g and %g", preset.Identity, scenario.Input.TotalParameters, scenario.Input.ActiveParameters, wantTotal, preset.Facts.ActiveParameters)
			}
			if scenario.Input.PrefillTokens/scenario.Input.DecodeTokens != scenario.PrefillDecodeRatio || scenario.Input.PrefillTokens%scenario.Input.DecodeTokens != 0 {
				t.Fatalf("%s context=%d ratio is %d:%d", preset.Identity, scenario.ContextTokens, scenario.Input.PrefillTokens, scenario.Input.DecodeTokens)
			}
			got, err := EstimateLongContextEnvelope(scenario.Input)
			if err != nil {
				t.Fatalf("%s context=%d ratio=%d: %v", preset.Identity, scenario.ContextTokens, scenario.PrefillDecodeRatio, err)
			}
			assertPresetOrderedEnvelope(t, got)
		}
		for _, contextTokens := range []uint64{35_000, 64_000, 128_000, 200_000} {
			for _, ratio := range []uint64{200, 300} {
				if !seen[[2]uint64{contextTokens, ratio}] {
					t.Errorf("%s missing context=%d ratio=%d:1", preset.Identity, contextTokens, ratio)
				}
			}
		}
	}
}

func TestLongContextScenariosRejectsUnversionedOrTooShortPreset(t *testing.T) {
	preset := qwen38FlashNextPreset()
	preset.Schema = ""
	if _, err := LongContextScenarios(preset, longContextPresetRuntime()); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("unversioned preset error = %v", err)
	}
	preset = qwen38FlashNextPreset()
	preset.Facts.MaxPositionEmbeddings = 128_000
	if _, err := LongContextScenarios(preset, longContextPresetRuntime()); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("short-context preset error = %v", err)
	}
}

func longContextPresetRuntime() LongContextEstimatorInput {
	return LongContextEstimatorInput{
		ResidentAgents: 1, Concurrency: 1,
		UsableMemoryBytes: 1e15, BandwidthBytesPerSec: ClosedRange{700e9, 900e9},
		ComputeFLOPS: ClosedRange{100e12, 140e12}, Efficiency: ClosedRange{0.45, 0.70},
		PrefillCacheHit: ClosedRange{0, 0},
	}
}

func assertPresetOrderedEnvelope(t *testing.T, got LongContextEnvelope) {
	t.Helper()
	ranges := []ClosedRange{
		got.ModelMemoryBytes, got.KVMemoryPerAgentBytes, got.KVMemoryBytes,
		got.TotalResidentMemoryBytes, got.HeadroomBytes, got.MaxResidentAgents,
		got.Prefill.ComputeSeconds, got.Prefill.MemorySeconds, got.Prefill.TimeSeconds,
		got.Decode.ComputeSeconds, got.Decode.MemorySeconds, got.Decode.TimeSeconds,
		got.ServiceTimeSeconds, got.ProcessedTokensPerSec, got.DecodeTokensPerSec,
	}
	for i, r := range ranges {
		if !finite(r.Min) || !finite(r.Max) || r.Min > r.Max {
			t.Fatalf("range %d is invalid: %+v", i, r)
		}
	}
}
