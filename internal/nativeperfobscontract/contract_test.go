package nativeperfobscontract

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestFrozenCatalogDoesNotClaimUnsupportedExports(t *testing.T) {
	data, err := JSON(Frozen())
	if err != nil {
		t.Fatal(err)
	}
	var c Contract
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatal(err)
	}
	wantMetrics := []string{
		"fak_native_model_load_seconds", "fak_native_admission_wait_seconds",
		"fak_native_tokenization_seconds", "fak_native_phase_seconds", "fak_native_tokens_total",
		"fak_native_kv_bytes", "fak_native_cache_events_total", "fak_native_transfer_bytes_total",
		"fak_native_kernel_launches_total", "fak_native_synchronization_seconds",
		"fak_native_memory_bytes", "fak_native_output_seconds",
		"fak_native_benchmark_gate_total", "fak_native_promotion_total",
	}
	var metricNames []string
	for _, m := range c.Metrics {
		metricNames = append(metricNames, m.Name)
		retired := false
		for _, s := range c.Surfaces {
			retired = retired || strings.Contains(s.Exclusion, m.Name)
		}
		if !retired {
			t.Errorf("catalog metric %s lacks an explicit retirement", m.Name)
		}
	}
	if !slices.Equal(metricNames, wantMetrics) {
		t.Fatalf("catalog names = %v, want %v", metricNames, wantMetrics)
	}
	wantSurfaces := []string{
		"model_loading", "admission_queueing", "tokenization", "prefill", "decode",
		"kv_cache", "cache", "transfers", "kernel_launches", "synchronization",
		"memory", "output", "benchmark_gates", "promotion_revert",
	}
	var surfaceIDs []string
	for _, s := range c.Surfaces {
		surfaceIDs = append(surfaceIDs, s.ID)
		if s.Status != StatusExcluded || s.Metric != "" || !strings.HasPrefix(s.Exclusion, "RETIRED:") {
			t.Errorf("surface %s claims unsupported export: status=%s metric=%s exclusion=%s", s.ID, s.Status, s.Metric, s.Exclusion)
		}
	}
	if !slices.Equal(surfaceIDs, wantSurfaces) {
		t.Fatalf("surface IDs = %v, want %v", surfaceIDs, wantSurfaces)
	}
}

func TestFrozenContractIsValidAndMachineReadable(t *testing.T) {
	want := Frozen()
	got, err := JSON(want)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Contract
	if err := json.Unmarshal(got, &decoded); err != nil {
		t.Fatalf("matrix is not JSON: %v", err)
	}
	if decoded.Engine != Engine {
		t.Fatalf("engine = %q, want %q", decoded.Engine, Engine)
	}
	if decoded.DefaultModel != "Qwen3.8" {
		t.Fatalf("default model = %q", decoded.DefaultModel)
	}
	if !slices.Equal(decoded.Surfaces, want.Surfaces) {
		t.Fatalf("decoded surfaces = %#v, want frozen contract surfaces %#v", decoded.Surfaces, want.Surfaces)
	}
	for _, s := range decoded.Surfaces {
		if s.Status != StatusExported && s.Status != StatusExcluded {
			t.Fatalf("surface %s has UNKNOWN status %q", s.ID, s.Status)
		}
	}
}

func TestValidationFailsWhenProducerLacksSignalOrExclusion(t *testing.T) {
	c := Frozen()
	c.Surfaces[0].Status = "UNKNOWN"
	c.Surfaces[0].Metric = ""
	err := Validate(c)
	if err == nil {
		t.Fatal("Validate succeeded for UNKNOWN coverage")
	}
	if !strings.Contains(err.Error(), "UNKNOWN is debt") {
		t.Fatalf("error %q does not identify UNKNOWN debt", err)
	}
}

func TestValidationRejectsSilentEngineFallback(t *testing.T) {
	c := Frozen()
	c.Engine = "llama.cpp"
	err := Validate(c)
	if err == nil {
		t.Fatal("Validate succeeded for non-native engine")
	}
	if !strings.Contains(err.Error(), `engine must be "fak-native"`) {
		t.Fatalf("error = %q", err)
	}
}

func TestValidationRejectsUnboundedLabels(t *testing.T) {
	c := Frozen()
	c.Metrics[0].Labels = append(c.Metrics[0].Labels, "request_id")
	err := Validate(c)
	if err == nil {
		t.Fatal("Validate succeeded for unbounded label")
	}
	if !strings.Contains(err.Error(), "unbounded label request_id") {
		t.Fatalf("error = %q", err)
	}
}
