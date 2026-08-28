package nativeperfobscontract

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFrozenContractIsValidAndMachineReadable(t *testing.T) {
	got, err := JSON(Frozen())
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
	if len(decoded.Surfaces) != 14 {
		t.Fatalf("surface count = %d, want 14", len(decoded.Surfaces))
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
