package harnessbench

import (
	"encoding/json"
	"testing"
)

func TestSpawningInvariantWitness(t *testing.T) {
	cycles := 5000
	if !testing.Short() {
		cycles = 10000
	}

	receipt, err := RunSpawningInvariantWitness(cycles, t.TempDir())
	if err != nil {
		t.Fatalf("RunSpawningInvariantWitness: %v", err)
	}

	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	t.Logf("Spawning Invariant Receipt:\n%s", string(raw))

	if !receipt.Pass {
		t.Fatalf("spawning invariant witness failed: %+v", receipt)
	}
	if receipt.Schema != SpawningInvariantSchema {
		t.Errorf("schema mismatch: %s", receipt.Schema)
	}
}

func TestThunderingHerdWitness(t *testing.T) {
	receipt, err := RunThunderingHerdWitness(1000, 16, 512)
	if err != nil {
		t.Fatalf("RunThunderingHerdWitness: %v", err)
	}

	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	t.Logf("Thundering Herd Receipt:\n%s", string(raw))

	if !receipt.Pass {
		t.Fatalf("thundering herd witness failed: %+v", receipt)
	}
	if receipt.EnqueuedCount != 512 {
		t.Errorf("expected 512 enqueued, got %d", receipt.EnqueuedCount)
	}
	if receipt.Rejected429 != 488 {
		t.Errorf("expected 488 rejected with 429, got %d", receipt.Rejected429)
	}
	if !receipt.CleanRecovery {
		t.Errorf("expected clean recovery")
	}
}

func TestThermalTelemetryWitness(t *testing.T) {
	receipt, err := RunThermalTelemetryWitness()
	if err != nil {
		t.Fatalf("RunThermalTelemetryWitness: %v", err)
	}

	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	t.Logf("Thermal Shedding Receipt:\n%s", string(raw))

	if !receipt.Pass {
		t.Fatalf("thermal telemetry witness failed: %+v", receipt)
	}
	if receipt.InitialK != 16 || receipt.ThrottledK != 8 || receipt.RestoredK != 16 {
		t.Errorf("unexpected concurrency progression: 16 -> %d -> %d", receipt.ThrottledK, receipt.RestoredK)
	}
	if !receipt.P3TasksPaused {
		t.Errorf("expected P3 tasks to be paused under thermal throttle")
	}
}
