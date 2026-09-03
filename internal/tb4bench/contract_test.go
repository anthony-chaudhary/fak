package tb4bench

import (
	"path/filepath"
	"testing"
)

func TestContractSchemaValidation(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "testdata", "tb4bench", "sample_contract.json")
	contract, err := LoadContractFile(fixturePath, true)
	if err != nil {
		t.Fatalf("failed to load valid contract: %v", err)
	}

	if contract.Schema != ContractSchema {
		t.Errorf("expected schema %s, got %s", ContractSchema, contract.Schema)
	}
	if contract.Determinism.Temperature != 0.0 {
		t.Errorf("expected temperature 0.0, got %f", contract.Determinism.Temperature)
	}
	if contract.Determinism.Seed != 42 {
		t.Errorf("expected seed 42, got %d", contract.Determinism.Seed)
	}

	digest, err := contract.Digest()
	if err != nil {
		t.Fatalf("failed to compute contract digest: %v", err)
	}
	if len(digest) != 64 {
		t.Errorf("expected 64-char hex digest, got %d chars", len(digest))
	}

	// Test strict determinism violation: temperature != 0.0
	contract.Determinism.Temperature = 0.7
	if err := contract.Validate(true); err == nil {
		t.Errorf("expected error with temperature=0.7 under strict determinism, got nil")
	}
	contract.Determinism.Temperature = 0.0

	// Test strict determinism violation: seed != 42
	contract.Determinism.Seed = 1234
	if err := contract.Validate(true); err == nil {
		t.Errorf("expected error with seed=1234 under strict determinism, got nil")
	}
	contract.Determinism.Seed = 42

	// Test parity violation: model hash mismatch
	contract.ArmB.Model.Sha256 = "different_hash"
	if err := contract.Validate(false); err == nil {
		t.Errorf("expected error when arm_b model hash differs from contract, got nil")
	}
}

func TestDefaultRunContract(t *testing.T) {
	tasks := []string{"task-b", "task-a"}
	contract := DefaultRunContract("test.gguf", "sha256:abcd", "Q4_K_M", tasks)
	if err := contract.Validate(true); err != nil {
		t.Fatalf("default contract should be valid: %v", err)
	}
	if len(contract.TaskSelection.Tasks) != 2 || contract.TaskSelection.Tasks[0] != "task-a" {
		t.Errorf("expected sorted tasks, got %v", contract.TaskSelection.Tasks)
	}
}
