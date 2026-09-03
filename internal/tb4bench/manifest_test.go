package tb4bench

import (
	"path/filepath"
	"testing"
)

func TestManifestValidation(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "testdata", "tb4bench", "sample_manifest.json")
	suite, err := LoadManifestFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to load valid manifest suite: %v", err)
	}

	if suite.Benchmark != BenchmarkName {
		t.Errorf("expected benchmark %s, got %s", BenchmarkName, suite.Benchmark)
	}
	if len(suite.Tasks) != 2 {
		t.Fatalf("expected 2 tasks, got %d", len(suite.Tasks))
	}

	task1 := suite.Tasks[0]
	if task1.Category != CategoryRefactor {
		t.Errorf("expected category %s, got %s", CategoryRefactor, task1.Category)
	}

	// Verify oracle script hash check
	script := []byte("#!/bin/bash\nexit 0\n")
	if err := task1.VerifyOracleScript(script); err != nil {
		t.Errorf("expected script hash verification to pass: %v", err)
	}

	tampered := []byte("#!/bin/bash\nexit 1\n")
	if err := task1.VerifyOracleScript(tampered); err == nil {
		t.Errorf("expected tampered script to fail hash verification")
	}

	// Test invalid category
	invalidTask := task1
	invalidTask.Category = "unknown-category"
	if err := invalidTask.Validate(); err == nil {
		t.Errorf("expected error for invalid category, got nil")
	}

	// Test non-immutable image
	invalidTask = task1
	invalidTask.EnvironmentImageDigest = "ubuntu:latest"
	if err := invalidTask.Validate(); err == nil {
		t.Errorf("expected error for non-sha256 image digest, got nil")
	}
}
