package tb4bench

import (
	"context"
	"testing"
)

func TestSandboxNetworkIsolation(t *testing.T) {
	mockEngine := NewMockContainerEngine()
	defer mockEngine.Close()

	ctx := context.Background()
	config := ContainerConfig{
		ImageDigest: "ghcr.io/fak/tb4-sandbox@sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
		Name:        "tb4-test-isolated",
		NetworkMode: NetworkModeNone,
		NanoCPUs:    4 * 1e9,
		MemoryBytes: 8 * 1024 * 1024 * 1024,
		PidsLimit:   512,
		WorkingDir:  "/workspace",
	}

	inst, err := mockEngine.CreateContainer(ctx, config)
	if err != nil {
		t.Fatalf("failed to create isolated container: %v", err)
	}

	if err := mockEngine.StartContainer(ctx, inst.ID); err != nil {
		t.Fatalf("failed to start container: %v", err)
	}

	// 1. Verify standard command works
	res, err := mockEngine.ExecCommand(ctx, inst.ID, ExecConfig{
		Cmd: []string{"echo", "hello"},
	})
	if err != nil {
		t.Fatalf("failed to exec local command: %v", err)
	}
	if res.ExitCode != 0 {
		t.Errorf("expected exit code 0, got %d", res.ExitCode)
	}

	// 2. Verify external network requests are refused / fail in isolated mode
	netRes, err := mockEngine.ExecCommand(ctx, inst.ID, ExecConfig{
		Cmd: []string{"curl", "https://example.com"},
	})
	if err != nil {
		t.Fatalf("unexpected exec error: %v", err)
	}
	if netRes.ExitCode == 0 {
		t.Errorf("expected curl in network_mode 'none' to fail, but it succeeded")
	}
	if len(netRes.Stderr) == 0 {
		t.Errorf("expected error output on stderr for airgapped network failure")
	}

	// 3. Test cleanup and reaping
	reaped, err := mockEngine.ReapOrphaned(ctx, "tb4-test-")
	if err != nil {
		t.Fatalf("failed to reap orphans: %v", err)
	}
	if reaped != 1 {
		t.Errorf("expected 1 reaped container, got %d", reaped)
	}
}

func TestValidateImageDigest(t *testing.T) {
	valid := "ghcr.io/fak/tb4@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	if err := ValidateImageDigest(valid); err != nil {
		t.Errorf("expected valid digest to pass: %v", err)
	}

	mutable := "ghcr.io/fak/tb4:latest"
	if err := ValidateImageDigest(mutable); err == nil {
		t.Errorf("expected :latest to fail image validation")
	}

	missingDigest := "ubuntu:22.04"
	if err := ValidateImageDigest(missingDigest); err == nil {
		t.Errorf("expected missing sha256 to fail image validation")
	}
}
