package fakpack

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func createTestFixtures(t testing.TB) (dir, lockPath, policyPath, assetsDir, binDir, modelPath string) {
	t.Helper()
	dir = t.TempDir()

	lockPath = filepath.Join(dir, "harness.lock.json")
	lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:test-lock-digest-placeholder",
  "platforms": [
    {"os": "linux", "arch": "amd64"},
    {"os": "darwin", "arch": "arm64"}
  ],
  "budget": {
    "context_tokens": 4096,
    "memory_mib": 512,
    "workers": 2
  },
  "components": [
    {
      "id": "my-worker",
      "version": "1.0.0",
      "digest": "sha256:dummyworkerhash",
      "source": "bin/my-worker"
    }
  ],
  "assets": [
    {
      "kind": "asset",
      "id": "prompt-template",
      "source": "assets/prompt.txt"
    },
    {
      "kind": "instruction",
      "id": "sys-prompt",
      "value": "You are a helpful assistant.",
      "source": "local"
    }
  ]
}`
	if err := os.WriteFile(lockPath, []byte(lockContent), 0o644); err != nil {
		t.Fatalf("writing lock: %v", err)
	}

	policyPath = filepath.Join(dir, "policy.json")
	if err := os.WriteFile(policyPath, []byte(`{"version":"v1","allow":["tool:read"]}`), 0o644); err != nil {
		t.Fatalf("writing policy: %v", err)
	}

	assetsDir = filepath.Join(dir, "assets")
	if err := os.MkdirAll(assetsDir, 0o755); err != nil {
		t.Fatalf("mkdir assets: %v", err)
	}
	if err := os.WriteFile(filepath.Join(assetsDir, "prompt.txt"), []byte("Hello airgap harness world!"), 0o644); err != nil {
		t.Fatalf("writing prompt: %v", err)
	}

	binDir = filepath.Join(dir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatalf("mkdir bin: %v", err)
	}
	if err := os.WriteFile(filepath.Join(binDir, "my-worker"), []byte("#!/bin/sh\necho worker\n"), 0o755); err != nil {
		t.Fatalf("writing bin: %v", err)
	}

	modelPath = filepath.Join(dir, "model.bin")
	if err := os.WriteFile(modelPath, []byte("fake-gguf-weights-data-bytes"), 0o644); err != nil {
		t.Fatalf("writing model: %v", err)
	}

	return dir, lockPath, policyPath, assetsDir, binDir, modelPath
}

func TestFakPackRoundtrip(t *testing.T) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(t)
	outBundle := filepath.Join(dir, "bundle.fakpack")

	createOpts := CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    outBundle,
	}

	res, err := Create(createOpts)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	if res.BundlePath != outBundle {
		t.Fatalf("unexpected bundle path: got %s, want %s", res.BundlePath, outBundle)
	}
	if !strings.HasPrefix(res.ManifestDigest, "sha256:") {
		t.Fatalf("bad manifest digest: %s", res.ManifestDigest)
	}
	if len(res.Layers) != 5 {
		t.Fatalf("expected 5 layers (lock, policy, assets, binaries, model), got %d", len(res.Layers))
	}
	if res.TotalSize <= 0 {
		t.Fatalf("expected positive total size, got %d", res.TotalSize)
	}

	verifyRes, err := Verify(VerifyOptions{
		BundlePath:       outBundle,
		ExpectedLockPath: lockPath,
	})
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	if !verifyRes.AirGapVerified {
		t.Fatal("expected air-gap verified")
	}
	if !verifyRes.LockMatches {
		t.Fatal("expected lock matches")
	}
	if verifyRes.LayersVerified != 5 {
		t.Fatalf("expected 5 layers verified, got %d", verifyRes.LayersVerified)
	}
}

func TestFakPackTamperWitness(t *testing.T) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(t)
	outBundle := filepath.Join(dir, "original.fakpack")

	_, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    outBundle,
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	data, err := os.ReadFile(outBundle)
	if err != nil {
		t.Fatalf("read bundle: %v", err)
	}

	// Tamper witness: find bytes in the payload of model layer and flip a bit
	needle := []byte("fake-gguf-weights-data-bytes")
	idx := bytes.Index(data, needle)
	if idx < 0 {
		t.Fatalf("could not find payload needle in archive")
	}

	tampered := make([]byte, len(data))
	copy(tampered, data)
	tampered[idx+5] ^= 0x01

	tamperedPath := filepath.Join(dir, "tampered.fakpack")
	if err := os.WriteFile(tamperedPath, tampered, 0o644); err != nil {
		t.Fatalf("write tampered bundle: %v", err)
	}

	_, err = Verify(VerifyOptions{
		BundlePath: tamperedPath,
	})
	if err == nil {
		t.Fatal("expected Verify to fail on tampered payload, got nil")
	}
	if !strings.Contains(err.Error(), ErrBundleDigestMismatch) {
		t.Fatalf("expected error containing %s, got: %v", ErrBundleDigestMismatch, err)
	}
}

func TestFakPackCorruptArchive(t *testing.T) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(t)
	outBundle := filepath.Join(dir, "original.fakpack")

	_, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    outBundle,
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	data, err := os.ReadFile(outBundle)
	if err != nil {
		t.Fatalf("read bundle: %v", err)
	}

	// Truncate archive to 64 bytes
	corruptPath := filepath.Join(dir, "corrupt.fakpack")
	if err := os.WriteFile(corruptPath, data[:64], 0o644); err != nil {
		t.Fatalf("write corrupt bundle: %v", err)
	}

	_, err = Verify(VerifyOptions{
		BundlePath: corruptPath,
	})
	if err == nil {
		t.Fatal("expected Verify to fail on truncated archive, got nil")
	}
	if !strings.Contains(err.Error(), ErrBundleCorrupt) {
		t.Fatalf("expected error containing %s, got: %v", ErrBundleCorrupt, err)
	}
}

func TestFakPackAirgapValidation(t *testing.T) {
	dir := t.TempDir()

	// 1. Lock with http:// asset reference
	badLockPath1 := filepath.Join(dir, "bad1.lock.json")
	badLock1 := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "bad-lock-1",
  "components": [{"id": "c1", "version": "1.0.0", "digest": "sha256:dummy", "source": "bin/c1"}],
  "assets": [{"kind": "asset", "id": "a1", "source": "http://external.site/prompt.txt"}]
}`
	_ = os.WriteFile(badLockPath1, []byte(badLock1), 0o644)
	out1 := filepath.Join(dir, "out1.fakpack")

	_, err := Create(CreateOptions{
		LockPath: badLockPath1,
		OutPath:  out1,
	})
	if err == nil {
		t.Fatal("expected creation to fail on http:// asset reference")
	}
	if !strings.Contains(err.Error(), ErrAirgapViolation) {
		t.Fatalf("expected error containing %s, got: %v", ErrAirgapViolation, err)
	}

	// 2. Lock with https:// component source
	badLockPath2 := filepath.Join(dir, "bad2.lock.json")
	badLock2 := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "bad-lock-2",
  "components": [{"id": "c2", "version": "1.0.0", "digest": "sha256:dummy", "source": "https://external.site/bin.tar.gz"}],
  "assets": [{"kind": "instruction", "id": "a2", "value": "test", "source": "local"}]
}`
	_ = os.WriteFile(badLockPath2, []byte(badLock2), 0o644)
	out2 := filepath.Join(dir, "out2.fakpack")

	_, err = Create(CreateOptions{
		LockPath: badLockPath2,
		OutPath:  out2,
	})
	if err == nil {
		t.Fatal("expected creation to fail on https:// component reference")
	}
	if !strings.Contains(err.Error(), ErrAirgapViolation) {
		t.Fatalf("expected error containing %s, got: %v", ErrAirgapViolation, err)
	}
}

func TestFakPackInspect(t *testing.T) {
	dir, lockPath, policyPath, assetsDir, binDir, modelPath := createTestFixtures(t)
	outBundle := filepath.Join(dir, "bundle.fakpack")

	createRes, err := Create(CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    outBundle,
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	inspectRes, err := Inspect(outBundle)
	if err != nil {
		t.Fatalf("Inspect failed: %v", err)
	}

	if inspectRes.BundlePath != outBundle {
		t.Fatalf("expected bundle path %s, got %s", outBundle, inspectRes.BundlePath)
	}
	if inspectRes.LockSummary.ID != createRes.LockID {
		t.Fatalf("expected lock ID %s, got %s", createRes.LockID, inspectRes.LockSummary.ID)
	}
	if len(inspectRes.Layers) != 5 {
		t.Fatalf("expected 5 layers, got %d", len(inspectRes.Layers))
	}
	if inspectRes.TotalSize <= 0 {
		t.Fatalf("expected positive total size, got %d", inspectRes.TotalSize)
	}
	if inspectRes.CreatedTime == "" {
		t.Fatal("expected non-empty created time")
	}
	if len(inspectRes.Platforms) != 2 {
		t.Fatalf("expected 2 platforms, got %d (%v)", len(inspectRes.Platforms), inspectRes.Platforms)
	}
}
