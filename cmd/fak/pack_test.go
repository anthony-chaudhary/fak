package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fakpack"
	"github.com/anthony-chaudhary/fak/internal/ociartifact"
)

func TestFakPackCreateAndVerify(t *testing.T) {
	dir := t.TempDir()

	lockPath := filepath.Join(dir, "harness.lock.json")
	lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "cli-test-lock-id",
  "budget": {
    "context_tokens": 2048,
    "memory_mib": 256,
    "workers": 1
  },
  "components": [
    {
      "id": "cli-worker",
      "version": "1.0.0",
      "digest": "sha256:cliworkerhash",
      "source": "bin/cli-worker"
    }
  ],
  "assets": [
    {
      "kind": "asset",
      "id": "cli-prompt",
      "source": "assets/prompt.txt"
    }
  ]
}`
	if err := os.WriteFile(lockPath, []byte(lockContent), 0o644); err != nil {
		t.Fatalf("writing lock: %v", err)
	}

	policyPath := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(policyPath, []byte(`{"version":"v1"}`), 0o644); err != nil {
		t.Fatalf("writing policy: %v", err)
	}

	assetsDir := filepath.Join(dir, "assets")
	_ = os.MkdirAll(assetsDir, 0o755)
	_ = os.WriteFile(filepath.Join(assetsDir, "prompt.txt"), []byte("cli prompt text"), 0o644)

	binDir := filepath.Join(dir, "bin")
	_ = os.MkdirAll(binDir, 0o755)
	_ = os.WriteFile(filepath.Join(binDir, "cli-worker"), []byte("#!/bin/sh\necho ok\n"), 0o755)

	bundlePath := filepath.Join(dir, "bundle.fakpack")

	var stdout, stderr bytes.Buffer

	// 1. fak pack create
	createArgs := []string{
		"create",
		"--lock", lockPath,
		"--policy", policyPath,
		"--assets", assetsDir,
		"--bin", binDir,
		"--out", bundlePath,
	}
	code := runPack(&stdout, &stderr, createArgs)
	if code != 0 {
		t.Fatalf("runPack create exited %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Created bundle:") {
		t.Fatalf("unexpected stdout on create: %s", stdout.String())
	}

	// 2. fak pack inspect
	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{"inspect", "--bundle", bundlePath, "--json"})
	if code != 0 {
		t.Fatalf("runPack inspect exited %d, stderr: %s", code, stderr.String())
	}
	var inspectRes fakpack.InspectResult
	if err := json.Unmarshal(stdout.Bytes(), &inspectRes); err != nil {
		t.Fatalf("failed to unmarshal inspect output: %v", err)
	}
	if len(inspectRes.Layers) != 4 { // lock, policy, assets, binaries
		t.Fatalf("expected 4 layers in inspect result, got %d", len(inspectRes.Layers))
	}

	// 3. fak pack verify (plain)
	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{"verify", "--bundle", bundlePath, "--lock", lockPath})
	if code != 0 {
		t.Fatalf("runPack verify exited %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Bundle verified:") {
		t.Fatalf("unexpected verify stdout: %s", stdout.String())
	}

	// 4. fak pack verify (--json)
	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{"verify", "--bundle", bundlePath, "--json"})
	if code != 0 {
		t.Fatalf("runPack verify --json exited %d, stderr: %s", code, stderr.String())
	}
	var verifyRes fakpack.VerifyResult
	if err := json.Unmarshal(stdout.Bytes(), &verifyRes); err != nil {
		t.Fatalf("failed to unmarshal verify output: %v", err)
	}
	if !verifyRes.AirGapVerified {
		t.Fatal("expected airgap verified")
	}

	// 5. Verification failure on tampered bundle
	tamperedPath := filepath.Join(dir, "tampered.fakpack")
	data, _ := os.ReadFile(bundlePath)
	idx := bytes.Index(data, []byte("cli-test-lock-id"))
	if idx >= 0 {
		data[idx] ^= 0x01
	} else {
		data[0] ^= 0x01
	}
	_ = os.WriteFile(tamperedPath, data, 0o644)

	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{"verify", "--bundle", tamperedPath})
	if code != 1 {
		t.Fatalf("expected verify on tampered bundle to exit 1, got %d", code)
	}

	// 6. Usage errors
	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{"create"})
	if code != 2 {
		t.Fatalf("expected usage error (exit 2), got %d", code)
	}

	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{"verify"})
	if code != 2 {
		t.Fatalf("expected usage error (exit 2), got %d", code)
	}

	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{"unknown"})
	if code != 2 {
		t.Fatalf("expected unknown subcommand (exit 2), got %d", code)
	}
}

func TestSignedBundle(t *testing.T) {
	TestSignedPrivateBundleExecution(t)
}

func TestSignedPrivateBundleExecution(t *testing.T) {
	dir := t.TempDir()

	lockPath := filepath.Join(dir, "harness.lock.json")
	lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "signed-bundle-lock-id",
  "platforms": [{"os": "linux", "arch": "amd64"}],
  "budget": {"context_tokens": 1024, "memory_mib": 256, "workers": 1},
  "components": [{"id": "worker", "version": "1.0.0", "digest": "sha256:workerhash", "source": "bin/worker"}],
  "assets": [{"kind": "asset", "id": "prompt", "source": "assets/prompt.txt"}]
}`
	if err := os.WriteFile(lockPath, []byte(lockContent), 0o644); err != nil {
		t.Fatal(err)
	}

	assetsDir := filepath.Join(dir, "assets")
	_ = os.MkdirAll(assetsDir, 0o755)
	_ = os.WriteFile(filepath.Join(assetsDir, "prompt.txt"), []byte("signed asset content"), 0o644)

	binDir := filepath.Join(dir, "bin")
	_ = os.MkdirAll(binDir, 0o755)
	_ = os.WriteFile(filepath.Join(binDir, "worker"), []byte("#!/bin/sh\nexit 0\n"), 0o755)

	bundlePath := filepath.Join(dir, "bundle.fakpack")

	var stdout, stderr bytes.Buffer
	code := runPack(&stdout, &stderr, []string{
		"create",
		"--lock", lockPath,
		"--assets", assetsDir,
		"--bin", binDir,
		"--out", bundlePath,
	})
	if code != 0 {
		t.Fatalf("create failed with %d: %s", code, stderr.String())
	}

	privKey, pubKey, err := ociartifact.GenerateEd25519KeyPair()
	if err != nil {
		t.Fatalf("generate key pair: %v", err)
	}

	privKeyPath := filepath.Join(dir, "cosign.key")
	pubKeyPath := filepath.Join(dir, "cosign.pub")
	if err := os.WriteFile(privKeyPath, []byte(privKey), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(pubKeyPath, []byte(pubKey), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1. Sign bundle
	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{
		"sign",
		"--bundle", bundlePath,
		"--key", privKeyPath,
	})
	if code != 0 {
		t.Fatalf("sign failed with %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Signed bundle:") {
		t.Fatalf("unexpected output: %s", stdout.String())
	}

	// 2. Verify bundle with correct public key
	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{
		"verify",
		"--bundle", bundlePath,
		"--verify-key", pubKeyPath,
	})
	if code != 0 {
		t.Fatalf("verify with key failed with %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Bundle verified:") {
		t.Fatalf("unexpected output: %s", stdout.String())
	}

	// 3. Verify with wrong public key fails
	_, wrongPub, _ := ociartifact.GenerateEd25519KeyPair()
	wrongPubPath := filepath.Join(dir, "wrong.pub")
	_ = os.WriteFile(wrongPubPath, []byte(wrongPub), 0o644)

	stdout.Reset()
	stderr.Reset()
	code = runPack(&stdout, &stderr, []string{
		"verify",
		"--bundle", bundlePath,
		"--verify-key", wrongPubPath,
	})
	if code != 1 {
		t.Fatalf("expected verify with wrong key to fail with 1, got %d", code)
	}
}
