package fakpack

import (
	"bytes"
	"encoding/json"
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

// writeBinClosureLock writes a minimal v2 product lock that declares exactly the
// named components (id==file name in the bin dir) and nothing else.
func writeBinClosureLock(t testing.TB, path string, comps ...string) {
	t.Helper()
	var sb strings.Builder
	sb.WriteString(`{"schema":"fak.harness-product-lock/v2","id":"sha256:declared-binary-closure-lock","components":[`)
	for i, id := range comps {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(`{"id":"` + id + `","version":"1.0.0","digest":"sha256:declared","source":"bin/` + id + `"}`)
	}
	sb.WriteString(`]}`)
	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		t.Fatalf("writing lock %s: %v", path, err)
	}
}

// binariesLayerDigest returns the manifest-declared digest of the binaries layer.
func binariesLayerDigest(t testing.TB, res *CreateResult) string {
	t.Helper()
	for _, l := range res.Layers {
		if l.MediaType == MediaTypeBinaries {
			return l.Digest
		}
	}
	t.Fatalf("no binaries layer in result: %+v", res.Layers)
	return ""
}

// binariesLayerEntries returns the set of names archived in the bundle's binaries
// layer, including a base-name alias for each entry.
func binariesLayerEntries(t testing.TB, bundlePath string) map[string]bool {
	t.Helper()
	files, err := readArchive(bundlePath)
	if err != nil {
		t.Fatalf("readArchive(%s): %v", bundlePath, err)
	}
	var manifest Manifest
	if err := json.Unmarshal(files["manifest.json"], &manifest); err != nil {
		t.Fatalf("unmarshal manifest: %v", err)
	}
	for _, layer := range manifest.Layers {
		if layer.MediaType != MediaTypeBinaries {
			continue
		}
		hexDigest := strings.TrimPrefix(layer.Digest, "sha256:")
		blob, ok := files["blobs/sha256/"+hexDigest]
		if !ok {
			t.Fatalf("binaries blob %s missing", layer.Digest)
		}
		entries, err := listTarGzFiles(blob)
		if err != nil {
			t.Fatalf("listTarGzFiles(binaries): %v", err)
		}
		return entries
	}
	t.Fatal("no binaries layer in manifest")
	return nil
}

// TestCreateDeclaredBinaryClosure witnesses that the packaged binaries layer is
// exactly the lock-declared component closure: unrelated executables never enter,
// a missing or escaping declared executable fails before publication, and
// identical inputs yield identical content digests.
func TestCreateDeclaredBinaryClosure(t *testing.T) {
	t.Run("declared_only", func(t *testing.T) {
		dir, lockPath, _, _, binDir, _ := createTestFixtures(t)
		if err := os.WriteFile(filepath.Join(binDir, "helper"), []byte("#!/bin/sh\necho helper\n"), 0o755); err != nil {
			t.Fatalf("write helper: %v", err)
		}
		// Unrelated executable that is NOT referenced by the lock.
		if err := os.WriteFile(filepath.Join(binDir, "rogue"), []byte("#!/bin/sh\necho rogue\n"), 0o755); err != nil {
			t.Fatalf("write rogue: %v", err)
		}
		writeBinClosureLock(t, lockPath, "my-worker", "helper")

		outBundle := filepath.Join(dir, "bundle.fakpack")
		if _, err := Create(CreateOptions{LockPath: lockPath, BinDir: binDir, OutPath: outBundle}); err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		entries := binariesLayerEntries(t, outBundle)
		for _, want := range []string{"my-worker", "helper"} {
			if !entries[want] {
				t.Fatalf("declared binary %q absent from binaries layer; got %v", want, entries)
			}
		}
		if entries["rogue"] {
			t.Fatalf("unrelated executable %q leaked into binaries layer; got %v", "rogue", entries)
		}
	})

	t.Run("missing_declared_fails", func(t *testing.T) {
		dir, lockPath, _, _, binDir, _ := createTestFixtures(t)
		writeBinClosureLock(t, lockPath, "ghost")
		outBundle := filepath.Join(dir, "ghost.fakpack")

		_, err := Create(CreateOptions{LockPath: lockPath, BinDir: binDir, OutPath: outBundle})
		if err == nil {
			t.Fatal("expected Create to refuse a missing declared binary")
		}
		if Code(err) != ErrComponentMissing {
			t.Fatalf("expected code %s, got %q (%v)", ErrComponentMissing, Code(err), err)
		}
		if _, statErr := os.Stat(outBundle); !os.IsNotExist(statErr) {
			t.Fatalf("archive must not be published on refusal, stat err=%v", statErr)
		}
	})

	t.Run("escape_refused", func(t *testing.T) {
		dir, lockPath, _, _, binDir, _ := createTestFixtures(t)
		// A file one level above BinDir, reachable only by ../ traversal.
		if err := os.WriteFile(filepath.Join(dir, "outside"), []byte("#!/bin/sh\necho outside\n"), 0o755); err != nil {
			t.Fatalf("write outside: %v", err)
		}
		writeBinClosureLock(t, lockPath, "escapee")
		// Force Source to be the traversal path, not the base name.
		lockTraversal := `{"schema":"fak.harness-product-lock/v2","id":"sha256:escape","components":[{"id":"escapee","version":"1.0.0","digest":"sha256:declared","source":"../outside"}]}`
		if err := os.WriteFile(lockPath, []byte(lockTraversal), 0o644); err != nil {
			t.Fatalf("write traversal lock: %v", err)
		}
		outBundle := filepath.Join(dir, "escape.fakpack")

		_, err := Create(CreateOptions{LockPath: lockPath, BinDir: binDir, OutPath: outBundle})
		if err == nil || Code(err) != ErrComponentMissing {
			t.Fatalf("expected traversal refusal %s, got %v", ErrComponentMissing, err)
		}
		if _, statErr := os.Stat(outBundle); !os.IsNotExist(statErr) {
			t.Fatalf("archive must not be published on traversal refusal, stat err=%v", statErr)
		}

		// Symlink escape, best effort: symlink creation may be unavailable.
		target := filepath.Join(dir, "link-target")
		if err := os.WriteFile(target, []byte("#!/bin/sh\necho link\n"), 0o755); err != nil {
			t.Fatalf("write symlink target: %v", err)
		}
		if err := os.Symlink(target, filepath.Join(binDir, "link-escape")); err != nil {
			t.Logf("symlink unsupported, skipping symlink escape: %v", err)
			return
		}
		if err := os.WriteFile(lockPath, []byte(`{"schema":"fak.harness-product-lock/v2","id":"sha256:link-escape","components":[{"id":"link-escape","version":"1.0.0","digest":"sha256:declared","source":"bin/link-escape"}]}`), 0o644); err != nil {
			t.Fatalf("write symlink lock: %v", err)
		}
		outSymlink := filepath.Join(dir, "symlink-escape.fakpack")
		_, err = Create(CreateOptions{LockPath: lockPath, BinDir: binDir, OutPath: outSymlink})
		if err == nil || Code(err) != ErrComponentMissing {
			t.Fatalf("expected symlink refusal %s, got %v", ErrComponentMissing, err)
		}
	})

	t.Run("deterministic_digest", func(t *testing.T) {
		dir, lockPath, _, _, binDir, _ := createTestFixtures(t)
		if err := os.WriteFile(filepath.Join(binDir, "helper"), []byte("#!/bin/sh\necho helper\n"), 0o755); err != nil {
			t.Fatalf("write helper: %v", err)
		}
		if err := os.WriteFile(filepath.Join(binDir, "rogue"), []byte("#!/bin/sh\necho rogue\n"), 0o755); err != nil {
			t.Fatalf("write rogue: %v", err)
		}
		writeBinClosureLock(t, lockPath, "my-worker", "helper")

		resA, err := Create(CreateOptions{LockPath: lockPath, BinDir: binDir, OutPath: filepath.Join(dir, "a.fakpack")})
		if err != nil {
			t.Fatalf("Create A failed: %v", err)
		}
		resB, err := Create(CreateOptions{LockPath: lockPath, BinDir: binDir, OutPath: filepath.Join(dir, "b.fakpack")})
		if err != nil {
			t.Fatalf("Create B failed: %v", err)
		}
		if digA, digB := binariesLayerDigest(t, resA), binariesLayerDigest(t, resB); digA != digB {
			t.Fatalf("binaries layer digest not deterministic: %s != %s", digA, digB)
		}
	})
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
