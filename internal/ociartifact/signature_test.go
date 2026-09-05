package ociartifact

import (
	"crypto/ed25519"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSignedBundle(t *testing.T) {
	TestSignedBundleLifecycle(t)
}

func TestSignedBundleLifecycle(t *testing.T) {
	privPEM, pubPEM, err := GenerateEd25519KeyPair()
	if err != nil {
		t.Fatalf("GenerateEd25519KeyPair failed: %v", err)
	}

	dir := t.TempDir()
	bundlePath := filepath.Join(dir, "test.fakpack")

	// Create a minimal bundle archive with manifest.json and a blob
	manifestData := []byte(`{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.fak.collection.config.v1+json",
    "digest": "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    "size": 15
  },
  "layers": [
    {
      "mediaType": "application/vnd.fak.policy.v1+json",
      "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "size": 0
    }
  ]
}`)

	entries := []archiveEntry{
		{
			name: "manifest.json",
			data: manifestData,
			mode: 0644,
		},
		{
			name: "blobs/sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
			data: []byte(`{"schema":"v1"}`),
			mode: 0644,
		},
	}

	archiveBytes, err := writeTarArchive(entries)
	if err != nil {
		t.Fatalf("writeTarArchive failed: %v", err)
	}

	if err := os.WriteFile(bundlePath, archiveBytes, 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	// 1. Verify before signing -> fails with BUNDLE_SIGNATURE_INVALID
	err = VerifyArtifact(bundlePath, pubPEM)
	if err == nil {
		t.Fatal("expected VerifyArtifact to fail on unsigned bundle, got nil")
	}
	if Code(err) != ErrBundleSignatureInvalid && !strings.Contains(err.Error(), ErrBundleSignatureInvalid) {
		t.Fatalf("expected BUNDLE_SIGNATURE_INVALID, got: %v", err)
	}

	// 2. Sign bundle with private key
	err = SignArtifact(bundlePath, privPEM)
	if err != nil {
		t.Fatalf("SignArtifact failed: %v", err)
	}

	// 3. Verify signed bundle with public key -> succeeds
	err = VerifyArtifact(bundlePath, pubPEM)
	if err != nil {
		t.Fatalf("VerifyArtifact failed on valid signed bundle: %v", err)
	}

	// 4. Verify with wrong public key -> fails with BUNDLE_SIGNATURE_INVALID
	_, otherPubPEM, err := GenerateEd25519KeyPair()
	if err != nil {
		t.Fatalf("generating second key pair: %v", err)
	}
	err = VerifyArtifact(bundlePath, otherPubPEM)
	if err == nil {
		t.Fatal("expected VerifyArtifact with wrong key to fail, got nil")
	}
	if Code(err) != ErrBundleSignatureInvalid && !strings.Contains(err.Error(), ErrBundleSignatureInvalid) {
		t.Fatalf("expected BUNDLE_SIGNATURE_INVALID for wrong key, got: %v", err)
	}

	// 5. Test Hex keys
	pubRaw, privRaw, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("ed25519.GenerateKey: %v", err)
	}
	privHex := hex.EncodeToString(privRaw)
	pubHex := hex.EncodeToString(pubRaw)

	bundlePathHex := filepath.Join(dir, "hex.fakpack")
	if err := os.WriteFile(bundlePathHex, archiveBytes, 0644); err != nil {
		t.Fatalf("WriteFile hex: %v", err)
	}
	if err := SignArtifact(bundlePathHex, privHex); err != nil {
		t.Fatalf("SignArtifact with hex key: %v", err)
	}
	if err := VerifyArtifact(bundlePathHex, pubHex); err != nil {
		t.Fatalf("VerifyArtifact with hex key: %v", err)
	}

	// 6. Tampering with manifest.json invalidates signature
	corruptBytes, err := os.ReadFile(bundlePath)
	if err != nil {
		t.Fatalf("reading signed bundle: %v", err)
	}
	corruptEntries, _, err := readTarArchive(corruptBytes)
	if err != nil {
		t.Fatalf("reading corrupt archive: %v", err)
	}
	for i := range corruptEntries {
		if corruptEntries[i].name == "manifest.json" {
			corruptEntries[i].data = append(corruptEntries[i].data, []byte("   ")...)
		}
	}
	repackedCorrupt, err := writeTarArchive(corruptEntries)
	if err != nil {
		t.Fatalf("repacking corrupt archive: %v", err)
	}
	corruptPath := filepath.Join(dir, "corrupt.fakpack")
	if err := os.WriteFile(corruptPath, repackedCorrupt, 0644); err != nil {
		t.Fatalf("writing corrupt bundle: %v", err)
	}
	err = VerifyArtifact(corruptPath, pubPEM)
	if err == nil {
		t.Fatal("expected VerifyArtifact to fail on tampered manifest, got nil")
	}
	if Code(err) != ErrBundleSignatureInvalid && !strings.Contains(err.Error(), ErrBundleSignatureInvalid) {
		t.Fatalf("expected BUNDLE_SIGNATURE_INVALID on tampered manifest, got: %v", err)
	}

	// 7. Directory layout signing & verification
	dirBundle := filepath.Join(dir, "dirbundle")
	if err := os.MkdirAll(dirBundle, 0755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dirBundle, "manifest.json"), manifestData, 0644); err != nil {
		t.Fatalf("writing dir manifest: %v", err)
	}
	if err := SignArtifact(dirBundle, privPEM); err != nil {
		t.Fatalf("SignArtifact on dir: %v", err)
	}
	if err := VerifyArtifact(dirBundle, pubPEM); err != nil {
		t.Fatalf("VerifyArtifact on dir: %v", err)
	}
}
