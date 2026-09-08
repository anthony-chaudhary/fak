package harnessresolve

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	publiclockv2 "github.com/anthony-chaudhary/fak/pkg/harnesskit/lockv2"
)

func TestProductLockV2_CanonicalIDMatchesHarnessKit(t *testing.T) {
	publicLock := &publiclockv2.Lock{
		Schema:    publiclockv2.ProductLockSchemaV2,
		Platforms: []publiclockv2.PlatformRequirement{{OS: "linux", Arch: "amd64", Contract: "v1"}},
		Budget:    publiclockv2.LockBudget{ContextTokens: 1024, MemoryMiB: 256, Workers: 1},
		Components: []publiclockv2.LockedComponent{{
			ID: "kernel", Version: "1.0.0", Digest: "sha256:kernel", Source: "registry/kernel",
		}},
		Assets: []publiclockv2.LockedAsset{{Kind: "instruction", ID: "guide", Value: "line 1\r\nline 2", Source: "fixture"}},
	}
	want, err := publiclockv2.CanonicalID(publicLock)
	if err != nil {
		t.Fatal(err)
	}
	publicLock.ID = want
	raw, err := json.Marshal(publicLock)
	if err != nil {
		t.Fatal(err)
	}

	got, err := CanonicalIDV2(raw)
	if err != nil {
		t.Fatalf("internal compatibility wrapper rejected public lock: %v", err)
	}
	if got != want {
		t.Fatalf("canonical ID drift: internal=%s public=%s", got, want)
	}
}

func TestProductLockV2_MultiPlatform(t *testing.T) {
	baseLock := func() ProductLockV2 {
		return ProductLockV2{
			Schema: LockSchemaV2,
			Platforms: []LockEnvironment{
				{OS: "linux", Arch: "amd64", Contract: "v1"},
				{OS: "darwin", Arch: "arm64", Contract: "v1"},
				{OS: "windows", Arch: "amd64", Contract: "v1"},
			},
			Components: []LockedComponentV2{
				{
					ID:      "kernel",
					Version: "1.0.0",
					Digest:  "sha256:kernel",
					Source:  "registry/kernel",
				},
			},
			Assets: []LockedAssetV2{
				{
					Kind:   "secret",
					ID:     "auth-token",
					Ref:    "env:AUTH_TOKEN",
					Source: "company",
				},
			},
		}
	}

	t.Run("compatible across linux darwin windows", func(t *testing.T) {
		lock := baseLock()
		data, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateProductLockV2(data); err != nil {
			t.Fatalf("expected valid multi-platform lock, got: %v", err)
		}
	})

	t.Run("component incompatible OS rejected", func(t *testing.T) {
		lock := baseLock()
		lock.Components = append(lock.Components, LockedComponentV2{
			ID:            "linux-only-daemon",
			Version:       "1.0.0",
			Digest:        "sha256:daemon",
			Source:        "registry/daemon",
			Compatibility: LockCompatibilityV2{OS: []string{"linux"}},
		})
		data, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidateProductLockV2(data)
		if err == nil {
			t.Fatal("expected error for incompatible OS, got nil")
		}
		if !strings.Contains(err.Error(), "incompatible OS") {
			t.Fatalf("expected incompatible OS error, got: %v", err)
		}
	})

	t.Run("component incompatible arch rejected", func(t *testing.T) {
		lock := baseLock()
		lock.Components = append(lock.Components, LockedComponentV2{
			ID:            "amd64-simd",
			Version:       "1.0.0",
			Digest:        "sha256:simd",
			Source:        "registry/simd",
			Compatibility: LockCompatibilityV2{Arch: []string{"amd64"}},
		})
		data, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidateProductLockV2(data)
		if err == nil {
			t.Fatal("expected error for incompatible arch, got nil")
		}
		if !strings.Contains(err.Error(), "darwin/arm64") || !strings.Contains(err.Error(), "incompatible arch") {
			t.Fatalf("expected darwin/arm64 incompatible arch error, got: %v", err)
		}
	})

	t.Run("contract mismatch rejected", func(t *testing.T) {
		lock := baseLock()
		lock.Components = append(lock.Components, LockedComponentV2{
			ID:            "contract-v2-adapter",
			Version:       "2.0.0",
			Digest:        "sha256:adapter",
			Source:        "registry/adapter",
			Compatibility: LockCompatibilityV2{Contract: "v2"},
		})
		data, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidateProductLockV2(data)
		if err == nil {
			t.Fatal("expected error for contract mismatch, got nil")
		}
		if !strings.Contains(err.Error(), "requires contract \"v2\", got \"v1\"") {
			t.Fatalf("expected contract mismatch error, got: %v", err)
		}
	})

	t.Run("empty platforms rejected", func(t *testing.T) {
		lock := baseLock()
		lock.Platforms = nil
		data, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidateProductLockV2(data)
		if err == nil {
			t.Fatal("expected error for missing platforms, got nil")
		}
		if !strings.Contains(err.Error(), "platforms must not be empty") {
			t.Fatalf("expected platforms empty error, got: %v", err)
		}
	})

	t.Run("duplicate platform rejected", func(t *testing.T) {
		lock := baseLock()
		lock.Platforms = append(lock.Platforms, LockEnvironment{OS: "linux", Arch: "amd64", Contract: "v1"})
		data, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidateProductLockV2(data)
		if err == nil {
			t.Fatal("expected error for duplicate platform, got nil")
		}
		if !strings.Contains(err.Error(), "duplicate platform") {
			t.Fatalf("expected duplicate platform error, got: %v", err)
		}
	})

	t.Run("serialized platform matrix remains compatible", func(t *testing.T) {
		wire := struct {
			Schema     string              `json:"schema"`
			ID         string              `json:"id,omitempty"`
			Matrix     *PlatformMatrix     `json:"matrix"`
			Components []LockedComponentV2 `json:"components"`
			Assets     []LockedAssetV2     `json:"assets"`
		}{
			Schema: LockSchemaV2,
			Matrix: &PlatformMatrix{
				OS:       []string{"linux", "darwin", "windows"},
				Arch:     []string{"amd64"},
				Contract: []string{"v1"},
			},
			Components: baseLock().Components,
			Assets:     baseLock().Assets,
		}
		data, err := json.Marshal(wire)
		if err != nil {
			t.Fatal(err)
		}
		wire.ID, err = CanonicalIDV2(data)
		if err != nil {
			t.Fatalf("canonicalize serialized matrix: %v", err)
		}
		data, err = json.Marshal(wire)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateProductLockV2(data); err != nil {
			t.Fatalf("serialized matrix lock rejected: %v", err)
		}
		parsed, err := ParseProductLockV2(data)
		if err != nil {
			t.Fatalf("parse serialized matrix: %v", err)
		}
		if len(parsed.Platforms) != 3 {
			t.Fatalf("matrix expanded to %d platforms, want 3", len(parsed.Platforms))
		}
	})

}

func TestProductLockV2_SecretPlaintextLeak(t *testing.T) {
	validLock := func() ProductLockV2 {
		return ProductLockV2{
			Schema: LockSchemaV2,
			Platforms: []LockEnvironment{
				{OS: "linux", Arch: "amd64", Contract: "v1"},
			},
			Components: []LockedComponentV2{
				{
					ID:      "kernel",
					Version: "1.0.0",
					Digest:  "sha256:kernel",
					Source:  "registry/kernel",
				},
			},
		}
	}

	t.Run("fails closed on non-empty plaintext value", func(t *testing.T) {
		lock := validLock()
		lock.Assets = []LockedAssetV2{
			{
				Kind:   "secret",
				ID:     "prod-db-credentials",
				Value:  "postgresql://admin:super_secret_pw@10.0.0.1/db",
				Ref:    "vault:secrets/prod/db#url",
				Source: "company",
			},
		}
		data, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}

		err = ValidateProductLockV2(data)
		if err == nil {
			t.Fatal("expected ValidateProductLockV2 to fail on plaintext secret, got nil")
		}

		if !strings.Contains(err.Error(), "SECRET_PLAINTEXT_LEAK") {
			t.Fatalf("expected error string to contain SECRET_PLAINTEXT_LEAK, got: %v", err)
		}

		var leakErr *SecretPlaintextLeakError
		if !errors.As(err, &leakErr) {
			t.Fatalf("expected typed *SecretPlaintextLeakError, got %T (%v)", err, err)
		}
		if leakErr.AssetID != "prod-db-credentials" {
			t.Fatalf("expected leak error asset ID 'prod-db-credentials', got %q", leakErr.AssetID)
		}
	})

	t.Run("passes with empty value and valid ref schemes", func(t *testing.T) {
		validRefs := []string{
			"env:API_KEY",
			"env:SECRET_123_VALUE",
			"file:/etc/secrets/token.key",
			"file:./relative/path/key.pem",
			"vault:secret/data/app/prod#token",
			"vault:kv/v2/database-credentials",
			"keyring:system/anthony#token",
		}
		for _, ref := range validRefs {
			lock := validLock()
			lock.Assets = []LockedAssetV2{
				{
					Kind:   "secret",
					ID:     "api-token",
					Value:  "",
					Ref:    ref,
					Source: "company",
				},
			}
			data, err := json.Marshal(lock)
			if err != nil {
				t.Fatal(err)
			}
			if err := ValidateProductLockV2(data); err != nil {
				t.Errorf("expected ref %q to pass validation, got: %v", ref, err)
			}
		}
	})

	t.Run("fails closed on invalid ref scheme or format", func(t *testing.T) {
		invalidRefs := []string{
			"",
			"plaintext_secret",
			"http://example.com/secret",
			"aws:secretsmanager:key",
			"env:",
			"vault:",
			"file:",
			"keyring:",
			"env:bad key with spaces",
		}
		for _, ref := range invalidRefs {
			lock := validLock()
			lock.Assets = []LockedAssetV2{
				{
					Kind:   "secret",
					ID:     "api-token",
					Value:  "",
					Ref:    ref,
					Source: "company",
				},
			}
			data, err := json.Marshal(lock)
			if err != nil {
				t.Fatal(err)
			}
			err = ValidateProductLockV2(data)
			if err == nil {
				t.Errorf("expected invalid ref %q to fail, got nil", ref)
			}
		}
	})

	t.Run("non-secret assets allow plaintext values", func(t *testing.T) {
		lock := validLock()
		lock.Assets = []LockedAssetV2{
			{
				Kind:   "instruction",
				ID:     "system-prompt",
				Value:  "You are a helpful assistant.",
				Source: "company",
			},
			{
				Kind:   "workflow",
				ID:     "audit-flow",
				Value:  "run audit",
				Source: "legal",
			},
		}
		data, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateProductLockV2(data); err != nil {
			t.Fatalf("expected non-secret assets with value to pass, got: %v", err)
		}
	})
}

func TestProductLockV2_CanonicalLF(t *testing.T) {
	lock := ProductLockV2{
		Schema: LockSchemaV2,
		Platforms: []LockEnvironment{
			{OS: "linux", Arch: "amd64", Contract: "v1"},
			{OS: "darwin", Arch: "arm64", Contract: "v1"},
			{OS: "windows", Arch: "amd64", Contract: "v1"},
		},
		Components: []LockedComponentV2{
			{
				ID:      "kernel",
				Version: "1.0.0",
				Digest:  "sha256:kernel",
				Source:  "registry/kernel",
			},
		},
		Assets: []LockedAssetV2{
			{
				Kind:   "secret",
				ID:     "auth-key",
				Ref:    "env:AUTH_KEY",
				Source: "company",
			},
			{
				Kind:   "instruction",
				ID:     "multi-line-guide",
				Value:  "Line 1: Initialize\nLine 2: Execute\nLine 3: Teardown",
				Source: "company",
			},
		},
	}

	canonicalID, err := CanonicalLockIDV2(lock)
	if err != nil {
		t.Fatalf("failed to compute canonical ID: %v", err)
	}
	if !strings.HasPrefix(canonicalID, "sha256:") {
		t.Fatalf("canonical ID missing sha256 prefix: %s", canonicalID)
	}
	lock.ID = canonicalID

	lfBytes, err := json.MarshalIndent(lock, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	lfBytes = CanonicalizeLF(lfBytes)

	crlfBytes := bytes.ReplaceAll(lfBytes, []byte("\n"), []byte("\r\n"))
	if !bytes.Contains(crlfBytes, []byte("\r\n")) {
		t.Fatal("crlfBytes should contain CRLF sequences")
	}
	if bytes.Contains(lfBytes, []byte("\r")) {
		t.Fatal("lfBytes should not contain CR bytes")
	}

	if err := ValidateProductLockV2(lfBytes); err != nil {
		t.Fatalf("LF payload validation failed: %v", err)
	}
	if err := ValidateProductLockV2(crlfBytes); err != nil {
		t.Fatalf("CRLF payload validation failed: %v", err)
	}

	idLF, err := CanonicalIDV2(lfBytes)
	if err != nil {
		t.Fatalf("LF canonical ID extraction failed: %v", err)
	}
	idCRLF, err := CanonicalIDV2(crlfBytes)
	if err != nil {
		t.Fatalf("CRLF canonical ID extraction failed: %v", err)
	}

	if idLF != canonicalID {
		t.Fatalf("LF canonical ID %s != expected %s", idLF, canonicalID)
	}
	if idCRLF != canonicalID {
		t.Fatalf("CRLF canonical ID %s != expected %s", idCRLF, canonicalID)
	}
	if idLF != idCRLF {
		t.Fatalf("CRLF and LF payloads produced different canonical IDs: %s vs %s", idCRLF, idLF)
	}

	t.Run("tampered ID rejected under both LF and CRLF", func(t *testing.T) {
		tamperedLock := lock
		tamperedLock.ID = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
		tamperedLF, err := json.MarshalIndent(tamperedLock, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		tamperedLF = CanonicalizeLF(tamperedLF)
		tamperedCRLF := bytes.ReplaceAll(tamperedLF, []byte("\n"), []byte("\r\n"))

		errLF := ValidateProductLockV2(tamperedLF)
		if errLF == nil || !strings.Contains(errLF.Error(), "lock id mismatch") {
			t.Fatalf("expected lock id mismatch on LF, got: %v", errLF)
		}

		errCRLF := ValidateProductLockV2(tamperedCRLF)
		if errCRLF == nil || !strings.Contains(errCRLF.Error(), "lock id mismatch") {
			t.Fatalf("expected lock id mismatch on CRLF, got: %v", errCRLF)
		}
	})
}
