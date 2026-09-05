package lockv2

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestProductLockV2(t *testing.T) {
	t.Run("crlf vs lf produces identical canonical ID", func(t *testing.T) {
		lockLF := &Lock{
			Schema: ProductLockSchemaV2,
			Platforms: []PlatformRequirement{
				{OS: "linux", Arch: "amd64"},
				{OS: "darwin", Arch: "arm64"},
				{OS: "windows", Arch: "amd64"},
			},
			Budget: LockBudget{ContextTokens: 1500, MemoryMiB: 320, Workers: 1},
			Components: []LockedComponent{
				{
					ID:            "kernel",
					Version:       "1.0.0",
					Digest:        "sha256:kernel-digest",
					Source:        "registry/kernel",
					Reason:        "root",
					Provider:      "system",
					Provides:      []string{"runtime"},
					Compatibility: LockCompatibility{OS: []string{"darwin", "linux", "windows"}, Arch: []string{"amd64", "arm64"}},
					Adapters:      []string{"instruction"},
				},
			},
			Assets: []LockedAsset{
				{
					Kind:   "instruction",
					ID:     "guidance",
					Value:  "line 1\nline 2\nline 3\n",
					Source: "legal",
				},
				{
					Kind:   "secret",
					ID:     "db-token",
					Ref:    "vault:secret/data/db#password",
					Source: "vault",
				},
			},
		}

		lockCRLF := &Lock{
			Schema: ProductLockSchemaV2,
			Platforms: []PlatformRequirement{
				{OS: "linux", Arch: "amd64"},
				{OS: "darwin", Arch: "arm64"},
				{OS: "windows", Arch: "amd64"},
			},
			Budget: LockBudget{ContextTokens: 1500, MemoryMiB: 320, Workers: 1},
			Components: []LockedComponent{
				{
					ID:            "kernel",
					Version:       "1.0.0",
					Digest:        "sha256:kernel-digest",
					Source:        "registry/kernel",
					Reason:        "root",
					Provider:      "system",
					Provides:      []string{"runtime"},
					Compatibility: LockCompatibility{OS: []string{"darwin", "linux", "windows"}, Arch: []string{"amd64", "arm64"}},
					Adapters:      []string{"instruction"},
				},
			},
			Assets: []LockedAsset{
				{
					Kind:   "instruction",
					ID:     "guidance",
					Value:  "line 1\r\nline 2\r\nline 3\r\n",
					Source: "legal",
				},
				{
					Kind:   "secret",
					ID:     "db-token",
					Ref:    "vault:secret/data/db#password",
					Source: "vault",
				},
			},
		}

		idLF, err := CanonicalID(lockLF)
		if err != nil {
			t.Fatalf("CanonicalID(lockLF) error: %v", err)
		}
		idCRLF, err := CanonicalID(lockCRLF)
		if err != nil {
			t.Fatalf("CanonicalID(lockCRLF) error: %v", err)
		}

		if idLF != idCRLF {
			t.Fatalf("Canonical ID drifted under CRLF:\nLF:   %s\nCRLF: %s", idLF, idCRLF)
		}
		if !strings.HasPrefix(idLF, "sha256:") {
			t.Fatalf("invalid ID prefix: %s", idLF)
		}

		// Also verify raw JSON bytes with CRLF separators parse to identical ID
		lockLF.ID = idLF
		rawLF, err := json.Marshal(lockLF)
		if err != nil {
			t.Fatal(err)
		}
		rawCRLF := []byte(strings.ReplaceAll(string(rawLF), "\n", "\r\n"))

		parsedLF, err := Parse(rawLF)
		if err != nil {
			t.Fatalf("Parse(rawLF) error: %v", err)
		}
		parsedCRLF, err := Parse(rawCRLF)
		if err != nil {
			t.Fatalf("Parse(rawCRLF) error: %v", err)
		}
		if parsedLF.ID != parsedCRLF.ID {
			t.Fatalf("Parsed IDs mismatch: %s vs %s", parsedLF.ID, parsedCRLF.ID)
		}
	})

	t.Run("secret with plaintext value fails with SECRET_PLAINTEXT_LEAK", func(t *testing.T) {
		lock := &Lock{
			Schema: ProductLockSchemaV2,
			Budget: LockBudget{ContextTokens: 500},
			Components: []LockedComponent{
				{ID: "c1", Version: "1.0.0", Digest: "sha256:abc", Source: "test"},
			},
			Assets: []LockedAsset{
				{
					Kind:   "secret",
					ID:     "api_key",
					Value:  "super-secret-token",
					Ref:    "env:MY_KEY",
					Source: "env",
				},
			},
		}

		err := ValidateSecretContracts(lock)
		if err == nil {
			t.Fatal("expected error for secret with non-empty plaintext value, got nil")
		}
		if !strings.Contains(err.Error(), SecretPlaintextLeakError) {
			t.Fatalf("expected error containing %q, got %q", SecretPlaintextLeakError, err.Error())
		}

		id, err := CanonicalID(lock)
		if err != nil {
			t.Fatal(err)
		}
		lock.ID = id
		raw, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		_, err = Parse(raw)
		if err == nil || !strings.Contains(err.Error(), SecretPlaintextLeakError) {
			t.Fatalf("Parse should fail with %s, got %v", SecretPlaintextLeakError, err)
		}
	})

	t.Run("secret opaque references validated", func(t *testing.T) {
		validRefs := []string{
			"env:API_KEY",
			"env:MY_SECRET_VAR_123",
			"file:/etc/secrets/token.txt",
			"file:./relative/token.key",
			"vault:secret/data/db#password",
			"vault:production/keys/stripe#key-id",
			"keyring:fak-service-prod",
		}
		for _, ref := range validRefs {
			lock := &Lock{
				Assets: []LockedAsset{
					{Kind: "secret", ID: "sec", Ref: ref, Source: "test"},
				},
			}
			if err := ValidateSecretContracts(lock); err != nil {
				t.Fatalf("valid ref %q failed validation: %v", ref, err)
			}
		}

		invalidRefs := []string{
			"",
			"plaintext:my-secret",
			"http://example.com/secret",
			"env:",
			"vault:",
			"invalid-scheme:foo",
		}
		for _, ref := range invalidRefs {
			lock := &Lock{
				Assets: []LockedAsset{
					{Kind: "secret", ID: "sec", Ref: ref, Source: "test"},
				},
			}
			if err := ValidateSecretContracts(lock); err == nil {
				t.Fatalf("invalid ref %q was accepted", ref)
			}
		}
	})

	t.Run("multi-platform compatibility validation", func(t *testing.T) {
		lock := &Lock{
			Schema: ProductLockSchemaV2,
			Platforms: []PlatformRequirement{
				{OS: "linux", Arch: "amd64"},
				{OS: "darwin", Arch: "arm64"},
				{OS: "windows", Arch: "amd64"},
			},
			Components: []LockedComponent{
				{
					ID:            "cross-plat",
					Version:       "1.0.0",
					Digest:        "sha256:cp",
					Source:        "reg",
					Compatibility: LockCompatibility{OS: []string{"darwin", "linux", "windows"}, Arch: []string{"amd64", "arm64"}},
				},
			},
		}

		if !lock.SupportsPlatform("linux", "amd64") {
			t.Fatal("expected linux/amd64 support")
		}
		if !lock.SupportsPlatform("darwin", "arm64") {
			t.Fatal("expected darwin/arm64 support")
		}
		if !lock.SupportsPlatform("windows", "amd64") {
			t.Fatal("expected windows/amd64 support")
		}
		if lock.SupportsPlatform("freebsd", "riscv64") {
			t.Fatal("did not expect freebsd/riscv64 support")
		}

		if err := lock.ValidatePlatforms(); err != nil {
			t.Fatalf("ValidatePlatforms() error = %v", err)
		}

		// Incompatible component should fail ValidatePlatforms
		lock.Components = append(lock.Components, LockedComponent{
			ID:            "linux-only",
			Version:       "1.0.0",
			Digest:        "sha256:lo",
			Source:        "reg",
			Compatibility: LockCompatibility{OS: []string{"linux"}, Arch: []string{"amd64"}},
		})
		err := lock.ValidatePlatforms()
		if err == nil || !strings.Contains(err.Error(), "incompatible with platform OS") {
			t.Fatalf("expected incompatible platform error, got: %v", err)
		}
	})

	t.Run("mixable requires v2 schema and facts", func(t *testing.T) {
		legacy := &Lock{
			Schema: ProductLockSchemaV1Alpha1,
			Components: []LockedComponent{
				{ID: "c1", Version: "1.0.0", Digest: "sha256:c1", Source: "src", Reason: "root", Provider: "sys", Adapters: []string{"instruction"}, Compatibility: LockCompatibility{Contract: "v1"}},
			},
		}
		if err := legacy.Mixable(); err == nil || !strings.Contains(err.Error(), "launchable but not mixable") {
			t.Fatalf("legacy lock should not be mixable: %v", err)
		}

		v2 := &Lock{
			Schema: ProductLockSchemaV2,
			Platforms: []PlatformRequirement{
				{OS: "linux", Arch: "amd64"},
			},
			Components: []LockedComponent{
				{
					ID:            "c1",
					Version:       "1.0.0",
					Digest:        "sha256:c1",
					Source:        "src",
					Reason:        "root",
					Provider:      "sys",
					Compatibility: LockCompatibility{Contract: "v1"},
					Adapters:      []string{"instruction"},
				},
			},
		}
		if err := v2.Mixable(); err != nil {
			t.Fatalf("valid v2 lock should be mixable: %v", err)
		}
	})
}
