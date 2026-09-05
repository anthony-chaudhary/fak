package harnesskit

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
)

func TestProductLockVerifiesAndProjectsLaunchReceipt(t *testing.T) {
	lock := ProductLock{Schema: ProductLockSchema, Environment: LockEnvironment{OS: "linux", Arch: "amd64", Contract: "v1"}, Components: []LockedComponent{{ID: "legal-pack", Version: "1.0.0", Digest: "sha256:legal", Source: "registry/legal"}}, Assets: []LockedAsset{{Kind: "policy", ID: "tools", Source: "company", Locked: true}, {Kind: "instruction", ID: "citations", Value: "cite-primary", Source: "legal"}}}
	lock.ID = lockDigest(t, lock)
	raw, _ := json.Marshal(lock)
	got, err := ParseProductLock(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got.Profile().ID != "legal" || got.InstructionText() != "cite-primary" {
		t.Fatalf("lock=%#v", got)
	}
	receipt := got.LaunchReceipt("product")
	if receipt.LockID != lock.ID || receipt.Profile != "legal" || len(receipt.Assets) != 2 {
		t.Fatalf("receipt=%#v", receipt)
	}
}

func TestProductLockRejectsTamperAndInlineSecret(t *testing.T) {
	lock := ProductLock{Schema: ProductLockSchema, Components: []LockedComponent{{ID: "p", Version: "1.0.0", Digest: "sha256:p", Source: "r", Reason: "selected", Provider: "x"}}, Assets: []LockedAsset{{Kind: "instruction", ID: "i", Value: "one", Source: "legal"}}}
	lock.ID = lockDigest(t, lock)
	raw, _ := json.Marshal(lock)
	raw = []byte(strings.Replace(string(raw), `"value":"one"`, `"value":"two"`, 1))
	if _, err := ParseProductLock(raw); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("err=%v", err)
	}
	lock.Assets = []LockedAsset{{Kind: "secret", ID: "db", Source: "company"}}
	lock.ID = lockDigest(t, lock)
	raw, _ = json.Marshal(lock)
	if _, err := ParseProductLock(raw); err == nil || !strings.Contains(err.Error(), "no opaque reference") {
		t.Fatalf("err=%v", err)
	}
}

func lockDigest(t *testing.T, lock ProductLock) string {
	t.Helper()
	lock.ID = ""
	raw, err := json.Marshal(lock)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256Sum(raw)
	return "sha256:" + sum
}
func sha256Sum(raw []byte) string {
	h := sha256.New()
	h.Write(raw)
	return hex.EncodeToString(h.Sum(nil))
}

func TestMixableRefusesLegacyAndMissingEvidence(t *testing.T) {
	base := ProductLock{Schema: LegacyProductLockSchema, Components: []LockedComponent{{ID: "x", Version: "1.0.0", Digest: "sha256:x", Source: "r", Reason: "selected", Provider: "x"}}}
	if err := base.Mixable(); err == nil || !strings.Contains(err.Error(), "launchable but not mixable") {
		t.Fatalf("legacy err=%v", err)
	}
	base.Schema = ProductLockSchema
	if err := base.Mixable(); err == nil || !strings.Contains(err.Error(), "compatibility contract") {
		t.Fatalf("compat err=%v", err)
	}
	base.Components[0].Compatibility.Contract = "v1"
	if err := base.Mixable(); err == nil || !strings.Contains(err.Error(), "adapter conformance") {
		t.Fatalf("adapter err=%v", err)
	}
	base.Components[0].Adapters = []string{"instruction"}
	if err := base.Mixable(); err != nil {
		t.Fatal(err)
	}
}

func TestProductLockV2(t *testing.T) {
	t.Run("synthetic lock with CRLF and LF produces exact same canonical ID", func(t *testing.T) {
		crlfJSON := []byte("{\r\n" +
			"  \"schema\": \"" + ProductLockSchemaV2 + "\",\r\n" +
			"  \"id\": \"placeholder\",\r\n" +
			"  \"platforms\": [\r\n" +
			"    {\"os\": \"linux\", \"arch\": \"amd64\", \"contract\": \"v1\"}\r\n" +
			"  ],\r\n" +
			"  \"components\": [\r\n" +
			"    {\"id\": \"p\", \"version\": \"1.0.0\", \"digest\": \"sha256:p\", \"source\": \"r\", \"reason\": \"selected\\r\\nline2\", \"provider\": \"x\"}\r\n" +
			"  ],\r\n" +
			"  \"assets\": [\r\n" +
			"    {\"kind\": \"instruction\", \"id\": \"inst\", \"source\": \"s\", \"value\": \"line1\\r\\nline2\"}\r\n" +
			"  ]\r\n" +
			"}")

		lfJSON := []byte("{\n" +
			"  \"schema\": \"" + ProductLockSchemaV2 + "\",\n" +
			"  \"id\": \"placeholder\",\n" +
			"  \"platforms\": [\n" +
			"    {\"os\": \"linux\", \"arch\": \"amd64\", \"contract\": \"v1\"}\n" +
			"  ],\n" +
			"  \"components\": [\n" +
			"    {\"id\": \"p\", \"version\": \"1.0.0\", \"digest\": \"sha256:p\", \"source\": \"r\", \"reason\": \"selected\\nline2\", \"provider\": \"x\"}\n" +
			"  ],\n" +
			"  \"assets\": [\n" +
			"    {\"kind\": \"instruction\", \"id\": \"inst\", \"source\": \"s\", \"value\": \"line1\\nline2\"}\n" +
			"  ]\n" +
			"}")

		var lockCRLF, lockLF ProductLock
		if err := json.Unmarshal(crlfJSON, &lockCRLF); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(lfJSON, &lockLF); err != nil {
			t.Fatal(err)
		}

		idCRLF, err := LockID(lockCRLF)
		if err != nil {
			t.Fatal(err)
		}
		idLF, err := LockID(lockLF)
		if err != nil {
			t.Fatal(err)
		}
		if idCRLF != idLF {
			t.Fatalf("CRLF and LF produced different IDs: crlf=%s lf=%s", idCRLF, idLF)
		}

		crlfWithID := []byte(strings.Replace(string(crlfJSON), "placeholder", idCRLF, 1))
		lfWithID := []byte(strings.Replace(string(lfJSON), "placeholder", idLF, 1))

		parsedCRLF, err := ParseProductLock(crlfWithID)
		if err != nil {
			t.Fatalf("failed to parse CRLF lock: %v", err)
		}
		parsedLF, err := ParseProductLock(lfWithID)
		if err != nil {
			t.Fatalf("failed to parse LF lock: %v", err)
		}
		if parsedCRLF.ID != parsedLF.ID || parsedCRLF.ID != idCRLF {
			t.Fatalf("parsed IDs do not match: crlf=%s lf=%s expected=%s", parsedCRLF.ID, parsedLF.ID, idCRLF)
		}
	})

	t.Run("secret with plaintext value fails with SECRET_PLAINTEXT_LEAK", func(t *testing.T) {
		lock := ProductLock{
			Schema: ProductLockSchemaV2,
			Components: []LockedComponent{
				{ID: "c", Version: "1.0.0", Digest: "sha256:c", Source: "s", Reason: "r", Provider: "p"},
			},
			Assets: []LockedAsset{
				{Kind: "secret", ID: "api-key", Source: "user", Value: "supersecret", Ref: "env:API_KEY"},
			},
		}
		id, err := LockID(lock)
		if err != nil {
			t.Fatal(err)
		}
		lock.ID = id
		raw, _ := json.Marshal(lock)
		_, err = ParseProductLock(raw)
		if err == nil || !strings.Contains(err.Error(), SecretPlaintextLeakError) {
			t.Fatalf("expected error containing %q, got %v", SecretPlaintextLeakError, err)
		}
	})

	t.Run("multi-platform compatibility validation", func(t *testing.T) {
		platforms := []PlatformRequirement{
			{OS: "linux", Arch: "amd64", Contract: "v1"},
			{OS: "darwin", Arch: "arm64", Contract: "v1"},
			{OS: "windows", Arch: "amd64", Contract: "v1"},
		}
		lock := ProductLock{
			Schema:    ProductLockSchemaV2,
			Platforms: platforms,
			Components: []LockedComponent{
				{
					ID:            "c",
					Version:       "1.0.0",
					Digest:        "sha256:c",
					Source:        "s",
					Reason:        "r",
					Provider:      "p",
					Compatibility: LockCompatibility{OS: []string{"darwin", "linux", "windows"}, Arch: []string{"amd64", "arm64"}},
				},
			},
			Assets: []LockedAsset{
				{Kind: "instruction", ID: "i", Source: "s", Value: "hello"},
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
		if err := lock.ValidatePlatforms(); err != nil {
			t.Fatalf("ValidatePlatforms failed: %v", err)
		}

		id, err := LockID(lock)
		if err != nil {
			t.Fatal(err)
		}
		lock.ID = id
		raw, err := json.Marshal(lock)
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := ParseProductLock(raw)
		if err != nil {
			t.Fatalf("multi-platform lock rejected: %v", err)
		}
		if len(parsed.Platforms) != 3 {
			t.Fatalf("expected 3 platforms, got %d", len(parsed.Platforms))
		}
		if parsed.Platforms[1].OS != "darwin" || parsed.Platforms[1].Arch != "arm64" {
			t.Fatalf("unexpected platform 1: %+v", parsed.Platforms[1])
		}
	})

	t.Run("secret opaque references validated", func(t *testing.T) {
		validRefs := []string{
			"env:API_KEY",
			"file:/run/secrets/key",
			"vault:secret/data#token",
			"keyring:fak-agent",
		}
		for _, ref := range validRefs {
			lock := ProductLock{
				Schema: ProductLockSchemaV2,
				Components: []LockedComponent{
					{ID: "c", Version: "1.0.0", Digest: "sha256:c", Source: "s", Reason: "r", Provider: "p"},
				},
				Assets: []LockedAsset{
					{Kind: "secret", ID: "api-key", Source: "user", Ref: ref},
				},
			}
			id, err := LockID(lock)
			if err != nil {
				t.Fatal(err)
			}
			lock.ID = id
			raw, _ := json.Marshal(lock)
			if _, err := ParseProductLock(raw); err != nil {
				t.Fatalf("valid ref %q rejected: %v", ref, err)
			}
		}

		invalidRefs := []string{
			"bare-string",
			":missing-scheme",
			"http://remote/secret",
			"env:",
			"file:",
		}
		for _, ref := range invalidRefs {
			lock := ProductLock{
				Schema: ProductLockSchemaV2,
				Components: []LockedComponent{
					{ID: "c", Version: "1.0.0", Digest: "sha256:c", Source: "s", Reason: "r", Provider: "p"},
				},
				Assets: []LockedAsset{
					{Kind: "secret", ID: "api-key", Source: "user", Ref: ref},
				},
			}
			id, err := LockID(lock)
			if err != nil {
				t.Fatal(err)
			}
			lock.ID = id
			raw, _ := json.Marshal(lock)
			if _, err := ParseProductLock(raw); err == nil || !strings.Contains(err.Error(), "invalid opaque reference") {
				t.Fatalf("expected error for ref %q, got %v", ref, err)
			}
		}
	})

	t.Run("backward compatibility with schemas", func(t *testing.T) {
		schemas := []string{
			LegacyProductLockSchema,
			ProductLockSchema,
			ProductLockSchemaV2,
		}
		for _, s := range schemas {
			lock := ProductLock{
				Schema: s,
				Components: []LockedComponent{
					{ID: "c", Version: "1.0.0", Digest: "sha256:c", Source: "s", Reason: "r", Provider: "p"},
				},
				Assets: []LockedAsset{
					{Kind: "instruction", ID: "i", Source: "s", Value: "hello"},
				},
			}
			id, err := LockID(lock)
			if err != nil {
				t.Fatal(err)
			}
			lock.ID = id
			raw, _ := json.Marshal(lock)
			parsed, err := ParseProductLock(raw)
			if err != nil {
				t.Fatalf("schema %q rejected: %v", s, err)
			}
			if parsed.Schema != s {
				t.Fatalf("parsed schema %q != expected %q", parsed.Schema, s)
			}
		}
	})
}
