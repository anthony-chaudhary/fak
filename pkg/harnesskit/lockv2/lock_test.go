package lockv2

import (
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseValidLock(t *testing.T) {
	validJSON := []byte(`{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "platforms": [
    {"os": "linux", "arch": "amd64", "variant": "v3"}
  ],
  "budget": {
    "context_tokens": 128000,
    "memory_mib": 8192,
    "workers": 4
  },
  "components": [
    {
      "id": "comp-core",
      "version": "1.0.0",
      "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "source": "fak/core",
      "provider": "builtin",
      "provides": ["kernel", "tool_dispatch"]
    }
  ],
  "assets": [
    {
      "name": "system_prompt",
      "kind": "prompt",
      "ref": "file:prompts/system.md",
      "digest": "sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
      "value": "You are a helpful assistant."
    }
  ],
  "secrets": [
    {
      "name": "api_key",
      "kind": "secret",
      "ref": "vault:secret/api-key",
      "value": "",
      "provider": "vault"
    }
  ],
  "tool_fingerprints": [
    {
      "server": "github",
      "tool": "create_issue",
      "schema_hash": "sha256:1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff"
    }
  ]
}`)

	lock, err := Parse(validJSON)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if lock.Schema != Schema {
		t.Errorf("schema = %q, want %q", lock.Schema, Schema)
	}
	if lock.ID != "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" {
		t.Errorf("id = %q", lock.ID)
	}
	if len(lock.Platforms) != 1 || lock.Platforms[0].OS != "linux" || lock.Platforms[0].Arch != "amd64" {
		t.Errorf("unexpected platforms: %#v", lock.Platforms)
	}
	if lock.Budget.ContextTokens != 128000 || lock.Budget.MemoryMiB != 8192 || lock.Budget.Workers != 4 {
		t.Errorf("unexpected budget: %#v", lock.Budget)
	}
	if len(lock.Components) != 1 || lock.Components[0].ID != "comp-core" || len(lock.Components[0].Provides) != 2 {
		t.Errorf("unexpected components: %#v", lock.Components)
	}
	if len(lock.Assets) != 1 || lock.Assets[0].Name != "system_prompt" || lock.Assets[0].Value != "You are a helpful assistant." {
		t.Errorf("unexpected assets: %#v", lock.Assets)
	}
	if len(lock.Secrets) != 1 || lock.Secrets[0].Name != "api_key" || lock.Secrets[0].Ref != "vault:secret/api-key" {
		t.Errorf("unexpected secrets: %#v", lock.Secrets)
	}
	if len(lock.ToolFingerprints) != 1 || lock.ToolFingerprints[0].Tool != "create_issue" {
		t.Errorf("unexpected tool fingerprints: %#v", lock.ToolFingerprints)
	}

	// Invalid JSON must fail
	if _, err := Parse([]byte("not-json")); err == nil {
		t.Errorf("expected error parsing invalid JSON, got nil")
	}

	// Wrong schema must fail
	wrongSchemaJSON := []byte(`{"schema": "fak.harness-product-lock/v1alpha2"}`)
	if _, err := Parse(wrongSchemaJSON); err == nil || !strings.Contains(err.Error(), "unsupported lock schema") {
		t.Errorf("expected unsupported schema error, got %v", err)
	}
}

func TestCanonicalIDProducesIdenticalHashRegardlessOfCRLFvsLF(t *testing.T) {
	// Case 1: CRLF vs LF within struct string fields
	lockLF := &Lock{
		Schema: Schema,
		Platforms: []PlatformRequirement{
			{OS: "linux", Arch: "amd64", Variant: "v3"},
		},
		Budget: LockBudget{ContextTokens: 4096, MemoryMiB: 1024, Workers: 2},
		Components: []LockedComponent{
			{
				ID:       "pkg1",
				Version:  "1.0.0",
				Digest:   "sha256:abc",
				Source:   "repo/pkg1",
				Provider: "std",
				Provides: []string{"srv1\nsrv2"},
			},
		},
		Assets: []LockedAsset{
			{
				Name:  "prompt",
				Kind:  "instruction",
				Value: "line 1\nline 2\nline 3\n",
			},
		},
	}

	lockCRLF := &Lock{
		Schema: Schema,
		Platforms: []PlatformRequirement{
			{OS: "linux", Arch: "amd64", Variant: "v3"},
		},
		Budget: LockBudget{ContextTokens: 4096, MemoryMiB: 1024, Workers: 2},
		Components: []LockedComponent{
			{
				ID:       "pkg1",
				Version:  "1.0.0",
				Digest:   "sha256:abc",
				Source:   "repo/pkg1",
				Provider: "std",
				Provides: []string{"srv1\r\nsrv2"},
			},
		},
		Assets: []LockedAsset{
			{
				Name:  "prompt",
				Kind:  "instruction",
				Value: "line 1\r\nline 2\r\nline 3\r\n",
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

	if !strings.HasPrefix(idLF, "sha256:") || len(idLF) != 71 {
		t.Fatalf("invalid canonical ID format: %q", idLF)
	}
	if idLF != idCRLF {
		t.Fatalf("hash mismatch for struct CRLF vs LF:\n  LF:   %s\n  CRLF: %s", idLF, idCRLF)
	}

	// Case 2: CRLF vs LF in serialized JSON input
	jsonLF := []byte("{\n  \"schema\": \"fak.harness-product-lock/v2\",\n  \"budget\": {\"workers\": 2},\n  \"assets\": [\n    {\"name\": \"a\", \"kind\": \"prompt\", \"value\": \"hello\\nworld\"}\n  ]\n}")
	jsonCRLF := []byte("{\r\n  \"schema\": \"fak.harness-product-lock/v2\",\r\n  \"budget\": {\"workers\": 2},\r\n  \"assets\": [\r\n    {\"name\": \"a\", \"kind\": \"prompt\", \"value\": \"hello\\r\\nworld\"}\r\n  ]\r\n}")

	parsedLF, err := Parse(jsonLF)
	if err != nil {
		t.Fatalf("Parse(jsonLF): %v", err)
	}
	parsedCRLF, err := Parse(jsonCRLF)
	if err != nil {
		t.Fatalf("Parse(jsonCRLF): %v", err)
	}

	idParsedLF, err := CanonicalID(parsedLF)
	if err != nil {
		t.Fatalf("CanonicalID(parsedLF): %v", err)
	}
	idParsedCRLF, err := CanonicalID(parsedCRLF)
	if err != nil {
		t.Fatalf("CanonicalID(parsedCRLF): %v", err)
	}
	if idParsedLF != idParsedCRLF {
		t.Fatalf("hash mismatch for parsed JSON CRLF vs LF:\n  LF:   %s\n  CRLF: %s", idParsedLF, idParsedCRLF)
	}

	// Case 3: CanonicalID is invariant whether ID is already set or empty
	lockLF.ID = idLF
	idWithID, err := CanonicalID(lockLF)
	if err != nil {
		t.Fatalf("CanonicalID with pre-set ID: %v", err)
	}
	if idWithID != idLF {
		t.Fatalf("pre-set ID altered canonical ID calculation: got %s, want %s", idWithID, idLF)
	}
}

func TestValidateSecretContractsRejectsNonEmptyValueWithSecretPlaintextLeak(t *testing.T) {
	// Valid lock: secret kind with empty value and valid ref
	validLock := &Lock{
		Schema: Schema,
		Assets: []LockedAsset{
			{Name: "api_key_asset", Kind: "secret", Ref: "env:OPENAI_API_KEY", Value: ""},
			{Name: "prompt_asset", Kind: "instruction", Value: "plaintext instructions are allowed"},
		},
		Secrets: []SecretContract{
			{Name: "db_pass", Kind: "secret", Ref: "vault:secret/data/db#password", Value: ""},
		},
	}
	if err := ValidateSecretContracts(validLock); err != nil {
		t.Fatalf("expected valid secret contracts, got error: %v", err)
	}

	// Reject locked asset with Kind == "secret" and non-empty Value
	leakedAssetLock := &Lock{
		Schema: Schema,
		Assets: []LockedAsset{
			{Name: "leaked_asset", Kind: "secret", Ref: "env:LEAKED_KEY", Value: "sk-proj-1234567890"},
		},
	}
	err := ValidateSecretContracts(leakedAssetLock)
	if err == nil {
		t.Fatalf("expected error for leaked asset plaintext, got nil")
	}
	if !strings.Contains(err.Error(), SecretPlaintextLeak) {
		t.Errorf("expected error to contain %s, got %v", SecretPlaintextLeak, err)
	}

	// Reject secret contract with Kind == "secret" and non-empty Value
	leakedSecretLock := &Lock{
		Schema: Schema,
		Secrets: []SecretContract{
			{Name: "leaked_secret", Kind: "secret", Ref: "vault:secret/token", Value: "super_secret_password"},
		},
	}
	err = ValidateSecretContracts(leakedSecretLock)
	if err == nil {
		t.Fatalf("expected error for leaked secret contract plaintext, got nil")
	}
	if !strings.Contains(err.Error(), SecretPlaintextLeak) {
		t.Errorf("expected error to contain %s, got %v", SecretPlaintextLeak, err)
	}
}

func TestValidateSecretContractsValidatesRefURIScheme(t *testing.T) {
	validRefs := []string{
		"env:DATABASE_URL",
		"file:/var/run/secrets/token",
		"file:relative/path/key.pem",
		"vault:secret/data/payment#api_key",
		"keyring:corp-service.user_1-main",
	}

	for _, ref := range validRefs {
		t.Run("valid_"+ref, func(t *testing.T) {
			l := &Lock{
				Schema: Schema,
				Assets: []LockedAsset{
					{Name: "asset1", Kind: "secret", Ref: ref, Value: ""},
				},
				Secrets: []SecretContract{
					{Name: "sec1", Kind: "secret", Ref: ref, Value: ""},
				},
			}
			if err := ValidateSecretContracts(l); err != nil {
				t.Errorf("expected valid ref %q, got error: %v", ref, err)
			}
		})
	}

	invalidRefs := []struct {
		name string
		ref  string
	}{
		{"empty", ""},
		{"unsupported_http", "http://example.com/secret"},
		{"unsupported_https", "https://vault.corp.internal/token"},
		{"unsupported_ftp", "ftp:secret"},
		{"bare_scheme_no_path", "env:"},
		{"contains_spaces", "vault:secret/data with space"},
		{"disallowed_special_char", "keyring:service$user"},
		{"backslash", "file:C:\\secrets\\token"},
	}

	for _, tc := range invalidRefs {
		t.Run("invalid_"+tc.name, func(t *testing.T) {
			lAsset := &Lock{
				Schema: Schema,
				Assets: []LockedAsset{
					{Name: "asset_invalid", Kind: "secret", Ref: tc.ref, Value: ""},
				},
			}
			if err := ValidateSecretContracts(lAsset); err == nil {
				t.Errorf("expected error for asset with invalid ref %q, got nil", tc.ref)
			}

			lSecret := &Lock{
				Schema: Schema,
				Secrets: []SecretContract{
					{Name: "sec_invalid", Kind: "secret", Ref: tc.ref, Value: ""},
				},
			}
			if err := ValidateSecretContracts(lSecret); err == nil {
				t.Errorf("expected error for secret contract with invalid ref %q, got nil", tc.ref)
			}
		})
	}
}

func TestZeroInternalImports(t *testing.T) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filepath.Join(".", "lock.go"), nil, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("failed to parse lock.go: %v", err)
	}

	allowed := map[string]bool{
		"crypto/sha256": true,
		"encoding/hex":  true,
		"encoding/json": true,
		"fmt":           true,
		"regexp":        true,
		"sort":          true,
		"strings":       true,
	}

	for _, imp := range node.Imports {
		path := strings.Trim(imp.Path.Value, `"`)
		if strings.Contains(path, "fak") || strings.Contains(path, "internal") {
			t.Errorf("internal or module import forbidden in pkg/harnesskit/lockv2: %q", path)
		}
		if !allowed[path] {
			t.Errorf("unexpected import in lock.go: %q (allowed: %v)", path, allowed)
		}
	}
}
