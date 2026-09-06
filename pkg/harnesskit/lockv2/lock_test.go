package lockv2

import (
	"encoding/json"
	"reflect"
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

	t.Run("IsMCP and IsLSP helper methods", func(t *testing.T) {
		tests := []struct {
			name      string
			component LockedComponent
			wantMCP   bool
			wantLSP   bool
		}{
			{
				name:      "zero value component",
				component: LockedComponent{},
				wantMCP:   false,
				wantLSP:   false,
			},
			{
				name:      "component with Kind ComponentKindMCP",
				component: LockedComponent{Kind: ComponentKindMCP},
				wantMCP:   true,
				wantLSP:   false,
			},
			{
				name:      "component with MCP metadata only",
				component: LockedComponent{MCP: &LockedMCPMetadata{Transport: "stdio"}},
				wantMCP:   true,
				wantLSP:   false,
			},
			{
				name:      "component with both Kind ComponentKindMCP and MCP metadata",
				component: LockedComponent{Kind: ComponentKindMCP, MCP: &LockedMCPMetadata{Transport: "stdio"}},
				wantMCP:   true,
				wantLSP:   false,
			},
			{
				name:      "component with Kind ComponentKindLSP",
				component: LockedComponent{Kind: ComponentKindLSP},
				wantMCP:   false,
				wantLSP:   true,
			},
			{
				name:      "component with LSP metadata only",
				component: LockedComponent{LSP: &LockedLSPMetadata{Language: "go"}},
				wantMCP:   false,
				wantLSP:   true,
			},
			{
				name:      "component with both Kind ComponentKindLSP and LSP metadata",
				component: LockedComponent{Kind: ComponentKindLSP, LSP: &LockedLSPMetadata{Language: "go"}},
				wantMCP:   false,
				wantLSP:   true,
			},
			{
				name:      "component with Kind ComponentKindRuntime",
				component: LockedComponent{Kind: ComponentKindRuntime},
				wantMCP:   false,
				wantLSP:   false,
			},
			{
				name:      "component with Kind ComponentKindTool",
				component: LockedComponent{Kind: ComponentKindTool},
				wantMCP:   false,
				wantLSP:   false,
			},
			{
				name:      "component with Kind ComponentKindEngine",
				component: LockedComponent{Kind: ComponentKindEngine},
				wantMCP:   false,
				wantLSP:   false,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				if got := tc.component.IsMCP(); got != tc.wantMCP {
					t.Errorf("IsMCP() = %v, want %v", got, tc.wantMCP)
				}
				if got := tc.component.IsLSP(); got != tc.wantLSP {
					t.Errorf("IsLSP() = %v, want %v", got, tc.wantLSP)
				}
			})
		}
	})

	t.Run("parsing and canonical ID calculation with ComponentKindMCP and ComponentKindLSP", func(t *testing.T) {
		lock := &Lock{
			Schema: ProductLockSchemaV2,
			Platforms: []PlatformRequirement{
				{OS: "linux", Arch: "amd64"},
			},
			Budget: LockBudget{ContextTokens: 2048, MemoryMiB: 512, Workers: 2},
			Components: []LockedComponent{
				{
					ID:       "mcp-filesystem",
					Version:  "1.0.0",
					Digest:   "sha256:1111111111111111111111111111111111111111111111111111111111111111",
					Source:   "registry/mcp/filesystem",
					Reason:   "declared mcp server",
					Provider: "mcp-org",
					Kind:     ComponentKindMCP,
					MCP: &LockedMCPMetadata{
						Transport: "stdio",
						Command:   []string{"npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"},
						Environment: map[string]string{
							"NODE_ENV":  "production",
							"LOG_LEVEL": "warn",
						},
						Policy: "read-only",
					},
				},
				{
					ID:       "lsp-gopls",
					Version:  "0.16.1",
					Digest:   "sha256:2222222222222222222222222222222222222222222222222222222222222222",
					Source:   "registry/lsp/gopls",
					Reason:   "language server for Go",
					Provider: "golang.org",
					Kind:     ComponentKindLSP,
					LSP: &LockedLSPMetadata{
						Language:       "go",
						Extensions:     []string{".go", ".mod"},
						Transport:      "stdio",
						Command:        []string{"gopls", "-mode=stdio"},
						Diagnostics:    true,
						Symbols:        true,
						Initialization: json.RawMessage(`{"formatting":true,"hover":true}`),
					},
				},
			},
		}

		cid, err := CanonicalID(lock)
		if err != nil {
			t.Fatalf("CanonicalID() error = %v", err)
		}
		if !strings.HasPrefix(cid, "sha256:") {
			t.Fatalf("invalid canonical ID: %s", cid)
		}

		lock.ID = cid
		raw, err := json.Marshal(lock)
		if err != nil {
			t.Fatalf("json.Marshal() error = %v", err)
		}

		parsed, err := Parse(raw)
		if err != nil {
			t.Fatalf("Parse() error = %v", err)
		}
		if parsed.ID != cid {
			t.Fatalf("parsed ID %s != expected %s", parsed.ID, cid)
		}
		if len(parsed.Components) != 2 {
			t.Fatalf("expected 2 components, got %d", len(parsed.Components))
		}

		// Verify MCP component
		mcpComp := parsed.Components[0]
		if mcpComp.Kind != ComponentKindMCP {
			t.Errorf("expected Kind %q, got %q", ComponentKindMCP, mcpComp.Kind)
		}
		if !mcpComp.IsMCP() {
			t.Error("expected IsMCP() to be true")
		}
		if mcpComp.IsLSP() {
			t.Error("expected IsLSP() to be false")
		}
		if mcpComp.MCP == nil {
			t.Fatal("expected non-nil MCP metadata")
		}
		if mcpComp.MCP.Transport != "stdio" {
			t.Errorf("expected transport stdio, got %s", mcpComp.MCP.Transport)
		}
		if !reflect.DeepEqual(mcpComp.MCP.Command, []string{"npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"}) {
			t.Errorf("command mismatch: %v", mcpComp.MCP.Command)
		}
		if mcpComp.MCP.Environment["NODE_ENV"] != "production" || mcpComp.MCP.Environment["LOG_LEVEL"] != "warn" {
			t.Errorf("environment mismatch: %v", mcpComp.MCP.Environment)
		}
		if mcpComp.MCP.Policy != "read-only" {
			t.Errorf("policy mismatch: %s", mcpComp.MCP.Policy)
		}

		// Verify LSP component
		lspComp := parsed.Components[1]
		if lspComp.Kind != ComponentKindLSP {
			t.Errorf("expected Kind %q, got %q", ComponentKindLSP, lspComp.Kind)
		}
		if !lspComp.IsLSP() {
			t.Error("expected IsLSP() to be true")
		}
		if lspComp.IsMCP() {
			t.Error("expected IsMCP() to be false")
		}
		if lspComp.LSP == nil {
			t.Fatal("expected non-nil LSP metadata")
		}
		if lspComp.LSP.Language != "go" {
			t.Errorf("expected language go, got %s", lspComp.LSP.Language)
		}
		if !reflect.DeepEqual(lspComp.LSP.Extensions, []string{".go", ".mod"}) {
			t.Errorf("extensions mismatch: %v", lspComp.LSP.Extensions)
		}
		if lspComp.LSP.Transport != "stdio" {
			t.Errorf("expected transport stdio, got %s", lspComp.LSP.Transport)
		}
		if !reflect.DeepEqual(lspComp.LSP.Command, []string{"gopls", "-mode=stdio"}) {
			t.Errorf("command mismatch: %v", lspComp.LSP.Command)
		}
		if !lspComp.LSP.Diagnostics || !lspComp.LSP.Symbols {
			t.Errorf("diagnostics/symbols mismatch: diag=%v symbols=%v", lspComp.LSP.Diagnostics, lspComp.LSP.Symbols)
		}
		var initPayload map[string]bool
		if err := json.Unmarshal(lspComp.LSP.Initialization, &initPayload); err != nil {
			t.Fatalf("unmarshal initialization error: %v", err)
		}
		if !initPayload["formatting"] || !initPayload["hover"] {
			t.Errorf("initialization payload mismatch: %v", initPayload)
		}

		// Verify canonical ID re-calculation matches
		recomputed, err := CanonicalID(parsed)
		if err != nil {
			t.Fatalf("CanonicalID(parsed) error = %v", err)
		}
		if recomputed != cid {
			t.Fatalf("recomputed canonical ID %s != original %s", recomputed, cid)
		}
	})

	t.Run("backwards compatibility: existing locks without Kind MCP or LSP continue to parse and calculate identical canonical IDs", func(t *testing.T) {
		// Existing lock without Kind, MCP, or LSP fields
		lock := &Lock{
			Schema: ProductLockSchemaV2,
			Platforms: []PlatformRequirement{
				{OS: "linux", Arch: "amd64"},
				{OS: "windows", Arch: "amd64"},
			},
			Budget: LockBudget{ContextTokens: 4096, MemoryMiB: 1024, Workers: 4},
			Components: []LockedComponent{
				{
					ID:       "core-runtime",
					Version:  "2.1.0",
					Digest:   "sha256:3333333333333333333333333333333333333333333333333333333333333333",
					Source:   "registry/runtime/core",
					Reason:   "execution core",
					Provider: "fak-team",
					Provides: []string{"runtime", "eval"},
					Compatibility: LockCompatibility{
						OS:   []string{"linux", "windows"},
						Arch: []string{"amd64"},
					},
					Adapters: []string{"instruction", "tool"},
				},
			},
			Assets: []LockedAsset{
				{
					Kind:   "instruction",
					ID:     "system-prompt",
					Value:  "You are a helpful assistant.",
					Source: "system",
				},
			},
		}

		cid, err := CanonicalID(lock)
		if err != nil {
			t.Fatalf("CanonicalID() error = %v", err)
		}
		lock.ID = cid

		raw, err := json.Marshal(lock)
		if err != nil {
			t.Fatalf("json.Marshal() error = %v", err)
		}

		// Ensure component JSON does not contain kind, mcp, or lsp
		var rawMap map[string]any
		if err := json.Unmarshal(raw, &rawMap); err != nil {
			t.Fatalf("json.Unmarshal error: %v", err)
		}
		compMap := rawMap["components"].([]any)[0].(map[string]any)
		if _, exists := compMap["kind"]; exists {
			t.Fatalf("expected component JSON to omit kind, got %v", compMap["kind"])
		}
		if _, exists := compMap["mcp"]; exists {
			t.Fatalf("expected component JSON to omit mcp, got %v", compMap["mcp"])
		}
		if _, exists := compMap["lsp"]; exists {
			t.Fatalf("expected component JSON to omit lsp, got %v", compMap["lsp"])
		}

		// Parse should succeed under DisallowUnknownFields
		parsed, err := Parse(raw)
		if err != nil {
			t.Fatalf("Parse() error = %v", err)
		}
		if parsed.ID != cid {
			t.Fatalf("parsed ID %s != %s", parsed.ID, cid)
		}

		comp := parsed.Components[0]
		if comp.Kind != "" {
			t.Errorf("expected empty Kind, got %q", comp.Kind)
		}
		if comp.MCP != nil {
			t.Errorf("expected nil MCP, got %+v", comp.MCP)
		}
		if comp.LSP != nil {
			t.Errorf("expected nil LSP, got %+v", comp.LSP)
		}
		if comp.IsMCP() {
			t.Error("expected IsMCP() to be false for legacy component")
		}
		if comp.IsLSP() {
			t.Error("expected IsLSP() to be false for legacy component")
		}

		// Recomputed CanonicalID must be byte-for-byte identical
		recomputed, err := CanonicalID(parsed)
		if err != nil {
			t.Fatalf("CanonicalID(parsed) error = %v", err)
		}
		if recomputed != cid {
			t.Fatalf("recomputed canonical ID %s != %s", recomputed, cid)
		}
	})
}
