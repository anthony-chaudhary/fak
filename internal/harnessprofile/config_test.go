package harnessprofile

import (
	"strings"
	"testing"
)

// TestResolveEmptyIsBuiltins proves an absent/empty config leaves the built-ins intact.
func TestResolveEmptyIsBuiltins(t *testing.T) {
	for _, raw := range [][]byte{nil, []byte(""), []byte("   \n")} {
		got, err := Resolve(raw)
		if err != nil {
			t.Fatalf("Resolve(%q) errored: %v", raw, err)
		}
		if len(got) != len(Builtins()) {
			t.Errorf("Resolve(empty) = %d profiles, want %d built-ins", len(got), len(Builtins()))
		}
	}
}

// TestConfigDeclaresNewHarness is the headline C6 acceptance: a config entry with a NOVEL
// detect name routes with zero Go change — the merged registry's Lookup finds it and returns
// its wire.
func TestConfigDeclaresNewHarness(t *testing.T) {
	raw := []byte(`{"harnesses":[{"name":"acme","names":["acme-cli"],"wire":"openai","repoint":["env"],"identity":"env-key"}]}`)
	merged, err := Resolve(raw)
	if err != nil {
		t.Fatalf("Resolve errored: %v", err)
	}
	p, ok := lookupIn(merged, "acme-cli")
	if !ok {
		t.Fatal("new harness acme-cli not found in merged registry")
	}
	if p.Wire != WireOpenAI {
		t.Errorf("acme-cli wire = %q, want openai", p.Wire)
	}
	// Built-ins are still present and unchanged.
	if _, ok := lookupIn(merged, "claude"); !ok {
		t.Error("built-in claude vanished after merge")
	}
}

// TestConfigOverridesBuiltinField proves a partial override repins ONE field (codex's default
// base URL) while keeping the built-in's wire/repoint intact — the field-level overlay.
func TestConfigOverridesBuiltinField(t *testing.T) {
	raw := []byte(`{"harnesses":[{"names":["codex"],"default_base_url":"https://my.gw/v1"}]}`)
	merged, err := Resolve(raw)
	if err != nil {
		t.Fatalf("Resolve errored: %v", err)
	}
	p, ok := lookupIn(merged, "codex")
	if !ok {
		t.Fatal("codex missing after override")
	}
	if p.DefaultBaseURL != "https://my.gw/v1" {
		t.Errorf("codex base URL = %q, want the override https://my.gw/v1", p.DefaultBaseURL)
	}
	if p.Wire != WireOpenAIResponses {
		t.Errorf("override clobbered codex wire: got %q, want openai-responses (a partial override keeps it)", p.Wire)
	}
	if !p.HasRepoint(RepointCLIConfig) {
		t.Error("override clobbered codex cli-config repoint (a partial override keeps it)")
	}
	// A registry with no override for a built-in leaves it exactly.
	if len(merged) != len(Builtins()) {
		t.Errorf("overriding an existing profile changed the count: %d vs %d", len(merged), len(Builtins()))
	}
}

// TestConfigRejectsUnknownVocabulary is the fail-loud fence: an unknown wire, mechanism, or a
// typo'd field is a named error at load, never a silent fallback.
func TestConfigRejectsUnknownVocabulary(t *testing.T) {
	cases := []struct {
		name, raw, wantSub string
	}{
		{"unknown wire", `{"harnesses":[{"names":["x"],"wire":"groq"}]}`, "unknown wire"},
		{"unknown mechanism", `{"harnesses":[{"names":["x"],"wire":"openai","repoint":["smoke"]}]}`, "unknown repoint mechanism"},
		{"unknown identity", `{"harnesses":[{"names":["x"],"wire":"openai","identity":"ldap"}]}`, "unknown identity"},
		{"unknown field", `{"harnesses":[{"names":["x"],"wire":"openai","bogus":true}]}`, "bogus"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Resolve([]byte(tc.raw)); err == nil {
				t.Fatalf("Resolve(%s) should fail loud, got nil error", tc.raw)
			} else if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error %q missing %q", err.Error(), tc.wantSub)
			}
		})
	}
}

// TestNewHarnessMustBeComplete proves a brand-new harness declaration missing a wire or detect
// name fails loud — a novel entry must be usable, not a half-declaration.
func TestNewHarnessMustBeComplete(t *testing.T) {
	// New name, no wire.
	if _, err := Resolve([]byte(`{"harnesses":[{"name":"x","names":["xcli"]}]}`)); err == nil {
		t.Error("a new harness with no wire should fail loud")
	}
}

// TestSetActiveIntegration proves the package-level Lookup consults the merged registry after
// SetActive, and ResetActive restores the built-ins.
func TestSetActiveIntegration(t *testing.T) {
	t.Cleanup(ResetActive)
	if _, ok := Lookup("acme-cli"); ok {
		t.Fatal("acme-cli should be unknown before SetActive")
	}
	merged, err := Resolve([]byte(`{"harnesses":[{"name":"acme","names":["acme-cli"],"wire":"openai","repoint":["env"]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	SetActive(merged)
	if _, ok := Lookup("acme-cli"); !ok {
		t.Error("Lookup should find acme-cli after SetActive(merged)")
	}
	if got := DefaultBaseURLForCommand("acme-cli"); got != "https://api.openai.com/v1" {
		t.Errorf("acme-cli base URL via command = %q, want the openai wire default", got)
	}
	ResetActive()
	if _, ok := Lookup("acme-cli"); ok {
		t.Error("ResetActive should drop acme-cli")
	}
}

// TestDumpJSONReflectsActive proves the dump surface shows the active (merged) set.
func TestDumpJSONReflectsActive(t *testing.T) {
	t.Cleanup(ResetActive)
	merged, err := Resolve([]byte(`{"harnesses":[{"name":"acme","names":["acme-cli"],"wire":"openai","repoint":["env"]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	SetActive(merged)
	b, err := DumpJSON()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "acme-cli") {
		t.Errorf("DumpJSON missing the config-declared harness:\n%s", b)
	}
}
