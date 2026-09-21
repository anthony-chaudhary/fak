package projectassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestEnsurePiDefaultProviderModelFresh: a missing settings.json is created with
// the two default keys and nothing else invented.
func TestEnsurePiDefaultProviderModelFresh(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")

	path, modified, err := EnsurePiDefaultProviderModel(target, "fak", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatal("modified = false, want true on a fresh file")
	}
	if path != target {
		t.Fatalf("path = %q, want %q", path, target)
	}

	raw := readSettings(t, path)
	if raw["defaultProvider"] != "fak" {
		t.Errorf("defaultProvider = %v, want fak", raw["defaultProvider"])
	}
	if raw["defaultModel"] != "qwen38:27b-q4" {
		t.Errorf("defaultModel = %v, want qwen38:27b-q4", raw["defaultModel"])
	}
}

// TestEnsurePiDefaultProviderModelPreservesForeignKeys: an existing settings.json
// keeps every unrelated key (theme, tools, compaction) — only the two defaults move.
func TestEnsurePiDefaultProviderModelPreservesForeignKeys(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	seed := `{
  "theme": "light",
  "defaultTools": ["read", "bash"],
  "compaction": {"enabled": true, "reserveTokens": 8192},
  "defaultProvider": "hive-ai",
  "defaultModel": "deepseek-ai/DeepSeek-V4.1-Flash"
}`
	if err := os.WriteFile(target, []byte(seed), 0644); err != nil {
		t.Fatal(err)
	}

	path, modified, err := EnsurePiDefaultProviderModel(target, "fak", "/var/lib/fak/models/Qwen3.8-27B-UD-Q2_K_XL.gguf")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatal("modified = false, want true when the defaults change")
	}

	raw := readSettings(t, path)
	if raw["defaultProvider"] != "fak" {
		t.Errorf("defaultProvider = %v, want fak", raw["defaultProvider"])
	}
	if raw["defaultModel"] != "/var/lib/fak/models/Qwen3.8-27B-UD-Q2_K_XL.gguf" {
		t.Errorf("defaultModel = %v", raw["defaultModel"])
	}
	// Foreign keys survive untouched.
	if raw["theme"] != "light" {
		t.Errorf("theme = %v, want light (must be preserved)", raw["theme"])
	}
	tools, ok := raw["defaultTools"].([]interface{})
	if !ok || len(tools) != 2 {
		t.Errorf("defaultTools = %v, want the 2 seeded tools preserved", raw["defaultTools"])
	}
	comp, ok := raw["compaction"].(map[string]interface{})
	if !ok || comp["enabled"] != true {
		t.Errorf("compaction = %v, want the seeded block preserved", raw["compaction"])
	}
}

// TestEnsurePiDefaultProviderModelIdempotent: a second call is a no-op.
func TestEnsurePiDefaultProviderModelIdempotent(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")

	if _, _, err := EnsurePiDefaultProviderModel(target, "fak", "qwen38:27b-q4"); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}

	_, modified, err := EnsurePiDefaultProviderModel(target, "fak", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if modified {
		t.Fatal("modified = true on the second identical call, want false (idempotent)")
	}
	after, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Errorf("file changed on an idempotent call:\nbefore=%s\nafter=%s", before, after)
	}
}

// TestEnsurePiDefaultProviderModelDefaultsProvider: an empty provider id falls back
// to the canonical fak provider rather than writing a blank key.
func TestEnsurePiDefaultProviderModelDefaultsProvider(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")

	if _, _, err := EnsurePiDefaultProviderModel(target, "   ", ""); err != nil {
		t.Fatal(err)
	}
	raw := readSettings(t, target)
	if raw["defaultProvider"] != DefaultPiProviderID {
		t.Errorf("defaultProvider = %v, want %q", raw["defaultProvider"], DefaultPiProviderID)
	}
	// An empty model normalizes to the canonical default, never the literal empty string.
	if raw["defaultModel"] != DefaultPiModelID {
		t.Errorf("defaultModel = %v, want %q", raw["defaultModel"], DefaultPiModelID)
	}
}

// TestEnsurePiDefaultProviderModelStripsBOM: a BOM-prefixed settings.json (a Windows
// tooling artifact) still parses, and the rewrite is BOM-free — the same repair the
// models.json path needed.
func TestEnsurePiDefaultProviderModelStripsBOM(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	bom := append([]byte{0xEF, 0xBB, 0xBF}, []byte(`{"defaultProvider":"hive-ai"}`)...)
	if err := os.WriteFile(target, bom, 0644); err != nil {
		t.Fatal(err)
	}

	if _, _, err := EnsurePiDefaultProviderModel(target, "fak", "qwen38:27b-q4"); err != nil {
		t.Fatalf("BOM-prefixed settings.json must still parse: %v", err)
	}
	after, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(string(after), "\ufeff") {
		t.Error("rewritten settings.json still carries a BOM prefix")
	}
	readSettings(t, target) // must remain valid JSON
}

// TestEnsurePiDefaultProviderModelIfAbsentPreservesExistingDefault is the core
// regression: a launcher whose model id came from a backend /healthz label must NOT
// overwrite a deliberately configured defaultModel. On a routing-mode router /healthz
// names the local planner engine, so adopting it would clobber the operator's route
// (e.g. DeepSeek V4.1 Flash -> hive-ai with fallbacks) and make the ladder unreachable.
func TestEnsurePiDefaultProviderModelIfAbsentPreservesExistingDefault(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	seed := `{
  "theme": "light",
  "defaultProvider": "fak",
  "defaultModel": "deepseek-ai/DeepSeek-V4.1-Flash"
}`
	if err := os.WriteFile(target, []byte(seed), 0644); err != nil {
		t.Fatal(err)
	}

	// Auto-detect reports the router's local engine label (a nemotron-class id here).
	path, modified, err := EnsurePiDefaultProviderModelIfAbsent(target, "fak", "nemotron-3-super-120b-a12b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if modified {
		t.Fatal("modified = true, want false: an existing default must be preserved")
	}

	raw := readSettings(t, path)
	if raw["defaultModel"] != "deepseek-ai/DeepSeek-V4.1-Flash" {
		t.Errorf("defaultModel = %v, want the deliberate default preserved", raw["defaultModel"])
	}
	if raw["defaultProvider"] != "fak" {
		t.Errorf("defaultProvider = %v, want fak", raw["defaultProvider"])
	}
}

// TestEnsurePiDefaultProviderModelIfAbsentStillRepointsProvider: even when the model
// default is preserved, the provider is still set to fak so a bare launch reaches the
// router rather than whatever provider was previously configured.
func TestEnsurePiDefaultProviderModelIfAbsentStillRepointsProvider(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	seed := `{
  "defaultProvider": "hive-ai",
  "defaultModel": "deepseek-ai/DeepSeek-V4.1-Flash"
}`
	if err := os.WriteFile(target, []byte(seed), 0644); err != nil {
		t.Fatal(err)
	}

	path, modified, err := EnsurePiDefaultProviderModelIfAbsent(target, "fak", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatal("modified = false, want true when the provider is repointed")
	}

	raw := readSettings(t, path)
	if raw["defaultProvider"] != "fak" {
		t.Errorf("defaultProvider = %v, want fak (provider must still be repointed)", raw["defaultProvider"])
	}
	if raw["defaultModel"] != "deepseek-ai/DeepSeek-V4.1-Flash" {
		t.Errorf("defaultModel = %v, want the deliberate default preserved", raw["defaultModel"])
	}
}

// TestEnsurePiDefaultProviderModelIfAbsentSeedsWhenEmpty: a first-ever config with no
// defaultModel is still seeded from the detected id, so auto-detect keeps working as
// a fallback for users who never pinned a default.
func TestEnsurePiDefaultProviderModelIfAbsentSeedsWhenEmpty(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	seed := `{"defaultProvider":"fak"}`
	if err := os.WriteFile(target, []byte(seed), 0644); err != nil {
		t.Fatal(err)
	}

	path, modified, err := EnsurePiDefaultProviderModelIfAbsent(target, "fak", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatal("modified = false, want true when seeding an absent default")
	}

	raw := readSettings(t, path)
	if raw["defaultModel"] != "qwen38:27b-q4" {
		t.Errorf("defaultModel = %v, want the detected seed qwen38:27b-q4", raw["defaultModel"])
	}
}

// TestEnsurePiDefaultProviderModelIfAbsentSeedsFreshFile: a missing settings.json is
// created and seeded (the fresh-install path).
func TestEnsurePiDefaultProviderModelIfAbsentSeedsFreshFile(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")

	path, modified, err := EnsurePiDefaultProviderModelIfAbsent(target, "fak", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatal("modified = false, want true on a fresh file")
	}
	raw := readSettings(t, path)
	if raw["defaultModel"] != "qwen38:27b-q4" {
		t.Errorf("defaultModel = %v, want qwen38:27b-q4", raw["defaultModel"])
	}
}

// TestPiDefaultModelIsDeliberate is the witness for the rule that decides whether an
// existing settings.json defaultModel survives auto-detect. The three cases are the three
// observed realities: a registry-known model is deliberate even when the probing backend
// (a routing-mode router whose /healthz names only the local engine) does not advertise it;
// an advertised id is deliberate; a stale placeholder that is neither is replaceable, which
// is what stops `custom-model` from being re-confirmed on every launch.
func TestPiDefaultModelIsDeliberate(t *testing.T) {
	nemotronCatalog := []string{"nemotron-3-super-120b-a12b"}
	tests := []struct {
		name       string
		modelID    string
		advertised []string
		want       bool
	}{
		{
			name:       "registry-known model survives an unrelated catalog",
			modelID:    "deepseek-ai/DeepSeek-V4.1-Flash",
			advertised: nemotronCatalog,
			want:       true,
		},
		{
			name:       "registry alias is deliberate",
			modelID:    "deepseek-v41-flash",
			advertised: nemotronCatalog,
			want:       true,
		},
		{
			name:       "advertised id is deliberate",
			modelID:    "nemotron-3-super-120b-a12b",
			advertised: nemotronCatalog,
			want:       true,
		},
		{
			name:       "stale placeholder is not deliberate when a real catalog omits it",
			modelID:    "custom-model",
			advertised: nemotronCatalog,
			want:       false,
		},
		{
			// No catalog observed is not evidence against the id: the launcher never
			// replaces a default on absence of evidence, only when a live catalog
			// positively omits it. The registry-known case above still returns true,
			// so this is about the placeholder path, not about the registry.
			name:       "placeholder is preserved when no catalog was observed",
			modelID:    "custom-model",
			advertised: nil,
			want:       true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := PiDefaultModelIsDeliberate(tc.modelID, tc.advertised); got != tc.want {
				t.Fatalf("PiDefaultModelIsDeliberate(%q, %v) = %v, want %v", tc.modelID, tc.advertised, got, tc.want)
			}
		})
	}
}

// TestPiDefaultModelIsDeliberateRegistryAuthority pins the authority the rule rests on: the
// per-model served-window registry (pi_model_windows.go) is what makes an id "known", so the
// registry and this rule cannot drift apart.
func TestPiDefaultModelIsDeliberateRegistryAuthority(t *testing.T) {
	for _, id := range []string{
		"deepseek-ai/DeepSeek-V4.1-Flash",
		"Qwen3.8-27B-UD-Q2_K_XL",
	} {
		if !IsPiKnownModel(id) {
			t.Errorf("IsPiKnownModel(%q) = false; the registry must name every routed default", id)
		}
		if !PiDefaultModelIsDeliberate(id, nil) {
			t.Errorf("PiDefaultModelIsDeliberate(%q, nil) = false; a registry-known id is deliberate", id)
		}
	}
	if IsPiKnownModel("custom-model") {
		t.Error("IsPiKnownModel(\"custom-model\") = true; a placeholder must not be registry-known")
	}
}

func readSettings(t *testing.T, path string) map[string]interface{} {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return raw
}
