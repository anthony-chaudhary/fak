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
