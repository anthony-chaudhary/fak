package categorybaseline

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestEvaluateCompletedLayerRedirect(t *testing.T) {
	r := Normalize(Registry{Categories: []Category{{Name: "serving", Layers: []string{"medium-model", "l2-cache", "l3-cache"}, CompletedLayer: "medium-model", NextLayer: "l2-cache", Witness: "fak serve selfcheck"}}})
	if got := Evaluate(r, "serving", "medium-model", false); !got.Hold || got.NextLayer != "l2-cache" {
		t.Fatalf("completed layer = %+v", got)
	}
	if got := Evaluate(r, "serving", "l2-cache", false); got.Hold {
		t.Fatalf("next layer held: %+v", got)
	}
	if got := Evaluate(r, "serving", "medium-model", true); got.Hold {
		t.Fatalf("regression held: %+v", got)
	}
	if got := Evaluate(r, "", "", false); got.Hold {
		t.Fatalf("undeclared held: %+v", got)
	}
}

func TestSaveUsesTrackedCanonicalPathAndLoadFallsBackToLegacy(t *testing.T) {
	root := t.TempDir()
	legacy := filepath.Join(root, filepath.FromSlash(LegacyPath))
	if err := os.MkdirAll(filepath.Dir(legacy), 0o755); err != nil {
		t.Fatal(err)
	}
	legacyRegistry := Normalize(Registry{Categories: []Category{{Name: "legacy", Layers: []string{"one", "two"}, CompletedLayer: "one", NextLayer: "two", Witness: "legacy witness"}}})
	legacyBytes, err := json.Marshal(legacyRegistry)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacy, legacyBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := Load(root); len(got.Categories) != 1 || got.Categories[0].Name != "legacy" {
		t.Fatalf("legacy fallback = %+v", got)
	}

	canonical := Normalize(Registry{Categories: []Category{{Name: "canonical", Layers: []string{"one", "two"}, CompletedLayer: "one", NextLayer: "two", Witness: "canonical witness"}}})
	if err := Save(root, canonical); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(DefaultPath))); err != nil {
		t.Fatalf("canonical registry not persisted: %v", err)
	}
	if got := Load(root); len(got.Categories) != 1 || got.Categories[0].Name != "canonical" {
		t.Fatalf("canonical did not override legacy = %+v", got)
	}
}
