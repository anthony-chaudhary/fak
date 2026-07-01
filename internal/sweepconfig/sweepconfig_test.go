package sweepconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func sampleProfile() SweepProfile {
	return SweepProfile{
		Name:        "nightly",
		Description: "the nightly sweep",
		Models: []ModelConfig{
			{
				Name:      "zai/glm-4.7-flash",
				Provider:  "zai",
				BaseURL:   "https://api.example/v1",
				APIKeyEnv: "ZAI_KEY",
				PriceHint: &PriceHint{Input: 0.5, Output: 1.5, Source: "docs"},
				Enabled:   true,
			},
			{Name: "local/qwen", Provider: "local", LocalShim: "shims/qwen.sh", Enabled: false},
		},
		Workload:  WorkloadConfig{MaxTurns: 20, Trials: 3, TimeoutS: 900, TranscriptPath: "t/x.jsonl"},
		OutputDir: "out/here",
		SkipAPI:   true,
		Tags:      []string{"a", "b"},
		Public:    false,
	}
}

func TestSaveThenLoadRecoversEveryField(t *testing.T) {
	p := sampleProfile()
	path := filepath.Join(t.TempDir(), "nightly.yaml")
	if err := SaveProfile(p, path); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, p) {
		t.Fatalf("profile mismatch\ngot:  %+v\nwant: %+v", got, p)
	}
}

func TestMinimalJSONProfileUsesDocumentedDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bare.json")
	b, _ := json.Marshal(map[string]any{"name": "bare"})
	if err := osWrite(path, b); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "bare" || got.Description != "" || len(got.Models) != 0 || got.OutputDir != "fak/experiments/agent-live/sweep" || got.SkipAPI || !got.Public {
		t.Fatalf("defaults wrong: %+v", got)
	}
	if got.Workload.MaxTurns != 12 || got.Workload.Trials != 1 || got.Workload.TimeoutS != 600 || got.Workload.TranscriptPath != "" {
		t.Fatalf("workload defaults wrong: %+v", got.Workload)
	}
}

func TestModelProviderDefaultsToUnknown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.json")
	b, _ := json.Marshal(map[string]any{"name": "p", "models": []any{map[string]any{"name": "x"}}})
	if err := osWrite(path, b); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Models) != 1 || got.Models[0].Provider != "unknown" || !got.Models[0].Enabled || got.Models[0].PriceHint != nil {
		t.Fatalf("model defaults wrong: %+v", got.Models)
	}
}

func TestProfilePaths(t *testing.T) {
	dir := t.TempDir()
	if err := osWrite(filepath.Join(dir, "p.yml"), []byte("name: p\n")); err != nil {
		t.Fatal(err)
	}
	if got := GetProfilePath("p", dir); got != filepath.Join(dir, "p.yml") {
		t.Fatalf("path = %q", got)
	}
	if err := osWrite(filepath.Join(dir, "p.yaml"), []byte("name: p\n")); err != nil {
		t.Fatal(err)
	}
	if got := GetProfilePath("p", dir); got != filepath.Join(dir, "p.yaml") {
		t.Fatalf("path = %q", got)
	}
	if got := GetProfilePath("nope", dir); got != filepath.Join(dir, "nope.yaml") {
		t.Fatalf("absent path = %q", got)
	}
}

func TestListProfilesLoadsAllInDir(t *testing.T) {
	dir := t.TempDir()
	if err := SaveProfile(DefaultProfile("one"), filepath.Join(dir, "one.yaml")); err != nil {
		t.Fatal(err)
	}
	if err := SaveProfile(DefaultProfile("two"), filepath.Join(dir, "two.yaml")); err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, p := range ListProfiles(dir) {
		names = append(names, p.Name)
	}
	sort.Strings(names)
	if !reflect.DeepEqual(names, []string{"one", "two"}) {
		t.Fatalf("names = %+v", names)
	}
}

func TestExistingSimpleYAMLShapeLoads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quick.yaml")
	body := []byte("name: quick\nworkload:\n  max_turns: 5\nmodels:\n  - name: m\n    provider: p\n    price_hint:\n      input: 0.07\n      output: 0.40\n      source: docs\ntags:\n  - smoke\npublic: true\n")
	if err := osWrite(path, body); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "quick" || got.Workload.MaxTurns != 5 || len(got.Models) != 1 || got.Models[0].PriceHint == nil || got.Tags[0] != "smoke" {
		t.Fatalf("parsed yaml wrong: %+v", got)
	}
}

func osWrite(path string, b []byte) error {
	return os.WriteFile(path, b, 0o644)
}
