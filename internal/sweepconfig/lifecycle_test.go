package sweepconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestSweepConfigLifecycle(t *testing.T) {
	dir := t.TempDir()
	yamlPath := filepath.Join(dir, "lifecycle_profile.yaml")
	jsonPath := filepath.Join(dir, "lifecycle_profile.json")

	// Phase 1: Initialize baseline profile with default invariants.
	profile := DefaultProfile("alpha-sweep")
	if profile.Name != "alpha-sweep" {
		t.Fatalf("expected profile name 'alpha-sweep', got %q", profile.Name)
	}
	if profile.Workload.MaxTurns != 12 || profile.Workload.Trials != 1 || profile.Workload.TimeoutS != 600 {
		t.Fatalf("unexpected workload defaults: %+v", profile.Workload)
	}
	if profile.OutputDir != "fak/experiments/agent-live/sweep" {
		t.Fatalf("unexpected default output dir: %q", profile.OutputDir)
	}
	if !profile.Public {
		t.Fatalf("expected DefaultProfile to be public by default")
	}

	// Phase 2: Mutate with explicit lifecycle configurations.
	profile.Description = "end-to-end lifecycle sweep test"
	profile.OutputDir = "artifacts/alpha-run"
	profile.SkipAPI = false
	profile.SkipOffline = true
	profile.SkipLocalShim = false
	profile.FailFast = true
	profile.Public = false
	profile.Tags = []string{"unit-test", "lifecycle", "v1"}
	profile.Workload = WorkloadConfig{
		MaxTurns:       30,
		Trials:         5,
		TimeoutS:       1200,
		TranscriptPath: "transcripts/alpha.jsonl",
	}
	profile.Models = []ModelConfig{
		{
			Name:      "hosted-llm-1",
			Provider:  "openai",
			BaseURL:   "https://api.openai.com/v1",
			APIKeyEnv: "OPENAI_API_KEY",
			PriceHint: &PriceHint{
				Input:  0.0015,
				Output: 0.002,
				Source: "pricing_page",
			},
			Enabled: true,
		},
		{
			Name:      "local-shim-candidate",
			Provider:  "local",
			LocalShim: "bin/eval_runner.sh",
			Enabled:   false,
		},
	}

	// Phase 3: Save to YAML and verify round-trip persistence.
	if err := SaveProfile(profile, yamlPath); err != nil {
		t.Fatalf("failed to save YAML profile: %v", err)
	}
	loadedYAML, err := LoadProfile(yamlPath)
	if err != nil {
		t.Fatalf("failed to load YAML profile: %v", err)
	}
	if !reflect.DeepEqual(loadedYAML, profile) {
		t.Fatalf("YAML roundtrip mismatch:\nGot:  %+v\nWant: %+v", loadedYAML, profile)
	}

	// Phase 4: Save to JSON and verify LoadProfile dual-format decoding.
	rawJSON, err := json.Marshal(profile)
	if err != nil {
		t.Fatalf("failed to marshal JSON profile: %v", err)
	}
	if err := os.WriteFile(jsonPath, rawJSON, 0o644); err != nil {
		t.Fatalf("failed to write JSON profile: %v", err)
	}
	loadedJSON, err := LoadProfile(jsonPath)
	if err != nil {
		t.Fatalf("failed to load JSON profile: %v", err)
	}
	if !reflect.DeepEqual(loadedJSON, profile) {
		t.Fatalf("JSON roundtrip mismatch:\nGot:  %+v\nWant: %+v", loadedJSON, profile)
	}

	// Phase 5: Mutate and advance lifecycle version.
	profile.Models = append(profile.Models, ModelConfig{
		Name:     "fallback-worker",
		Provider: "anthropic",
		Enabled:  true,
	})
	profile.Workload.Trials = 10
	if err := SaveProfile(profile, yamlPath); err != nil {
		t.Fatalf("failed to overwrite updated YAML profile: %v", err)
	}
	reloaded, err := LoadProfile(yamlPath)
	if err != nil {
		t.Fatalf("failed to reload updated profile: %v", err)
	}
	if len(reloaded.Models) != 3 || reloaded.Workload.Trials != 10 {
		t.Fatalf("state evolution failed to persist: %+v", reloaded)
	}
}

func TestSweepConfigValidationAndFailClosed(t *testing.T) {
	dir := t.TempDir()

	t.Run("empty name in JSON fails closed", func(t *testing.T) {
		path := filepath.Join(dir, "empty_name.json")
		_ = os.WriteFile(path, []byte(`{"description": "missing name"}`), 0o644)
		_, err := LoadProfile(path)
		if err == nil {
			t.Fatal("expected error on JSON profile with empty name, got nil")
		}
	})

	t.Run("empty name in YAML fails closed", func(t *testing.T) {
		path := filepath.Join(dir, "empty_name.yaml")
		_ = os.WriteFile(path, []byte("description: \"no name specified\"\n"), 0o644)
		_, err := LoadProfile(path)
		if err == nil {
			t.Fatal("expected error on YAML profile with empty name, got nil")
		}
	})

	t.Run("non-existent file fails closed", func(t *testing.T) {
		path := filepath.Join(dir, "does_not_exist.yaml")
		_, err := LoadProfile(path)
		if err == nil {
			t.Fatal("expected error loading non-existent file, got nil")
		}
	})

	t.Run("corrupt yaml syntax fails closed", func(t *testing.T) {
		path := filepath.Join(dir, "corrupt.yaml")
		_ = os.WriteFile(path, []byte("name: valid\nworkload:\n  max_turns: [invalid flow syntax\n"), 0o644)
		_, err := LoadProfile(path)
		if err == nil {
			t.Fatal("expected error on corrupted YAML syntax, got nil")
		}
	})

	t.Run("model without name in YAML fails closed", func(t *testing.T) {
		path := filepath.Join(dir, "unnamed_model.yaml")
		content := "name: profile\nmodels:\n  - provider: local\n    enabled: true\n"
		_ = os.WriteFile(path, []byte(content), 0o644)
		_, err := LoadProfile(path)
		if err == nil {
			t.Fatal("expected error for model with missing name, got nil")
		}
	})

	t.Run("duplicate root key in YAML fails closed", func(t *testing.T) {
		path := filepath.Join(dir, "dup_key.yaml")
		content := "name: test\nname: test2\n"
		_ = os.WriteFile(path, []byte(content), 0o644)
		_, err := LoadProfile(path)
		if err == nil {
			t.Fatal("expected error on duplicate YAML root key, got nil")
		}
	})

	t.Run("duplicate workload key fails closed", func(t *testing.T) {
		path := filepath.Join(dir, "dup_workload.yaml")
		content := "name: test\nworkload:\n  max_turns: 5\n  max_turns: 10\n"
		_ = os.WriteFile(path, []byte(content), 0o644)
		_, err := LoadProfile(path)
		if err == nil {
			t.Fatal("expected error on duplicate workload key, got nil")
		}
	})

	t.Run("tabs in yaml indentation fail closed", func(t *testing.T) {
		path := filepath.Join(dir, "tabs.yaml")
		content := "name: test\nworkload:\n\tmax_turns: 5\n"
		_ = os.WriteFile(path, []byte(content), 0o644)
		_, err := LoadProfile(path)
		if err == nil {
			t.Fatal("expected error on tab indentation, got nil")
		}
	})
}

func TestSweepConfigEdgeCases(t *testing.T) {
	dir := t.TempDir()

	t.Run("subdirectories auto-created on SaveProfile", func(t *testing.T) {
		nestedPath := filepath.Join(dir, "sub1", "sub2", "nested.yaml")
		profile := DefaultProfile("nested")
		if err := SaveProfile(profile, nestedPath); err != nil {
			t.Fatalf("expected SaveProfile to create parent directories, got: %v", err)
		}
		if _, err := os.Stat(nestedPath); err != nil {
			t.Fatalf("nested file was not written: %v", err)
		}
	})

	t.Run("zero price hints and default source", func(t *testing.T) {
		path := filepath.Join(dir, "zero_price.yaml")
		content := `name: zero-cost
models:
  - name: free-model
    provider: local
    price_hint:
      input: 0
      output: 0
`
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		loaded, err := LoadProfile(path)
		if err != nil {
			t.Fatalf("unexpected error loading zero-cost model: %v", err)
		}
		if len(loaded.Models) != 1 {
			t.Fatalf("expected 1 model, got %d", len(loaded.Models))
		}
		ph := loaded.Models[0].PriceHint
		if ph == nil {
			t.Fatal("expected price_hint to be populated")
		}
		if ph.Input != 0 || ph.Output != 0 {
			t.Fatalf("expected 0 input/output rates, got input=%f output=%f", ph.Input, ph.Output)
		}
		if ph.Source != "manual" {
			t.Fatalf("expected default price_hint source 'manual', got %q", ph.Source)
		}
	})

	t.Run("list profiles in empty or non-existent directory", func(t *testing.T) {
		emptyDir := filepath.Join(dir, "empty_dir")
		_ = os.MkdirAll(emptyDir, 0o755)
		list := ListProfiles(emptyDir)
		if len(list) != 0 {
			t.Fatalf("expected 0 profiles in empty dir, got %d", len(list))
		}

		nonExistentDir := filepath.Join(dir, "does_not_exist")
		listNonExist := ListProfiles(nonExistentDir)
		if len(listNonExist) != 0 {
			t.Fatalf("expected 0 profiles in non-existent dir, got %d", len(listNonExist))
		}
	})

	t.Run("list profiles skips invalid files", func(t *testing.T) {
		mixedDir := filepath.Join(dir, "mixed")
		_ = os.MkdirAll(mixedDir, 0o755)

		validProfile := DefaultProfile("valid-profile")
		if err := SaveProfile(validProfile, filepath.Join(mixedDir, "valid.yaml")); err != nil {
			t.Fatal(err)
		}
		_ = os.WriteFile(filepath.Join(mixedDir, "invalid.yaml"), []byte("invalid content: [\n"), 0o644)

		results := ListProfiles(mixedDir)
		if len(results) != 1 {
			t.Fatalf("expected exactly 1 valid profile, got %d", len(results))
		}
		if results[0].Name != "valid-profile" {
			t.Fatalf("expected profile name 'valid-profile', got %q", results[0].Name)
		}
	})

	t.Run("concurrent reads and writes", func(t *testing.T) {
		concurrentDir := filepath.Join(dir, "concurrent")
		_ = os.MkdirAll(concurrentDir, 0o755)
		basePath := filepath.Join(concurrentDir, "profile.yaml")
		_ = SaveProfile(DefaultProfile("concurrent-base"), basePath)

		var wg sync.WaitGroup
		for i := 0; i < 20; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, _ = LoadProfile(basePath)
			}()
		}
		wg.Wait()
	})
}

func BenchmarkSweepConfig(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "bench_profile.yaml")
	profile := sampleProfile()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := SaveProfile(profile, path); err != nil {
			b.Fatalf("SaveProfile failed: %v", err)
		}
		if _, err := LoadProfile(path); err != nil {
			b.Fatalf("LoadProfile failed: %v", err)
		}
	}
}

func BenchmarkSaveProfile(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "bench_save.yaml")
	profile := sampleProfile()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := SaveProfile(profile, path); err != nil {
			b.Fatalf("SaveProfile failed: %v", err)
		}
	}
}

func BenchmarkLoadProfileYAML(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "bench_load.yaml")
	profile := sampleProfile()
	if err := SaveProfile(profile, path); err != nil {
		b.Fatalf("setup SaveProfile failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LoadProfile(path); err != nil {
			b.Fatalf("LoadProfile failed: %v", err)
		}
	}
}

func BenchmarkLoadProfileJSON(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "bench_load.json")
	profile := sampleProfile()
	raw, err := json.Marshal(profile)
	if err != nil {
		b.Fatalf("marshal failed: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		b.Fatalf("write failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LoadProfile(path); err != nil {
			b.Fatalf("LoadProfile JSON failed: %v", err)
		}
	}
}
