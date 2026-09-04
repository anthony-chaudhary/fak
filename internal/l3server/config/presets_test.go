package config

import (
	"testing"
)

func TestApplyPresetStatic(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "static"
	cfg.ModelPageBytes = 2097152

	if err := ApplyPreset(&cfg, nil); err != nil {
		t.Fatalf("ApplyPreset: %v", err)
	}

	if cfg.SlabDistribution != "model" {
		t.Errorf("expected slab_distribution=model, got %s", cfg.SlabDistribution)
	}
	if cfg.WarmupOps != 0 {
		t.Errorf("expected warmup_ops=0, got %d", cfg.WarmupOps)
	}
	if cfg.AutoTuneSlabs {
		t.Error("expected auto_tune_slabs=false")
	}
	if cfg.VacuumEnabled {
		t.Error("expected vacuum_enabled=false")
	}
}

func TestApplyPresetAuto(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "auto"

	if err := ApplyPreset(&cfg, nil); err != nil {
		t.Fatalf("ApplyPreset: %v", err)
	}

	if cfg.SlabDistribution != "auto" {
		t.Errorf("expected slab_distribution=auto, got %s", cfg.SlabDistribution)
	}
	if cfg.WarmupOps != 1000 {
		t.Errorf("expected warmup_ops=1000, got %d", cfg.WarmupOps)
	}
	if !cfg.AutoTuneSlabs {
		t.Error("expected auto_tune_slabs=true")
	}
	if !cfg.VacuumEnabled {
		t.Error("expected vacuum_enabled=true")
	}
	if !cfg.VacuumPressureRebalancing {
		t.Error("expected vacuum_pressure_rebalancing=true")
	}
}

func TestApplyPresetBenchmark(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "benchmark"
	cfg.ModelPageBytes = 2097152

	if err := ApplyPreset(&cfg, nil); err != nil {
		t.Fatalf("ApplyPreset: %v", err)
	}

	if cfg.SlabDistribution != "model" {
		t.Errorf("expected slab_distribution=model, got %s", cfg.SlabDistribution)
	}
	if cfg.VacuumEnabled {
		t.Error("expected vacuum_enabled=false")
	}
	if cfg.VacuumMinAgeSeconds != 0 {
		t.Errorf("expected vacuum_min_age_seconds=0, got %d", cfg.VacuumMinAgeSeconds)
	}
}

func TestApplyPresetUnknown(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "nope"

	err := ApplyPreset(&cfg, nil)
	if err == nil {
		t.Fatal("expected error for unknown preset")
	}
}

func TestPresetExplicitOverride(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "static"
	cfg.ModelPageBytes = 2097152
	cfg.VacuumEnabled = true

	overrides := map[string]bool{"vacuum_enabled": true}
	if err := ApplyPreset(&cfg, overrides); err != nil {
		t.Fatalf("ApplyPreset: %v", err)
	}

	if !cfg.VacuumEnabled {
		t.Error("explicit vacuum_enabled=true should override preset")
	}
	if cfg.SlabDistribution != "model" {
		t.Errorf("expected slab_distribution=model, got %s", cfg.SlabDistribution)
	}
}

func TestValidatePresetStaticNoModelPage(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "static"
	cfg.ModelPageBytes = 0

	err := ValidatePreset(&cfg)
	if err == nil {
		t.Fatal("expected error for static preset without model_page_bytes")
	}
}

func TestValidatePresetBenchmarkNoModelPage(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "benchmark"
	cfg.ModelPageBytes = 0

	err := ValidatePreset(&cfg)
	if err == nil {
		t.Fatal("expected error for benchmark preset without model_page_bytes")
	}
}

func TestValidatePresetAutoNoModelPage(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "auto"
	cfg.ModelPageBytes = 0

	err := ValidatePreset(&cfg)
	if err != nil {
		t.Fatalf("auto preset should not require model_page_bytes: %v", err)
	}
}

func TestApplyPresetSGLang(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "sglang"
	cfg.ModelPageBytes = 5242880

	if err := ApplyPreset(&cfg, nil); err != nil {
		t.Fatalf("ApplyPreset: %v", err)
	}

	if cfg.SlabDistribution != "dedicated" {
		t.Errorf("expected slab_distribution=dedicated, got %s", cfg.SlabDistribution)
	}
	if cfg.WarmupOps != 0 {
		t.Errorf("expected warmup_ops=0, got %d", cfg.WarmupOps)
	}
	if cfg.AutoTuneSlabs {
		t.Error("expected auto_tune_slabs=false")
	}
	if !cfg.VacuumEnabled {
		t.Error("expected vacuum_enabled=true")
	}
	if !cfg.VacuumPressureRebalancing {
		t.Error("expected vacuum_pressure_rebalancing=true")
	}
}

func TestValidatePresetSGLangNoModelPage(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "sglang"
	cfg.ModelPageBytes = 0

	err := ValidatePreset(&cfg)
	if err == nil {
		t.Fatal("expected error for sglang preset without model_page_bytes")
	}
}

func TestPresetSGLangExplicitOverride(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "sglang"
	cfg.ModelPageBytes = 5242880
	cfg.VacuumEnabled = false

	overrides := map[string]bool{"vacuum_enabled": true}
	if err := ApplyPreset(&cfg, overrides); err != nil {
		t.Fatalf("ApplyPreset: %v", err)
	}

	if cfg.VacuumEnabled {
		t.Error("explicit vacuum_enabled=false should override preset")
	}
	if cfg.SlabDistribution != "dedicated" {
		t.Errorf("expected slab_distribution=dedicated, got %s", cfg.SlabDistribution)
	}
}

func TestPresetSGLangViaValidate(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "sglang"
	cfg.ModelPageBytes = 5242880
	cfg.MaxMemoryGB = 1
	cfg.NumShards = 1
	cfg.MaxKeys = 1000

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	if cfg.SlabDistribution != "dedicated" {
		t.Errorf("expected slab_distribution=dedicated after Validate, got %s", cfg.SlabDistribution)
	}
	if !cfg.VacuumEnabled {
		t.Error("expected vacuum_enabled=true after Validate with sglang preset")
	}
}

func TestPresetViaValidate(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SlabPreset = "static"
	cfg.ModelPageBytes = 2097152
	cfg.MaxMemoryGB = 1
	cfg.NumShards = 1
	cfg.MaxKeys = 1000

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	if cfg.SlabDistribution != "model" {
		t.Errorf("expected slab_distribution=model after Validate, got %s", cfg.SlabDistribution)
	}
	if cfg.VacuumEnabled {
		t.Error("expected vacuum_enabled=false after Validate with static preset")
	}
}
