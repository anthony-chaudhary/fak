package config

import (
	"fmt"
	"log"
)

// ValidPresets lists the recognized slab_preset values.
var ValidPresets = []string{"static", "auto", "benchmark", "sglang"}

// presetDescription returns a human-readable summary for startup logging.
func presetDescription(name string) string {
	switch name {
	case "static":
		return "static sizing, no vacuum"
	case "auto":
		return "full auto-tune with vacuum rebalancing"
	case "benchmark":
		return "static sizing, no vacuum, zero warmup delay"
	case "sglang":
		return "dedicated sizing with vacuum pressure rebalancing (SGLang)"
	default:
		return name
	}
}

// PresetDescription returns a human-readable summary of a preset for logging.
func PresetDescription(name string) string {
	return presetDescription(name)
}

// ApplyPreset applies a named slab preset to the config, setting fields that
// were not explicitly overridden by config or CLI flags.
func ApplyPreset(c *Config, overrides map[string]bool) error {
	if c.SlabPreset == "" {
		return nil
	}

	switch c.SlabPreset {
	case "static":
		applyStatic(c, overrides)
	case "auto":
		applyAuto(c, overrides)
	case "benchmark":
		applyBenchmark(c, overrides)
	case "sglang":
		applySGLang(c, overrides)
	default:
		return fmt.Errorf("unknown slab_preset: %q (valid: static, auto, benchmark, sglang)", c.SlabPreset)
	}

	return nil
}

// ValidatePreset checks preset-specific requirements after all overrides are applied.
func ValidatePreset(c *Config) error {
	switch c.SlabPreset {
	case "static", "benchmark", "sglang":
		if c.ModelPageBytes == 0 {
			return fmt.Errorf(
				"slab_preset=%q requires model_page_bytes > 0 "+
					"(set it to your dominant value size, e.g. 2097152 for ~2 MB values)",
				c.SlabPreset)
		}
	}
	return nil
}

func applyStatic(c *Config, overrides map[string]bool) {
	setIfNotOverridden(overrides, "slab_distribution", func() {
		c.SlabDistribution = "model"
	}, c.SlabDistribution, "model")

	setIfNotOverridden(overrides, "warmup_ops", func() {
		c.WarmupOps = 0
	}, c.WarmupOps, 0)

	setIfNotOverridden(overrides, "auto_tune_slabs", func() {
		c.AutoTuneSlabs = false
	}, c.AutoTuneSlabs, false)

	setIfNotOverridden(overrides, "vacuum_enabled", func() {
		c.VacuumEnabled = false
	}, c.VacuumEnabled, false)
}

func applyAuto(c *Config, overrides map[string]bool) {
	setIfNotOverridden(overrides, "slab_distribution", func() {
		c.SlabDistribution = "auto"
	}, c.SlabDistribution, "auto")

	setIfNotOverridden(overrides, "warmup_ops", func() {
		c.WarmupOps = 1000
	}, c.WarmupOps, 1000)

	setIfNotOverridden(overrides, "auto_tune_slabs", func() {
		c.AutoTuneSlabs = true
	}, c.AutoTuneSlabs, true)

	setIfNotOverridden(overrides, "vacuum_enabled", func() {
		c.VacuumEnabled = true
	}, c.VacuumEnabled, true)

	setIfNotOverridden(overrides, "vacuum_pressure_rebalancing", func() {
		c.VacuumPressureRebalancing = true
	}, c.VacuumPressureRebalancing, true)
}

func applyBenchmark(c *Config, overrides map[string]bool) {
	applyStatic(c, overrides)
	setIfNotOverridden(overrides, "vacuum_min_age_seconds", func() {
		c.VacuumMinAgeSeconds = 0
	}, c.VacuumMinAgeSeconds, 0)
}

func applySGLang(c *Config, overrides map[string]bool) {
	setIfNotOverridden(overrides, "slab_distribution", func() {
		c.SlabDistribution = "dedicated"
	}, c.SlabDistribution, "dedicated")

	setIfNotOverridden(overrides, "warmup_ops", func() {
		c.WarmupOps = 0
	}, c.WarmupOps, 0)

	setIfNotOverridden(overrides, "auto_tune_slabs", func() {
		c.AutoTuneSlabs = false
	}, c.AutoTuneSlabs, false)

	setIfNotOverridden(overrides, "vacuum_enabled", func() {
		c.VacuumEnabled = true
	}, c.VacuumEnabled, true)

	setIfNotOverridden(overrides, "vacuum_pressure_rebalancing", func() {
		c.VacuumPressureRebalancing = true
	}, c.VacuumPressureRebalancing, true)
}

func setIfNotOverridden[T comparable](overrides map[string]bool, field string, apply func(), current T, presetVal T) {
	if overrides[field] {
		if current != presetVal {
			log.Printf("[l3server] note: slab_preset sets %s but explicit config overrides it", field)
		}
		return
	}
	apply()
}
