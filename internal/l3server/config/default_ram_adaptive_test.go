package config

import "testing"

// TestDefaultConfigValidOn16GB simulates a 16 GB CI runner (the environment
// where the debug-listener tests fail) and asserts DefaultConfig().Validate()
// passes. It guards the RAM-adaptive defaults against regressing to a
// hard-coded appliance-sized value that would fail the per-field and aggregate
// RAM checks on small hosts.
func TestDefaultConfigValidOn16GB(t *testing.T) {
	orig := systemMemoryGB
	systemMemoryGB = func() float64 { return 16.0 }
	t.Cleanup(func() { systemMemoryGB = orig })

	cfg := DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("DefaultConfig().Validate() on 16 GB = %v, want nil", err)
	}
	t.Logf("16 GB: max_memory_gb=%d max_keys=%d headroom=%.1f",
		cfg.MaxMemoryGB, cfg.MaxKeys, cfg.BudgetHeadroomGB)
}

// TestDefaultConfigValidOn8GB checks the smallest realistic runner.
func TestDefaultConfigValidOn8GB(t *testing.T) {
	orig := systemMemoryGB
	systemMemoryGB = func() float64 { return 8.0 }
	t.Cleanup(func() { systemMemoryGB = orig })

	cfg := DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("DefaultConfig().Validate() on 8 GB = %v, want nil", err)
	}
	t.Logf("8 GB: max_memory_gb=%d max_keys=%d headroom=%.1f",
		cfg.MaxMemoryGB, cfg.MaxKeys, cfg.BudgetHeadroomGB)
}
