package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// TestMicroConfigPrecedence pins the flags > env > file > defaults ladder the issue
// (#2029) requires: a file value survives where nothing above it sets the field, and
// an env var overrides the file for the field it names.
func TestMicroConfigPrecedence(t *testing.T) {
	cfg := defaultMicroConfig(true) // host mode: agents defaults to the worker count
	if cfg.Engine != "mock" || cfg.Isolation != microagent.BackendGoroutine {
		t.Fatalf("defaults: engine=%q isolation=%q", cfg.Engine, cfg.Isolation)
	}
	if cfg.Agents != microagent.DefaultWorkers {
		t.Fatalf("host default agents=%d, want %d", cfg.Agents, microagent.DefaultWorkers)
	}

	// File overlay: workers=4, turns=5 (below env, above defaults).
	dir := t.TempDir()
	path := filepath.Join(dir, "micro.json")
	if err := os.WriteFile(path, []byte(`{"workers":4,"turns":5}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := loadMicroConfigFile(path, &cfg); err != nil {
		t.Fatalf("loadMicroConfigFile: %v", err)
	}
	if cfg.Workers != 4 || cfg.Turns != 5 {
		t.Fatalf("after file: workers=%d turns=%d, want 4/5", cfg.Workers, cfg.Turns)
	}

	// Env over file: FAK_MICRO_WORKERS beats the file's workers; turns (no env) stays.
	t.Setenv("FAK_MICRO_WORKERS", "9")
	applyMicroEnv(&cfg)
	if cfg.Workers != 9 {
		t.Errorf("env should override file: workers=%d, want 9", cfg.Workers)
	}
	if cfg.Turns != 5 {
		t.Errorf("field with no env should keep file value: turns=%d, want 5", cfg.Turns)
	}
}

// TestMicroValidate pins the isolation-backend guard: only a registered ToolExec
// backend name is accepted, and the numeric caps must be sane.
func TestMicroValidate(t *testing.T) {
	cfg := defaultMicroConfig(false)
	if err := validateMicroConfig(cfg); err != nil {
		t.Fatalf("default config should validate: %v", err)
	}
	bad := cfg
	bad.Isolation = "bogus"
	if err := validateMicroConfig(bad); err == nil {
		t.Error("unknown isolation backend should be refused")
	}
	bad = cfg
	bad.Agents = 0
	if err := validateMicroConfig(bad); err == nil {
		t.Error("agents<1 should be refused")
	}
	bad = cfg
	bad.AdmissionTokenBudget = -1
	if err := validateMicroConfig(bad); err == nil {
		t.Error("negative admission cap should be refused")
	}
}

// TestMicroSlots pins the effective slot-pool derivation: an explicit seat count
// wins, otherwise it falls back to the worker count.
func TestMicroSlots(t *testing.T) {
	if got := (microConfig{Workers: 8, Seats: 0}).slots(); got != 8 {
		t.Errorf("seats=0 should derive from workers: got %d, want 8", got)
	}
	if got := (microConfig{Workers: 8, Seats: 3}).slots(); got != 3 {
		t.Errorf("explicit seats should win: got %d, want 3", got)
	}
}

// TestMicroRunEndToEndOnMock is the #2029 acceptance witness: `fak micro` runs
// agents end-to-end on the Mock engine through the real microagent.Host, the slot
// scheduler, and one audit sink — every spawned agent retires done with the
// configured number of steps.
func TestMicroRunEndToEndOnMock(t *testing.T) {
	cfg := defaultMicroConfig(false) // bare `fak micro`: one agent
	cfg.Turns = 2
	if cfg.Agents != 1 {
		t.Fatalf("bare micro should default to 1 agent, got %d", cfg.Agents)
	}
	if err := runMicro(cfg, false, true); err != nil {
		t.Fatalf("runMicro (1 agent): %v", err)
	}

	// A small fleet also completes cleanly through the host.
	fleet := defaultMicroConfig(true)
	fleet.Agents = 5
	fleet.Turns = 3
	fleet.Seats = 2
	if err := runMicro(fleet, true, true); err != nil {
		t.Fatalf("runMicro (fleet): %v", err)
	}
}
