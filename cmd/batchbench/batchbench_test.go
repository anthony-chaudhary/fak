// Package main unit tests for the pure, deterministic helpers in batchbench:
// capPositive (clamp with a >=1 floor), bestMS (least-contended minimum
// duration in ms), and lcgIDs (the deterministic LCG token-id generator). All
// are resource-free: no model file, GPU, network, or subprocess is touched.
// (parseInts moved to internal/intlist under #776; see intlist_test.go.)
package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCapPositive(t *testing.T) {
	tests := []struct {
		name   string
		n, cap int
		want   int
	}{
		{"under cap passes through", 5, 10, 5},
		{"over cap is clamped", 50, 10, 10},
		{"equal to cap passes through", 10, 10, 10},
		{"zero cap disables clamp", 50, 0, 50},
		{"negative cap disables clamp", 50, -1, 50},
		{"n below one floors to one", 0, 10, 1},
		{"negative n floors to one", -7, 10, 1},
		{"n below one with cap zero still floors", 0, 0, 1},
		{"one stays one", 1, 0, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := capPositive(tt.n, tt.cap); got != tt.want {
				t.Fatalf("capPositive(%d, %d) = %d, want %d", tt.n, tt.cap, got, tt.want)
			}
		})
	}
}

func TestBestMS(t *testing.T) {
	// bestMS returns the MINIMUM duration converted to milliseconds. 2_000_000 ns = 2 ms.
	ds := []time.Duration{
		5 * time.Millisecond,
		2 * time.Millisecond,
		9 * time.Millisecond,
	}
	if got := bestMS(ds); got != 2.0 {
		t.Fatalf("bestMS = %v, want 2.0", got)
	}

	// Single element: returns that element in ms.
	if got := bestMS([]time.Duration{3500 * time.Microsecond}); got != 3.5 {
		t.Fatalf("bestMS(single) = %v, want 3.5", got)
	}

	// bestMS must not mutate the caller's slice order (it copies before sorting).
	orig := []time.Duration{
		5 * time.Millisecond,
		2 * time.Millisecond,
		9 * time.Millisecond,
	}
	_ = bestMS(orig)
	if orig[0] != 5*time.Millisecond || orig[1] != 2*time.Millisecond || orig[2] != 9*time.Millisecond {
		t.Fatalf("bestMS mutated input slice: %v", orig)
	}
}

func TestReadHFConfigFillsHeadDim(t *testing.T) {
	dir := t.TempDir()
	config := []byte(`{
		"model_type": "qwen2",
		"hidden_size": 1536,
		"num_attention_heads": 12,
		"num_key_value_heads": 2,
		"num_hidden_layers": 28,
		"intermediate_size": 8960,
		"vocab_size": 151936
	}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), config, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := readHFConfig(dir)
	if err != nil {
		t.Fatalf("readHFConfig: %v", err)
	}
	if cfg.ModelType != "qwen2" {
		t.Fatalf("ModelType = %q, want qwen2", cfg.ModelType)
	}
	if cfg.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want hidden_size/num_attention_heads = 128", cfg.HeadDim)
	}
}

func TestHFNameIncludesModelTypeWhenPresent(t *testing.T) {
	dir := filepath.Join("models", "snapshots", "abc123") + string(os.PathSeparator)
	cfg, err := readHFConfig(writeHFConfig(t, `{"model_type":"llama","hidden_size":64,"num_attention_heads":4}`))
	if err != nil {
		t.Fatalf("readHFConfig: %v", err)
	}
	if got := hfName(cfg, dir); got != "abc123 (llama)" {
		t.Fatalf("hfName with type = %q, want abc123 (llama)", got)
	}

	cfg.ModelType = ""
	if got := hfName(cfg, dir); got != "abc123" {
		t.Fatalf("hfName without type = %q, want abc123", got)
	}
}

func TestHFSourcePrefersShardIndex(t *testing.T) {
	dir := t.TempDir()
	if got := hfSource(dir); got != filepath.Join(dir, "model.safetensors") {
		t.Fatalf("hfSource without index = %q", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(`{"weight_map":{}}`), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	if got := hfSource(dir); got != filepath.Join(dir, "model.safetensors.index.json") {
		t.Fatalf("hfSource with index = %q", got)
	}
}

func writeHFConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return dir
}

func TestLCGIDs(t *testing.T) {
	// Length, range, and determinism are all contract; the first two values are
	// computed directly from the LCG recurrence with seed 0, vocab 1000:
	//   state0 = 2463534242
	//   state1 = (state0*1103515245 + 12345) & 0x7fffffff = 1266642227 -> %1000 = 227
	//   state2 = (state1*1103515245 + 12345) & 0x7fffffff           -> %1000 = 776
	const vocab = 1000
	got := lcgIDs(5, vocab, 0)
	if len(got) != 5 {
		t.Fatalf("lcgIDs returned len %d, want 5", len(got))
	}
	if got[0] != 227 {
		t.Fatalf("lcgIDs[0] = %d, want 227", got[0])
	}
	if got[1] != 776 {
		t.Fatalf("lcgIDs[1] = %d, want 776", got[1])
	}
	for i, id := range got {
		if id < 0 || id >= vocab {
			t.Fatalf("lcgIDs[%d] = %d out of range [0,%d)", i, id, vocab)
		}
	}

	// Determinism: same (n, vocab, seed) yields identical output.
	again := lcgIDs(5, vocab, 0)
	for i := range got {
		if got[i] != again[i] {
			t.Fatalf("lcgIDs not deterministic at %d: %d vs %d", i, got[i], again[i])
		}
	}

	// A different seed produces a different sequence (here, a different first id).
	other := lcgIDs(5, vocab, 1)
	if other[0] == got[0] {
		t.Fatalf("lcgIDs seed 1 unexpectedly matched seed 0 first id %d", got[0])
	}

	// Zero length is a valid empty result.
	if z := lcgIDs(0, vocab, 0); len(z) != 0 {
		t.Fatalf("lcgIDs(0,...) len = %d, want 0", len(z))
	}
}
