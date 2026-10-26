package main

import (
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestCLIFlagsExpandHomeModelPath(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	cfg, err := parseFlags([]string{
		"-mode", "producer",
		"-gguf", "~/weights.gguf",
		"-prompt-ids", "1,2",
	})
	if err != nil {
		t.Fatalf("parseFlags returned an error: %v", err)
	}
	if want := filepath.Join(home, "weights.gguf"); cfg.gguf != want {
		t.Fatalf("model path = %q, want %q", cfg.gguf, want)
	}
}
