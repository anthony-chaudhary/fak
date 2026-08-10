package lightgapport

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func repoRoot(t *testing.T) string {
	_, f, _, _ := runtime.Caller(0)
	return filepath.Clean(filepath.Join(filepath.Dir(f), "..", ".."))
}
func TestCommittedSwapWitnessesExist(t *testing.T) {
	r, err := Load(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Swaps) != 5 {
		t.Fatalf("swaps=%d", len(r.Swaps))
	}
	for _, s := range r.Swaps {
		if s.Fak.Path == "" || s.Fak.Test == "" {
			t.Fatalf("%s has no fak witness", s.ID)
		}
	}
}
func TestClaimedAlternativeIsNotCountedAsWitness(t *testing.T) {
	r := Contract()
	for _, s := range r.Swaps {
		for alt, w := range s.Alternatives {
			if w.Path != "" || w.Test != "" {
				t.Fatalf("%s/%s unexpectedly witnessed", s.ID, alt)
			}
		}
	}
}
func TestMissingWitnessFailsClosed(t *testing.T) {
	root := t.TempDir()
	r := Contract()
	w := r.Swaps[0].Fak
	if err := checkWitness(root, w); err == nil {
		t.Fatal("missing witness accepted")
	}
	if err := os.MkdirAll(filepath.Join(root, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
}
