package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/headroom"
)

// TestBenchInputsFromPaths pins the real-corpus reader: it picks up a directory's
// top-level files plus explicit args, sorts them, skips subdirectories, and skips
// any file over the size cap.
func TestBenchInputsFromPaths(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	write("a.txt", "alpha")
	write("b.txt", "bravo")
	write("big.txt", "xxxxxxxxxxxxxxxxxxxx") // 20 bytes, over the cap below
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}

	inputs, err := benchInputsFromPaths(dir, nil, 10) // cap 10 bytes -> big.txt skipped
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) != 2 {
		t.Fatalf("got %d inputs, want 2 (a.txt, b.txt); %+v", len(inputs), names(inputs))
	}
	// sorted by path -> a.txt before b.txt
	if inputs[0].Name != "a.txt" || inputs[1].Name != "b.txt" {
		t.Fatalf("order/names wrong: %v", names(inputs))
	}
	if string(inputs[0].Bytes) != "alpha" {
		t.Fatalf("a.txt content wrong: %q", inputs[0].Bytes)
	}
	for _, in := range inputs {
		if in.Name == "big.txt" {
			t.Fatal("big.txt should have been skipped by the size cap")
		}
		if in.Name == "sub" {
			t.Fatal("subdirectory should not be read as a file")
		}
	}

	// explicit file args are read too (no cap hit at 0 = unlimited).
	got, err := benchInputsFromPaths("", []string{filepath.Join(dir, "big.txt")}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != "big.txt" {
		t.Fatalf("explicit arg not read: %v", names(got))
	}

	// a missing dir is an error; missing explicit files are skipped, not fatal.
	if _, err := benchInputsFromPaths(filepath.Join(dir, "nope"), nil, 0); err == nil {
		t.Fatal("expected an error for a missing --dir")
	}
}

func names(in []headroom.BenchInput) []string {
	out := make([]string, len(in))
	for i, x := range in {
		out[i] = x.Name
	}
	return out
}
