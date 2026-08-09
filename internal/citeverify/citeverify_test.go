package citeverify

import (
	"os"
	"path/filepath"
	"testing"
)

func TestVerifyConservativeCitationOutcomes(t *testing.T) {
	root := t.TempDir()
	mustWrite := func(rel, body string) {
		t.Helper()
		path := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite("pkg/source.go", "package pkg\nfunc Target() {}\n\n")
	mustWrite("one/shared.go", "package one\n")
	mustWrite("two/shared.go", "package two\n")
	mustWrite("pkg/empty.go", "package pkg\n\n")

	tests := []struct {
		name, claim string
		evidence    []string
		want        Status
	}{
		{"supports exact symbol", "`Target` exists", []string{"pkg/source.go:2"}, Supports},
		{"resolved line lacks symbol", "`Missing` exists", []string{"pkg/source.go:2"}, Contradicts},
		{"out of range is contradiction", "`Target` exists", []string{"pkg/source.go:99"}, Contradicts},
		{"ambiguous basename is unknown", "`Target` exists", []string{"shared.go:1"}, Unknown},
		{"empty line is unknown", "`Target` exists", []string{"pkg/empty.go:2"}, Unknown},
		{"unresolved is unknown", "`Target` exists", []string{"missing.go:1"}, Unknown},
		{"mixed strong outcomes", "`Target` exists", []string{"pkg/source.go:2", "pkg/source.go:99"}, Mixed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Verify(tt.claim, tt.evidence, root); got != tt.want {
				t.Fatalf("Verify() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestVerifyRejectsUnsafeSources(t *testing.T) {
	root := t.TempDir()
	for _, rel := range []string{".env.go", "secret.go", "note.txt"} {
		if err := os.WriteFile(filepath.Join(root, rel), []byte("Target\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := Verify("`Target`", []string{rel + ":1"}, root); got != Unknown {
			t.Fatalf("%s: got %q, want unknown", rel, got)
		}
	}
	outside := filepath.Join(t.TempDir(), "outside.go")
	if err := os.WriteFile(outside, []byte("Target\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := Verify("`Target`", []string{outside + ":1"}, root); got != Unknown {
		t.Fatalf("outside root: got %q", got)
	}
}
