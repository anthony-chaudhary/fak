package hooks

import (
	"os"
	"path/filepath"
	"testing"
)

func TestUndeclaredLeaves(t *testing.T) {
	root := t.TempDir()
	dosToml := `[lanes]
concurrent = ["gateway", "policy"]
[lanes.trees]
gateway = ["internal/gateway/**"]
policy  = ["internal/policy/**"]
`
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}
	// gateway/policy are declared Go packages; newleaf is an undeclared Go package; docsonly has
	// no .go file (not a leaf); declared-but-empty would also be skipped (no Go file).
	mk := func(rel string, withGo bool) {
		d := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if withGo {
			if err := os.WriteFile(filepath.Join(d, "x.go"), []byte("package x\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	mk("internal/gateway", true)
	mk("internal/policy", true)
	mk("internal/newleaf", true)   // real Go pkg, no lane -> expected gap
	mk("internal/another", true)   // real Go pkg, no lane -> expected gap
	mk("internal/docsonly", false) // no Go file -> not a leaf

	gaps, err := UndeclaredLeaves(root)
	if err != nil {
		t.Fatalf("UndeclaredLeaves: %v", err)
	}
	got := map[string]bool{}
	for _, g := range gaps {
		got[g.Leaf] = true
		if g.Base != "internal" {
			t.Errorf("unexpected base %q for %q", g.Base, g.Leaf)
		}
	}
	if !got["newleaf"] || !got["another"] {
		t.Errorf("want newleaf+another flagged, got %v", got)
	}
	if got["gateway"] || got["policy"] {
		t.Errorf("declared lanes must not be flagged, got %v", got)
	}
	if got["docsonly"] {
		t.Errorf("a non-Go dir is not a leaf and must not be flagged")
	}
	// Result must be sorted.
	for i := 1; i < len(gaps); i++ {
		if gaps[i-1].Leaf > gaps[i].Leaf {
			t.Errorf("gaps not sorted: %v", gaps)
		}
	}
}

func TestUndeclaredLeaves_noDosTomlIsError(t *testing.T) {
	root := t.TempDir() // no dos.toml
	if _, err := UndeclaredLeaves(root); err == nil {
		t.Fatal("want an error when dos.toml is unreadable (could-not-run, not a clean zero)")
	}
}
