package hooks

import (
	"os"
	"path/filepath"
	"testing"
)

func TestUndeclaredLeaves(t *testing.T) {
	root := t.TempDir()
	dosToml := `[lanes]
concurrent = ["gateway", "policy", "harnessconformance"]
[lanes.trees]
gateway = ["internal/gateway/**"]
policy  = ["internal/policy/**"]
studyreceipt = ["internal/study/**", "cmd/fak/study.go"]
harnessconformance = ["pkg/harnessconformance/**"]
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
	mk("internal/study", true)         // composite lane name differs from package basename
	mk("internal/newleaf", true)       // real Go pkg in internal/, no lane -> expected gap
	mk("internal/another", true)       // real Go pkg in internal/, no lane -> expected gap
	mk("internal/docsonly", false)     // no Go file -> not a leaf
	mk("pkg/harnessconformance", true) // declared Go pkg in pkg/ -> no gap
	mk("pkg/newpkgleaf", true)         // real Go pkg in pkg/, no lane -> expected gap
	mk("pkg/pkgdocs", false)           // no Go file -> not a leaf

	gaps, err := UndeclaredLeaves(root)
	if err != nil {
		t.Fatalf("UndeclaredLeaves: %v", err)
	}
	got := map[string]string{}
	for _, g := range gaps {
		got[g.Leaf] = g.Base
	}
	if got["newleaf"] != "internal" || got["another"] != "internal" {
		t.Errorf("want internal gaps for newleaf and another, got %v", got)
	}
	if got["newpkgleaf"] != "pkg" {
		t.Errorf("want pkg gap for newpkgleaf, got %v", got)
	}
	if _, ok := got["gateway"]; ok {
		t.Errorf("gateway is declared and must not be flagged")
	}
	if _, ok := got["policy"]; ok {
		t.Errorf("policy is declared and must not be flagged")
	}
	if _, ok := got["study"]; ok {
		t.Errorf("study has explicit tree owner and must not be flagged")
	}
	if _, ok := got["harnessconformance"]; ok {
		t.Errorf("harnessconformance is declared and must not be flagged")
	}
	if _, ok := got["docsonly"]; ok {
		t.Errorf("a non-Go dir is not a leaf and must not be flagged")
	}
	if _, ok := got["pkgdocs"]; ok {
		t.Errorf("a non-Go dir in pkg/ is not a leaf and must not be flagged")
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
