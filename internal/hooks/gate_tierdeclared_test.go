package hooks

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestTierDeclared_LiveTreeClean: once the four leaves (dispatchorder/dojo/looprecover/
// nightrun) are declared, the gate must find ZERO undeclared internal packages on the
// real tracked tree — the same end-to-end witness internal/architest's
// TestEveryPackageDeclaresTier provides, one boundary earlier.
func TestTierDeclared_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateTierDeclaredTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != 0 {
		t.Fatalf("undeclared internal leaf on the tracked tree: %+v", findings)
	}
}

// TestTierDeclared_FiresOnUndeclaredLeaf: a synthetic tree with a tier table that does
// NOT list internal/synthundeclared must produce exactly one TIER_DECLARED finding;
// adding the row clears it.
func TestTierDeclared_FiresOnUndeclaredLeaf(t *testing.T) {
	tierBody := func(declareSynth bool) string {
		rows := `	"abi": 0,
	"hooks": 1,
`
		if declareSynth {
			rows += "\t\"synthundeclared\": 1,\n"
		}
		return "package architest\n\nvar tier = map[string]int{\n" + rows + "}\n"
	}

	build := func(declareSynth bool) *TrackedTree {
		root := t.TempDir()
		write := func(rel, body string) {
			p := filepath.Join(root, filepath.FromSlash(rel))
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		write(tierTableFile, tierBody(declareSynth))
		write("internal/synthundeclared/x.go", "package synthundeclared\n")
		// an _test.go-only dir must NOT count as a leaf needing a tier
		write("internal/onlytests/x_test.go", "package onlytests\n")
		return &TrackedTree{
			Root: root,
			Paths: []string{
				tierTableFile,
				"internal/synthundeclared/x.go",
				"internal/onlytests/x_test.go",
			},
			fileCache: map[string]fileEntry{},
		}
	}

	// Undeclared -> exactly one finding, naming the leaf.
	findings, err := gateTierDeclaredTree(build(false))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("want 1 TIER_DECLARED finding, got %d: %+v", len(findings), findings)
	}
	if findings[0].Gate != "TIER_DECLARED" || findings[0].File != "internal/synthundeclared/" {
		t.Fatalf("finding wrong: %+v", findings[0])
	}

	// Declared -> clean.
	findings, err = gateTierDeclaredTree(build(true))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("declared leaf should be clean, got %+v", findings)
	}
}

// TestTierDeclared_FailsOpenOnUnreadableTable: with no tier table on the tree, the gate
// returns ErrCouldNotRun (fail open) rather than flagging every package.
func TestTierDeclared_FailsOpenOnUnreadableTable(t *testing.T) {
	tree := &TrackedTree{
		Root:      t.TempDir(),
		Paths:     []string{"internal/foo/x.go"},
		fileCache: map[string]fileEntry{},
	}
	if _, err := gateTierDeclaredTree(tree); err != ErrCouldNotRun {
		t.Fatalf("want ErrCouldNotRun on a missing tier table, got %v", err)
	}
}
