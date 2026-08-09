package devindex

import (
	"os"
	"path/filepath"
	"testing"
)

// writeMultiLineLaneRepo lays down a repo whose [lanes.trees] table uses the
// multi-line array spelling TOML allows. fak's own dos.toml writes every tree
// inline, so this shape is unexercised here — it was reported by a downstream
// adopter whose dos.toml wraps its wider doc trees across lines.
func writeMultiLineLaneRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := `[lanes]
concurrent = [
  "gateway",
  "docsroot",
]

[lanes.trees]
gateway = ["internal/gateway/**"]
docsroot = [
  "docs/**",
  "README.md",
  "INDEX.md",
] # every root doc plus the docs subtree
tools = ["tools/**"]
`
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "INDEX.md"), []byte("# INDEX\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

// TestParseLanesMultiLineTree pins the silent-in-the-safe-direction failure:
// parseLanes is a line scanner, so a multi-line array puts the lane's NAME and
// its GLOBS on different lines. The name alone marks the lane declared, which is
// what every "is this lane real?" check consults — so the leaf passes validation
// while contributing no prefixes and no exact entries, and LaneForPath falls
// through to `unknown` for every path the lane owns. Nothing errors; the gate
// that reads it just quietly stops resolving.
func TestParseLanesMultiLineTree(t *testing.T) {
	c, err := Load(writeMultiLineLaneRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	leaf, ok := c.LeafByName("docsroot")
	if !ok {
		t.Fatalf("multi-line lane `docsroot` produced no leaf; got %v", names(c.Leaves))
	}
	if leaf.Desc != "every root doc plus the docs subtree" {
		t.Errorf("docsroot desc = %q, want the comment trailing the closing bracket", leaf.Desc)
	}

	// The inline neighbours must keep working — the join must not eat them.
	for _, tc := range []struct{ path, want string }{
		{"docs/notes/x.md", "docsroot"},      // subtree glob, continuation line
		{"README.md", "docsroot"},            // exact entry, continuation line
		{"INDEX.md", "docsroot"},             // exact entry, line before the close
		{"internal/gateway/a.go", "gateway"}, // inline entry above the multi-line one
		{"tools/x.py", "tools"},              // inline entry below the multi-line one
	} {
		if got := c.LaneForPath(tc.path); got != tc.want {
			t.Errorf("LaneForPath(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

// TestParseLanesMultiLineConcurrent pins the same shape in the [lanes] table,
// whose values are lane NAMES rather than globs. This half already worked (every
// quoted token on every line is read), and the fix must not regress it.
func TestParseLanesMultiLineConcurrent(t *testing.T) {
	c, err := Load(writeMultiLineLaneRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for _, lane := range []string{"gateway", "docsroot", "tools"} {
		if !c.declared[lane] {
			t.Errorf("lane %q not declared; declared set = %v", lane, c.declared)
		}
	}
}
