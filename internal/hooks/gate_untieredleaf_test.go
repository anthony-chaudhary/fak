package hooks

import (
	"strings"
	"testing"
)

// A synthetic architest tier table: the parser keys on the `var tier = map[string]int{` marker,
// the packed `"name": <int>` rows, the closing `}`, and the `// new-leaf:tier` insert anchor.
const fakeTierTable = `package architest

var tier = map[string]int{
	"abi":    0,
	"hooks":  1, "newleaf": 1,
	// new-leaf:tier
}
`

// stagedWithTable builds a StagedDiff whose tier-table read is pre-seeded from src (no disk), so
// a gate test exercises the parse + admission logic in isolation.
func stagedWithTable(src string, added ...string) *StagedDiff {
	return &StagedDiff{
		AddedPaths: added,
		fileCache:  map[string]fileEntry{tierTableFile: {data: []byte(src), exists: src != ""}},
	}
}

// A commit that adds a new internal/<leaf>/ non-test .go absent from the tier table is refused,
// and the finding names the leaf, the insert marker, and the tier-table file — acceptance #1's
// "stage a synthetic leaf and assert the commit is refused".
func TestGateUntieredLeafRefusesNewLeaf(t *testing.T) {
	d := stagedWithTable(fakeTierTable, "internal/widget/widget.go")
	findings, err := gateUntieredLeaf(d)
	if err != nil {
		t.Fatalf("gateUntieredLeaf returned error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("want 1 finding, got %d: %+v", len(findings), findings)
	}
	f := findings[0]
	if f.Gate != "UNTIERED_LEAF" {
		t.Errorf("gate = %q, want UNTIERED_LEAF", f.Gate)
	}
	if f.File != "internal/widget/" {
		t.Errorf("file = %q, want internal/widget/", f.File)
	}
	for _, want := range []string{"widget", tierInsertMarker, tierTableFile, "fak new-leaf widget"} {
		if !strings.Contains(f.Detail, want) {
			t.Errorf("detail %q missing %q", f.Detail, want)
		}
	}
}

// A leaf already carrying a tier row does not fire — re-touching a declared package is clean.
func TestGateUntieredLeafAllowsDeclaredLeaf(t *testing.T) {
	d := stagedWithTable(fakeTierTable, "internal/hooks/newfile.go")
	findings, err := gateUntieredLeaf(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("declared leaf should not fire, got %+v", findings)
	}
}

// A test-only file cannot make a leaf declarable, and architest is excluded from its own scan —
// neither fires, matching the whole-tree TIER_DECLARED gate's on-disk rules.
func TestGateUntieredLeafIgnoresTestOnlyAndArchitest(t *testing.T) {
	d := stagedWithTable(fakeTierTable,
		"internal/widget/widget_test.go", // test-only: not a declarable source
		"internal/architest/extra.go",    // architest excludes itself
		"internal/widget/sub/deep.go",    // nested: seg[2] is a dir, not a source file
		"docs/notes/x.md",                // outside internal/
	)
	findings, err := gateUntieredLeaf(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("want no findings, got %+v", findings)
	}
}

// Two new untiered leaves in one commit each fire, sorted by path for stable operator output.
func TestGateUntieredLeafSortsMultiple(t *testing.T) {
	d := stagedWithTable(fakeTierTable, "internal/zed/zed.go", "internal/alpha/alpha.go")
	findings, err := gateUntieredLeaf(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 2 {
		t.Fatalf("want 2 findings, got %d: %+v", len(findings), findings)
	}
	if findings[0].File != "internal/alpha/" || findings[1].File != "internal/zed/" {
		t.Errorf("findings not sorted by path: %q, %q", findings[0].File, findings[1].File)
	}
}

// An unreadable / shape-changed tier table fails OPEN (ErrCouldNotRun), never a false
// UNTIERED_LEAF — the whole-tree gate and architest's test remain the backstop.
func TestGateUntieredLeafFailsOpen(t *testing.T) {
	// Missing table.
	if _, err := gateUntieredLeaf(stagedWithTable("", "internal/widget/widget.go")); err != ErrCouldNotRun {
		t.Errorf("missing table: err = %v, want ErrCouldNotRun", err)
	}
	// Marker present but zero parseable keys (shape changed).
	garbled := "package architest\n\nvar tier = map[string]int{\n}\n"
	if _, err := gateUntieredLeaf(stagedWithTable(garbled, "internal/widget/widget.go")); err != ErrCouldNotRun {
		t.Errorf("empty table: err = %v, want ErrCouldNotRun", err)
	}
}
