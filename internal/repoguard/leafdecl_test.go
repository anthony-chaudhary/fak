package repoguard

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testDosTOML = `[lanes]
concurrent = [
  "adjudicator", "agent",
]

[lanes.trees]
adjudicator = ["internal/adjudicator/**"]
repoguard = ["internal/repoguard/**"] # the guard's own leaf
scorecard   = ["pkg/scorecard/**"]

[reasons.OUT_OF_TREE_WRITE]
summary = "not a lane tree row"
`

const testArchitest = `package architest

var tier = map[string]int{
	"abi": 0,

	"adjudicator":      3, // composer
	"repoguard":        1, // pure classifier
}

var tierName = []string{"root", "foundation"}
`

func TestLeafForWritePath(t *testing.T) {
	const ws = "C:/work/fak"
	cases := []struct {
		path string
		want string
		ok   bool
	}{
		{`C:\work\fak\internal\newthing\x.go`, "newthing", true},
		{"internal/repoguard/guard.go", "repoguard", true},
		{"internal/deep/sub/pkg/y.go", "deep", true},
		{"internal/loose.go", "", false},  // directly in internal/, not a leaf
		{"internal", "", false},           // the dir itself
		{"cmd/fak/main.go", "", false},    // not an internal leaf
		{"pkg/scorecard/s.go", "", false}, // pkg/ leaves are not this rung's business
		{"C:/other/repo/internal/x/y.go", "", false},
		{"", "", false},
	}
	for _, tc := range cases {
		got, ok := LeafForWritePath(tc.path, ws)
		if got != tc.want || ok != tc.ok {
			t.Errorf("LeafForWritePath(%q) = %q,%v want %q,%v", tc.path, got, ok, tc.want, tc.ok)
		}
	}
}

func TestParseLaneTrees(t *testing.T) {
	lanes := ParseLaneTrees(strings.NewReader(testDosTOML))
	for _, want := range []string{"adjudicator", "repoguard", "scorecard"} {
		if !lanes[want] {
			t.Errorf("ParseLaneTrees = %v, missing %q", lanes, want)
		}
	}
	// Section-scoped: the `[lanes]` list above and the `[reasons.*]` block below
	// must not leak keys into the tree taxonomy.
	for _, absent := range []string{"concurrent", "summary"} {
		if lanes[absent] {
			t.Errorf("ParseLaneTrees = %v, must not include %q (wrong section)", lanes, absent)
		}
	}
}

func TestParseArchTiers(t *testing.T) {
	tiers := ParseArchTiers(strings.NewReader(testArchitest))
	for _, want := range []string{"abi", "adjudicator", "repoguard"} {
		if !tiers[want] {
			t.Errorf("ParseArchTiers = %v, missing %q", tiers, want)
		}
	}
	// The table ends at the closing brace — later `var` blocks are not tier rows.
	if tiers["tiername"] || tiers["root"] {
		t.Errorf("ParseArchTiers = %v, leaked past the tier map", tiers)
	}
}

func testDeclarations() LeafDeclarations {
	return LeafDeclarations{
		Lanes: ParseLaneTrees(strings.NewReader(testDosTOML)),
		Tiers: ParseArchTiers(strings.NewReader(testArchitest)),
	}
}

func writeCall(path string) map[string]any { return map[string]any{"file_path": path} }

// The issue's first Done-when clause: an edit that would create an undeclared
// leaf produces an advisory AT EDIT TIME, naming both missing declarations.
func TestUndeclaredLeafAdvisoryOnNewLeaf(t *testing.T) {
	const ws = "C:/work/fak"
	for _, tool := range []string{"Write", "Edit", "MultiEdit", "NotebookEdit"} {
		v := EvaluateWithHints(tool, writeCall("internal/newthing/newthing.go"), ws,
			SafeRootsForWorkspace(ws), Hints{LeafDeclarations: testDeclarations()})
		if len(v) != 1 || v[0].Reason != ReasonUndeclaredLeaf {
			t.Fatalf("%s(new leaf) = %+v, want one %s", tool, v, ReasonUndeclaredLeaf)
		}
		if v[0].Resolved != "internal/newthing" {
			t.Errorf("%s resolved = %q, want internal/newthing", tool, v[0].Resolved)
		}
		if !strings.Contains(v[0].Why, "[lanes.trees]") || !strings.Contains(v[0].Why, "architest tier row") {
			t.Errorf("%s why = %q, want both missing declarations named", tool, v[0].Why)
		}
		if !strings.Contains(v[0].Fix, "fak new-leaf newthing") {
			t.Errorf("%s fix = %q, want the runnable new-leaf verb", tool, v[0].Fix)
		}
	}
}

// The second Done-when clause: clean edits are SILENT.
func TestDeclaredLeafIsSilent(t *testing.T) {
	const ws = "C:/work/fak"
	for _, path := range []string{
		"internal/repoguard/guard.go", // declared in both tables
		"internal/adjudicator/x.go",
		"cmd/fak/main.go",    // not an internal leaf at all
		"docs/AGENTS.md",     // not Go
		"pkg/scorecard/s.go", // pkg leaf, out of scope
	} {
		if v := EvaluateWithHints("Write", writeCall(path), ws, SafeRootsForWorkspace(ws),
			Hints{LeafDeclarations: testDeclarations()}); len(v) != 0 {
			t.Errorf("Write(%q) = %+v, want silence", path, v)
		}
	}
}

// A leaf that has a lane row but no tier row is the architest-drift case that
// wedges trunk at commit time; it must still be caught (and vice versa).
func TestUndeclaredLeafHalfDeclared(t *testing.T) {
	const ws = "C:/work/fak"
	cases := []struct {
		name string
		decl LeafDeclarations
		why  string
	}{
		{"lane row only", LeafDeclarations{
			Lanes: map[string]bool{"half": true},
			Tiers: map[string]bool{"repoguard": true},
		}, "no architest tier row"},
		{"tier row only", LeafDeclarations{
			Lanes: map[string]bool{"repoguard": true},
			Tiers: map[string]bool{"half": true},
		}, "no dos.toml [lanes.trees] row"},
	}
	for _, tc := range cases {
		v := EvaluateWithHints("Write", writeCall("internal/half/h.go"), ws, SafeRootsForWorkspace(ws),
			Hints{LeafDeclarations: tc.decl})
		if len(v) != 1 || v[0].Why != tc.why {
			t.Errorf("%s: got %+v, want one violation why=%q", tc.name, v, tc.why)
		}
	}
}

// Fail-open: with no taxonomy loaded (unreadable dos.toml, or a caller that
// never resolved one) the rung must stay quiet rather than flag every leaf.
func TestUndeclaredLeafSilentWithoutTaxonomy(t *testing.T) {
	const ws = "C:/work/fak"
	if v := EvaluateWithHints("Write", writeCall("internal/newthing/x.go"), ws,
		SafeRootsForWorkspace(ws), Hints{}); len(v) != 0 {
		t.Errorf("no taxonomy = %+v, want silence", v)
	}
	if v := Evaluate("Write", writeCall("internal/newthing/x.go"), ws, SafeRootsForWorkspace(ws)); len(v) != 0 {
		t.Errorf("Evaluate (hintless) = %+v, want silence", v)
	}
}

// Advisory only — the issue is explicit that the edit is never blocked.
func TestUndeclaredLeafSeverityIsAdvisory(t *testing.T) {
	if got := DefaultSeverity(ReasonUndeclaredLeaf); got != SeverityWarn {
		t.Fatalf("DefaultSeverity(%s) = %v, want warn", ReasonUndeclaredLeaf, got)
	}
	if got := ResolveSeverity(ReasonUndeclaredLeaf, nil, "warn"); got != SeverityWarn {
		t.Errorf("master-switch warn = %v, want warn", got)
	}
	if got := ResolveSeverity(ReasonUndeclaredLeaf, nil, "off"); got != SeverityOff {
		t.Errorf("master-switch off = %v, want off", got)
	}
}

func TestRenderUndeclaredLeafReason(t *testing.T) {
	const ws = "C:/work/fak"
	v := EvaluateWithHints("Write", writeCall("internal/newthing/x.go"), ws,
		SafeRootsForWorkspace(ws), Hints{LeafDeclarations: testDeclarations()})
	reason := RenderReason(v)
	for _, want := range []string{ReasonUndeclaredLeaf, "internal/newthing", "fak new-leaf newthing", "Advisory only"} {
		if !strings.Contains(reason, want) {
			t.Errorf("RenderReason = %q, missing %q", reason, want)
		}
	}
}

// An out-of-tree write into a sibling repo's internal/ tree keeps reporting the
// containment finding and must not also grow a leaf advisory for a tree that is
// not ours to declare.
func TestUndeclaredLeafSkipsOutOfTreeWrite(t *testing.T) {
	const ws = "C:/work/fak"
	v := EvaluateWithHints("Write", writeCall("C:/work/other/internal/newthing/x.go"), ws,
		SafeRootsForWorkspace(ws), Hints{LeafDeclarations: testDeclarations()})
	if len(v) != 1 || v[0].Reason != Reason {
		t.Fatalf("sibling write = %+v, want one %s only", v, Reason)
	}
}

func TestLeafDeclarationsForWorkspace(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(testDosTOML), 0o644); err != nil {
		t.Fatal(err)
	}
	archDir := filepath.Join(root, "internal", "architest")
	if err := os.MkdirAll(archDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(archDir, "architest_test.go"), []byte(testArchitest), 0o644); err != nil {
		t.Fatal(err)
	}
	decl := LeafDeclarationsForWorkspace(root)
	if !decl.Loaded() || !decl.Lanes["repoguard"] || !decl.Tiers["repoguard"] {
		t.Fatalf("LeafDeclarationsForWorkspace = %+v, want both tables loaded", decl)
	}
	// A workspace with neither file loads nothing and therefore judges nothing.
	if empty := LeafDeclarationsForWorkspace(t.TempDir()); empty.Loaded() {
		t.Errorf("empty workspace = %+v, want not loaded", empty)
	}
}
