package unwiredscore

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// writeGo writes a Go source file under root at repo-relative rel, creating parents.
func writeGo(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// fixtureTree builds a minimal fake module under a temp dir exercising every branch: a wired
// package (imported by a main), an orphan (imported by nothing), a bigger orphan (worst-first),
// a doc-only package (no decl -> not a candidate), and a testdata package (ignored).
func fixtureTree(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	// A binary that imports internal/wired -> wired is WIRED.
	writeGo(t, root, "cmd/app/main.go",
		"package main\nimport _ \""+ModulePrefix+"internal/wired\"\nfunc main() {}\n")
	writeGo(t, root, "internal/wired/wired.go",
		"package wired\nfunc Foo() int { return 1 }\n")
	// Two orphans imported by nothing; bigorphan has more source lines -> sorts first.
	writeGo(t, root, "internal/orphan/orphan.go",
		"package orphan\nfunc Bar() {}\n")
	writeGo(t, root, "internal/orphan/orphan_test.go",
		"package orphan\nimport \"testing\"\nfunc TestBar(t *testing.T) { Bar() }\n")
	writeGo(t, root, "internal/bigorphan/big.go",
		"package bigorphan\n"+strings.Repeat("// filler line\n", 20)+"func Big() {}\nvar X = 1\n")
	// A doc-only package: only a package clause + comment, no declaration -> NOT a candidate.
	writeGo(t, root, "internal/doconly/doc.go",
		"// Package doconly documents something.\npackage doconly\n")
	// A package that only self-registers under testdata -> ignored entirely by the go toolchain.
	writeGo(t, root, "internal/hasdata/testdata/x/x.go",
		"package x\nfunc Y() {}\n")
	return root
}

func scanByDir(pkgs []Pkg) map[string]Pkg {
	m := map[string]Pkg{}
	for _, p := range pkgs {
		m[p.Dir] = p
	}
	return m
}

func TestScanClassifiesWiredAndOrphans(t *testing.T) {
	root := fixtureTree(t)
	pkgs := Scan(root)
	by := scanByDir(pkgs)

	if _, ok := by["internal/doconly"]; ok {
		t.Errorf("doc-only package (no top-level decl) must not be a candidate, got %+v", by["internal/doconly"])
	}
	for _, p := range pkgs {
		if strings.Contains(p.Dir, "testdata") {
			t.Errorf("testdata package must be ignored, got candidate %s", p.Dir)
		}
	}

	w, ok := by["internal/wired"]
	if !ok {
		t.Fatal("internal/wired missing from candidates")
	}
	if !w.Wired {
		t.Errorf("internal/wired is imported by cmd/app -> must be Wired; got %+v", w)
	}
	if w.Unwired() {
		t.Errorf("internal/wired must not count as debt")
	}

	o, ok := by["internal/orphan"]
	if !ok {
		t.Fatal("internal/orphan missing from candidates")
	}
	if o.Wired || !o.Unwired() {
		t.Errorf("internal/orphan is imported by nothing -> must be unwired debt; got %+v", o)
	}
	if !o.HasTest {
		t.Errorf("internal/orphan has an orphan_test.go -> HasTest must be true; got %+v", o)
	}
}

func TestScanWorstFirstOrder(t *testing.T) {
	root := fixtureTree(t)
	pkgs := Scan(root)
	// The two unwired packages must precede the wired one, and the bigger orphan precede the
	// smaller (worst-first == biggest stranded investment first).
	var order []string
	for _, p := range pkgs {
		order = append(order, p.Dir)
	}
	posBig := indexOf(order, "internal/bigorphan")
	posSmall := indexOf(order, "internal/orphan")
	posWired := indexOf(order, "internal/wired")
	if posBig < 0 || posSmall < 0 || posWired < 0 {
		t.Fatalf("missing expected packages in order %v", order)
	}
	if !(posBig < posSmall && posSmall < posWired) {
		t.Errorf("expected bigorphan < orphan < wired, got order %v", order)
	}
}

func TestBuildFoldsUnwiredDebt(t *testing.T) {
	root := fixtureTree(t)
	payload := Build(root)
	if payload.Schema != Schema {
		t.Errorf("schema: got %q want %q", payload.Schema, Schema)
	}
	// Two orphans (orphan, bigorphan); wired and doconly are not debt.
	debt, ok := payload.Corpus[DebtKey]
	if !ok {
		t.Fatalf("corpus missing %s: %+v", DebtKey, payload.Corpus)
	}
	if debt != 2 {
		t.Errorf("unwired_debt: got %v want 2 (orphan + bigorphan)", debt)
	}
	if payload.OK {
		t.Errorf("payload.OK must be false when there is unwired debt")
	}
	if payload.Verdict != "ACTION" {
		t.Errorf("verdict: got %q want ACTION", payload.Verdict)
	}
}

func TestBuildCleanTree(t *testing.T) {
	root := t.TempDir()
	// A single package imported by a binary: zero debt, OK verdict.
	writeGo(t, root, "cmd/app/main.go",
		"package main\nimport _ \""+ModulePrefix+"internal/only\"\nfunc main() {}\n")
	writeGo(t, root, "internal/only/only.go", "package only\nfunc F() {}\n")
	payload := Build(root)
	if !payload.OK {
		t.Errorf("clean tree must be OK; corpus=%+v", payload.Corpus)
	}
	if payload.Corpus[DebtKey] != 0 {
		t.Errorf("clean tree unwired_debt: got %v want 0", payload.Corpus[DebtKey])
	}
}

func TestUnwiredPredicate(t *testing.T) {
	cases := []struct {
		p    Pkg
		want bool
	}{
		{Pkg{Wired: true}, false},
		{Pkg{Wired: false}, true},
		{Pkg{Wired: false, AllowReason: "public test harness"}, false},
		{Pkg{Wired: true, AllowReason: "x"}, false},
	}
	for _, c := range cases {
		if got := c.p.Unwired(); got != c.want {
			t.Errorf("Unwired(%+v) = %v, want %v", c.p, got, c.want)
		}
	}
}

func TestGapsAndActionItems(t *testing.T) {
	root := fixtureTree(t)
	gaps := Gaps(root)
	if len(gaps) != 2 {
		t.Fatalf("Gaps: got %d want 2 (%+v)", len(gaps), gaps)
	}
	for _, g := range gaps {
		if !g.Unwired() {
			t.Errorf("Gaps must return only unwired packages; got %+v", g)
		}
	}
	items := ActionItems(gaps, "fak unwired-scorecard --json")
	if len(items) != 2 {
		t.Fatalf("ActionItems: got %d want 2", len(items))
	}
	// Keys are content-stable (derived from the dir, no timestamp) so re-runs dedup.
	seen := map[string]bool{}
	for _, it := range items {
		if it.Key == "" || seen[it.Key] {
			t.Errorf("ActionItem key not unique/stable: %q", it.Key)
		}
		seen[it.Key] = true
		if it.DebtName != DebtKey || it.DebtCount != 1 {
			t.Errorf("ActionItem debt fields wrong: %+v", it)
		}
		if len(it.Paths) == 0 || !strings.HasSuffix(it.Paths[0], "/**") {
			t.Errorf("ActionItem must route by the package dir tree: %+v", it.Paths)
		}
	}
	// The bigger orphan's key is stable across the exact dir.
	wantKey := "unwired-debt/internal-bigorphan"
	if !seen[wantKey] {
		t.Errorf("missing expected stable key %q in %v", wantKey, keysOf(seen))
	}
}

// TestLiveTreeDeterministicAndConsistent runs the card over the REAL repo tree (this test file
// lives at internal/unwiredscore) and asserts the machinery is deterministic and internally
// consistent -- without pinning a specific orphan count (the fleet churns packages), so it never
// flaps. It also proves the end-to-end wiring signal: this very package is imported by
// cmd/fak/unwiredscore.go, so it must classify as WIRED (never flag itself).
func TestLiveTreeDeterministicAndConsistent(t *testing.T) {
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Skip("cannot locate self")
	}
	root := filepath.Dir(filepath.Dir(filepath.Dir(self))) // internal/unwiredscore -> internal -> repo root

	a := Scan(root)
	b := Scan(root)
	if len(a) != len(b) {
		t.Fatalf("Scan is not deterministic: %d vs %d candidates", len(a), len(b))
	}

	by := scanByDir(a)
	// The detector must not flag itself: cmd/fak imports internal/unwiredscore.
	if me, ok := by["internal/unwiredscore"]; ok {
		if !me.Wired {
			t.Errorf("internal/unwiredscore is imported by cmd/fak -> must be Wired, got %+v", me)
		}
	}
	// architest is doc.go + _test.go only (no top-level decl) -> never a candidate.
	if _, ok := by["internal/architest"]; ok {
		t.Errorf("internal/architest is a doc-only/test-harness package and must not be a candidate")
	}
	// No candidate may live under a testdata segment.
	for _, p := range a {
		if strings.Contains(p.Dir, "/testdata/") || strings.HasSuffix(p.Dir, "/testdata") {
			t.Errorf("testdata package leaked into candidates: %s", p.Dir)
		}
	}
	// Build's debt equals the count of Unwired packages (internal consistency).
	unwired := 0
	for _, p := range a {
		if p.Unwired() {
			unwired++
		}
	}
	payload := Build(root)
	if got := payload.Corpus[DebtKey]; got != unwired {
		t.Errorf("Build unwired_debt=%v but Scan counts %d unwired", got, unwired)
	}
}

func indexOf(ss []string, want string) int {
	for i, s := range ss {
		if s == want {
			return i
		}
	}
	return -1
}

func keysOf(m map[string]bool) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}
