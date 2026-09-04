package godfileceiling

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// repoRoot is two levels up from internal/godfileceiling.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	return root
}

var testCaps = map[string]int{
	"internal/big/monster.go": 3000,
	"cmd/x/large.go":          1800,
}

func TestLineCount(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"", 0},
		{"a\n", 1},
		{"a\nb\n", 2},
		{"a\nb", 2}, // final line without trailing newline still counts
		{"\n\n", 2}, // two blank lines
		{"one", 1},
	}
	for _, c := range cases {
		if got := LineCount([]byte(c.in)); got != c.want {
			t.Errorf("LineCount(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestExcluded(t *testing.T) {
	if !Excluded("vendor/x/y.go") {
		t.Error("vendor path should be excluded")
	}
	if !Excluded(".claude/worktrees/w/internal/a/b.go") {
		t.Error(".claude checkout path should be excluded")
	}
	if Excluded("internal/gateway/metrics.go") {
		t.Error("first-party path should not be excluded")
	}
}

func TestEvaluateCleanTree(t *testing.T) {
	measured := map[string]int{
		"internal/big/monster.go": 3000, // exactly at cap — ok
		"cmd/x/large.go":          1750, // under cap — ok (re-pin opportunity)
		"internal/small/a.go":     400,  // unpinned, under ceiling — ok
		"internal/small/b.go":     1500, // unpinned, exactly at ceiling — ok
	}
	v := Evaluate(measured, testCaps)
	if !v.OK {
		t.Fatalf("clean tree should pass, got violations %+v", v.Violations)
	}
	if len(v.Shrunk) != 1 || v.Shrunk[0].Path != "cmd/x/large.go" || v.Shrunk[0].Under != 50 {
		t.Errorf("expected cmd/x/large.go shrunk by 50, got %+v", v.Shrunk)
	}
}

func TestEvaluateNewGodFileFails(t *testing.T) {
	v := Evaluate(map[string]int{"internal/new/whale.go": 1600}, testCaps)
	if v.OK {
		t.Fatal("a new 1600-line file must fail the no-new-god-file rule")
	}
	if len(v.Violations) != 1 || v.Violations[0].Kind != "new-god-file" || v.Violations[0].Over != 100 {
		t.Errorf("expected one new-god-file over-by-100, got %+v", v.Violations)
	}
}

func TestEvaluatePinnedGrewFails(t *testing.T) {
	measured := map[string]int{"internal/big/monster.go": 3001, "cmd/x/large.go": 1800}
	v := Evaluate(measured, testCaps)
	if v.OK {
		t.Fatal("a pinned god-file growing past its cap must fail the ratchet")
	}
	if len(v.Violations) != 1 || v.Violations[0].Kind != "grew-past-cap" || v.Violations[0].Over != 1 {
		t.Errorf("expected one grew-past-cap over-by-1, got %+v", v.Violations)
	}
}

func TestEvaluatePinnedShrankPasses(t *testing.T) {
	measured := map[string]int{"internal/big/monster.go": 2500, "cmd/x/large.go": 1800}
	v := Evaluate(measured, testCaps)
	if !v.OK {
		t.Fatalf("a shrinking god-file must not fail the gate, got %+v", v.Violations)
	}
	if len(v.Shrunk) != 1 || v.Shrunk[0].Under != 500 {
		t.Errorf("expected shrink surfaced by 500, got %+v", v.Shrunk)
	}
}

func TestEvaluateStalePin(t *testing.T) {
	// cmd/x/large.go is pinned but absent from the measured tree.
	v := Evaluate(map[string]int{"internal/big/monster.go": 3000}, testCaps)
	if !v.OK {
		t.Fatalf("a stale pin is not a violation, got %+v", v.Violations)
	}
	if len(v.StalePins) != 1 || v.StalePins[0] != "cmd/x/large.go" {
		t.Errorf("expected cmd/x/large.go stale, got %+v", v.StalePins)
	}
}

func TestRepinRatchetsDownOnly(t *testing.T) {
	// A file shrank below its cap: repin accepts the lower cap.
	measured := map[string]int{"internal/big/monster.go": 2900, "cmd/x/large.go": 1750}
	accepted, refusals := Repin(measured, testCaps)
	if len(refusals) != 0 {
		t.Fatalf("a pure shrink must be accepted, got refusals %v", refusals)
	}
	if accepted["internal/big/monster.go"] != 2900 || accepted["cmd/x/large.go"] != 1750 {
		t.Errorf("accepted baseline did not ratchet down: %+v", accepted)
	}
}

func TestRepinRefusesRaisingCap(t *testing.T) {
	measured := map[string]int{"internal/big/monster.go": 3100, "cmd/x/large.go": 1800}
	accepted, refusals := Repin(measured, testCaps)
	if accepted != nil {
		t.Fatal("repin must refuse to raise a cap")
	}
	if len(refusals) != 1 || !contains(refusals[0], "RAISE") {
		t.Errorf("expected a RAISE refusal, got %v", refusals)
	}
}

func TestRepinRefusesNewOffender(t *testing.T) {
	measured := map[string]int{
		"internal/big/monster.go": 3000,
		"cmd/x/large.go":          1800,
		"internal/new/whale.go":   1600, // new file over the ceiling
	}
	accepted, refusals := Repin(measured, testCaps)
	if accepted != nil {
		t.Fatal("repin must refuse to pin a brand-new offender")
	}
	if len(refusals) != 1 || !contains(refusals[0], "NEW file") {
		t.Errorf("expected a NEW-file refusal, got %v", refusals)
	}
}

// TestLiveTreeUnderCeiling IS the gate: it measures the real repo tree and asserts the
// shipped Baseline holds — no unpinned file over the ceiling, no pinned file grown past
// its cap. `make ci` runs this via `go test ./...`. A failure means someone landed a new
// monolith or grew an existing god-file; split it (`/modularize`) or, if a file shrank,
// re-pin lower by regenerating baseline.go (MeasureTree->Repin->FormatBaseline; see doc.go).
func TestLiveTreeUnderCeiling(t *testing.T) {
	root := repoRoot(t)
	measured, err := MeasureTree(root)
	if err != nil {
		t.Skipf("cannot measure tree (git unavailable?): %v", err)
	}
	if len(measured) == 0 {
		t.Skip("no tracked .go files measured — not a git checkout")
	}
	v := Evaluate(measured, Baseline)
	if !v.OK {
		for _, viol := range v.Violations {
			t.Errorf("%s: %s — %d lines > cap %d (+%d)",
				viol.Path, viol.Kind, viol.Lines, viol.Cap, viol.Over)
		}
		t.Fatalf("god-file ceiling violated by %d file(s); split them or re-pin (see doc.go)",
			len(v.Violations))
	}
}

// TestBaselineIsConsistent guards the shipped baseline: every pin must be a real god-file
// (over the hard ceiling) and never a _test.go (those are graded by the tests KPI, not
// architecture, and churn per new leaf — see doc.go), else it does not belong in the ratchet.
func TestBaselineIsConsistent(t *testing.T) {
	for rel, cap := range Baseline {
		if cap <= HardCeiling {
			t.Errorf("pinned %s cap %d is at/under HardCeiling %d — it should not be pinned",
				rel, cap, HardCeiling)
		}
		if rel != filepath.ToSlash(rel) {
			t.Errorf("baseline path %q must be forward-slashed", rel)
		}
		if strings.HasSuffix(rel, "_test.go") {
			t.Errorf("baseline pins %s: a _test.go file must not be pinned (tests KPI grades them; MeasureTree excludes them)", rel)
		}
	}
}

// TestMeasureTreeExcludesTests covers the _test.go exclusion end-to-end: MeasureTree over
// the real tracked tree must return zero _test.go paths, so an over-ceiling test file (a
// shared *_test.go that gains a row per new leaf) can never red the gate. This is the
// fix for the spurious grew-past-cap that a pinned architest_test.go produced.
func TestMeasureTreeExcludesTests(t *testing.T) {
	measured, err := MeasureTree(repoRoot(t))
	if err != nil {
		t.Skipf("cannot measure tree (git unavailable?): %v", err)
	}
	if len(measured) == 0 {
		t.Skip("no tracked .go files measured — not a git checkout")
	}
	for rel := range measured {
		if strings.HasSuffix(rel, "_test.go") {
			t.Errorf("MeasureTree included a _test.go file %q — tests must be excluded from the god-file corpus", rel)
		}
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// TestGodFileCeilingMaturityContractsAndDocs asserts that internal/godfileceiling maintains
// substantive contract comments and full exported symbol documentation.
func TestGodFileCeilingMaturityContractsAndDocs(t *testing.T) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "godfileceiling.go", nil, parser.ParseComments)
	if err != nil {
		t.Fatalf("failed to parse godfileceiling.go: %v", err)
	}

	contractCount := 0
	for _, cg := range node.Comments {
		if isSubstantiveContractComment(cg) {
			contractCount++
		}
	}
	if contractCount == 0 {
		t.Error("godfileceiling.go: expected at least one substantive contract comment")
	}

	exported := 0
	documented := 0
	for _, decl := range node.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if ast.IsExported(d.Name.Name) {
				exported++
				if d.Doc != nil && len(strings.TrimSpace(d.Doc.Text())) > 0 {
					documented++
				} else {
					t.Errorf("undocumented exported func: %s", d.Name.Name)
				}
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if ast.IsExported(s.Name.Name) {
						exported++
						doc := s.Doc
						if doc == nil {
							doc = d.Doc
						}
						if doc != nil && len(strings.TrimSpace(doc.Text())) > 0 {
							documented++
						} else {
							t.Errorf("undocumented exported type: %s", s.Name.Name)
						}
					}
				case *ast.ValueSpec:
					for _, name := range s.Names {
						if ast.IsExported(name.Name) {
							exported++
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if doc != nil && len(strings.TrimSpace(doc.Text())) > 0 {
								documented++
							} else {
								t.Errorf("undocumented exported const/var: %s", name.Name)
							}
						}
					}
				}
			}
		}
	}
	if documented < exported {
		t.Errorf("documented exports %d/%d is below 100%%", documented, exported)
	}
}

func isSubstantiveContractComment(cg *ast.CommentGroup) bool {
	if cg == nil {
		return false
	}
	text := strings.TrimSpace(cg.Text())
	if len(text) < 35 {
		return false
	}
	lower := strings.ToLower(text)

	hasContractMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "assumption:") ||
		strings.Contains(lower, "assumptions:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.Contains(lower, "precondition:") ||
		strings.Contains(lower, "postcondition:") ||
		strings.Contains(lower, "guard:")
	if !hasContractMarker {
		return false
	}

	words := strings.Fields(lower)
	if len(words) < 6 {
		return false
	}

	keywordCount := 0
	for _, w := range words {
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.4 {
		return false
	}
	return true
}
