package taskfixture

import (
	"go/parser"
	"go/token"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestSuiteSelection(t *testing.T) {
	def, err := Suite("", false)
	if err != nil || len(def) != len(Cases(false)) {
		t.Fatalf("default suite = %d fixtures, %v; want Cases(false)", len(def), err)
	}
	ext, err := Suite(SuiteExtended, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(ext) < 32 {
		t.Fatalf("extended suite with heldout = %d fixtures, want >= 32", len(ext))
	}
	if extNoHeldout, _ := Suite(SuiteExtended, false); len(extNoHeldout) < 30 {
		t.Fatalf("extended suite without heldout = %d fixtures, want >= 30", len(extNoHeldout))
	}
	if _, err := Suite("bogus", false); err == nil {
		t.Fatal("unknown suite accepted")
	}
	for _, f := range ext {
		if got, ok := ByID(f.ID); !ok || got.ID != f.ID {
			t.Fatalf("ByID(%q) missed", f.ID)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestExtendedFixturesAreWellFormed(t *testing.T) {
	seen := map[string]bool{}
	for _, f := range Cases(true) {
		seen[f.ID] = true
	}
	families := map[string]map[string]bool{}
	for _, f := range ExtendedCases() {
		if seen[f.ID] {
			t.Fatalf("duplicate fixture id %q", f.ID)
		}
		seen[f.ID] = true
		if families[f.Family] == nil {
			families[f.Family] = map[string]bool{}
		}
		families[f.Family][f.Workflow.Kind] = true
		if f.BrokenSource == f.FixedSource || f.Prompt == "" || f.TargetFile == "" || len(f.Workflow.RequiredSteps) == 0 {
			t.Fatalf("%s: incomplete fixture", f.ID)
		}
		for name, src := range map[string]string{f.TargetFile: f.BrokenSource, "fixed.go": f.FixedSource, "visible_test.go": f.VisibleTest, "oracle_test.go": f.OracleTest} {
			if _, err := parser.ParseFile(token.NewFileSet(), name, src, parser.AllErrors); err != nil {
				t.Fatalf("%s: %s does not parse: %v", f.ID, name, err)
			}
		}
	}
	if len(ExtendedCases()) < 24 || len(families) < 6 {
		t.Fatalf("extended suite = %d tasks over %d families, want >= 24 over >= 6", len(ExtendedCases()), len(families))
	}
	for family, kinds := range families {
		for _, kind := range []string{"behavior_fix", "tdd", "stale_reread"} {
			if !kinds[kind] {
				t.Fatalf("family %s lacks workflow %s", family, kind)
			}
		}
	}
}
