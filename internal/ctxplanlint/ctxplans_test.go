package ctxplanlint

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const fixtureRoot = "testdata/repo"

// TestScanFixture pins the walker against the committed fixture tree: three
// declared context verbs (session, vcache, guard), two undeclared context verbs
// (headroom + recall, whose directive is PARTIAL so it does not count), one
// undeclared context skill (ctx-overlay), and the non-context verb (widget) and
// skill (quality-report) correctly ignored.
func TestScanFixture(t *testing.T) {
	rep, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if rep.DeclaredVerbs != 3 {
		t.Errorf("DeclaredVerbs = %d, want 3", rep.DeclaredVerbs)
	}
	if rep.Debt != 3 {
		t.Errorf("Debt = %d, want 3 (headroom, recall, ctx-overlay)", rep.Debt)
	}
	// Partition invariant: every surface is either a declared verb or debt, so the
	// surface count is exactly DeclaredVerbs+Debt (3+3=6 for this fixture, which holds
	// no declared skills). This is the relation the frozen magic count stood for.
	if len(rep.Surfaces) != rep.DeclaredVerbs+rep.Debt {
		t.Fatalf("len(Surfaces) = %d, want DeclaredVerbs+Debt = %d+%d = %d: %+v",
			len(rep.Surfaces), rep.DeclaredVerbs, rep.Debt, rep.DeclaredVerbs+rep.Debt, rep.Surfaces)
	}

	got := map[string]Surface{}
	for _, s := range rep.Surfaces {
		got[string(s.Kind)+":"+s.Name] = s
	}
	for _, want := range []struct {
		key      string
		declared bool
	}{
		{"verb:session", true},
		{"verb:vcache", true},
		{"verb:guard", true},
		{"verb:headroom", false},
		{"verb:recall", false}, // partial directive (no warms=) is not a real declaration
		{"skill:ctx-overlay", false},
	} {
		s, ok := got[want.key]
		if !ok {
			t.Errorf("missing expected surface %q", want.key)
			continue
		}
		if s.Declared != want.declared {
			t.Errorf("%q declared = %v, want %v", want.key, s.Declared, want.declared)
		}
		// A declared surface carries its plan fields and the directive site.
		if s.Declared && (s.Enters == "" || s.Pages == "" || s.Warms == "") {
			t.Errorf("%q declared but missing a plan field: %+v", want.key, s)
		}
		if s.Declared && (s.File == "" || s.Line == 0) {
			t.Errorf("%q declared but missing file:line provenance: %+v", want.key, s)
		}
	}
	// Over-match guards: neither the non-context verb nor the non-context skill is a surface.
	for _, bad := range []string{"verb:widget", "skill:quality-report"} {
		if _, ok := got[bad]; ok {
			t.Errorf("walker over-matched a non-context surface: %q", bad)
		}
	}
}

// TestScanDeterministic is the "scan twice → identical output" witness: two scans
// of the same tree marshal to byte-identical JSON.
func TestScanDeterministic(t *testing.T) {
	a, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan a: %v", err)
	}
	b, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan b: %v", err)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Errorf("scan not deterministic:\n a=%s\n b=%s", ja, jb)
	}
}

// TestFixtureVerbRaisesDebtByExactlyOne is THE #2202 witness: a synthetic tree with
// N context verbs is scanned, one MORE context verb with no declaration is added,
// and the debt count rises by exactly 1 — and adding a DECLARED verb, or a
// non-context verb, does not move it.
func TestFixtureVerbRaisesDebtByExactlyOne(t *testing.T) {
	dir := t.TempDir()
	base := map[string]bool{ // verb -> declared
		"session":  true,  // context, declared
		"headroom": false, // context, undeclared -> debt
	}
	writeTree(t, dir, base)

	before, err := Scan(dir)
	if err != nil {
		t.Fatalf("Scan before: %v", err)
	}

	// Add one undeclared context verb.
	base["recall"] = false
	writeTree(t, dir, base)
	after, err := Scan(dir)
	if err != nil {
		t.Fatalf("Scan after: %v", err)
	}
	if after.Debt != before.Debt+1 {
		t.Fatalf("adding one undeclared context verb: debt %d -> %d, want +1", before.Debt, after.Debt)
	}

	// Adding a DECLARED context verb must NOT raise debt.
	base["resume"] = true
	writeTree(t, dir, base)
	decl, err := Scan(dir)
	if err != nil {
		t.Fatalf("Scan decl: %v", err)
	}
	if decl.Debt != after.Debt {
		t.Errorf("adding a DECLARED verb changed debt %d -> %d, want unchanged", after.Debt, decl.Debt)
	}

	// Adding a NON-context verb must NOT raise debt.
	base["widget"] = false
	writeTree(t, dir, base)
	non, err := Scan(dir)
	if err != nil {
		t.Fatalf("Scan non-context: %v", err)
	}
	if non.Debt != decl.Debt {
		t.Errorf("adding a non-context verb changed debt %d -> %d, want unchanged", decl.Debt, non.Debt)
	}
}

// TestLiveContextPlanFloorAndDebt is the LIVE, ADVISORY rung under `make ci`
// (#2202 done-condition). It asserts the done-condition FLOOR — at least the 10
// context-touching verbs from the spine's survey carry a real declaration — and
// EMITS the undeclared-surface count. The debt is advisory: it is logged, never
// gated (this rung is never a hard gate).
func TestLiveContextPlanFloorAndDebt(t *testing.T) {
	root := repoRoot(t)
	rep, err := Scan(root)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	const floor = 10
	if rep.DeclaredVerbs < floor {
		t.Errorf("only %d context verbs carry a real context-plan declaration, want >= %d.\n"+
			"Add a `//fak:ctxplan verb=<name> enters=\"…\" pages=\"…\" warms=\"…\"` directive "+
			"to the verb's cmd/fak handler file.", rep.DeclaredVerbs, floor)
		for _, s := range rep.Surfaces {
			if s.Kind == KindVerb && s.Declared {
				t.Logf("  declared: %s (%s:%d)", s.Name, s.File, s.Line)
			}
		}
	}
	// Advisory: emit the undeclared-surface count and name the debt. Never fails.
	t.Logf("context-plan debt (advisory): %d undeclared context surface(s); %d verbs declared",
		rep.Debt, rep.DeclaredVerbs)
	for _, s := range rep.Surfaces {
		if !s.Declared {
			t.Logf("  undeclared %s: %s (%s)", s.Kind, s.Name, s.File)
		}
	}
}

// TestSwitchMentionedInCommentDoesNotDerailWalker is the regression for the live-scan
// zeroing bug: cmd/fak/main.go documents its own dispatch in a `//` comment that quotes
// `switch os.Args[1]` a few lines ABOVE the real switch. The walker used to latch onto
// that zero-depth comment line, see depth<=0 on the next line, and break before reaching
// a single case — so the whole scan reported 0 verbs. The header match must skip the
// comment half and only fire on the real switch statement.
func TestDispatchVerbsScansSplitRouterSwitches(t *testing.T) {
	dir := t.TempDir()
	mainPath := filepath.Join(dir, "main.go")
	src := `package main
func dispatchA(name string) bool {
 switch name {
 case "memory":
  return true
 }
 return false
}
func dispatchB(name string) bool {
 switch name {
 case "recall":
  return true
 case "memory":
  return true
 }
 return false
}
`
	if err := os.WriteFile(mainPath, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := dispatchVerbs(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"memory", "recall"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("dispatch verbs = %v, want %v", got, want)
	}
}

func TestSwitchMentionedInCommentDoesNotDerailWalker(t *testing.T) {
	dir := t.TempDir()
	fakDir := filepath.Join(dir, "cmd", "fak")
	if err := os.MkdirAll(fakDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	main := "package main\n\nimport \"os\"\n\nfunc main() {\n" +
		"\t// The devindex scanner is keyed on this file's `switch os.Args[1]` header;\n" +
		"\t// this comment MENTIONS it a few lines above the real one and must be ignored.\n" +
		"\tif os.Args[1] == \"dev\" {\n\t\treturn\n\t}\n" +
		"\tswitch os.Args[1] {\n" +
		"\tcase \"session\":\n\t\tcmdSession(os.Args[2:])\n" +
		"\tcase \"recall\":\n\t\tcmdRecall(os.Args[2:])\n" +
		"\t}\n}\n"
	if err := os.WriteFile(filepath.Join(fakDir, "main.go"), []byte(main), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	// A declared handler for one of the verbs, so the scan yields a real declaration
	// only reachable if the walker got past the comment to the real switch body.
	handler := "package main\n\n//fak:ctxplan verb=session enters=\"x\" pages=\"y\" warms=\"z\"\nfunc cmdSession(argv []string) {}\n"
	if err := os.WriteFile(filepath.Join(fakDir, "session.go"), []byte(handler), 0o644); err != nil {
		t.Fatalf("write session.go: %v", err)
	}

	rep, err := Scan(dir)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	// Both context verbs must be enumerated (session declared, recall debt) — proof the
	// walker reached the real switch body rather than breaking at the comment.
	if rep.DeclaredVerbs != 1 {
		t.Errorf("DeclaredVerbs = %d, want 1 (session); the comment mention broke the walker", rep.DeclaredVerbs)
	}
	got := map[string]bool{}
	for _, s := range rep.Surfaces {
		if s.Kind == KindVerb {
			got[s.Name] = s.Declared
		}
	}
	if decl, ok := got["session"]; !ok || !decl {
		t.Errorf("verb:session missing or undeclared: present=%v declared=%v", ok, decl)
	}
	if decl, ok := got["recall"]; !ok || decl {
		t.Errorf("verb:recall missing or wrongly declared: present=%v declared=%v", ok, decl)
	}
}

// writeTree (re)writes a minimal cmd/fak tree: a main.go dispatch switch over the
// given verbs, plus one handler file per verb carrying a complete //fak:ctxplan
// directive when declared. Rewriting is idempotent so a test can grow the tree and
// re-scan.
func writeTree(t *testing.T, root string, verbs map[string]bool) {
	t.Helper()
	dir := filepath.Join(root, "cmd", "fak")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	var b strings.Builder
	b.WriteString("package main\n\nimport \"os\"\n\nfunc main() {\n\tswitch os.Args[1] {\n")
	for name := range verbs {
		fmt.Fprintf(&b, "\tcase %q:\n\t\tcmd_%s(os.Args[2:])\n", name, name)
	}
	b.WriteString("\t}\n}\n")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	for name, declared := range verbs {
		var f strings.Builder
		f.WriteString("package main\n\n")
		if declared {
			fmt.Fprintf(&f, "//fak:ctxplan verb=%s enters=\"x\" pages=\"y\" warms=\"z\"\n", name)
		}
		fmt.Fprintf(&f, "func cmd_%s(argv []string) {}\n", name)
		if err := os.WriteFile(filepath.Join(dir, name+".go"), []byte(f.String()), 0o644); err != nil {
			t.Fatalf("write %s.go: %v", name, err)
		}
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
