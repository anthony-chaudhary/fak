package architest

import (
	"fmt"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// #6071: one file in internal/compute declared `package cudaacceptance_test`. Go permits
// exactly two package clauses in a directory — the package itself (P) and its external
// test package (P_test) — so the toolchain stopped being able to LOAD internal/compute at
// all and reported "found packages compute (anchorwarm.go) and cudaacceptance
// (cuda_acceptance_scripts_test.go)". That is not a test-only break: an unloadable
// directory takes out the ordinary `go build` of every importer (internal/cachemeta,
// internal/gateway, ...), so a single mistyped clause dark-fails the whole tree and every
// concurrent worker's ship gate behind it.
//
// The compiler never catches this class early enough to be useful: the file that carries
// the bad clause compiles fine in isolation, and the error surfaces at whatever unrelated
// package happens to import the directory — which is why #6071 read as "internal/cachemeta
// is broken" rather than "internal/compute has a typo". This gate re-derives the rule from
// the tree so the class is caught at its source, by name, with the fix in the message.

// clauseViolation is one file whose package clause the directory cannot legally hold.
type clauseViolation struct {
	Dir  string // module-relative directory, slash-separated
	File string // base name
	Got  string // the package clause the file declares
	Want string // the package the directory declares
}

func (v clauseViolation) String() string {
	return fmt.Sprintf("%s/%s declares package %s; the directory is package %s, so the only legal clauses are %s and %s_test",
		v.Dir, v.File, v.Got, v.Want, v.Want, v.Want)
}

// checkDirClauses applies Go's one-package-per-directory rule to a single directory.
// clauses maps a base file name to the package clause that file declares.
//
// The directory's package P is taken from its non-test files (they must all agree). A
// test-only directory has no non-test files, so P is recovered from the test clauses:
// the un-suffixed clause if one is present, else the shared base of the _test-suffixed
// ones. Every *_test.go file may then declare P or P+"_test" and nothing else.
func checkDirClauses(dir string, clauses map[string]string) []clauseViolation {
	names := make([]string, 0, len(clauses))
	for name := range clauses {
		names = append(names, name)
	}
	sort.Strings(names)

	// The package the directory declares, per its non-test files.
	want := ""
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		if want == "" {
			want = clauses[name]
		}
	}
	if want == "" {
		want = testOnlyDirPackage(names, clauses)
	}
	if want == "" {
		return nil
	}

	var out []clauseViolation
	for _, name := range names {
		got := clauses[name]
		if got == want {
			continue
		}
		if strings.HasSuffix(name, "_test.go") && got == want+"_test" {
			continue
		}
		out = append(out, clauseViolation{Dir: dir, File: name, Got: got, Want: want})
	}
	return out
}

// testOnlyDirPackage recovers the nominal package of a directory that holds only _test.go
// files (internal/architest is one). Such a directory still has a single legal identity;
// without recovering it, a stray clause in a test-only directory would go unchecked.
func testOnlyDirPackage(names []string, clauses map[string]string) string {
	counts := map[string]int{}
	for _, name := range names {
		got := clauses[name]
		if got == "" {
			continue
		}
		counts[strings.TrimSuffix(got, "_test")]++
	}
	best, bestN := "", 0
	bases := make([]string, 0, len(counts))
	for base := range counts {
		bases = append(bases, base)
	}
	sort.Strings(bases)
	for _, base := range bases {
		if counts[base] > bestN {
			best, bestN = base, counts[base]
		}
	}
	return best
}

// scanPackageClauses walks root and returns every file whose package clause its directory
// cannot legally hold. Directories the go tool itself ignores (a leading "_" or ".", and
// "testdata") are skipped, because their contents are not part of any package.
func scanPackageClauses(root string) ([]clauseViolation, error) {
	var out []clauseViolation
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			return nil
		}
		name := d.Name()
		if path != root && (name == "testdata" || strings.HasPrefix(name, "_") || strings.HasPrefix(name, ".")) {
			return filepath.SkipDir
		}
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}
		clauses := map[string]string{}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
				continue
			}
			// The go tool ignores files whose name starts with "_" or "." exactly as it
			// ignores such directories, so they belong to no package and carry no rule.
			if strings.HasPrefix(e.Name(), "_") || strings.HasPrefix(e.Name(), ".") {
				continue
			}
			file := filepath.Join(path, e.Name())
			pkg, err := packageClause(file)
			if err != nil {
				// An unparseable file is a different failure with its own signal; this
				// gate is only about the clause, so it stays out of the way.
				continue
			}
			if buildExcluded(file) {
				continue
			}
			clauses[e.Name()] = pkg
		}
		if len(clauses) == 0 {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		out = append(out, checkDirClauses(filepath.ToSlash(rel), clauses)...)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// buildExcluded reports whether a file's //go:build constraint keeps it out of every
// ordinary build — the `//go:build ignore` standalone-script convention being the case that
// matters here. Such a file is not part of the directory's package, so Go does not hold it
// to the directory's clause and neither does this gate. Tags other than "ignore" are treated
// as satisfiable, so a platform-specific file stays in scope.
func buildExcluded(path string) bool {
	f, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.PackageClauseOnly|parser.ParseComments)
	if err != nil {
		return false
	}
	for _, group := range f.Comments {
		if group.Pos() > f.Package {
			break
		}
		for _, c := range group.List {
			if !constraint.IsGoBuild(c.Text) {
				continue
			}
			expr, err := constraint.Parse(c.Text)
			if err != nil {
				continue
			}
			if !expr.Eval(func(tag string) bool { return tag != "ignore" }) {
				return true
			}
		}
	}
	return false
}

// packageClause reads just the package clause of a Go file — no bodies, no imports — so
// scanning the whole tree stays cheap.
func packageClause(path string) (string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.PackageClauseOnly)
	if err != nil {
		return "", err
	}
	return f.Name.Name, nil
}

// TestPackageClausesAreLoadable is the #6071 recurrence gate over the real tree: every
// directory in the module holds at most its own package clause and that package's external
// test clause. It is the whole-tree half of the floor — the synthetic half
// (TestCheckDirClausesFlagsForeignTestPackage) proves the check bites; this one proves the
// tree is currently clean, and turns RED the moment a file reintroduces the class.
//
// Seeded GREEN. When it fails, the message names the file and the two clauses that file is
// allowed to use, which is the whole fix.
func TestPackageClausesAreLoadable(t *testing.T) {
	root := repoRoot(t)
	bad, err := scanPackageClauses(root)
	if err != nil {
		t.Fatalf("scan package clauses under %s: %v", root, err)
	}
	for _, v := range bad {
		t.Errorf("unloadable directory (#6071 class): %s\n"+
			"\tGo permits exactly two package clauses per directory, so this makes the WHOLE "+
			"directory unloadable — `go build` of every importer fails, not just its tests.\n"+
			"\tFix: rename the clause to %s or %s_test, or move the file into its own directory "+
			"if the separate package name is deliberate.", v, v.Want, v.Want)
	}
}

// TestCheckDirClausesFlagsForeignTestPackage replays #6071's exact shape on disk: a
// directory of `package compute` files plus a legitimate `package compute_test` external
// test, into which one file lands declaring `package cudaacceptance_test`. The scanner must
// flag that one file and nothing else.
//
// This is the half of the gate that proves the check BITES. Without it,
// TestPackageClausesAreLoadable could be silently vacuous — a scanner that returned nil
// unconditionally would pass over a clean tree forever and never notice the break it exists
// to catch.
func TestCheckDirClausesFlagsForeignTestPackage(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"anchorwarm.go":                   "package compute",
		"cuda.go":                         "package compute",
		"compute_ext_test.go":             "package compute_test",
		"anchorwarm_test.go":              "package compute",
		"cuda_acceptance_scripts_test.go": "package cudaacceptance_test",
	}
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src+"\n\nfunc init() {}\n"), 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	bad, err := scanPackageClauses(dir)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(bad) != 1 {
		t.Fatalf("scanPackageClauses flagged %d files, want exactly 1 (the cudaacceptance outlier): %v", len(bad), bad)
	}
	got := bad[0]
	if got.File != "cuda_acceptance_scripts_test.go" {
		t.Errorf("flagged %q, want cuda_acceptance_scripts_test.go", got.File)
	}
	if got.Got != "cudaacceptance_test" || got.Want != "compute" {
		t.Errorf("violation = got %q want %q; expected got cudaacceptance_test / want compute", got.Got, got.Want)
	}
	if !strings.Contains(got.String(), "compute_test") {
		t.Errorf("violation message %q does not name the legal external-test clause compute_test", got)
	}
}

// TestCheckDirClausesAcceptsLegalShapes pins the other side of the rule so the gate cannot
// be "fixed" into rejecting legitimate layouts: the in-package tests, the external test
// package, a main package, and a directory that holds only _test.go files (architest's own
// shape) are all legal.
func TestCheckDirClausesAcceptsLegalShapes(t *testing.T) {
	cases := []struct {
		name    string
		clauses map[string]string
	}{
		{"package plus in-package and external tests", map[string]string{
			"compute.go":      "compute",
			"compute_test.go": "compute",
			"ext_test.go":     "compute_test",
		}},
		{"main package", map[string]string{
			"main.go":      "main",
			"main_test.go": "main",
		}},
		{"test-only directory", map[string]string{
			"doc.go":            "architest",
			"architest_test.go": "architest",
		}},
		{"test-only directory with no non-test file", map[string]string{
			"a_test.go": "boundarylint_test",
			"b_test.go": "boundarylint_test",
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if bad := checkDirClauses("d", tc.clauses); len(bad) != 0 {
				t.Errorf("checkDirClauses flagged a legal layout: %v", bad)
			}
		})
	}
}
