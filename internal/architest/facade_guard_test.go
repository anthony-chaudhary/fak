package architest

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// facadeOffList names blank imports in the shipped registration surfaces whose package
// declares NO init() and therefore performs no self-registration when the fak binary
// links it. Such an import runs nothing; it only makes reachability readers count the
// package as wired.
//
// The cmd/fak/registered_leaves.go entries are a burn-down list, not a reviewed keep.
// Their "fak version modules lanes" reason never held: internal/modver.Snapshot scans the
// git tree, so a runtime proof of the form `fak version modules` -> "internal/<pkg>"
// passes whether or not the package is linked. Each entry leaves this list when its
// package gains a production caller (wire), loses the blank import because its consumer
// is a CI test (unlink), or is deleted (reap); none may be added.
//
// Derivation contract (issue #12873): this list must stay in lockstep with the tree.
// A leaf that grows a real init() registration graduates off this list automatically by
// the test below re-deriving self-registration from the AST; a NEW leaf that blank-imports
// without init() fails the build until it is listed here for a stated reason.
var facadeOffList = map[string]string{
	"advmodel":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"breathgate":            "reachability-roster entry for fak version modules lanes; no self-registration",
	"cavemansafety":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"fakrpc":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"guardcorpus":           "reachability-roster entry for fak version modules lanes; no self-registration",
	"knownenv":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"lightgapport":          "reachability-roster entry for fak version modules lanes; no self-registration",
	"localappux":            "reachability-roster entry for fak version modules lanes; no self-registration",
	"lookahead":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"market":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"nativeperfobscontract": "reachability-roster entry for fak version modules lanes; no self-registration",
	"opensweharder":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"pkg/harnesssidecar":    "public SDK intentionally linked on the reachability roster; consumers import it explicitly and it has no init side effects (fak-private#2609)",
	"pkg/managedharness":    "public SDK intentionally linked on the reachability roster; consumers import it explicitly and it has no init side effects (fak-private#2609)",
	"portabilityswitch":     "reachability-roster entry for fak version modules lanes; no self-registration",
	"projectionspine":       "reachability-roster entry for fak version modules lanes; no self-registration",
	"reflexagent":           "reachability-roster entry for fak version modules lanes; no self-registration",

	// fak-private#2609: internal/registrations is also a shipped registration surface.
	// These imports are deliberate reachability-roster links and have no init side effects.
	"codetools":        "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"guardcompile":     "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"kvquantmeta":      "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"operatorquestion": "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"operatorresolve":  "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"planresolve":      "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"quantdetect":      "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"quantmeta":        "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"quantprov":        "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"tokenprofile":     "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"toolgrammar":      "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
	"worktreewitness":  "intentional registration-surface reachability entry; no init side effects (fak-private#2609)",
}

// reapedFacadeLeaves pins the leaves #12873 deleted as inert blank-import facades.
// Re-adding any of them re-opens the island: a blank import that fires no init() links
// dead code into every fak build while fooling roster tooling into counting the package
// as reached.
var reapedFacadeLeaves = []string{"macromailbox", "macrostate"}

// facadeSurfacePackages are registration aggregators scanned as surfaces themselves, so a
// blank import of one is checked through its own imports rather than its (absent) init().
var facadeSurfacePackages = map[string]bool{"internal/registrations": true}

// facadeRegistrationFiles returns every non-test .go file directly under cmd/fak, so a
// facade cannot regrow in a new file, plus the registrations aggregator.
func facadeRegistrationFiles(t *testing.T, internal string) []string {
	t.Helper()
	cmdFiles, err := filepath.Glob(filepath.Join(internal, "..", "cmd", "fak", "*.go"))
	if err != nil {
		t.Fatalf("glob cmd/fak: %v", err)
	}
	var out []string
	for _, f := range cmdFiles {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		out = append(out, filepath.Join("cmd", "fak", filepath.Base(f)))
	}
	return append(out, filepath.Join("internal", "registrations", "registrations.go"))
}

type facadeImport struct {
	source      string
	importPath  string
	packagePath string
	leaf        string
}

func (f facadeImport) offListKey() string {
	if strings.HasPrefix(f.packagePath, "pkg/") {
		return f.packagePath
	}
	return f.leaf
}

// blankImportFacades parses every shipped registration surface and returns each blank
// import under this module in file order. internal/* keeps the historical direct-child
// contract; pkg/* is accepted as a first-class public SDK shape. Reading the AST means a
// commented-out or renamed import is never counted.
func blankImportFacades(t *testing.T, internal string) []facadeImport {
	t.Helper()
	var out []facadeImport
	for _, registrationFile := range facadeRegistrationFiles(t, internal) {
		out = append(out, blankImportFacadesFromFile(t, internal, registrationFile)...)
	}
	return out
}

func blankImportFacadesFromFile(t *testing.T, internal, registrationFile string) []facadeImport {
	t.Helper()
	repoPrefix := strings.TrimSuffix(modPrefix, "/internal/") + "/"
	path := filepath.Join(internal, "..", registrationFile)
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	var out []facadeImport
	for _, spec := range f.Imports {
		if spec.Name == nil || spec.Name.Name != "_" {
			continue
		}
		importPath := strings.Trim(spec.Path.Value, `"`)
		if !strings.HasPrefix(importPath, repoPrefix) {
			continue
		}
		packagePath := strings.TrimPrefix(importPath, repoPrefix)
		var leaf string
		switch {
		case strings.HasPrefix(packagePath, "internal/"):
			leaf = strings.TrimPrefix(packagePath, "internal/")
			if leaf == "" || strings.Contains(leaf, "/") {
				t.Errorf("blank import %q in %s does not name an internal package directly under internal/", importPath, filepath.ToSlash(registrationFile))
				continue
			}
		case strings.HasPrefix(packagePath, "pkg/"):
			parts := strings.Split(strings.TrimPrefix(packagePath, "pkg/"), "/")
			if len(parts) == 0 || parts[len(parts)-1] == "" {
				t.Errorf("blank import %q in %s does not name a package under pkg/", importPath, filepath.ToSlash(registrationFile))
				continue
			}
			leaf = parts[len(parts)-1]
		default:
			t.Errorf("blank import %q in %s does not name an internal/* or pkg/* package", importPath, filepath.ToSlash(registrationFile))
			continue
		}
		out = append(out, facadeImport{
			source:      filepath.ToSlash(registrationFile),
			importPath:  importPath,
			packagePath: packagePath,
			leaf:        leaf,
		})
	}
	return out
}

// blankImportLeaves retains the leaf-name view used by the reaped-package regression
// while delegating parsing to the widened registration-surface scanner.
func blankImportLeaves(t *testing.T, internal string) []string {
	t.Helper()
	imports := blankImportFacades(t, internal)
	leaves := make([]string, 0, len(imports))
	for _, imp := range imports {
		leaves = append(leaves, imp.leaf)
	}
	return leaves
}

// declaresInitFunc reports whether the imported package's non-test .go files declare a
// func init() (via AST walk, so a doc comment mentioning init is correctly not counted).
func declaresInitFunc(t *testing.T, internal string, imp facadeImport) bool {
	t.Helper()
	dir := filepath.Join(internal, "..", filepath.FromSlash(imp.packagePath))
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, dir,
		func(fi os.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	for _, p := range parsed {
		for _, f := range p.Files {
			found := false
			ast.Inspect(f, func(n ast.Node) bool {
				if fd, ok := n.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "init" {
					found = true
				}
				return true
			})
			if found {
				return true
			}
		}
	}
	return false
}

func potentialFacadeMessage(imp facadeImport) string {
	return "blank import _ \"" + imp.importPath + "\" in " + imp.source +
		" is a potential facade: " + imp.packagePath + " declares no func init(), so the import links the package in without running anything from it " +
		"(an island that roster/reachability tooling counts as reached). Per the wire-or-delete-or-allow-list doctrine " +
		"(internal/unwiredscore/dispatch.go), add real init registration, wire a production caller, delete the package, " +
		"or add " + imp.offListKey() + " to facadeOffList with a stated reason."
}

// TestRegisteredLeavesFacadeImports closes the facade hole across both shipped
// registration surfaces. A blank import of a package that declares no init() links
// NOTHING — the package compiles into the binary and roster/reachability tooling counts
// it as reached, but no code runs. That is an island masquerading as a connection.
// #12873 reaped macromailbox and macrostate for exactly this; fak-private#2609 extends
// the same ratchet to internal/registrations and first-class pkg/* SDK imports:
//
//  1. every blank-imported leaf either self-registers (declares func init()) — OK, or
//  2. is consciously listed in facadeOffList with a reason — OK, or the contributor fixes
//     the linkage (see the failure message below), and
//  3. the reaped macromailbox/macrostate facades stay gone: not blank-imported, not on
//     the off-list, directories deleted.
//
// The self-registration set is re-derived from the AST on every run, so a leaf that grows
// a real init() graduates off facadeOffList automatically, and a new unlinked blank import
// fails the build instead of silently islanding.
func TestRegisteredLeavesFacadeImports(t *testing.T) {
	internal := internalDir(t)

	imports := blankImportFacades(t, internal)

	// Regression pin on the #12873 reap: the deleted facade imports must not return,
	// must not be allow-listed, and their directories must stay deleted. If you are
	// re-adding one of these, you are re-opening the island #12873 closed — add a real
	// init() registration or a production caller instead, or file the reap issue for
	// an honest re-review with evidence the package actually runs.
	for _, reaped := range reapedFacadeLeaves {
		for _, imp := range imports {
			if imp.leaf != reaped {
				continue
			}
			t.Errorf("facade regression: %q is blank-imported in %s again. "+
				"It was reaped by issue #12873 as an inert blank-import facade (no init(), no production caller); "+
				"re-adding it re-opens the island. Give internal/%s a real init() registration, wire a production "+
				"caller, or delete it again.", reaped, imp.source, reaped)
		}
		if reason, ok := facadeOffList[reaped]; ok {
			t.Errorf("facade regression: %q is listed in facadeOffList (reason %q) but was reaped by issue "+
				"#12873 as an inert blank-import facade. Remove it from facadeOffList and keep the package deleted; "+
				"an allow-list entry must name a package that exists and is deliberately linked.", reaped, reason)
		}
		dir := filepath.Join(internal, reaped)
		if _, err := os.Stat(dir); err == nil {
			t.Errorf("facade regression: directory %s exists again. It was deleted by issue #12873 after "+
				"macromailbox/macrostate proved to be inert blank-import facades. Re-adding the directory without "+
				"a real init() registration re-opens the island; delete it or wire it for real.", dir)
		} else if !os.IsNotExist(err) {
			t.Fatalf("stat %s: %v", dir, err)
		}
	}

	requiredPublicFacades := map[string]bool{
		"pkg/harnesssidecar": false,
		"pkg/managedharness": false,
	}
	var selfRegistered, offListed int
	observed := map[string]bool{}
	for _, imp := range imports {
		observed[imp.offListKey()] = true
		if _, required := requiredPublicFacades[imp.offListKey()]; required {
			requiredPublicFacades[imp.offListKey()] = true
		}
		if facadeSurfacePackages[imp.packagePath] || declaresInitFunc(t, internal, imp) {
			selfRegistered++
			continue
		}
		if _, ok := facadeOffList[imp.offListKey()]; ok {
			offListed++
			continue
		}
		t.Error(potentialFacadeMessage(imp))
	}
	for key := range facadeOffList {
		if !observed[key] {
			t.Errorf("stale facadeOffList entry %q: no registration surface blank-imports it any more; delete the entry", key)
		}
	}
	for packagePath, seen := range requiredPublicFacades {
		if !seen {
			t.Errorf("registration-surface regression: expected blank import %q was not observed; "+
				"the widened facade guard no longer covers internal/registrations/registrations.go", packagePath)
		}
	}
	t.Logf("facade ratchet: %d blank imports checked, %d self-register, %d on facadeOffList",
		selfRegistered+offListed, selfRegistered, offListed)
}

func TestBlankImportLeavesCoversRegistrationPackages(t *testing.T) {
	got := blankImportLeaves(t, internalDir(t))
	for _, want := range []string{"harnesssidecar", "managedharness"} {
		found := false
		for _, leaf := range got {
			if leaf == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("blankImportLeaves() omitted registration package %q", want)
		}
	}
}

func TestPotentialFacadeDiagnosticAcceptsUnlistedPkgImport(t *testing.T) {
	root := t.TempDir()
	internal := filepath.Join(root, "internal")
	registrationFile := filepath.Join("internal", "registrations", "registrations.go")
	path := filepath.Join(root, registrationFile)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	contents := "package registrations\n\nimport _ \"github.com/anthony-chaudhary/fak/pkg/unlistedfixture\"\n"
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}

	imports := blankImportFacadesFromFile(t, internal, registrationFile)
	if len(imports) != 1 {
		t.Fatalf("blankImportFacadesFromFile() returned %d imports, want 1", len(imports))
	}
	message := potentialFacadeMessage(imports[0])
	if !strings.Contains(message, "pkg/unlistedfixture") || !strings.Contains(message, "is a potential facade") {
		t.Fatalf("potentialFacadeMessage() = %q, want pkg fixture facade diagnostic", message)
	}
	if strings.Contains(message, "does not name an internal package") {
		t.Fatalf("potentialFacadeMessage() = %q, pkg/* must be accepted as a leaf shape", message)
	}
}
