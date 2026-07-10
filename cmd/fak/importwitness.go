package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"path"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// cmd/fak/importwitness.go — `fak hooks import-witness`: the CHEAP structural half of the
// tree-level build witness (#1325 acceptance criterion 2).
//
// The recurring way trunk goes — and stays — red on the shared always-on tree is a caller
// committed without its callee: a tracked `.go` imports an internal package whose files were
// never `git add`ed, so `git ls-tree HEAD internal/<pkg>/` is empty. A clean clone cannot build
// the importer at all ("no required module provides package …"), yet the author's disk built
// fine, so nothing looked wrong locally. The compile-integrity gate (`fak hooks pre-push`,
// #1338) catches this only by running the full/cone `go build` — heavy, and its raw compiler
// error is opaque about the fix.
//
// This witness catches the SAME failure structurally and cheaply, with a precise fix. It reads
// only the COMMITTED tree (a git rev, default HEAD) — never the working tree — so it is immune
// to peers' in-flight uncommitted churn: it archives nothing and compiles nothing. It:
//
//  1. `git ls-tree -r <rev>` → the set of package directories that have ≥1 tracked, non-test
//     `.go` file (the packages `go build` can actually compile);
//  2. one `git grep <rev>` for every module-local import line → the committed importer edges;
//  3. flags each edge whose target package directory is NOT in set (1) — a committed importer of
//     a package nobody committed — and names the exact `git add` that greens it.
//
// The parse + detection are pure functions (no git, no toolchain) so the whole verdict is
// unit-testable without a build. Exit codes mirror the hooks contract: 0 = clean, 1 =
// IMPORT_OF_UNCOMMITTED_PACKAGE (block-worthy), 2 = could-not-run (fail-open).

// modulePkgPrefix is this module's path with a trailing slash. An import beginning with it is
// module-local, and the remainder is the package's repo-relative directory (import paths are
// always slash-separated, matching the ls-tree dirs).
const modulePkgPrefix = "github.com/anthony-chaudhary/fak/"

// importEdge is one committed importer file importing one module-local package.
type importEdge struct {
	Importer   string // repo-relative .go path (slash-separated), committed
	ImportPath string // full module-local import path
	PkgDir     string // repo-relative package dir (ImportPath minus the module prefix)
}

// uncommittedImport is a violation: a committed importer references a package dir with zero
// tracked non-test .go files at the witnessed rev — a forgotten `git add` of the whole package.
type uncommittedImport struct {
	Importer   string `json:"importer"`
	ImportPath string `json:"import_path"`
	PkgDir     string `json:"package_dir"`
}

// importSpecRE matches a gofmt import-spec line: an optional (`_` blank / named) qualifier, then
// a quoted path, then only an optional line comment. It deliberately rejects any line that is not
// purely an import spec (assignments, calls, struct tags that merely CONTAIN a package path), so
// the witness never fires on a string literal that happens to hold an import path.
var importSpecRE = regexp.MustCompile(`^\s*(?:_\s+|[A-Za-z_][A-Za-z0-9_]*\s+)?"([^"]+)"\s*(?://.*)?$`)

// packageDirsWithTrackedSource returns the set of repo-relative directories that contain at least
// one tracked, non-test `.go` file. A package importable by `go build` must have one — a dir with
// only `_test.go` (or nothing) cannot satisfy a build-time import. Pure.
func packageDirsWithTrackedSource(trackedPaths []string) map[string]bool {
	dirs := map[string]bool{}
	for _, p := range trackedPaths {
		p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			continue
		}
		dirs[path.Dir(p)] = true
	}
	return dirs
}

// parseImportEdges turns `git grep -n <rev> -- '*.go'` output into module-local import edges.
// Each line is `<rev>:<file>:<lineno>:<content>`. Only genuine import-spec lines (importSpecRE)
// importing the module prefix are kept; `_test.go` importers are excluded because they never red
// `go build ./...`. De-duplicates by (file, import). Pure — takes the grep bytes, does no I/O.
func parseImportEdges(grepOut string) []importEdge {
	var edges []importEdge
	seen := map[string]bool{}
	sc := bufio.NewScanner(strings.NewReader(grepOut))
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		// SplitN on ":" with limit 4 — the committed path never contains ":" (git repo-relative,
		// forward-slash, no drive letter), and the content (last field) keeps any ":" it holds.
		parts := strings.SplitN(sc.Text(), ":", 4)
		if len(parts) < 4 {
			continue
		}
		file := strings.ReplaceAll(parts[1], "\\", "/")
		if !strings.HasSuffix(file, ".go") || strings.HasSuffix(file, "_test.go") {
			continue
		}
		m := importSpecRE.FindStringSubmatch(parts[3])
		if m == nil {
			continue
		}
		imp := m[1]
		if !strings.HasPrefix(imp, modulePkgPrefix) {
			continue
		}
		key := file + "\x00" + imp
		if seen[key] {
			continue
		}
		seen[key] = true
		edges = append(edges, importEdge{
			Importer:   file,
			ImportPath: imp,
			PkgDir:     strings.TrimPrefix(imp, modulePkgPrefix),
		})
	}
	return edges
}

// detectUncommittedImports returns, sorted by (package dir, importer), every import edge whose
// target package dir has no tracked non-test source — the committed caller of a package nobody
// committed. Pure over its two inputs, so the whole decision unit-tests without git or a compiler.
func detectUncommittedImports(edges []importEdge, pkgDirsWithSource map[string]bool) []uncommittedImport {
	var out []uncommittedImport
	for _, e := range edges {
		if pkgDirsWithSource[e.PkgDir] {
			continue
		}
		out = append(out, uncommittedImport{Importer: e.Importer, ImportPath: e.ImportPath, PkgDir: e.PkgDir})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].PkgDir != out[j].PkgDir {
			return out[i].PkgDir < out[j].PkgDir
		}
		return out[i].Importer < out[j].Importer
	})
	return out
}

// --------------------------------------------------------------------------- I/O

// gitCapture runs git in root and returns stdout plus the raw error. Callers decide whether a
// non-zero exit is fatal (git grep exits 1 on "no matches", which is not an error here).
func gitCapture(root string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	return string(out), err
}

func runHooksImportWitness(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks import-witness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	rev := fs.String("rev", "HEAD", "the committed tree to witness (any git rev)")
	asJSON := fs.Bool("json", false, "emit the verdict as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		return 2 // not in a repo => could-not-run => shell wrapper falls open
	}

	tracked, err := gitCapture(r, "ls-tree", "-r", "--name-only", *rev)
	if err != nil {
		fmt.Fprintf(stderr, "fak hooks import-witness: git ls-tree %s: %v\n", *rev, err)
		return 2
	}
	pkgDirs := packageDirsWithTrackedSource(strings.Split(tracked, "\n"))

	// One fixed-string grep over the committed tree for every module-local import line. `git grep
	// <rev>` searches the rev's tree (committed blobs), not the working tree — the churn-immune view.
	grepOut, gerr := gitCapture(r, "grep", "-n", "-I", "-F", "-e", modulePkgPrefix, *rev, "--", "*.go")
	if gerr != nil {
		// git grep exits 1 with no output when nothing matches — expected, not a failure.
		if ee, ok := gerr.(*exec.ExitError); !ok || ee.ExitCode() != 1 {
			fmt.Fprintf(stderr, "fak hooks import-witness: git grep %s: %v\n", *rev, gerr)
			return 2
		}
	}
	violations := detectUncommittedImports(parseImportEdges(grepOut), pkgDirs)

	if *asJSON {
		if encErr := writeIndentedJSON(stdout, map[string]any{
			"rev":        *rev,
			"verdict":    importWitnessVerdict(violations),
			"violations": violationsForJSON(violations),
			"count":      len(violations),
		}); encErr != nil {
			fmt.Fprintf(stderr, "fak hooks import-witness: %v\n", encErr)
			return 2
		}
	} else if len(violations) == 0 {
		fmt.Fprintln(stdout, "import-witness: OK — every module-local import resolves to a package with tracked source.")
	} else {
		fmt.Fprintf(stdout, "import-witness: IMPORT_OF_UNCOMMITTED_PACKAGE — %d committed import(s) reference a package with NO tracked source (forgotten `git add`):\n", len(violations))
		for _, v := range violations {
			fmt.Fprintf(stdout, "  - %s imports %s — dir %s/ has zero tracked non-test .go files; `git add %s/` (or revert the importer until the package lands)\n",
				v.Importer, v.ImportPath, v.PkgDir, v.PkgDir)
		}
	}
	if len(violations) > 0 {
		return 1
	}
	return 0
}

func importWitnessVerdict(violations []uncommittedImport) string {
	if len(violations) > 0 {
		return "IMPORT_OF_UNCOMMITTED_PACKAGE"
	}
	return "OK"
}

// violationsForJSON returns a non-nil slice so the JSON payload always has an array.
func violationsForJSON(violations []uncommittedImport) []uncommittedImport {
	if violations == nil {
		return []uncommittedImport{}
	}
	return violations
}
