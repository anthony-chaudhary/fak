package main

// `fak affected` -- the fast inner-loop test gate. It runs `go test` for ONLY the
// packages your working-tree change can affect -- the changed packages plus every
// package that (transitively, test imports included) imports one -- instead of the full
// `go test ./...`. For a fleet editing one leaf at a time, that turns the multi-minute
// green gate into a seconds-long pre-commit check you can afford to run on every change,
// without dropping coverage on what you touched.
//
//	fak affected                 run go test on the affected packages (working tree vs HEAD)
//	fak affected --list          print the affected import paths and exit (no tests run)
//	fak affected --json          print the selection plan as JSON and exit
//	fak affected --base origin/main   select everything changed since a base ref
//	fak affected --short -- -run TestX   pass-through flags to go test (after --)
//
// It is the impure shell over internal/affectedtests: it gathers the changed files
// (`git diff`), the import graph (`go list -json ./...`), folds them through the pure
// affectedtests.Select, and execs `go test`. `make ci` still runs the full `go test
// ./...` as the authoritative oracle -- this is the fast inner loop, not a replacement
// (see the package doc's stated limit).

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
)

func cmdAffected(argv []string) { os.Exit(runAffected(os.Stdout, os.Stderr, argv)) }

// affectedPlan is the JSON view (-json): the deterministic selection the shell derived
// before running any test, so a peer can reproduce it from the same tree.
type affectedPlan struct {
	Base             string   `json:"base"`
	ChangedFiles     []string `json:"changed_files"`
	ChangedPackages  []string `json:"changed_packages"`
	SelectedPackages []string `json:"selected_packages"`
	TotalPackages    int      `json:"total_packages"`
}

func runAffected(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak affected", flag.ContinueOnError)
	fs.SetOutput(stderr)
	base := fs.String("base", "", "git ref to diff against (default: working tree vs HEAD). e.g. --base origin/main selects everything changed since the base")
	list := fs.Bool("list", false, "print the affected package import paths and exit without running tests")
	asJSON := fs.Bool("json", false, "print the selection plan as JSON and exit without running tests")
	short := fs.Bool("short", false, "pass -short to go test")
	verbose := fs.Bool("v", false, "pass -v to go test")
	run := fs.String("run", "", "pass -run <regexp> to go test")
	count := fs.Int("count", 0, "pass -count <n> to go test (0 = omit; -count=1 bypasses the test cache)")
	timeout := fs.String("timeout", "", "pass -timeout <dur> to go test (e.g. 120s)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	passthrough := fs.Args() // anything after -- (flag stops at the first non-flag/--)

	root := repoRoot()

	changedFiles, err := gitChangedFiles(root, *base)
	if err != nil {
		fmt.Fprintf(stderr, "fak affected: discovering changed files: %v\n", err)
		return 1
	}

	fileToPkg, edges, total, err := goListGraph(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak affected: %v\n", err)
		return 1
	}

	changedPkgs := affectedtests.ChangedPackages(fileToPkg, changedFiles)
	selected := affectedtests.Select(edges, changedPkgs)

	if *asJSON {
		plan := affectedPlan{
			Base:             *base,
			ChangedFiles:     changedFiles,
			ChangedPackages:  changedPkgs,
			SelectedPackages: selected,
			TotalPackages:    total,
		}
		_ = writeIndentedJSONNoEscape(stdout, plan)
		return 0
	}

	if *list {
		for _, p := range selected {
			fmt.Fprintln(stdout, p)
		}
		return 0
	}

	if len(selected) == 0 {
		fmt.Fprintf(stderr, "fak affected: no Go packages affected by the change (%d changed file(s)) -- nothing to test\n", len(changedFiles))
		return 0
	}

	fmt.Fprintf(stderr, "fak affected: testing %d/%d package(s) affected by %d changed file(s)%s\n",
		len(selected), total, len(changedFiles), baseNote(*base))

	args := []string{"test"}
	if *short {
		args = append(args, "-short")
	}
	if *verbose {
		args = append(args, "-v")
	}
	if *run != "" {
		args = append(args, "-run", *run)
	}
	if *count != 0 {
		args = append(args, fmt.Sprintf("-count=%d", *count))
	}
	if *timeout != "" {
		args = append(args, "-timeout", *timeout)
	}
	args = append(args, passthrough...)
	args = append(args, selected...)

	cmd := exec.Command("go", args...)
	cmd.Dir = root
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak affected: running go test: %v\n", err)
		return 1
	}
	return 0
}

func baseNote(base string) string {
	if base == "" {
		return " (working tree vs HEAD)"
	}
	return " (since " + base + ")"
}

// gitChangedFiles returns the repo-relative, slash-separated paths changed relative to
// the base. With base == "" it is the working tree vs HEAD (staged + unstaged tracked
// changes) plus untracked files; with a base ref it is everything in the working tree
// that differs from that ref. Git already emits forward-slash paths on every platform,
// so the result keys directly into the go-list file index.
func gitChangedFiles(root, base string) ([]string, error) {
	set := map[string]bool{}
	add := func(out string) {
		for _, line := range strings.Split(out, "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				set[filepath.ToSlash(line)] = true
			}
		}
	}

	ref := base
	if ref == "" {
		ref = "HEAD"
	}
	// `git diff --name-only <ref>` compares <ref> to the WORKING TREE, so it covers both
	// committed and uncommitted differences from the ref -- "everything I'd be
	// introducing", which is what the affected set should cover.
	diff, err := gitOut(root, "diff", "--name-only", ref)
	if err != nil {
		return nil, err
	}
	add(diff)
	// Untracked-but-not-ignored files (a brand-new package the diff misses). New
	// relative to ANY ref, so include them whether diffing HEAD or an explicit base.
	others, err := gitOut(root, "ls-files", "--others", "--exclude-standard")
	if err != nil {
		return nil, err
	}
	add(others)

	out := make([]string, 0, len(set))
	for f := range set {
		out = append(out, f)
	}
	sort.Strings(out)
	return out, nil
}

func gitOut(root string, args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = io.Discard
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	return buf.String(), nil
}

// goPkg is the subset of `go list -json` fields the selector needs. The file-list
// fields (relative base names within Dir) are what map a changed FILE to its package;
// the import lists build the dependency edges.
type goPkg struct {
	ImportPath string
	Dir        string
	Module     *struct{ Path, Dir string }

	GoFiles           []string
	CgoFiles          []string
	CFiles            []string
	TestGoFiles       []string
	XTestGoFiles      []string
	EmbedFiles        []string
	TestEmbedFiles    []string
	XTestEmbedFiles   []string
	IgnoredGoFiles    []string // files excluded by build constraints -- still belong to the package
	IgnoredOtherFiles []string

	Imports      []string
	TestImports  []string
	XTestImports []string
}

// goListGraph runs `go list -e -json ./...` and folds it into (fileToPkg, edges, total):
//   - fileToPkg maps each package's repo-relative slash SOURCE/EMBED file to its import path.
//   - edges maps each import path to the intra-module packages it imports, INCLUDING
//     test imports (so a package whose _test.go imports the changed one is selected).
//   - total is the package count (for the "N/M selected" saving line).
//
// `-e` tolerates a package that does not compile (a peer's mid-edit tree) so the gate
// still selects sensibly instead of erroring out.
func goListGraph(root string) (fileToPkg map[string]string, edges map[string][]string, total int, err error) {
	cmd := exec.Command("go", "list", "-e", "-json", "./...")
	cmd.Dir = root
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	// A non-zero exit with `-e` still emits valid JSON for the packages it could load;
	// only treat it as fatal if we parsed nothing.
	runErr := cmd.Run()

	fileToPkg, edges, total, err = parseGoList(&out)
	if err != nil {
		return nil, nil, 0, err
	}
	if total == 0 {
		if runErr != nil {
			return nil, nil, 0, fmt.Errorf("go list produced no packages: %w", runErr)
		}
		return nil, nil, 0, fmt.Errorf("go list produced no packages")
	}
	return fileToPkg, edges, total, nil
}

// parseGoList folds a `go list -json` object stream into (fileToPkg, edges, total). It is
// the pure half of goListGraph (no exec), so the fold is unit-testable against a fixture
// stream. The first package carrying a Module fixes the module path/dir used to make
// file paths repo-relative and to keep only intra-module import edges. Every category of
// a package's files (incl. build-constrained ones) is indexed so a change to any of them
// selects the package; a file that belongs to no package (a doc, a top-level Makefile) is
// simply absent from the index and selects nothing.
func parseGoList(r io.Reader) (fileToPkg map[string]string, edges map[string][]string, total int, err error) {
	fileToPkg = map[string]string{}
	edges = map[string][]string{}
	var modPath, modDir string

	dec := json.NewDecoder(r)
	for {
		var p goPkg
		if decErr := dec.Decode(&p); decErr != nil {
			if decErr == io.EOF {
				break
			}
			return nil, nil, 0, fmt.Errorf("parsing go list json: %w", decErr)
		}
		if p.Module != nil && modPath == "" {
			modPath, modDir = p.Module.Path, p.Module.Dir
		}
		total++
		if p.Dir != "" && modDir != "" {
			if rel, relErr := filepath.Rel(modDir, p.Dir); relErr == nil {
				relSlash := filepath.ToSlash(rel)
				for _, group := range [][]string{
					p.GoFiles, p.CgoFiles, p.CFiles, p.TestGoFiles, p.XTestGoFiles,
					p.EmbedFiles, p.TestEmbedFiles, p.XTestEmbedFiles,
					p.IgnoredGoFiles, p.IgnoredOtherFiles,
				} {
					for _, f := range group {
						key := f
						if relSlash != "." {
							key = relSlash + "/" + f
						}
						fileToPkg[key] = p.ImportPath
					}
				}
			}
		}
		var deps []string
		for _, group := range [][]string{p.Imports, p.TestImports, p.XTestImports} {
			for _, imp := range group {
				if modPath != "" && strings.HasPrefix(imp, modPath) {
					deps = append(deps, imp)
				}
			}
		}
		if len(deps) > 0 {
			edges[p.ImportPath] = deps
		}
	}
	return fileToPkg, edges, total, nil
}
