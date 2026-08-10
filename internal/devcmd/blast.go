package devcmd

// `fak blast` -- the blast-radius estimator (epic #2712, W3). Given a broken package
// or tree, it reports the AFFECTED SET: the live leases and queued issues whose
// declared tree intersects the broken package's DEPENDENCY blast radius -- the package
// plus every package that transitively imports it -- and, as the witness, the disjoint
// ones it excludes. It joins two things that already existed but had never been joined:
// the `internal/affectedtests` import graph (who depends on the broken tree) and the
// live dos lease ledger (`internal/leaseref`, who is holding which tree). It only
// REPORTS the set; holding it (W4), electing one fixer (W5), and rendering the operator
// card (W7) act on it.
//
//	fak blast estimate <path|package>            print the affected leases/issues (human)
//	fak blast estimate <path|package> --json     print the AffectedSet as JSON
//	fak blast estimate <path> --leases FILE      read leases from a JSONL fixture instead of the live ledger
//	fak blast estimate <path> --issues FILE      join queued-issue declared paths from a JSONL file
//
// Impure shell over internal/blastradius: it gathers the import graph (`go list`, keyed
// by repo-relative package DIR so a package node and a lease tree compare in the same
// namespace), the live lease set (leaseref), and the queued-issue paths, then folds
// them through the pure blastradius.Estimate. The `--leases` fixture path is what makes
// an estimate capturable offline (the witness) and what the cmd tests drive.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastlease"
	"github.com/anthony-chaudhary/fak/internal/blastradius"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Seams the tests inject to avoid a real `go list` / git lease read.
var (
	blastDirGraph    = goListDirGraph
	blastLeaseSource = blastlease.Live
	blastNow         = time.Now
)

func RunBlast(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak blast: expected a subcommand (estimate)")
		return 2
	}
	switch argv[0] {
	case "estimate":
		return runBlastEstimate(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak blast: unknown subcommand %q (want: estimate)\n", argv[0])
		return 2
	}
}

func runBlastEstimate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak blast estimate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "blast")
	asJSON := fs.Bool("json", false, "print the affected set as JSON")
	leasesFile := fs.String("leases", "", "read the lease set from a JSONL fixture ({lane,tree_globs} per line) instead of the live dos lease ledger")
	issuesFile := fs.String("issues", "", "join queued-issue declared paths from a JSONL file ({id,paths} per line)")
	// Interspersed parse: pull the single <path|package> positional out no matter where
	// it sits, so both `estimate internal/a --json` and `estimate --json internal/a` work
	// (Go's flag package otherwise stops at the first non-flag token).
	var positionals []string
	rest := argv
	for {
		if err := fs.Parse(rest); err != nil {
			return 2 // the FlagSet already rendered the error/usage
		}
		if fs.NArg() == 0 {
			break
		}
		positionals = append(positionals, fs.Arg(0))
		rest = fs.Args()[1:]
	}
	if len(positionals) != 1 {
		fmt.Fprintln(stderr, "fak blast estimate: expected exactly one <path|package> to expand (e.g. internal/knownbad)")
		return 2
	}
	target := positionals[0]

	root := repoRoot()

	edges, modPath, _, err := blastDirGraph(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak blast estimate: building import graph: %v\n", err)
		return 1
	}

	// Reconcile the broken arg to the graph's node namespace: canonicalize the tree
	// (knownbad's glob rule -- the same one the leases were normalized through) and
	// strip the module prefix so a full import path resolves to its repo-relative dir.
	broken := trimModulePrefix(knownbad.NormalizeTree(target), modPath)
	if broken == "" {
		fmt.Fprintf(stderr, "fak blast estimate: %q is not a repo-relative path/package\n", target)
		return 2
	}

	var leases []blastradius.Lease
	if *leasesFile != "" {
		leases, err = blastlease.Read(*leasesFile)
	} else {
		leases, err = blastLeaseSource(root, blastNow())
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak blast estimate: reading leases: %v\n", err)
		return 1
	}

	var issues []blastradius.Issue
	if *issuesFile != "" {
		if issues, err = readBlastIssues(*issuesFile); err != nil {
			fmt.Fprintf(stderr, "fak blast estimate: reading issues: %v\n", err)
			return 1
		}
	}

	set := blastradius.Estimate(edges, broken, leases, issues)

	if *asJSON {
		out := struct {
			Schema string `json:"schema"`
			blastradius.AffectedSet
		}{Schema: blastradius.Schema, AffectedSet: set}
		if err := writeIndentedJSONNoEscape(stdout, out); err != nil {
			fmt.Fprintf(stderr, "fak blast estimate: %v\n", err)
			return 1
		}
		return 0
	}

	writeBlastText(stdout, set)
	return 0
}

// writeBlastText renders the affected set for a human operator: the radius, then the
// held leases/issues with the trees that put them in the hold set, then the disjoint
// ones that are free to keep running (the witness the done-condition asks for).
func writeBlastText(w io.Writer, set blastradius.AffectedSet) {
	fmt.Fprintf(w, "blast radius of %s: %d package(s)\n", set.Broken, len(set.Radius))
	for _, p := range set.Radius {
		fmt.Fprintf(w, "  %s\n", p)
	}
	fmt.Fprintf(w, "held (affected -- intersect the blast radius): %d lease(s), %d issue(s)\n", len(set.Leases), len(set.Issues))
	for _, l := range set.Leases {
		fmt.Fprintf(w, "  lease %s\tvia %s\n", l.Lane, strings.Join(l.Matched, " "))
	}
	for _, is := range set.Issues {
		fmt.Fprintf(w, "  issue %s\tvia %s\n", is.ID, strings.Join(is.Matched, " "))
	}
	fmt.Fprintf(w, "excluded (disjoint -- free to run): %d lease(s), %d issue(s)\n", len(set.ExcludedLeases), len(set.ExcludedIssues))
	for _, lane := range set.ExcludedLeases {
		fmt.Fprintf(w, "  lease %s\n", lane)
	}
	for _, id := range set.ExcludedIssues {
		fmt.Fprintf(w, "  issue %s\n", id)
	}
}

// readBlastJSONL decodes one JSONL fixture into a slice of T. Blank lines are skipped so a
// trailing newline is not read as a record, while a malformed line stays a hard error.
func readBlastJSONL[T any](path string) ([]T, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []T
	for i, line := range strings.Split(string(data), "\n") {
		s := strings.TrimSpace(line)
		if s == "" {
			continue
		}
		var v T
		if err := json.Unmarshal([]byte(s), &v); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, i+1, err)
		}
		out = append(out, v)
	}
	return out, nil
}

func readBlastIssues(path string) ([]blastradius.Issue, error) {
	return readBlastJSONL[blastradius.Issue](path)
}

// trimModulePrefix maps a full import path down to its repo-relative dir so the broken
// arg lands in the same namespace as the graph nodes and lease trees. A bare
// repo-relative path (no module prefix) passes through unchanged.
func trimModulePrefix(tree, modPath string) string {
	if modPath == "" || tree == "" {
		return tree
	}
	if tree == modPath {
		return "."
	}
	return strings.TrimPrefix(tree, modPath+"/")
}

// goListDirGraph runs `go list -e -json ./...` and folds it into a repo-relative-
// DIRECTORY import graph: edges[dir] is the set of intra-module package directories dir
// imports (Imports + TestImports + XTestImports). It mirrors affected.go's goListGraph
// but in tree space -- keying by repo-relative dir rather than import path is what lets
// the blast radius intersect a lease tree, since both are then the same repo-relative
// form knownbad.TreesIntersect compares. `-e` tolerates a package that does not compile
// (a peer's mid-edit tree) so the estimate still lands. It also returns the module path
// so the shell can strip it off a full import path passed as the broken arg.
func goListDirGraph(root string) (edges map[string][]string, modPath string, total int, err error) {
	cmd := exec.Command("go", "list", "-e", "-json", "./...")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	// A non-zero exit with `-e` still emits valid JSON for the packages it could load;
	// only treat it as fatal if we parsed nothing.
	runErr := cmd.Run()

	edges, modPath, total, err = parseGoListDirs(&out)
	if err != nil {
		return nil, "", 0, err
	}
	if total == 0 {
		if runErr != nil {
			return nil, "", 0, fmt.Errorf("go list produced no packages: %w", runErr)
		}
		return nil, "", 0, fmt.Errorf("go list produced no packages")
	}
	return edges, modPath, total, nil
}

// parseGoListDirs is the pure fold at the heart of goListDirGraph (no exec), so it is
// unit-testable against a fixture stream. Two passes: first map every import path to its
// repo-relative dir (relative to the module root), then translate each package's
// intra-module imports into dir->dir edges. The first package carrying a Module fixes
// the module path/dir. A package whose dir cannot be made relative is skipped rather
// than mis-keyed.
type blastGoPkg struct {
	ImportPath   string
	Dir          string
	Module       *struct{ Path, Dir string }
	Imports      []string
	TestImports  []string
	XTestImports []string
}

func parseGoListDirs(r io.Reader) (edges map[string][]string, modPath string, total int, err error) {
	var pkgs []blastGoPkg
	var modDir string
	dec := json.NewDecoder(r)
	for {
		var p blastGoPkg
		if decErr := dec.Decode(&p); decErr != nil {
			if decErr == io.EOF {
				break
			}
			return nil, "", 0, fmt.Errorf("parsing go list json: %w", decErr)
		}
		if p.Module != nil && modPath == "" {
			modPath, modDir = p.Module.Path, p.Module.Dir
		}
		pkgs = append(pkgs, p)
	}
	total = len(pkgs)

	dirOf := make(map[string]string, len(pkgs))
	for _, p := range pkgs {
		if p.Dir == "" || modDir == "" {
			continue
		}
		if rel, relErr := filepath.Rel(modDir, p.Dir); relErr == nil {
			dirOf[p.ImportPath] = filepath.ToSlash(rel)
		}
	}

	edges = make(map[string][]string, len(pkgs))
	for _, p := range pkgs {
		dir, ok := dirOf[p.ImportPath]
		if !ok {
			continue
		}
		var deps []string
		for _, group := range [][]string{p.Imports, p.TestImports, p.XTestImports} {
			for _, imp := range group {
				if modPath != "" && strings.HasPrefix(imp, modPath) {
					if d, ok := dirOf[imp]; ok {
						deps = append(deps, d)
					}
				}
			}
		}
		if len(deps) > 0 {
			edges[dir] = deps
		}
	}
	return edges, modPath, total, nil
}
