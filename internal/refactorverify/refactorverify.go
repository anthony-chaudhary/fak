// Package refactorverify proves a god-split / code-motion refactor dropped NO
// top-level definition — the Go port of the retired tools/refactor_verify.py
// (fak pythongate).
//
// The /modularize skill splits a monolith by MOVING top-level declarations between
// files in the SAME package — a semantic no-op in Go, because package (not file) is
// the scope. internal/godsplitplan PLANS the cut; this leaf VERIFIES the result. It
// answers the one question `go build` cannot: did any top-level declaration silently
// disappear?
//
// `go build`/`go vet` catch a REFERENCED symbol that went missing (undefined: X) or a
// duplicated one (X redeclared). But a top-level decl that was DROPPED and happens to
// be unreferenced in-module — dead-but-real, or an exported symbol whose only
// consumers are out-of-tree or a JSON wire contract — compiles clean and vanishes
// without a trace. THAT is the "god module then missing a definition" failure: an
// incomplete split that still builds green. This leaf closes that gap mechanically.
//
// How: it folds every touched package's top-level declaration MULTISET before (a git
// ref, default HEAD) and after (the working tree), and diffs them. A pure code-motion
// split is declaration-set-preserving by construction, so the correct diff is EMPTY.
// Anything in the diff is RELOCATED (moved to another touched package — informational),
// DROPPED (reappeared nowhere — the missing definition), or OVER-SPLIT (a new file
// carrying a single decl — the file-per-function anti-pattern the skill forbids).
//
// Decl extraction reuses godsplitplan.Compute (raw-string/comment-aware), so this leaf
// inherits the same tested, hazard-aware fold rather than standing up a second parser.
// const/var are normalized to one "value" kind so a top-level const↔var flip (the
// standard make-it-test-overridable move) is not misread as a drop.
//
// Scope (honest boundaries): DECL-level only — it proves no whole top-level decl was
// dropped, NOT that struct fields survived a `type X = pkg.Y` alias consolidation
// (that needs go/types). Grouped `var (`/`const (` blocks collapse to one "(group)"
// decl, counted and footnoted but never classified.
//
// The Verify core is pure (driven by in-memory strings); Run adds the git I/O. It
// never edits, moves, or commits.
package refactorverify

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/godsplitplan"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// DeclID is the (kind, name) identity of a top-level declaration, with const/var
// normalized to "value".
type DeclID struct {
	Kind string
	Name string
}

// DeclRec is a dropped or relocated declaration. To is populated (and marshaled) only
// for relocations — a dropped decl has no destination, matching the Python payload.
type DeclRec struct {
	Pkg   string   `json:"pkg"`
	Kind  string   `json:"kind"`
	Name  string   `json:"name"`
	Count int      `json:"count"`
	To    []string `json:"to,omitempty"`
}

// OverRec is a new file that carries a single top-level decl (the file-per-function
// smell).
type OverRec struct {
	Pkg  string `json:"pkg"`
	File string `json:"file"`
}

// Report is the completeness verdict for a change.
type Report struct {
	Packages       []string  `json:"packages"`
	Dropped        []DeclRec `json:"dropped"`
	Relocated      []DeclRec `json:"relocated"`
	Oversplit      []OverRec `json:"oversplit"`
	GroupedSkipped int       `json:"grouped_skipped"`
}

// DeclsOf returns the (kind, name) identity of every top-level decl in one Go source,
// via godsplitplan.Compute. const/var collapse to "value"; func/method/type stay
// distinct.
func DeclsOf(text string) []DeclID {
	out := []DeclID{}
	for _, d := range godsplitplan.Compute(text).Decls {
		kind := d.Kind
		if kind == "const" || kind == "var" {
			kind = "value"
		}
		out = append(out, DeclID{Kind: kind, Name: d.Name})
	}
	return out
}

// PackageDecls folds the declaration multiset of a package (all its .go files) into
// (kind, name) -> count.
func PackageDecls(files map[string]string) map[DeclID]int {
	c := map[DeclID]int{}
	for _, text := range files {
		for _, id := range DeclsOf(text) {
			c[id]++
		}
	}
	return c
}

// subtractCounts is Counter subtraction: keys of a with a-b > 0 (zero/negative dropped).
func subtractCounts(a, b map[DeclID]int) map[DeclID]int {
	out := map[DeclID]int{}
	for k, v := range a {
		if d := v - b[k]; d > 0 {
			out[k] = d
		}
	}
	return out
}

// Verify is the pure fold. before/after map package-dir -> {filename: source-text}.
// It returns the completeness report; no I/O, so a test drives it with strings.
func Verify(before, after map[string]map[string]string) Report {
	pkgSet := map[string]bool{}
	for p := range before {
		pkgSet[p] = true
	}
	for p := range after {
		pkgSet[p] = true
	}
	pkgs := sortedStrings(pkgSet)

	bdec := map[string]map[DeclID]int{}
	adec := map[string]map[DeclID]int{}
	removed := map[string]map[DeclID]int{}
	added := map[string]map[DeclID]int{}
	for _, p := range pkgs {
		bdec[p] = PackageDecls(before[p])
		adec[p] = PackageDecls(after[p])
	}
	for _, p := range pkgs {
		removed[p] = subtractCounts(bdec[p], adec[p])
		added[p] = subtractCounts(adec[p], bdec[p])
	}

	// A grouped var/const block collapses to (kind, "(group)") — not unique, so it can
	// never be matched across packages without fabricating a bogus relocation. Count and
	// footnote it; never classify it.
	groupedSkipped := 0
	for _, p := range pkgs {
		for id, c := range removed[p] {
			if id.Name == "(group)" {
				groupedSkipped += c
			}
		}
	}

	dropped := []DeclRec{}
	relocated := []DeclRec{}
	for _, p := range pkgs {
		for _, id := range sortedDeclIDs(removed[p]) {
			if id.Name == "(group)" {
				continue
			}
			cnt := removed[p][id]
			var targets []string
			for _, q := range pkgs {
				if q != p && added[q][id] > 0 {
					targets = append(targets, q)
				}
			}
			rec := DeclRec{Pkg: p, Kind: id.Kind, Name: id.Name, Count: cnt}
			if len(targets) > 0 {
				rec.To = targets
				relocated = append(relocated, rec)
			} else {
				dropped = append(dropped, rec)
			}
		}
	}

	oversplit := []OverRec{}
	for _, p := range pkgs {
		existing := map[string]bool{}
		for f := range before[p] {
			existing[f] = true
		}
		afiles := after[p]
		fnames := make([]string, 0, len(afiles))
		for f := range afiles {
			fnames = append(fnames, f)
		}
		sort.Strings(fnames)
		for _, fname := range fnames {
			if existing[fname] {
				continue
			}
			fp := godsplitplan.Compute(afiles[fname])
			if len(fp.Decls) != 1 {
				continue
			}
			// A per-OS build-tagged stub and a `func main` entrypoint carry one decl by
			// design, not by over-splitting a god-file.
			if len(fp.Hazards.BuildTags) > 0 {
				continue
			}
			if fp.Decls[0].Kind == "func" && fp.Decls[0].Name == "main" {
				continue
			}
			oversplit = append(oversplit, OverRec{Pkg: p, File: p + "/" + fname})
		}
	}

	return Report{
		Packages:       pkgs,
		Dropped:        dropped,
		Relocated:      relocated,
		Oversplit:      oversplit,
		GroupedSkipped: groupedSkipped,
	}
}

func sortedStrings(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

// sortedDeclIDs orders a decl multiset's keys by (kind, name) — the Python
// sorted(Counter.items()) tuple order, so list output is deterministic and matches.
func sortedDeclIDs(m map[DeclID]int) []DeclID {
	out := make([]DeclID, 0, len(m))
	for id := range m {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// Render is the human table (the non-JSON output).
func Render(rep Report, expectMotion bool) string {
	var out []string
	out = append(out, fmt.Sprintf("refactor-verify: %d package(s) touched", len(rep.Packages)))
	if len(rep.Dropped) > 0 {
		out = append(out, "")
		out = append(out, "DROPPED - a definition left a package and reappeared NOWHERE (missing definition):")
		for _, d := range rep.Dropped {
			mult := ""
			if d.Count > 1 {
				mult = fmt.Sprintf(" x%d", d.Count)
			}
			out = append(out, fmt.Sprintf("  !! %s: %s %s%s", d.Pkg, d.Kind, d.Name, mult))
		}
		out = append(out, "  -> a pure code-motion split must preserve every decl. If this deletion is")
		out = append(out, "    intentional, say so in the commit body; otherwise the split is INCOMPLETE.")
	}
	if len(rep.Relocated) > 0 {
		out = append(out, "")
		verb := "relocated (cross-package consolidation)"
		if expectMotion {
			verb = "!! (expect-motion) RELOCATED"
		}
		out = append(out, verb+" - a decl moved between packages:")
		for _, r := range rep.Relocated {
			out = append(out, fmt.Sprintf("  %s: %s %s  ->  %s", r.Pkg, r.Kind, r.Name, strings.Join(r.To, ", ")))
		}
		if expectMotion {
			out = append(out, "  -> --expect-motion asserts a PURE in-package split; a cross-package move fails it.")
		}
	}
	if len(rep.Oversplit) > 0 {
		out = append(out, "")
		out = append(out, "OVER-SPLIT - a new file carries a single decl (file-per-function anti-pattern):")
		for _, o := range rep.Oversplit {
			out = append(out, "  ~ "+o.File)
		}
		out = append(out, "  -> group related decls into one cohesive concern file; don't split per-function.")
	}
	fail := len(rep.Dropped) > 0 || (expectMotion && len(rep.Relocated) > 0)
	if !fail && len(rep.Oversplit) == 0 {
		out = append(out, "clean - declaration set preserved; no definition dropped.")
	}
	if rep.GroupedSkipped > 0 {
		out = append(out, "")
		out = append(out, fmt.Sprintf("note: %d grouped var/const block(s) not tracked "+
			"(members unnamed at this resolution).", rep.GroupedSkipped))
	}
	return strings.Join(out, "\n")
}

// --------------------------------------------------------------------------- git I/O

func gitOutput(root string, args ...string) (string, int) {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd) // windowless git child on Windows: no console flash
	cmd.Dir = root
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	rc := 0
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			rc = ee.ExitCode()
		} else {
			rc = 1
		}
	}
	return out.String(), rc
}

// normalizeNewlines mirrors Python's text-mode universal-newline read, so a CRLF blob
// (git autocrlf / a Windows working-tree file) folds identically to its LF form.
func normalizeNewlines(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "\r\n", "\n"), "\r", "\n")
}

func pkgOf(p string) string {
	return path.Dir(strings.ReplaceAll(p, "\\", "/"))
}

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
}

// touchedPkgs is the set of package dirs a change touched: explicit .go paths if given,
// else every .go that differs from ref PLUS untracked .go (a split's new concern files
// are untracked until committed, which git diff alone would miss).
func touchedPkgs(root, ref string, paths []string) []string {
	set := map[string]bool{}
	var files []string
	if len(paths) > 0 {
		for _, p := range paths {
			if strings.HasSuffix(p, ".go") {
				files = append(files, p)
			}
		}
	} else {
		changed, _ := gitOutput(root, "diff", "--name-only", ref, "--", "*.go")
		untracked, _ := gitOutput(root, "ls-files", "--others", "--exclude-standard", "--", "*.go")
		files = append(splitLines(changed), splitLines(untracked)...)
	}
	for _, f := range files {
		f = strings.TrimSpace(f)
		if f != "" {
			set[pkgOf(f)] = true
		}
	}
	return sortedStrings(set)
}

// afterFiles reads every top-level .go file of pkgdir in the WORKING TREE
// (non-recursive — a package is one directory).
func afterFiles(root, pkgdir string) map[string]string {
	files := map[string]string{}
	dir := filepath.Join(root, filepath.FromSlash(pkgdir))
	entries, err := os.ReadDir(dir)
	if err != nil {
		return files
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		if b, err := os.ReadFile(filepath.Join(dir, e.Name())); err == nil {
			files[e.Name()] = normalizeNewlines(string(b))
		}
	}
	return files
}

// beforeFiles reads every top-level .go file of pkgdir at ref (git ls-tree is
// non-recursive, so it lists exactly the package's own files, not sub-packages').
func beforeFiles(root, ref, pkgdir string) map[string]string {
	files := map[string]string{}
	listing, _ := gitOutput(root, "ls-tree", "--name-only", ref, pkgdir+"/")
	for _, p := range splitLines(listing) {
		p = strings.TrimSpace(p)
		if !strings.HasSuffix(p, ".go") {
			continue
		}
		blob, rc := gitOutput(root, "show", ref+":"+p)
		if rc == 0 {
			base := p
			if i := strings.LastIndex(p, "/"); i >= 0 {
				base = p[i+1:]
			}
			files[base] = normalizeNewlines(blob)
		}
	}
	return files
}

// Run is the CLI entry point: `refactor-verify [paths...] [--ref REF] [--expect-motion]
// [--json]`. Exit 0 clean, 1 on a dropped decl (or a relocation under --expect-motion),
// 2 if REF is not a commit — mirroring the Python main.
func Run(stdout, stderr io.Writer, argv []string) int {
	ref := "HEAD"
	expectMotion := false
	asJSON := false
	var paths []string
	for i := 0; i < len(argv); i++ {
		a := argv[i]
		switch {
		case a == "--ref":
			if i+1 < len(argv) {
				i++
				ref = argv[i]
			}
		case strings.HasPrefix(a, "--ref="):
			ref = a[len("--ref="):]
		case a == "--expect-motion":
			expectMotion = true
		case a == "--json":
			asJSON = true
		case strings.HasPrefix(a, "-"):
			// ignore unknown flags
		default:
			paths = append(paths, a)
		}
	}

	rootDir := "."
	if top, rc := gitOutput(".", "rev-parse", "--show-toplevel"); rc == 0 {
		if t := strings.TrimSpace(top); t != "" {
			rootDir = t
		}
	}
	if _, rc := gitOutput(rootDir, "rev-parse", "--verify", "--quiet", ref+"^{commit}"); rc != 0 {
		fmt.Fprintf(stderr, "refactor-verify: not a commit: %s\n", ref)
		return 2
	}

	pkgs := touchedPkgs(rootDir, ref, paths)
	before := map[string]map[string]string{}
	after := map[string]map[string]string{}
	for _, p := range pkgs {
		before[p] = beforeFiles(rootDir, ref, p)
		after[p] = afterFiles(rootDir, p)
	}
	rep := Verify(before, after)

	if asJSON {
		b, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintln(stdout, Render(rep, expectMotion))
	}

	if len(rep.Dropped) > 0 || (expectMotion && len(rep.Relocated) > 0) {
		return 1
	}
	return 0
}
