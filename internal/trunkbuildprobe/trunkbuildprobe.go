// Package trunkbuildprobe diagnoses *why* the release gate's ci-fast subset is
// red: is it a forgotten `git add`?
//
// Under always-on-dev the commonest way trunk goes — and stays — red is a
// coherence break: a commit lands a caller (or a modified file) that references
// a symbol whose definition lives only in an UNCOMMITTED sibling file the author
// forgot to stage. The whole tree builds on the author's disk, so nothing looks
// wrong locally; committed HEAD does not build and the gate correctly holds — but
// the gate's reason ("ci-fast is red") is opaque, so the freeze can sit for days.
//
// This probe makes that break legible. It is READ-ONLY and never edits, stages,
// commits, or pushes. It:
//
//  1. archives committed HEAD to a temp tree — exactly what CI checks out, no
//     uncommitted files — and runs `go build ./...` there;
//  2. if the build fails, parses the compiler errors into (failing package,
//     missing symbol) pairs;
//  3. searches the working tree's uncommitted files (modified + untracked) for a
//     definition of each missing symbol; a hit means committed code references a
//     symbol defined only in an uncommitted file — a forgotten `git add`.
//
// The verdict distinguishes a fixable coherence break (BUILD_BROKEN_COHERENCE,
// with the exact files to stage) from a genuine compile error with no uncommitted
// source to explain it (BUILD_BROKEN_OTHER). The error-parsing and forgotten-file
// heuristics are pure functions that unit-test without a Go toolchain; the build
// itself is a thin wrapper, replaceable by --build-log FILE to diagnose a captured
// CI log offline.
//
// Go port of the retired tools/trunk_build_probe.py (fak pythongate). Exit codes
// mirror release_decide: 0 = builds, 2 = broken, 1 = usage/probe failure.
package trunkbuildprobe

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// A Go build error block looks like:
//
//	# github.com/anthony-chaudhary/fak/cmd/fak
//	cmd\fak\main.go:52:16: undefined: parseVerbArgv
//	cmd\fak\guard.go:1093:8: srv.X undefined (type *T has no field or method X)
//	cmd\fak\slack_outbox.go:305:73: unknown field RetainDead in struct literal ...
var pkgRE = regexp.MustCompile(`^#\s+(\S+)`)

// symbolPatterns is every shape the compiler uses to say "this name has no
// definition". Each capture group is the bare identifier we then hunt for.
var symbolPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\bundefined:\s+(?:[\w./]+\.)?(\w+)\b`),
	regexp.MustCompile(`\bhas no field or method\s+(\w+)\b`),
	regexp.MustCompile(`\bunknown field\s+(\w+)\b`),
}

// diagRE matches the `path:line:col:` prefix opening a diagnostic line (Windows
// `\` or POSIX `/`).
var diagRE = regexp.MustCompile(`^(\S+\.go):(\d+):(?:\d+:)?\s`)

// defTemplates are uncommitted-file DEFINITION probes for a bare identifier
// (substituted for %s). Deliberately permissive on the definition shapes (a false
// "defined here" is cheap — the user eyeballs the named file — while a miss hides
// the forgotten add), but every template matches a DEFINITION, never a mere use:
// a struct-literal key `S:` or a field assignment is a use and is intentionally
// NOT probed, else any file that merely references the symbol would be mis-reported.
var defTemplates = []string{
	`func\s+%s\s*[\(\[]`,             // func S(...)  / func S[T any](...)
	`func\s+\([^)]*\)\s+%s\s*[\(\[]`, // func (r T) S(...)  (method)
	`type\s+%s\b`,                    // type S ...
	`(?:const|var)\s+\(?\s*%s\b`,     // const S = ... / var S T = ...  (keyword-prefixed)
	`(?m)^\s*%s\s+[\w\*\[\]\.]+`,     // struct field / typed block member:  S SomeType
	`(?m)^\s*%s\s*=`,                 // untyped const/var block member:  S = ...
}

// MissingSymbol is one undefined identifier and where the compiler flagged it.
type MissingSymbol struct {
	Symbol       string `json:"symbol"`
	ReferencedIn string `json:"referenced_in"`
	At           string `json:"at"`
}

// BuildErrors is the parse of `go build ./...` stderr.
type BuildErrors struct {
	FailingPackages []string        `json:"failing_packages"`
	MissingSymbols  []MissingSymbol `json:"missing_symbols"`
}

// ForgottenFile is one uncommitted file that defines missing symbol(s).
type ForgottenFile struct {
	Path    string   `json:"path"`
	Defines []string `json:"defines"`
}

// Verdict is the full diagnosis. Field order matches the Python JSON payload.
type Verdict struct {
	Head            string          `json:"head"`
	Builds          bool            `json:"builds"`
	Verdict         string          `json:"verdict"`
	FailingPackages []string        `json:"failing_packages"`
	MissingSymbols  []MissingSymbol `json:"missing_symbols"`
	ForgottenFiles  []ForgottenFile `json:"forgotten_files"`
	Summary         string          `json:"summary"`
}

// ParseBuildErrors parses `go build ./...` stderr into failing packages and
// missing symbols. ReferencedIn is the package (last `# pkg` header seen); At is
// the file:line the compiler flagged. Pure — no I/O.
func ParseBuildErrors(stderr string) BuildErrors {
	packages := []string{}
	symbols := []MissingSymbol{}
	seen := map[string]bool{}
	currentPkg := ""
	for _, raw := range strings.Split(stderr, "\n") {
		line := strings.TrimRight(raw, " \t\r\n\v\f")
		if m := pkgRE.FindStringSubmatch(line); m != nil {
			currentPkg = m[1]
			if !contains(packages, currentPkg) {
				packages = append(packages, currentPkg)
			}
			continue
		}
		at := ""
		if m := diagRE.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
			at = m[1] + ":" + m[2]
		}
		for _, pat := range symbolPatterns {
			for _, mm := range pat.FindAllStringSubmatch(line, -1) {
				sym := mm[1]
				key := currentPkg + "\x00" + sym
				if seen[key] {
					continue
				}
				seen[key] = true
				symbols = append(symbols, MissingSymbol{Symbol: sym, ReferencedIn: currentPkg, At: at})
			}
		}
	}
	return BuildErrors{FailingPackages: packages, MissingSymbols: symbols}
}

// DefinesSymbol reports whether content looks like it DEFINES the bare
// identifier symbol. Over-matches on purpose (see defTemplates): a spurious hit
// costs a glance, a miss costs the whole diagnosis.
func DefinesSymbol(content, symbol string) bool {
	if symbol == "" {
		return false
	}
	esc := regexp.QuoteMeta(symbol)
	for _, tmpl := range defTemplates {
		rx, err := regexp.Compile(fmt.Sprintf(tmpl, esc))
		if err != nil {
			continue
		}
		if rx.MatchString(content) {
			return true
		}
	}
	return false
}

// FindForgottenFiles maps each missing symbol to the uncommitted file(s) that
// define it. uncommitted maps a repo-relative path to its working-tree content.
// Returns one entry per implicated file, sorted by path (defines sorted). A
// symbol with no uncommitted definer is simply absent — the signal that the break
// is NOT a forgotten add.
func FindForgottenFiles(missing []MissingSymbol, uncommitted map[string]string) []ForgottenFile {
	byPath := map[string]map[string]bool{}
	for _, entry := range missing {
		sym := entry.Symbol
		for path, content := range uncommitted {
			// go build ./... never compiles _test.go, so a build-time reference
			// can never resolve to a test file.
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				continue
			}
			if DefinesSymbol(content, sym) {
				if byPath[path] == nil {
					byPath[path] = map[string]bool{}
				}
				byPath[path][sym] = true
			}
		}
	}
	paths := make([]string, 0, len(byPath))
	for p := range byPath {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	out := make([]ForgottenFile, 0, len(paths))
	for _, p := range paths {
		defs := make([]string, 0, len(byPath[p]))
		for s := range byPath[p] {
			defs = append(defs, s)
		}
		sort.Strings(defs)
		out = append(out, ForgottenFile{Path: p, Defines: defs})
	}
	return out
}

// Classify folds the verdict token.
func Classify(builds bool, forgotten []ForgottenFile, missing []MissingSymbol) string {
	if builds {
		return "BUILD_OK"
	}
	if len(forgotten) > 0 {
		return "BUILD_BROKEN_COHERENCE"
	}
	return "BUILD_BROKEN_OTHER"
}

// Diagnose assembles the full verdict from a build result + uncommitted contents.
// Pure over its inputs, so the whole decision is unit-testable without Go or git.
func Diagnose(builds bool, stderr string, uncommitted map[string]string, head string) Verdict {
	src := stderr
	if builds {
		src = ""
	}
	parsed := ParseBuildErrors(src)
	forgotten := []ForgottenFile{}
	if !builds {
		forgotten = FindForgottenFiles(parsed.MissingSymbols, uncommitted)
	}
	verdict := Classify(builds, forgotten, parsed.MissingSymbols)
	var summary string
	switch verdict {
	case "BUILD_OK":
		summary = "committed HEAD builds; the ci-fast red is not a build break"
	case "BUILD_BROKEN_COHERENCE":
		names := make([]string, len(forgotten))
		for i, f := range forgotten {
			names[i] = f.Path
		}
		summary = fmt.Sprintf(
			"committed HEAD does not build; %d missing symbol(s) are defined only in "+
				"uncommitted file(s): %s. This is a forgotten `git add` — stage and commit "+
				"those files to green the gate.",
			len(parsed.MissingSymbols), strings.Join(names, ", "))
	default:
		summary = fmt.Sprintf(
			"committed HEAD does not build (%d package(s)) but no uncommitted file defines "+
				"the missing symbol(s); this is a genuine compile error, not a forgotten add "+
				"— inspect the flagged sites.",
			len(parsed.FailingPackages))
	}
	return Verdict{
		Head:            head,
		Builds:          builds,
		Verdict:         verdict,
		FailingPackages: parsed.FailingPackages,
		MissingSymbols:  parsed.MissingSymbols,
		ForgottenFiles:  forgotten,
		Summary:         summary,
	}
}

// IsGoBuildablePath reports whether `go build ./...` would compile this path's
// package. Go's package discovery skips any directory component beginning with
// `.` or `_` and the testdata/vendor trees; a definition under such a path can
// never satisfy a committed reference, so it must never be reported as a
// forgotten file. Mirrors the compiler's own rule.
func IsGoBuildablePath(path string) bool {
	if !strings.HasSuffix(path, ".go") {
		return false
	}
	parts := strings.Split(strings.ReplaceAll(path, "\\", "/"), "/")
	for _, comp := range parts[:len(parts)-1] {
		if strings.HasPrefix(comp, ".") || strings.HasPrefix(comp, "_") || comp == "testdata" || comp == "vendor" {
			return false
		}
	}
	return true
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// --------------------------------------------------------------------------- I/O

// RepoRoot returns the git top-level, falling back to the working directory.
func RepoRoot() string {
	cmd := windowgate.Command("git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	if out, err := cmd.Output(); err == nil {
		if s := strings.TrimSpace(string(out)); s != "" {
			return s
		}
	}
	wd, _ := os.Getwd()
	return wd
}

// HeadSHA returns the committed HEAD sha ("" if unavailable).
func HeadSHA(root string) string {
	cmd := windowgate.Command("git", "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// UncommittedFiles maps repo-relative path -> working-tree content for every
// uncommitted .go file go build actually compiles (modified/added/renamed plus
// untracked-but-not-ignored). Reads the working-tree bytes, where a forgotten
// definition lives.
func UncommittedFiles(root string) map[string]string {
	cmd := windowgate.Command("git", "status", "--porcelain", "--untracked-files=all", "-z")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return map[string]string{}
	}
	result := map[string]string{}
	for _, record := range strings.Split(string(out), "\x00") {
		if len(record) < 4 {
			continue
		}
		path := record[3:]
		if !IsGoBuildablePath(path) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			continue
		}
		result[path] = string(data)
	}
	return result
}

// BuildCommittedHead archives committed HEAD to a temp tree and runs
// `go build ./...` there — the same blind base CI checks out, not the author's
// dirty working tree. Returns (ok, combined output).
func BuildCommittedHead(root string) (bool, string, error) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		return false, "", fmt.Errorf("go toolchain not found on PATH")
	}
	tmp, err := os.MkdirTemp("", "trunk_build_probe_")
	if err != nil {
		return false, "", err
	}
	defer os.RemoveAll(tmp)

	archive := windowgate.Command("git", "archive", "--format=tar", "HEAD")
	windowgate.ConfigureBackgroundCommand(archive)
	archive.Dir = root
	var arBuf, arErr bytes.Buffer
	archive.Stdout = &arBuf
	archive.Stderr = &arErr
	if err := archive.Run(); err != nil {
		return false, "", fmt.Errorf("git archive: %s", trunc(arErr.String(), 400))
	}
	if err := extractTar(&arBuf, tmp); err != nil {
		return false, "", fmt.Errorf("extract: %v", err)
	}

	build := windowgate.Command(goBin, "build", "./...")
	build.Dir = tmp
	var so, se bytes.Buffer
	build.Stdout = &so
	build.Stderr = &se
	runErr := build.Run()
	return runErr == nil, se.String() + so.String(), nil
}

// extractTar writes every regular file of a tar stream under dest.
func extractTar(r io.Reader, dest string) error {
	tr := tar.NewReader(r)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		target := filepath.Join(dest, filepath.FromSlash(hdr.Name))
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		f, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return err
		}
		if _, err := io.Copy(f, tr); err != nil {
			f.Close()
			return err
		}
		f.Close()
	}
}

func trunc(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// Run is the CLI entry point (exit codes: 0 builds, 2 broken, 1 probe failure).
func Run(stdout, stderr io.Writer, argv []string) int {
	asJSON := false
	buildLog := ""
	for i := 0; i < len(argv); i++ {
		switch {
		case argv[i] == "--json":
			asJSON = true
		case argv[i] == "--build-log":
			if i+1 < len(argv) {
				buildLog = argv[i+1]
				i++
			}
		case strings.HasPrefix(argv[i], "--build-log="):
			buildLog = argv[i][len("--build-log="):]
		}
	}

	root := RepoRoot()
	head := HeadSHA(root)
	var builds bool
	var out string
	if buildLog != "" {
		data, err := os.ReadFile(buildLog)
		if err != nil {
			fmt.Fprintf(stderr, "trunk-build-probe: could not probe HEAD: %v\n", err)
			return 1
		}
		builds, out = false, string(data)
	} else {
		var err error
		builds, out, err = BuildCommittedHead(root)
		if err != nil {
			fmt.Fprintf(stderr, "trunk-build-probe: could not probe HEAD: %v\n", err)
			return 1
		}
	}

	v := Diagnose(builds, out, UncommittedFiles(root), head)
	if asJSON {
		b, _ := json.MarshalIndent(v, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintf(stdout, "trunk-build-probe: %s — %s\n", v.Verdict, v.Summary)
		for _, f := range v.ForgottenFiles {
			fmt.Fprintf(stdout, "  forgotten: %s  (defines %s)\n", f.Path, strings.Join(f.Defines, ", "))
		}
	}
	if v.Builds {
		return 0
	}
	return 2
}
