package devindex

// The REACHABILITY half of the executable audit (#5648). Kept beside execaudit.go
// because it carries the one rule that decides whether the whole instrument measures
// anything: what counts as evidence that something outside a package invokes it.
//
// The vacuity trap this file exists to avoid: an audit that greps the tree for a
// package's name finds the package's own source, its own catalog row, and its own
// audit output — and reports everything reachable. So a name is never enough. A line
// is evidence only when it carries an INVOCATION FORM:
//
//	A. a `go build|run|install|test|vet` naming the package path or import path;
//	B. a `./<binary>` (or `<binary>.exe`) command token;
//	C. the package path inside a STRING LITERAL on a non-comment line of Go code
//	   outside the package — a main package cannot be imported, so a Go file that
//	   names its path in a literal is spawning or registering it.
//
// and only when it lives OUTSIDE the package's own directory. A markdown table row is
// never evidence (that is the shape of an inventory), and machine-readable inventory
// files (.json/.jsonl — including this audit's own witness) are never scanned at all.
// Together those two rules are what make "documented runnable example" mean a command
// a reader could run, not a row in a list of things that exist.

import (
	"bufio"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// execRunFormRE matches a `go build|run|install|test|vet` invocation anywhere on a
// line (after a shell prompt, inside a Makefile recipe, in a fenced doc block, or as
// an exec.Command argument list).
var execRunFormRE = regexp.MustCompile(`(?:^|[^\w-])go\s+(?:build|run|install|test|vet)(?:[^\w-]|$)`)

// execGoInstallRE distinguishes the installer class from the generic build target.
var execGoInstallRE = regexp.MustCompile(`(?:^|[^\w-])go\s+install(?:[^\w-]|$)`)

// execScanSkipDirs are directories that carry no reachability evidence: VCS and
// kernel runtime state, dependency mirrors, and the scratch/duplicate-checkout trees
// the Go toolchain itself already refuses to walk. Any directory whose name starts
// with "_" is skipped for the same reason. ".github" is deliberately NOT skipped — a
// CI workflow that builds a binary is exactly a build-target edge.
var execScanSkipDirs = map[string]bool{
	".git": true, ".dos": true, ".dispatch-runs": true,
	"node_modules": true, "vendor": true, "__pycache__": true, "testdata": true,
}

// execScanExts are the file kinds that can carry an invocation. `.json`/`.jsonl` are
// absent on purpose: a machine-readable inventory (this audit's own witness among
// them) lists packages, it does not invoke them.
var execScanExts = map[string]ExecEvidenceClass{
	".go":   ExecEvidenceDispatch,
	".mk":   ExecEvidenceBuildTarget,
	".yml":  ExecEvidenceBuildTarget,
	".yaml": ExecEvidenceBuildTarget,
	".sh":   ExecEvidenceScript,
	".bash": ExecEvidenceScript,
	".ps1":  ExecEvidenceScript,
	".bat":  ExecEvidenceScript,
	".cmd":  ExecEvidenceScript,
	".py":   ExecEvidenceScript,
	".md":   ExecEvidenceDocExample,
	".txt":  ExecEvidenceDocExample,
}

// maxExecScanBytes caps a single scanned file. A generated blob larger than this is
// not a hand-written invocation edge.
const maxExecScanBytes = 4 << 20

// scanExecReachability sweeps the corpus once and attaches to each package the FIRST
// evidence found per class. One sweep with a reverse lookup keeps the cost linear in
// the tree rather than (tree x packages). The corpus is resolved by the caller, so
// the domain filter and the evidence sweep are guaranteed to share one view of what
// belongs to the repository.
func scanExecReachability(root string, corpus []string, pkgs []ExecPackage) {
	if len(pkgs) == 0 {
		return
	}
	idx := newExecIndex(pkgs)

	found := make([]map[ExecEvidenceClass]ExecEvidence, len(pkgs))
	for i := range found {
		found[i] = map[ExecEvidenceClass]ExecEvidence{}
	}

	for _, relFile := range corpus {
		class, ok := execScanClass(path.Base(relFile))
		if !ok {
			continue
		}
		dirRel := path.Dir(relFile)
		if dirRel == "." {
			dirRel = ""
		}
		abs := filepath.Join(root, filepath.FromSlash(relFile))
		scanExecFile(abs, relFile, dirRel, class, pkgs, idx, found)
	}

	for i := range pkgs {
		pkgs[i].Evidence = sortedEvidence(found[i])
	}
}

// execScanClass maps a file name to the evidence class it can carry.
func execScanClass(name string) (ExecEvidenceClass, bool) {
	if name == "Makefile" {
		return ExecEvidenceBuildTarget, true
	}
	class, ok := execScanExts[strings.ToLower(filepath.Ext(name))]
	return class, ok
}

// execAuditCorpus returns the repo-relative, slash-separated files that may carry a
// reachability edge, plus whether that set came from version control (which is what
// lets the caller apply the same repository rule to the executable DOMAIN).
//
// It prefers the set of VERSION-CONTROLLED files, and that preference is a
// correctness rule before it is a speed one. This module is developed in a shared
// checkout that is permanently littered with per-worker scratch trees, build
// overlays and archived dispatch runs — each a partial COPY of the repository. A
// plain directory walk reads those copies, so a duplicated Makefile or README under
// a scratch tree fabricates a build-target edge for a package that nothing in the
// actual repository builds, and reports it with a file locator no reader can act on.
// Reachability means the REPOSITORY wires the package up; a throwaway copy of the
// repository is not that. Restricting the corpus to tracked files also makes the
// audit deterministic on a dirty tree — a peer's uncommitted file can no longer
// silently create or remove another package's evidence.
//
// The filesystem walk remains the fallback for a tree that is not a git checkout
// (the synthetic fixtures, and any consumer auditing an exported source tree), where
// every file present is the tree's own.
func execAuditCorpus(root string) ([]string, bool) {
	if tracked, ok := trackedScanCorpus(root); ok {
		return tracked, true
	}
	return walkScanCorpus(root), false
}

// keepTrackedExecPackages restricts the executable DOMAIN to packages the repository
// actually carries, using the same tracked-file corpus the evidence sweep uses.
//
// `go list ./...` answers for the DIRECTORY TREE, not for the repository, and on this
// shared checkout those differ: a peer's untracked scratch `package main` — a one-line
// stub left behind by a build overlay or a refutation probe — is a real main package
// on disk and lands in the domain. Counting it does two kinds of damage. It inflates
// the denominator with a package that is nobody's to fix, and because nothing tracked
// invokes a file that is not in the repository, it always reports as a fresh `orphan`
// FAILURE. That makes the audit's verdict depend on whatever scratch directories
// happened to exist when it ran, which is the one property a measured invariant cannot
// have.
//
// A directory counts as the repository's when it holds at least one tracked `.go`
// file. Outside a checkout the caller passes tracked=false and the set is kept whole,
// for the same reason the corpus falls back to the walk: there, every file present is
// the tree's own.
func keepTrackedExecPackages(corpus []string, pkgs []ExecPackage) []ExecPackage {
	trackedDirs := make(map[string]bool, len(corpus))
	for _, rel := range corpus {
		if !strings.HasSuffix(rel, ".go") {
			continue
		}
		dir := path.Dir(rel)
		if dir == "." {
			dir = ""
		}
		trackedDirs[dir] = true
	}
	out := pkgs[:0]
	for _, p := range pkgs {
		if trackedDirs[p.Dir] {
			out = append(out, p)
		}
	}
	return out
}

// trackedScanCorpus lists the tree's tracked files. It reports ok=false when git is
// unavailable or the tree is not a checkout, so the caller can fall back rather than
// silently auditing against an empty corpus — an empty corpus would report every
// executable unreachable, which is a false RED, not a false green.
func trackedScanCorpus(root string) ([]string, bool) {
	cmd := exec.Command("git", "-C", root, "ls-files", "-z")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, false
	}
	var files []string
	for _, rel := range strings.Split(string(out), "\x00") {
		if rel == "" || execScanSkipRel(rel) {
			continue
		}
		files = append(files, rel)
	}
	if len(files) == 0 {
		return nil, false
	}
	sort.Strings(files)
	return files, true
}

// walkScanCorpus is the non-git fallback: the whole tree minus the directories that
// carry no invocation edge.
func walkScanCorpus(root string) []string {
	var files []string
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil //nolint:nilerr // an unreadable subtree is not evidence, not a fatal audit error
		}
		name := d.Name()
		if d.IsDir() {
			if p == root {
				return nil
			}
			if execScanSkipDirs[name] || strings.HasPrefix(name, "_") {
				return filepath.SkipDir
			}
			return nil
		}
		rel := relSlash(root, p)
		if rel == "" {
			return nil
		}
		files = append(files, rel)
		return nil
	})
	sort.Strings(files)
	return files
}

// execScanSkipRel applies the skip-directory rules to a repo-relative path, so the
// tracked corpus and the walk fallback agree on what carries no evidence.
func execScanSkipRel(rel string) bool {
	for _, seg := range strings.Split(path.Dir(rel), "/") {
		if seg == "." || seg == "" {
			continue
		}
		if execScanSkipDirs[seg] || strings.HasPrefix(seg, "_") {
			return true
		}
	}
	return false
}

func sortedEvidence(m map[ExecEvidenceClass]ExecEvidence) []ExecEvidence {
	if len(m) == 0 {
		return nil
	}
	out := make([]ExecEvidence, 0, len(m))
	for _, e := range m {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Class != out[j].Class {
			return out[i].Class < out[j].Class
		}
		return out[i].File < out[j].File
	})
	return out
}

// execIndex is the reverse lookup that keeps the sweep linear in the tree. Without
// it every candidate line is re-tested against every executable package, which on a
// module with ~100 executables and a million lines dominates the whole audit.
type execIndex struct {
	prefixes []string         // the parent dirs executables live under: "cmd/", "experiments/", …
	dirs     map[string][]int // "cmd/foo" -> package rows
	bins     map[string][]int // "foo"     -> package rows
}

func newExecIndex(pkgs []ExecPackage) *execIndex {
	ix := &execIndex{dirs: map[string][]int{}, bins: map[string][]int{}}
	seen := map[string]bool{}
	for i, p := range pkgs {
		ix.dirs[p.Dir] = append(ix.dirs[p.Dir], i)
		ix.bins[p.Binary] = append(ix.bins[p.Binary], i)
		if j := strings.Index(p.Dir, "/"); j > 0 && !seen[p.Dir[:j+1]] {
			seen[p.Dir[:j+1]] = true
			ix.prefixes = append(ix.prefixes, p.Dir[:j+1])
		}
	}
	sort.Strings(ix.prefixes)
	return ix
}

// candidates returns the package rows a line could possibly be evidence for, by
// extracting the path- and binary-shaped tokens the line actually contains and
// looking them up. It is a FILTER, not the decision: every hit is still re-tested
// against the precise invocation-form rules in execLineInvokes.
func (ix *execIndex) candidates(line string, out []int) []int {
	out = out[:0]
	// Path-shaped tokens: from each "cmd/"-style prefix occurrence, take the maximal
	// path run and try it, then each shorter parent, so a nested package path
	// ("cmd/a/b/c") is found without scanning every package.
	for _, pre := range ix.prefixes {
		for at := 0; ; {
			i := strings.Index(line[at:], pre)
			if i < 0 {
				break
			}
			start := at + i
			end := start
			for end < len(line) && (isPathNameByte(line[end]) || line[end] == '/') {
				end++
			}
			for run := strings.TrimRight(line[start:end], "/"); run != ""; {
				out = appendUnique(out, ix.dirs[run]...)
				j := strings.LastIndexByte(run, '/')
				if j <= 0 {
					break
				}
				run = run[:j]
			}
			at = start + len(pre)
		}
	}
	// Binary-shaped tokens: the name after a "./" and the name before a ".exe".
	for at := 0; ; {
		i := strings.Index(line[at:], "./")
		if i < 0 {
			break
		}
		start := at + i + 2
		end := start
		for end < len(line) && isPathNameByte(line[end]) {
			end++
		}
		out = appendUnique(out, ix.bins[line[start:end]]...)
		at = start
	}
	for at := 0; ; {
		i := strings.Index(line[at:], ".exe")
		if i < 0 {
			break
		}
		end := at + i
		start := end
		for start > 0 && isPathNameByte(line[start-1]) && line[start-1] != '.' {
			start--
		}
		out = appendUnique(out, ix.bins[line[start:end]]...)
		at = end + 4
	}
	return out
}

func appendUnique(dst []int, add ...int) []int {
	for _, v := range add {
		dup := false
		for _, have := range dst {
			if have == v {
				dup = true
				break
			}
		}
		if !dup {
			dst = append(dst, v)
		}
	}
	return dst
}

// scanExecFile reads one file and records evidence for every package it invokes.
// dirRel is the file's repo-relative directory: a file at or below a package's own
// directory can never be that package's reachability evidence.
func scanExecFile(abs, relFile, dirRel string, class ExecEvidenceClass, pkgs []ExecPackage, ix *execIndex, found []map[ExecEvidenceClass]ExecEvidence) {
	fi, err := os.Stat(abs)
	if err != nil || fi.Size() == 0 || fi.Size() > maxExecScanBytes {
		return
	}
	// A package with evidence in every class needs nothing more from this file, but
	// checking that per file is not worth the bookkeeping; the per-class first-write
	// below is already idempotent.
	f, err := os.Open(abs)
	if err != nil {
		return
	}
	defer f.Close()

	isGo := strings.HasSuffix(relFile, ".go")
	isMarkdown := strings.HasSuffix(relFile, ".md")

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	cand := make([]int, 0, 8)
	for n := 1; sc.Scan(); n++ {
		line := sc.Text()
		if len(line) > 4096 {
			continue
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		// An inventory row is the shape of a markdown table. It names packages; it
		// does not run them, and treating it as evidence is what would make this
		// audit satisfiable by a list of its own subjects.
		if isMarkdown && strings.HasPrefix(trimmed, "|") {
			continue
		}
		if isGo && (strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "*")) {
			continue // a comment naming a package is prose, not a dispatch edge
		}
		cand = ix.candidates(line, cand)
		if len(cand) == 0 {
			continue
		}
		runForm := execRunFormRE.MatchString(line)
		lineClass := class
		if execGoInstallRE.MatchString(line) {
			lineClass = ExecEvidenceInstaller
		}
		for _, i := range cand {
			p := &pkgs[i]
			if dirRel == p.Dir || strings.HasPrefix(dirRel, p.Dir+"/") {
				continue // self-reference: the package's own tree
			}
			if _, done := found[i][lineClass]; done {
				continue
			}
			if !execLineInvokes(line, trimmed, *p, runForm, isGo) {
				continue
			}
			found[i][lineClass] = ExecEvidence{
				Class: lineClass,
				File:  relFile,
				Line:  n,
				Text:  clipEvidence(trimmed),
			}
		}
	}
}

// execLineInvokes applies the three invocation forms. It is the whole self-reference
// exclusion: a line that merely NAMES the package matches none of them.
func execLineInvokes(line, trimmed string, p ExecPackage, runForm, isGo bool) bool {
	// (A) a go build/run/install/test/vet naming the package path or import path.
	if runForm && (containsPathToken(line, p.Dir) || containsPathToken(line, p.ImportPath)) {
		return true
	}
	// (B) an executed binary: ./name or name.exe as a command token.
	if execInvokesBinary(line, p.Binary) {
		return true
	}
	// (C) Go code outside the package naming its path in a string literal — the
	// spawn/registration edge, since a main package cannot be imported.
	if isGo && !strings.HasPrefix(trimmed, "//") {
		if quotedContainsPath(line, p.Dir) || quotedContainsPath(line, p.ImportPath) {
			return true
		}
	}
	return false
}

// execInvokesBinary matches `./name`, `./name.exe` or a bare `name.exe` command
// token. A bare unqualified word is deliberately NOT matched: binary names like
// "codesearch" or "quality" occur constantly in prose, and counting those would
// re-open the vacuity hole this file exists to close.
func execInvokesBinary(line, bin string) bool {
	if bin == "" {
		return false
	}
	for _, form := range []string{"./" + bin, bin + ".exe"} {
		idx := 0
		for {
			i := strings.Index(line[idx:], form)
			if i < 0 {
				break
			}
			at := idx + i
			end := at + len(form)
			before := byte(' ')
			if at > 0 {
				before = line[at-1]
			}
			if !isPathNameByte(before) && (end == len(line) || !isPathNameByte(line[end])) {
				return true
			}
			idx = at + 1
		}
	}
	return false
}

// containsPathToken reports whether tok appears in line as a WHOLE package path.
// The boundary check is what stops "cmd/fak" from matching "./cmd/fakchat" or
// "cmd/fak/main.go" — a file inside a package is not an invocation of the package.
func containsPathToken(line, tok string) bool {
	if tok == "" {
		return false
	}
	idx := 0
	for {
		i := strings.Index(line[idx:], tok)
		if i < 0 {
			return false
		}
		at := idx + i
		end := at + len(tok)
		okBefore := at == 0 || !isPathNameByte(line[at-1])
		okAfter := end == len(line) || (!isPathNameByte(line[end]) && line[end] != '/')
		if okBefore && okAfter {
			return true
		}
		idx = at + 1
	}
}

// isPathNameByte reports whether b can be part of a package's directory name, so a
// token boundary is a byte that cannot: "cmd/fak" inside "cmd/fak-deepswe-runner"
// is not the package cmd/fak.
func isPathNameByte(b byte) bool {
	switch {
	case b >= 'a' && b <= 'z', b >= 'A' && b <= 'Z', b >= '0' && b <= '9':
		return true
	case b == '_' || b == '-' || b == '.':
		return true
	}
	return false
}

// quotedContainsPath reports whether tok appears inside a double-quoted or
// backquoted string on the line — the difference between Go code that WIRES a path
// and Go prose that mentions it.
func quotedContainsPath(line, tok string) bool {
	for _, q := range []byte{'"', '`'} {
		rest := line
		for {
			i := strings.IndexByte(rest, q)
			if i < 0 {
				break
			}
			j := strings.IndexByte(rest[i+1:], q)
			if j < 0 {
				break
			}
			if containsPathToken(rest[i+1:i+1+j], tok) {
				return true
			}
			rest = rest[i+1+j+1:]
		}
	}
	return false
}

func clipEvidence(s string) string {
	const max = 160
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
