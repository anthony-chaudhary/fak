package selfquery

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// grounding.go — the pre-apply diff-grounding scan (#3363). fak already carries a
// queryable self-index of every reference it knows about (the FeatureCard catalog +
// the underlying devindex leaves/verbs). Query ranks that index by relevance but never
// checks reference EXISTENCE, so the classic hallucinated-reference failure — an agent
// proposes a diff that imports a package the tree does not have — is only caught after
// apply, at compile/vet time. ScanPatch closes that gap: it takes a candidate patch,
// extracts the references its ADDED lines introduce, and resolves each against the
// index the patch is measured against — deny-by-structure, the same discipline
// dos_commit_audit uses when it grades a diff by what git recorded, not by the commit
// message. It is READ-ONLY and ADVISORY: it produces a report and neither mutates the
// tree nor blocks apply. Wiring the UNGROUNDED residual into a pre-apply normgate as a
// refusal gate is a follow-on, not this rung.

// ModulePath is fak's Go module path — the prefix that marks an import as one of fak's
// OWN packages (as opposed to the standard library or a third-party module). A
// module-local import is decidable against the self-index; an external import is not
// the index's to judge (see GroundingExternal).
const ModulePath = "github.com/anthony-chaudhary/fak"

// GroundingVerdict classifies one extracted reference against the self-index.
type GroundingVerdict string

const (
	// GroundingPresent: the reference resolves in the current index — a declared lane
	// tree covers its package, or a feature card names it.
	GroundingPresent GroundingVerdict = "PRESENT"
	// GroundingUngrounded: a module-local reference with NO resolution in the current
	// index — the residual a pre-apply gate can refuse on. Honest fence: this means
	// "no resolution in the current index", NOT "cannot exist anywhere".
	GroundingUngrounded GroundingVerdict = "UNGROUNDED"
	// GroundingExternal: an import outside fak's module (standard library or a
	// third-party dependency). The self-index has no opinion on it, so it is surfaced
	// but never counted as ungrounded.
	GroundingExternal GroundingVerdict = "EXTERNAL"
)

// GroundingRef is one reference extracted from a candidate patch's added lines, with
// its witnessed resolution against the index.
type GroundingRef struct {
	Kind    string           `json:"kind"` // the reference class; "import" today
	Ref     string           `json:"ref"`  // the raw reference (an import path)
	File    string           `json:"file"` // the added-side file it appeared in
	Line    int              `json:"line"` // its new-file line number
	Verdict GroundingVerdict `json:"verdict"`
	Detail  string           `json:"detail"` // how it resolved (or why it did not)
}

// GroundingReport is ScanPatch's verdict over a candidate diff: the per-reference
// resolutions plus the folded counts. Every value is derived mechanically from the
// index the patch is measured against, so the report is reproducible: the same diff
// over the same index yields the same report.
type GroundingReport struct {
	References []GroundingRef `json:"references"`
	Present    int            `json:"present"`
	Ungrounded int            `json:"ungrounded"`
	External   int            `json:"external"`
	// ContentOnly is true when the diff added lines but yielded NO groundable
	// references (no imports) — a pure content/text change the structural scan has
	// nothing to resolve. Distinct from an empty diff (no added lines at all), which
	// leaves this false. It is the Martin Loop content-only-diff signal.
	ContentOnly bool `json:"content_only"`
	// Grounded is true when no module-local reference came back UNGROUNDED — the
	// advisory pass/refuse bit a pre-apply gate reads. EXTERNAL imports never flip it.
	Grounded bool `json:"grounded"`
}

// ScanPatch parses a unified diff, extracts the references its ADDED lines introduce,
// and resolves each against the loaded self-index — returning a witnessed
// PRESENT/UNGROUNDED/EXTERNAL verdict per reference plus the content-only-diff signal.
//
// Today it grounds Go IMPORT references (the highest-signal, unambiguous class): a
// module-local import (under [ModulePath]) resolves to PRESENT when a declared lane
// tree covers its package or a feature card names it, else UNGROUNDED; a
// standard-library or third-party import is EXTERNAL, which the index cannot judge.
// Resolution is lane-level: a fabricated whole package (internal/doesnotexist) is
// caught, while a fabricated subpackage under a real lane resolves to that lane —
// deeper symbol/file-path grounding is a deliberate follow-on (#3363), not folded in.
func (c *Catalog) ScanPatch(diff string) GroundingReport {
	added := parseAddedLines(diff)
	var rep GroundingReport
	seen := map[string]bool{}
	sawAdded := false
	for _, al := range added {
		sawAdded = true
		path, ok := importPathOf(al.Text)
		if !ok {
			continue
		}
		key := path + "\x00" + al.File
		if seen[key] {
			continue
		}
		seen[key] = true
		ref := GroundingRef{Kind: "import", Ref: path, File: al.File, Line: al.Line}
		ref.Verdict, ref.Detail = c.resolveImport(path)
		switch ref.Verdict {
		case GroundingPresent:
			rep.Present++
		case GroundingUngrounded:
			rep.Ungrounded++
		case GroundingExternal:
			rep.External++
		}
		rep.References = append(rep.References, ref)
	}
	sort.SliceStable(rep.References, func(i, j int) bool {
		if rep.References[i].File != rep.References[j].File {
			return rep.References[i].File < rep.References[j].File
		}
		if rep.References[i].Line != rep.References[j].Line {
			return rep.References[i].Line < rep.References[j].Line
		}
		return rep.References[i].Ref < rep.References[j].Ref
	})
	rep.ContentOnly = sawAdded && len(rep.References) == 0
	rep.Grounded = rep.Ungrounded == 0
	return rep
}

// resolveImport classifies a single import path against the index. An import outside
// fak's module is EXTERNAL (not the self-index's to judge). A module-local import is
// PRESENT when a declared lane tree covers its package directory or a feature card's
// detail ref names it, and UNGROUNDED otherwise.
func (c *Catalog) resolveImport(path string) (GroundingVerdict, string) {
	if path != ModulePath && !strings.HasPrefix(path, ModulePath+"/") {
		return GroundingExternal, "external import (outside " + ModulePath + "); the self-index does not judge it"
	}
	rel := strings.TrimPrefix(strings.TrimPrefix(path, ModulePath), "/")
	if rel == "" {
		return GroundingPresent, "the module root"
	}
	if c.dev != nil {
		for _, l := range c.dev.Leaves {
			for _, tree := range splitTrees(l.Tree) {
				if treeCovers(tree, rel) {
					return GroundingPresent, "resolves to leaf " + l.Name + " (tree " + tree + ")"
				}
			}
		}
	}
	for _, fc := range c.Cards(PlaneAll) {
		if dr := fc.DetailRef; dr == rel || strings.HasPrefix(dr, rel+"/") {
			return GroundingPresent, "named by feature card " + fc.Name
		}
	}
	return GroundingUngrounded, "no leaf tree or feature card in the current index covers package " + rel
}

// splitTrees splits a leaf's comma-joined tree glob list ("internal/x/**, cmd/x/**")
// into its individual globs, trimming blanks.
func splitTrees(tree string) []string {
	var out []string
	for _, t := range strings.Split(tree, ",") {
		if t = strings.TrimSpace(t); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// treeCovers reports whether a lane glob covers the repo-relative package path rel. A
// subtree glob ("internal/x/**") covers its prefix dir and everything under it; a bare
// path covers itself and files/dirs under it. Wildcards other than a trailing "**" are
// treated as an opaque prefix boundary (the lane taxonomy uses only "**" tails).
func treeCovers(tree, rel string) bool {
	tree = strings.TrimSpace(tree)
	if strings.HasSuffix(tree, "**") {
		prefix := strings.TrimSuffix(strings.TrimSuffix(tree, "**"), "/")
		if prefix == "" || strings.Contains(prefix, "*") {
			return prefix == ""
		}
		return rel == prefix || strings.HasPrefix(rel, prefix+"/")
	}
	if strings.Contains(tree, "*") {
		return false
	}
	return rel == tree || strings.HasPrefix(rel, tree+"/")
}

// importLineRE matches an added line that is a Go import spec — a bare, aliased, dot,
// or blank-imported quoted path, as it appears one-per-line inside an import block or
// on a single `import "..."`. It requires the line to be ONLY the (optionally aliased)
// quoted string, so an ordinary string literal mid-statement (`x := "foo/bar"`) never
// matches.
var importLineRE = regexp.MustCompile(`^\s*(?:import\s+)?(?:(?:[A-Za-z_][A-Za-z0-9_]*|\.|_)\s+)?"([^"]+)"\s*$`)

// importPathOf returns the import path an added line introduces, if the line is a Go
// import spec (a bare or aliased quoted path on its own). The bool is false for any
// other line, so a normal string literal is never mistaken for an import.
func importPathOf(line string) (string, bool) {
	m := importLineRE.FindStringSubmatch(line)
	if m == nil {
		return "", false
	}
	p := strings.TrimSpace(m[1])
	if p == "" || strings.ContainsAny(p, " \t") {
		return "", false
	}
	return p, true
}

// addedLine is one new-side line of a unified diff: the file it lands in, its new-file
// line number, and its text (the leading "+" stripped).
type addedLine struct {
	File string
	Line int
	Text string
}

// hunkHeaderRE matches a unified-diff hunk header and captures the new-file start line.
var hunkHeaderRE = regexp.MustCompile(`^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@`)

// parseAddedLines folds a unified diff into its added (new-side) lines with file and
// new-file line number. A minimal, self-contained walk: a `+++ b/<path>` sets the
// current file, a hunk header resets the new-line counter, a `+` line (not `+++`)
// records (file, line, text) and advances the counter, and context lines advance it.
// (internal/hooks has the same walk but keeps its parser unexported and git-backed, so
// grounding a caller-supplied diff string needs this local, pure copy.)
func parseAddedLines(diff string) []addedLine {
	var out []addedLine
	var file string
	newLine := 0
	haveHunk := false
	for _, raw := range strings.Split(diff, "\n") {
		switch {
		case strings.HasPrefix(raw, "+++ b/"):
			file = raw[len("+++ b/"):]
			haveHunk = false
		case strings.HasPrefix(raw, "+++ "):
			// "+++ /dev/null" (a deletion) — no new-side file.
			file = ""
			haveHunk = false
		case strings.HasPrefix(raw, "@@"):
			if m := hunkHeaderRE.FindStringSubmatch(raw); m != nil {
				if n, err := strconv.Atoi(m[1]); err == nil {
					newLine = n
					haveHunk = true
				}
			}
		case strings.HasPrefix(raw, "+"):
			if file == "" || !haveHunk {
				continue
			}
			out = append(out, addedLine{File: file, Line: newLine, Text: raw[1:]})
			newLine++
		case strings.HasPrefix(raw, "-"):
			// removed line: no new-side advance.
		case strings.HasPrefix(raw, " "):
			if haveHunk {
				newLine++
			}
		}
	}
	return out
}
