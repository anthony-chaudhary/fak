// Package workflowaudit classifies every git-branch / tag reference in the project's
// GitHub Actions workflows against the branch-role contract (internal/branchrole), so the
// dev->main front-door migration (#1697 / #1701) has a checkable map of what each branch
// filter is FOR -- and a gate that reds the moment a new, unclassified development-path
// `main`/`master` reference is introduced.
//
// The repo is migrating from "main is the hot shared trunk" to a long-lived `dev`
// integration branch plus a clean `main` release front-door. Branch references are
// scattered across .github/workflows/*.yml in three idioms:
//
//   - branches: filters on push / pull_request triggers      -> ClassDevelopment
//   - tags: filters (e.g. "v*")                              -> ClassTag
//   - if: github.ref_name == 'main' / github.ref == refs/... -> ClassReleaseFrontDoor
//
// plus hidden development-trunk assumptions inside run-step shells (the canonical one is
// `gh run list --branch main` that fetches the benchmark baseline regardless of the
// current branch). Every reference is classified; anything that is a development-path
// branch matching no configured role and not on the intentional allowlist is
// ClassUnclassified -- the gate fires on those.
//
// The package is tier-1: it imports only internal/branchrole (also tier-1) and stdlib, so
// it stays off the hot path and can back both the `fak workflow-audit` CLI and the live-
// tree regression test.
package workflowaudit

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
)

// RefClass is the closed set of roles a workflow branch/tag reference can play.
type RefClass string

const (
	// ClassDevelopment is a push/PR branch filter naming a configured development branch.
	ClassDevelopment RefClass = "development"
	// ClassReleaseFrontDoor is an if-gate that runs a job only on the release / public
	// front-door branch (live posting, baseline upload, release cadence).
	ClassReleaseFrontDoor RefClass = "release-front-door"
	// ClassTag is a tags: filter -- the tag-driven release surface.
	ClassTag RefClass = "tag"
	// ClassLegacy is an intentional compatibility reference to an old trunk name
	// (`master`, `fak-v0.1`) that is not a currently-configured role.
	ClassLegacy RefClass = "legacy"
	// ClassUnclassified is a development-path reference matching no role and not on the
	// intentional allowlist -- the regression the gate exists to catch.
	ClassUnclassified RefClass = "unclassified"
)

// Reference kinds -- the syntactic shape the reference took in the YAML.
const (
	KindBranchesFilter = "branches-filter" // branches: [ ... ] / branches:\n  - X
	KindTagsFilter     = "tags-filter"     // tags: [ ... ]
	KindRefNameGate    = "ref-name-gate"   // github.ref_name == 'X'
	KindRefHeadsGate   = "ref-heads-gate"  // github.ref == 'refs/heads/X'
	KindHiddenShellRef = "hidden-shell-ref"
)

// Ref is one classified branch/tag reference found in a workflow file.
type Ref struct {
	File  string   `json:"file"`  // workflow file name (base, e.g. "ci.yml")
	Line  int      `json:"line"`  // 1-indexed line the reference sits on
	Kind  string   `json:"kind"`  // one of the Kind* constants
	Raw   string   `json:"raw"`   // the literal token, e.g. "main", "master", "fak-v0.1", "v*"
	Class RefClass `json:"class"` // the classification verdict
}

// Report is the full audit over a workflows directory.
type Report struct {
	Roles        branchrole.Roles `json:"roles"`        // contract the audit classified against
	Refs         []Ref            `json:"refs"`         // every reference, sorted by (file, line)
	ByClass      map[RefClass]int `json:"by_class"`     // count per class
	Unclassified []Ref            `json:"unclassified"` // the development-path refs the gate fails on
	Files        int              `json:"files"`        // number of workflow files scanned
}

// Clean reports whether the audit found no unclassified development-path references.
func (r Report) Clean() bool { return len(r.Unclassified) == 0 }

// classCount sums per-class for ByClass, in a stable order for rendering.
var renderOrder = []RefClass{ClassDevelopment, ClassReleaseFrontDoor, ClassTag, ClassLegacy, ClassUnclassified}

// --- regexes for the three reference idioms (verified against the real tree) ---

var (
	// branches: [main, master, fak-v0.1]  (inline list form)
	reBranchesInline = regexp.MustCompile(`^\s*branches:\s*\[([^\]]*)\]`)
	// tags: ["v*"]  (inline list form)
	reTagsInline = regexp.MustCompile(`^\s*tags:\s*\[([^\]]*)\]`)
	// a `branches:` / `tags:` key that opens a YAML block list on following lines
	reBranchesKey = regexp.MustCompile(`^\s*branches:\s*$`)
	reTagsKey     = regexp.MustCompile(`^\s*tags:\s*$`)
	// `  - main` style YAML list item
	reListItem = regexp.MustCompile(`^\s*-\s*(.+?)\s*$`)
	// github.ref_name == 'main'   (may appear several times on one if: line)
	reRefName = regexp.MustCompile(`github\.ref_name\s*==\s*'([^']+)'`)
	// github.ref == 'refs/heads/main'
	reRefHeads = regexp.MustCompile(`github\.ref\s*==\s*'refs/heads/([^']+)'`)
	// hidden shell ref: `--branch main` inside a run: step
	reShellBranch = regexp.MustCompile(`--branch\s+([A-Za-z0-9._/-]+)`)
)

// Audit scans every *.yml under workflowsDir, classifies each branch/tag reference
// against roles, and returns the folded Report. allow names intentional non-role
// references that must not red the gate (legacy compat, the documented hidden baseline
// fetch). It never panics: an unreadable file is skipped with its name recorded nowhere
// (the directory read error is the only hard error).
func Audit(workflowsDir string, roles branchrole.Roles, allow Allowlist) (Report, error) {
	entries, err := os.ReadDir(workflowsDir)
	if err != nil {
		return Report{}, fmt.Errorf("workflowaudit: read %s: %w", workflowsDir, err)
	}
	rep := Report{Roles: roles, ByClass: map[RefClass]int{}}
	roleSet := roleBranchSet(roles)

	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".yml") && !strings.HasSuffix(name, ".yaml") {
			continue
		}
		files = append(files, name)
	}
	sort.Strings(files)

	for _, name := range files {
		refs, scanErr := scanFile(filepath.Join(workflowsDir, name), name, roleSet, allow)
		if scanErr != nil {
			return Report{}, scanErr
		}
		rep.Refs = append(rep.Refs, refs...)
		rep.Files++
	}

	sort.SliceStable(rep.Refs, func(i, j int) bool {
		if rep.Refs[i].File != rep.Refs[j].File {
			return rep.Refs[i].File < rep.Refs[j].File
		}
		return rep.Refs[i].Line < rep.Refs[j].Line
	})
	for _, r := range rep.Refs {
		rep.ByClass[r.Class]++
		if r.Class == ClassUnclassified {
			rep.Unclassified = append(rep.Unclassified, r)
		}
	}
	return rep, nil
}

// roleBranchSet is the set of branch names that currently play a configured role.
func roleBranchSet(roles branchrole.Roles) map[string]bool {
	return map[string]bool{
		strings.TrimSpace(roles.DevelopmentBranch): true,
		strings.TrimSpace(roles.ReleaseBranch):     true,
		strings.TrimSpace(roles.ReleaseSource):     true,
		strings.TrimSpace(roles.PublicFrontDoor):   true,
	}
}

// scanFile reads one workflow file line-by-line and emits classified references.
func scanFile(path, name string, roleSet map[string]bool, allow Allowlist) ([]Ref, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("workflowaudit: open %s: %w", path, err)
	}
	defer f.Close()

	var out []Ref
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	lineNo := 0
	inBranchesBlock, inTagsBlock := false, false
	for sc.Scan() {
		lineNo++
		line := sc.Text()

		// A YAML block list opened by a bare `branches:` / `tags:` key continues until a
		// line that is not a list item. Handle the continuation first.
		if inBranchesBlock || inTagsBlock {
			if m := reListItem.FindStringSubmatch(line); m != nil && !strings.Contains(line, ":") {
				tok := unquote(m[1])
				kind := KindBranchesFilter
				if inTagsBlock {
					kind = KindTagsFilter
				}
				out = append(out, classifyToken(name, lineNo, kind, tok, roleSet, allow))
				continue
			}
			inBranchesBlock, inTagsBlock = false, false
			// fall through: this line may itself open a new construct
		}

		if m := reBranchesInline.FindStringSubmatch(line); m != nil {
			for _, tok := range splitList(m[1]) {
				out = append(out, classifyToken(name, lineNo, KindBranchesFilter, tok, roleSet, allow))
			}
			continue
		}
		if m := reTagsInline.FindStringSubmatch(line); m != nil {
			for _, tok := range splitList(m[1]) {
				out = append(out, classifyToken(name, lineNo, KindTagsFilter, tok, roleSet, allow))
			}
			continue
		}
		if reBranchesKey.MatchString(line) {
			inBranchesBlock = true
			continue
		}
		if reTagsKey.MatchString(line) {
			inTagsBlock = true
			continue
		}
		for _, m := range reRefName.FindAllStringSubmatch(line, -1) {
			out = append(out, classifyToken(name, lineNo, KindRefNameGate, m[1], roleSet, allow))
		}
		for _, m := range reRefHeads.FindAllStringSubmatch(line, -1) {
			out = append(out, classifyToken(name, lineNo, KindRefHeadsGate, m[1], roleSet, allow))
		}
		for _, m := range reShellBranch.FindAllStringSubmatch(line, -1) {
			out = append(out, classifyToken(name, lineNo, KindHiddenShellRef, m[1], roleSet, allow))
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("workflowaudit: scan %s: %w", path, err)
	}
	return out, nil
}

// classifyToken assigns a RefClass to one reference token.
//
// Decision order:
//  1. A tags-filter is always ClassTag (the release surface), regardless of token.
//  2. An if-gate (ref-name / ref-heads) on the release or front-door branch is
//     ClassReleaseFrontDoor.
//  3. A token naming a currently-configured role is ClassDevelopment when it arrived as a
//     branches-filter, else ClassReleaseFrontDoor (gate on the role branch).
//  4. An allowlisted (file, token) reference is ClassLegacy -- an intentional, reviewed
//     compatibility or hidden-baseline reference.
//  5. Everything else on a development path is ClassUnclassified.
func classifyToken(file string, line int, kind, raw string, roleSet map[string]bool, allow Allowlist) Ref {
	tok := strings.TrimSpace(raw)
	ref := Ref{File: file, Line: line, Kind: kind, Raw: tok}

	switch kind {
	case KindTagsFilter:
		ref.Class = ClassTag
		return ref
	case KindRefNameGate, KindRefHeadsGate:
		// An if-gate that runs a job only on a role branch is the front door by
		// construction (live posting / baseline upload / release cadence).
		if roleSet[tok] {
			ref.Class = ClassReleaseFrontDoor
			return ref
		}
	case KindBranchesFilter:
		if roleSet[tok] {
			ref.Class = ClassDevelopment
			return ref
		}
	case KindHiddenShellRef:
		// A shell --branch ref is never a trigger filter; it is only acceptable when the
		// allowlist documents it (the bench baseline fetch). Fall through to allowlist.
	}

	if allow.Has(file, tok) {
		ref.Class = ClassLegacy
		return ref
	}
	ref.Class = ClassUnclassified
	return ref
}

// splitList splits an inline YAML list body ("main, master, fak-v0.1") into tokens.
func splitList(body string) []string {
	var out []string
	for _, part := range strings.Split(body, ",") {
		tok := unquote(strings.TrimSpace(part))
		if tok != "" {
			out = append(out, tok)
		}
	}
	return out
}

// unquote strips surrounding single or double quotes from a YAML scalar.
func unquote(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}
