package antipattern

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/orphanscan"
	"github.com/anthony-chaudhary/fak/internal/unwiredscore"
)

// scanRoots are the module subtrees walked for orphaned funcs. Mirrors unwiredscore's set;
// tools/ is Python and .git/.dos are not code.
var antipatternScanRoots = []string{"cmd", "internal", "pkg", "experiments"}

// Collect runs every wired detector against the tree at root plus the git-history window
// `commits`, and returns the flat finding list with the per-class universe counts Fold uses
// for its legacy KPI scores. It is the impure boundary (filesystem reads); the pure Fold and
// DetectRedundantRework do the judging. A nil/empty commits slice simply yields no
// REDUNDANT_REWORK findings -- the card degrades honestly when git history is unavailable.
func Collect(root string, commits []Commit) ([]Finding, map[Class]int) {
	var findings []Finding
	universe := map[Class]int{}

	findings = append(findings, DetectRedundantRework(commits)...)

	up, upUniverse := unwiredPkgFindings(root)
	findings = append(findings, up...)
	universe[ClassUnwiredPkg] = upUniverse

	findings = append(findings, orphanFuncFindings(root)...)

	SortFindings(findings)
	return findings, universe
}

// unwiredPkgFindings folds internal/unwiredscore's orphaned-package scan into UNWIRED_PKG
// findings, weighted by stranded source lines (biggest investment first). The second return
// is the candidate-package universe, for the KPI's clean fraction.
func unwiredPkgFindings(root string) ([]Finding, int) {
	pkgs := unwiredscore.Scan(root)
	var out []Finding
	for _, p := range pkgs {
		if !p.Unwired() {
			continue
		}
		tested := "no"
		if p.HasTest {
			tested = "yes"
		}
		out = append(out, Finding{
			Class:  ClassUnwiredPkg,
			Ref:    p.Dir,
			Detail: fmt.Sprintf("code-complete (%d source line(s), tests: %s) but imported by no .go file in the module", p.SourceLines, tested),
			Weight: p.SourceLines,
		})
	}
	return out, len(pkgs)
}

// orphanFuncFindings walks each Go package dir under the scan roots and folds
// internal/orphanscan's per-package result into ORPHAN_FUNC findings. Each orphan weighs 1
// (a syntactic, count-based debt).
func orphanFuncFindings(root string) []Finding {
	dirs := goPackageDirs(root)
	var out []Finding
	for _, dir := range dirs {
		rel := relSlash(root, dir)
		orphans, err := orphanscan.ScanDir(dir, rel)
		if err != nil {
			continue // an unreadable dir contributes nothing, never aborts the scan
		}
		for _, o := range orphans {
			out = append(out, Finding{
				Class:  ClassOrphanFunc,
				Ref:    fmt.Sprintf("%s:%d", o.File, o.Line),
				Detail: fmt.Sprintf("func %s is defined but never referenced in its package", o.Name),
				Weight: 1,
			})
		}
	}
	return out
}

// goPackageDirs returns every directory under the scan roots that contains at least one
// non-testdata .go file -- the set of Go packages orphanscan is run over.
func goPackageDirs(root string) []string {
	seen := map[string]struct{}{}
	var dirs []string
	for _, sub := range antipatternScanRoots {
		base := filepath.Join(root, sub)
		_ = filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if d.Name() == "testdata" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(d.Name(), ".go") {
				return nil
			}
			dir := filepath.Dir(path)
			if _, ok := seen[dir]; ok {
				return nil
			}
			seen[dir] = struct{}{}
			dirs = append(dirs, dir)
			return nil
		})
	}
	return dirs
}

func relSlash(root, path string) string {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		rel = path
	}
	return filepath.ToSlash(rel)
}
