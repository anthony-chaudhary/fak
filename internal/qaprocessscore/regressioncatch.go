// Package qaprocessscore holds the QA-process scorecard's KPI folds -- the
// E-testing-quality track's "is our test process honest?" signals. Each KPI is a pure
// function over git/history facts that returns a pkg/scorecard.KPI, so the fold is
// deterministic and fixture-testable with no toolchain in the loop. This package is the
// Go home the qa-process dogfood keys (qa-process-score/*) fold into; per the scorecard
// family convention the per-KPI logic lives here and the shared grade/fold/markdown
// skeleton stays in pkg/scorecard.
package qaprocessscore

import (
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/brittleness"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// RegressionCatchKey is both the KPI key and the qa-process dogfood finding name
// (qa-process-score/regression-catch). #3841.
const RegressionCatchKey = "regression_catch"

// regressionGapClass is the closed-vocabulary work-list token each gap renders under --
// the same grep-able "<CLASS> <ref>: <detail>" discipline internal/brittleness uses.
const regressionGapClass = "REGRESSION_CATCH_GAP"

// RegressionGap is one reverted landing that came back off the trunk with NO accompanying
// regression test. A revert is a natural experiment: it names a bug that reached master.
// Asking "would a test have caught this?" turns each such revert into the highest-value,
// worst-first place to add a test.
type RegressionGap struct {
	RevertSHA string   `json:"revert_sha"`
	Subject   string   `json:"subject"`
	Packages  []string `json:"packages"` // reverted code packages with no co-located _test.go change
}

// workItem renders a gap as a single closed-vocabulary work-list entry.
func (g RegressionGap) workItem() string {
	return fmt.Sprintf("%s %s: revert %q left %s with no accompanying _test.go change -- add a regression test in %s",
		regressionGapClass, g.RevertSHA, g.Subject, strings.Join(g.Packages, ", "), g.Packages[0])
}

// RegressionGaps returns every reverted landing whose reverted code packages saw NO
// accompanying regression test, worst-first (newest revert first, the window's own order),
// alongside examined -- the count of code-reverting reverts inspected, which is the
// denominator of the catch rate. It is the structured half of RegressionCatch, split out so
// the dispatcher (dispatch.go) can route each gap to its owning package as a GitHub issue
// without re-deriving reverts; RegressionCatch renders the same gaps into KPI Defects.
//
// commits is the git-log window NEWEST-FIRST, exactly as internal/brittleness reads it
// (readCommitWindowForBrittleness). "The revert or its follow-up" is read as: the revert
// commit itself, or any NEWER commit in the window (a re-land-with-test that landed after
// the revert). Reverts are identified via brittleness.DetectReverts so "what counts as a
// revert" has one source of truth. The heuristic is deliberately coarse -- a same-package
// _test.go change, per the issue -- and does NOT try to prove the test reproduces the bug
// (that is mutation/repro territory, #3845).
func RegressionGaps(commits []brittleness.Commit) (gaps []RegressionGap, examined int) {
	revertSHAs := map[string]bool{}
	for _, f := range brittleness.DetectReverts(commits) {
		revertSHAs[f.Ref] = true
	}
	for i, c := range commits {
		if !revertSHAs[c.SHA] {
			continue
		}
		pkgs := revertedCodePackages(c.Files)
		if len(pkgs) == 0 {
			continue // a revert that touched no Go code package (docs/config) -- nothing to cover
		}
		examined++
		// commits[:i+1] is the revert (index i) plus everything newer than it (indices < i),
		// i.e. the revert and its follow-ups in this newest-first window.
		if hasAccompanyingTest(commits[:i+1], pkgs) {
			continue
		}
		gaps = append(gaps, RegressionGap{
			RevertSHA: c.SHA,
			Subject:   strings.TrimSpace(c.Subject),
			Packages:  pkgs,
		})
	}
	return gaps, examined
}

// RegressionCatch folds the commit window into the regression_catch HARD KPI: for every
// reverted landing, did a regression test land with it or in a follow-up? A revert whose
// reverted code packages saw no co-located _test.go change is one unit of debt -- the bug
// reached the trunk and nothing was added to catch its return. A window with no
// code-reverting reverts scores 100: nothing to catch.
func RegressionCatch(commits []brittleness.Commit) scorecard.KPI {
	gaps, examined := RegressionGaps(commits)

	caught := examined - len(gaps)
	score := 100.0
	detail := "no code-reverting reverts in window"
	if examined > 0 {
		score = scorecard.Round1(100 * float64(caught) / float64(examined))
		detail = fmt.Sprintf("%d/%d reverts landed with an accompanying regression test (%.1f%%)",
			caught, examined, score)
	}

	defects := make([]string, 0, len(gaps))
	for _, g := range gaps {
		defects = append(defects, g.workItem())
	}
	return scorecard.KPI{
		Key:     RegressionCatchKey,
		Group:   "regression",
		Score:   score,
		Detail:  detail,
		Defects: defects,
	}
}

// revertedCodePackages returns the sorted, unique package dirs of the non-test .go files a
// commit touched. Test files are excluded: a revert that only moved test code reverts no
// production behavior to cover. Paths are git's forward-slash form, so path.Dir (not
// filepath.Dir) is correct on every OS.
func revertedCodePackages(files []string) []string {
	seen := map[string]bool{}
	for _, f := range files {
		if !strings.HasSuffix(f, ".go") || strings.HasSuffix(f, "_test.go") {
			continue
		}
		seen[path.Dir(f)] = true
	}
	pkgs := make([]string, 0, len(seen))
	for p := range seen {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)
	return pkgs
}

// hasAccompanyingTest reports whether any commit in window touched a _test.go file living
// in one of the reverted packages -- the coarse "a regression test landed for this" signal.
func hasAccompanyingTest(window []brittleness.Commit, pkgs []string) bool {
	pkgSet := map[string]bool{}
	for _, p := range pkgs {
		pkgSet[p] = true
	}
	for _, c := range window {
		for _, f := range c.Files {
			if strings.HasSuffix(f, "_test.go") && pkgSet[path.Dir(f)] {
				return true
			}
		}
	}
	return false
}
