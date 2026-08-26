package qaprocessscore

// dispatch.go is the fan-out half of the qa-process card (#3844): it turns each HARD
// qa_process_debt gap -- a reverted landing with no regression test (#3841), a package below
// its coverage ratchet (#3834) -- into a deduped, dispatchable GitHub issue by mapping onto the
// EXISTING internal/dogfoodissues backlog bridge, the same ToActionItem -> BuildPlan -> Sync
// path internal/unwiredscore and internal/propagationscore established. There is no new
// gh-issue code here.
//
// The dedup Key is content-addressed on the STABLE {kpi, class, ref} triple -- the revert SHA
// or the package path -- and deliberately NOT the rendered Detail, which embeds run-varying
// coverage percentages. So a re-run UPDATES the same issue in place instead of opening a
// duplicate, and a recurrence after a "fixed" close re-files the same key. Each generated child
// hangs under the epic (ParentRef #3831) and routes to the lane that owns the gap's files, so
// the worker who closes it is the one who owns the code.

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/brittleness"
	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// qaProcessEpicRef is the tracker epic these gaps fan out under (#3831, "is our test process
// honest?"). Each generated issue carries it as ParentRef so the children hang under the epic.
const qaProcessEpicRef = "#3831"

// Gap is one HARD qa_process_debt defect in dispatchable, structured form: the KPI it belongs
// to, its closed-vocabulary class token, the STABLE content-address anchor (revert SHA / package
// path) the dedup key is built from, the run-varying human Detail, the repo Paths it routes to,
// and a size grade. It is the union shape the two KPI folds (regression_catch, coverage_discipline)
// project their gaps into so the dispatcher stays KPI-agnostic.
type Gap struct {
	KPI    string   // owning KPI key (RegressionCatchKey, CoverageDisciplineKey)
	Class  string   // closed-vocabulary work-list token (REGRESSION_CATCH_GAP, COVERAGE_BELOW_FLOOR, ...)
	Ref    string   // STABLE dedup anchor: the revert SHA or the package path (never the drifting Detail)
	Detail string   // the human-readable, possibly run-varying work-list line
	Paths  []string // repo path globs the gap routes to (the owning package)
	Grade  string   // size/severity grade for the issue header
}

// Key is the CONTENT-STABLE dedup identity: {kpi, class, ref}. It excludes the rendered Detail
// (which embeds run-varying coverage percentages), so a re-run of the same gap UPDATES the same
// issue in place instead of opening a duplicate.
func (g Gap) Key() string {
	return "qa-process-debt/" + g.KPI + "/" + slug(g.Class+"-"+g.Ref)
}

// Title is the one-line issue subject.
func (g Gap) Title() string {
	return "close qa-process debt: " + g.Class + " (" + g.Ref + ")"
}

// ToActionItem maps a gap onto the EXISTING dogfoodissues.ActionItem so the GitHub-issue half is
// pure reuse (IssueBody/BuildPlanWithOptions/Sync). Routing rides Paths (the owning package), so
// dispatch lands the issue in the lane that owns the leaking code. All scope fields the
// dispatchability review requires (in/out scope, done condition, witness, acceptance gate,
// closure binding) are populated, so a dry-run plans the item rather than skipping it.
func (g Gap) ToActionItem(evidencePath string) dogfoodissues.ActionItem {
	host := "the affected package"
	if len(g.Paths) > 0 {
		host = strings.TrimSuffix(g.Paths[0], "/**")
	}
	return dogfoodissues.ActionItem{
		Key:          g.Key(),
		Title:        g.Title(),
		SourceProbe:  "qa-process scorecard",
		ScoreName:    g.KPI,
		Score:        g.Class,
		Grade:        g.Grade,
		DebtName:     DebtKey,
		DebtCount:    1,
		EvidencePath: evidencePath,
		NextAction:   g.NextAction(host),
		Finding:      g.Key(),
		ParentRef:    qaProcessEpicRef,
		CurrentState: fmt.Sprintf("The qa-process scorecard folded a HARD %s defect: %s. This is one unit of qa_process_debt -- a real, in-tree-mendable hole in the test process (not a failing test), captured worst-first.",
			g.KPI, g.Detail),
		WhyNow:         "A green suite is not an honest suite: a bug that reached the trunk with no regression test, or a package that slipped below its coverage ratchet, is a hole the next regression falls through. Nothing breaks while the gap sits, so only a tracked issue surfaces it.",
		WorkingSpine:   g.WorkingSpine(host),
		WorkUnit:       "leaf",
		ExpectedSteps:  4,
		Assumptions:    []string{"The named defect points at a single owning package, so the fix is a scoped, one-worker change its owner can make from the code."},
		ConfusionRisks: []string{"Do not batch multiple qa-process gaps into one worker issue.", "Do not weaken the KPI's floor/ratchet or delete the reverted code to make the gap disappear -- add the missing test / raise the coverage."},
		Coordination:   []string{"One generated issue owns one qa-process gap (stable marker key)."},
		Trigger:        "fak score qa-process reports " + g.Class + " for " + g.Ref + ".",
		BatchPolicy:    "One issue per stable qa-process gap key; reruns update the existing marker instead of opening duplicates.",
		InScope:        g.InScope(host),
		OutOfScope:     "Do not change the qa-process KPI thresholds or grading, and do not touch other qa-process gaps in the same change.",
		DoneCondition:  "A re-run of `fak score qa-process --json` no longer lists the `" + g.Key() + "` gap.",
		Witness:        "fak score qa-process --json",
		AcceptanceGate: "go build ./... && go test ./internal/qaprocessscore/...",
		Lane:           "",
		Paths:          g.Paths,
		Labels:         []string{"track/E-testing-quality", "testing"},
		BoundaryNotes:  []string{"Public test-process evidence only; no private or lab-local artifacts."},
		ClosureBinding: "Resolving commit cites `#N` in the subject and carries a `(fak <leaf>)` trailer for the package's lane.",
	}
}

type gapInstructions struct {
	nextAction   string
	workingSpine string
	inScope      string
}

// instructions keeps the three operator-facing fields aligned on one KPI decision. A
// new debt class cannot accidentally update the action while leaving its working spine
// or scope on the generic fallback.
func (g Gap) instructions(host string) gapInstructions {
	switch g.KPI {
	case RegressionCatchKey:
		return gapInstructions{
			nextAction:   "Add a regression test in " + host + " that reproduces the bug the revert named.",
			workingSpine: "Write a failing test in " + host + " that reproduces the reverted bug, confirm it fails on the reverted code, then land it alongside (or ahead of) the re-land.",
			inScope:      "Add a regression test in " + host + " covering the reverted behavior.",
		}
	case CoverageDisciplineKey:
		return gapInstructions{
			nextAction:   "Raise " + host + " back over its coverage ratchet with tests for the uncovered statements.",
			workingSpine: "Identify the uncovered statements in " + host + " (go tool cover -func), add tests that exercise them, and confirm coverage clears the ratchet.",
			inScope:      "Add tests raising " + host + " over its coverage floor/ratchet.",
		}
	case FlakeQuarantineKey:
		return gapInstructions{
			nextAction:   "Make the quarantined test in " + host + " deterministic (or skip it behind a tracking ticket); do not raise the rerun budget.",
			workingSpine: "Reproduce the non-determinism in " + host + " (run the test in a loop / under -race), fix the shared state, ordering, timing or seed that causes it, then confirm it holds green without --rerun-fail.",
			inScope:      "De-flake the one named test identity in " + host + " so it passes deterministically without a rerun.",
		}
	}
	return gapInstructions{
		nextAction:   "Close the qa-process gap in " + host + ".",
		workingSpine: "Fix the underlying qa-process gap in " + host + " and re-run the card.",
		inScope:      "Close the named qa-process gap in " + host + ".",
	}
}

// NextAction / WorkingSpine / InScope are per-KPI so the generated issue tells its worker the
// concrete fix (add a regression test vs raise coverage), not a generic "close the defect".
func (g Gap) NextAction(host string) string { return g.instructions(host).nextAction }

// WorkingSpine describes the shape of the fix for the issue body.
func (g Gap) WorkingSpine(host string) string { return g.instructions(host).workingSpine }

// InScope is the one-package fix boundary for the issue body.
func (g Gap) InScope(host string) string { return g.instructions(host).inScope }

// Gaps collects every HARD qa_process_debt gap worst-first: regression gaps first (a bug reached
// the trunk -- the loudest process leak), then coverage gaps (below-floor before regressed). cov
// is empty when no --coverprofile was supplied, in which case only regression gaps are returned.
func Gaps(commits []brittleness.Commit, cov map[string]float64, base CoverageBaseline) []Gap {
	var gaps []Gap

	regr, _ := RegressionGaps(commits)
	for _, rg := range regr {
		if len(rg.Packages) == 0 {
			continue // workItem indexes Packages[0]; a package-less revert routes nowhere
		}
		paths := make([]string, 0, len(rg.Packages))
		for _, p := range rg.Packages {
			paths = append(paths, p+"/**")
		}
		gaps = append(gaps, Gap{
			KPI:    RegressionCatchKey,
			Class:  regressionGapClass,
			Ref:    rg.RevertSHA,
			Detail: rg.workItem(),
			Paths:  paths,
			Grade:  "D",
		})
	}

	for _, cg := range CoverageGaps(cov, base) {
		grade := "C"
		if cg.Class == coverageBelowFloorClass {
			grade = "D" // a package under the absolute floor is worse than one that merely slipped
		}
		gaps = append(gaps, Gap{
			KPI:    CoverageDisciplineKey,
			Class:  cg.Class,
			Ref:    cg.Pkg,
			Detail: cg.workItem(),
			Paths:  []string{cg.Pkg + "/**"},
			Grade:  grade,
		})
	}
	return gaps
}

// ActionItems maps a set of gaps onto dogfoodissues.ActionItems, the backlog bridge input.
func ActionItems(gaps []Gap, evidencePath string) []dogfoodissues.ActionItem {
	items := make([]dogfoodissues.ActionItem, 0, len(gaps))
	for _, g := range gaps {
		items = append(items, g.ToActionItem(evidencePath))
	}
	return items
}

// slug lowercases and dash-collapses an arbitrary string for a stable, marker-safe issue key
// (the dogfoodissues marker regex forbids whitespace and `>`). Mirrors internal/unwiredscore.slug.
func slug(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "gap"
	}
	return out
}
