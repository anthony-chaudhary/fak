package unwiredscore

// dispatch.go is the fan-out half: it turns each unwired package (code complete, imported by
// nothing) into a deduped, dispatchable GitHub issue by mapping onto the EXISTING
// internal/dogfoodissues backlog bridge -- the same ToActionItem -> BuildPlan -> Sync path
// internal/propagationscore and internal/guardroute established, so there is no new gh-issue
// code here. The stable Key is content-addressed (the package dir, no timestamp), so a re-run
// UPDATES the same issue in place instead of opening a duplicate, and a recurrence after a
// "fixed" close re-files the same key. This is the "10 auto-created tickets beat the operator
// remembering" mechanism: one tracked, scoped issue per orphan, routed to the lane that owns
// the package's files -- the worker who wires or retires it is the one who owns the code.

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// Gaps re-scans the tree and returns every unwired (debt) package, worst-first (Scan's order).
func Gaps(root string) []Pkg {
	var gaps []Pkg
	for _, p := range Scan(root) {
		if p.Unwired() {
			gaps = append(gaps, p)
		}
	}
	return gaps
}

// base returns the last path segment of a package dir ("internal/foo/bar" -> "bar").
func base(dir string) string {
	if i := strings.LastIndexByte(dir, '/'); i >= 0 {
		return dir[i+1:]
	}
	return dir
}

// Key is the CONTENT-STABLE dedup identity (no run-id / timestamp): the package dir.
func (p Pkg) Key() string { return "unwired-debt/" + slug(p.Dir) }

// Title is the one-line issue subject.
func (p Pkg) Title() string {
	return "wire or retire internal/" + base(p.Dir) + " (code-complete, imported by nothing)"
}

// ToActionItem maps an unwired package onto the EXISTING dogfoodissues.ActionItem so the
// GitHub-issue half is pure reuse (IssueBody/BuildPlanWithOptions/Sync). evidencePath is the
// scorecard JSON the issue cites. Routing rides Paths (the package dir), so dispatch lands the
// issue in the lane that owns the code.
func (p Pkg) ToActionItem(evidencePath string) dogfoodissues.ActionItem {
	tested := "no tests"
	if p.HasTest {
		tested = "has tests"
	}
	return dogfoodissues.ActionItem{
		Key:          p.Key(),
		Title:        p.Title(),
		SourceProbe:  "unwired-scorecard",
		ScoreName:    "source_lines_stranded",
		Score:        fmt.Sprintf("%d", p.SourceLines),
		Grade:        strandedGrade(p),
		DebtName:     DebtKey,
		DebtCount:    1,
		EvidencePath: evidencePath,
		NextAction:   "Wire internal/" + base(p.Dir) + " into a default path, or retire it.",
		Finding:      p.Key(),
		ParentRef:    "fak unwired-scorecard",
		CurrentState: fmt.Sprintf("The unwired scorecard found internal/%s is code-complete (%d source line(s), %s) but its import path %q is referenced by no .go file anywhere in the module -- it ships in no binary and runs on no default path. It passes every architest layering rule; being imported by nothing is exactly the invariant architest does not catch.",
			base(p.Dir), p.SourceLines, tested, p.ImportPath),
		WhyNow:         "Code-complete-but-unwired is where 10x-100x of the productization work silently stalls: the package delivers zero durable value until it is wired to a verb, dogfooded, and benchmarked -- or retired. Nothing breaks while it rots, so only a tracked issue surfaces it.",
		WorkingSpine:   "Decide wire-or-retire: either add a cmd/fak verb (or an existing default path) that imports and calls internal/" + base(p.Dir) + ", or delete the package by explicit path if it is obsolete.",
		WorkUnit:       "leaf",
		ExpectedSteps:  4,
		Assumptions:    []string{"The package's exported API names its intended use, so wire-vs-retire is a judgment its owner can make from the code."},
		ConfusionRisks: []string{"Do not batch multiple orphaned packages into one worker issue.", "If this is a public test/fixture harness meant for external consumers' _test.go, the fix is an AllowUnwired entry in internal/unwiredscore with a reason -- not a fake caller."},
		Coordination:   []string{"One generated issue owns one orphaned package."},
		Trigger:        "Unwired scorecard reports internal/" + base(p.Dir) + " as code-complete but imported by nothing.",
		BatchPolicy:    "One issue per unwired package dir; reruns update by stable marker.",
		InScope:        "Wire internal/" + base(p.Dir) + " to a default path (a cmd/fak shell that imports and calls it, or a blank-import into internal/registrations if it self-registers), OR delete it by explicit path, OR (public test harness only) allow-list it with a documented reason.",
		OutOfScope:     "Do not add a caller whose only purpose is to defeat the detector, change the unwired scorecard's KPI or grade thresholds, or touch other orphaned packages in the same change.",
		DoneCondition:  "A re-run of `fak unwired-scorecard --json` no longer lists internal/" + base(p.Dir) + " as unwired (the `" + p.Key() + "` gap is gone).",
		Witness:        "fak unwired-scorecard --json | grep -c " + base(p.Dir),
		AcceptanceGate: "go build ./... && go test ./internal/unwiredscore",
		Lane:           "",
		Paths:          []string{p.Dir + "/**"},
		Labels:         []string{"unwired-debt", "tech-debt"},
		BoundaryNotes:  []string{"Public package-wiring evidence only; no private or lab-local artifacts."},
		ClosureBinding: "Resolving commit cites `#N` in the subject and carries a `(fak <leaf>)` trailer for the package's lane.",
	}
}

// ActionItems maps a set of unwired packages onto dogfoodissues.ActionItems, the backlog input.
func ActionItems(gaps []Pkg, evidencePath string) []dogfoodissues.ActionItem {
	items := make([]dogfoodissues.ActionItem, 0, len(gaps))
	for _, p := range gaps {
		items = append(items, p.ToActionItem(evidencePath))
	}
	return items
}

// strandedGrade grades the size of the stranded investment (bigger = worse): a large,
// tested, unwired package is the loudest debt. F for the biggest orphans, up to C for a tiny
// one; a stranded package is never "good", so the ceiling is C.
func strandedGrade(p Pkg) string {
	switch {
	case p.SourceLines >= 400:
		return "F"
	case p.SourceLines >= 150:
		return "D"
	default:
		return "C"
	}
}

// slug lowercases and dash-collapses an arbitrary string for a stable issue key.
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
		return "package"
	}
	return out
}
