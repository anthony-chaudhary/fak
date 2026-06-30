package milestonereport

// Live runners for `fak milestone`: the CLIMB dimension reads the in-process
// covmatrix grid (no shell), the ROADMAP dimension resolves each tracked epic's
// child completion via internal/epicprogress (the extracted, independently-tested
// resolver). Kept separate from the pure fold so milestonereport.go stays
// unit-testable without a process or a repo.

import (
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/epicprogress"
)

// EpicSpec is one tracked roadmap milestone. It is an alias of
// epicprogress.EpicSpec so the resolver and the report share one type with no
// conversion; TrackedEpics and the `--epics-from` override both build these.
type EpicSpec = epicprogress.EpicSpec

// Runner is the injectable `gh` seam, an alias of epicprogress.Runner so callers
// (and tests) wire one runner type through both the resolver and Collect.
type Runner = epicprogress.Runner

// TrackedEpics is the data-driven list of epics the milestone roadmap dimension
// tracks — the analog of covmatrix.Families. Each entry's child-completion signal
// is resolved by epicprogress.Counts (track label, then body checklist).
// Seeded from the live fleet epics; edit this slice (not the logic) to track more.
// A `--epics-from` JSON file overrides it for ad-hoc tracking.
var TrackedEpics = []EpicSpec{
	{Number: 1243, Title: "support-maturity disambiguation", Label: "support-maturity"},
	{Number: 1315, Title: "native agent harness"},
	{Number: 1301, Title: "cache-value P&L roll-up"},
	{Number: 1178, Title: "first-class time horizons"},
	{Number: 1010, Title: "GLM-5.2 through fak's kernel"},
	{Number: 1354, Title: "release at agentic speed"},
}

// Collect measures both dimensions. The maturity climb is pure (covmatrix.Grid() is
// in-process and never errors); the roadmap resolves each tracked epic's children
// via epicprogress.Counts. A nil runner uses the real `gh`. The repo defaults to the
// current checkout's `gh` context, so `repo` is "" unless an override is wired.
func Collect(repo string, runner Runner) (Maturity, Epics) {
	if runner == nil {
		runner = epicprogress.DefaultRunner
	}
	maturity := InterpretMaturity(covmatrix.Grid())
	counts := make([]EpicCounts, 0, len(TrackedEpics))
	for _, spec := range TrackedEpics {
		counts = append(counts, asEpicCounts(epicprogress.Counts(runner, repo, spec)))
	}
	return maturity, InterpretEpics(TrackedEpics, counts, "")
}

// asEpicCounts adapts the resolver's epicprogress.EpicCounts into the report-local
// EpicCounts the pure InterpretEpics folds. The two are field-identical; this thin
// adapter is the seam that keeps epicprogress free of any milestonereport import.
func asEpicCounts(c epicprogress.EpicCounts) EpicCounts {
	return EpicCounts{
		Number: c.Number,
		Closed: c.Closed,
		Total:  c.Total,
		Source: c.Source,
		Err:    c.Err,
	}
}

// countTaskList re-exports epicprogress.CountTaskList under the name the in-package
// roadmap tests already use, so the task-list grammar has one implementation.
func countTaskList(body string) (total, checked int) {
	return epicprogress.CountTaskList(body)
}

// HeadCommit returns the short HEAD commit of root, or "unknown" — the same git
// plumbing gardenbundle.HeadCommit / cadencereport.HeadCommit use, inlined so this
// leaf imports no sibling composer.
func HeadCommit(root string) string {
	cmd := exec.Command("git", "rev-parse", "--short", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return "unknown"
	}
	return s
}

// CountsFromSpecs is a tiny helper for the `--epics-from` path: given pre-resolved
// specs + counts decoded from a JSON file, the caller folds them with the pure
// InterpretEpics. It exists so the CLI override path never reaches for `gh`.
func CountsFromSpecs(specs []EpicSpec, counts []EpicCounts) Epics {
	return InterpretEpics(specs, counts, "")
}
