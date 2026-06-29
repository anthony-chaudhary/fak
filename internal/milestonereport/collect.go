package milestonereport

// Live runners for `fak milestone`: the CLIMB dimension reads the in-process
// covmatrix grid (no shell), the ROADMAP dimension shells to `gh` for each tracked
// epic's child completion. Kept separate from the pure fold so milestonereport.go
// stays unit-testable without a process or a repo.

import (
	"encoding/json"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
)

// EpicSpec is one tracked roadmap milestone: its issue number, a human title, and
// the optional track LABEL whose open/closed child issues measure its completion.
// When Label is empty the collector falls back to the epic body's task-list
// checklist; when neither resolves, the epic is an honest ERRORED row.
type EpicSpec struct {
	Number int
	Title  string
	Label  string
}

// TrackedEpics is the data-driven list of epics the milestone roadmap dimension
// tracks — the analog of covmatrix.Families. Each entry's child-completion signal
// is resolved by the collector's priority chain (track label, then body checklist).
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

// Runner runs a `gh` subprocess and returns its stdout, stderr, and an ok flag
// (true when the process exited 0). It is injectable so Collect is testable without
// a real gh, mirroring internal/dogfoodissues.Runner.
type Runner func(args []string) (stdout, stderr string, ok bool)

// defaultRunner shells out to the real `gh` CLI.
func defaultRunner(args []string) (string, string, bool) {
	cmd := exec.Command("gh", args...)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err == nil
}

// Collect measures both dimensions. The maturity climb is pure (covmatrix.Grid() is
// in-process and never errors); the roadmap shells to `gh` per tracked epic. A nil
// runner uses the real `gh`. The repo defaults to the current checkout's `gh`
// context, so `repo` is "" unless an override is wired.
func Collect(repo string, runner Runner) (Maturity, Epics) {
	if runner == nil {
		runner = defaultRunner
	}
	maturity := InterpretMaturity(covmatrix.Grid())
	counts := make([]EpicCounts, 0, len(TrackedEpics))
	for _, spec := range TrackedEpics {
		counts = append(counts, epicCounts(runner, repo, spec))
	}
	return maturity, InterpretEpics(TrackedEpics, counts, "")
}

// epicCounts resolves one epic's child completion via the provenance-honest
// priority chain: a track LABEL (open/closed children carrying it), then the epic
// body's task-list CHECKLIST. Whichever answers first wins and stamps Source. When
// neither resolves the result carries Err — never a fabricated {Total: 0}, the seam
// that lets the interpreter tell "0 of N done" from "could not read".
func epicCounts(runner Runner, repo string, spec EpicSpec) EpicCounts {
	if spec.Label != "" {
		if c, ok := countByLabel(runner, repo, spec); ok {
			return c
		}
	}
	if c, ok := countByChecklist(runner, repo, spec); ok {
		return c
	}
	return EpicCounts{Number: spec.Number, Err: "no child signal (no track label children, no body checklist)"}
}

// countByLabel counts the open vs closed issues carrying the epic's track label.
// Closed children / all children is the completion. Returns ok=false when the query
// fails or the label has no children (so the chain falls through to the checklist).
func countByLabel(runner Runner, repo string, spec EpicSpec) (EpicCounts, bool) {
	args := []string{"issue", "list", "--label", spec.Label, "--state", "all", "--limit", "500", "--json", "number,state"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, _, ok := runner(args)
	if !ok {
		return EpicCounts{}, false
	}
	var issues []struct {
		Number int    `json:"number"`
		State  string `json:"state"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(stdout)), &issues); err != nil {
		return EpicCounts{}, false
	}
	// Exclude the epic issue itself so its own state never skews its completion.
	var total, closed int
	for _, iss := range issues {
		if iss.Number == spec.Number {
			continue
		}
		total++
		if strings.EqualFold(iss.State, "closed") {
			closed++
		}
	}
	if total == 0 {
		return EpicCounts{}, false
	}
	return EpicCounts{Number: spec.Number, Closed: closed, Total: total, Source: "label"}, true
}

// countByChecklist reads the epic issue body and counts its GitHub task-list items
// (- [ ] / - [x]). Checked items / all items is the completion. Returns ok=false
// when the body cannot be read or carries no task-list, so the chain ends in an
// honest errored row rather than a fabricated 0%.
func countByChecklist(runner Runner, repo string, spec EpicSpec) (EpicCounts, bool) {
	args := []string{"issue", "view", strconv.Itoa(spec.Number), "--json", "body"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, _, ok := runner(args)
	if !ok {
		return EpicCounts{}, false
	}
	var payload struct {
		Body string `json:"body"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(stdout)), &payload); err != nil {
		return EpicCounts{}, false
	}
	total, checked := countTaskList(payload.Body)
	if total == 0 {
		return EpicCounts{}, false
	}
	return EpicCounts{Number: spec.Number, Closed: checked, Total: total, Source: "checklist"}, true
}

// countTaskList counts GitHub markdown task-list items in body: total items and the
// checked subset. A task-list item is a line whose first non-space content is
// "- [ ]" or "- [x]" (case-insensitive on the mark), the same grammar GitHub
// renders as a checkbox.
func countTaskList(body string) (total, checked int) {
	for _, raw := range strings.Split(body, "\n") {
		ln := strings.TrimSpace(raw)
		if !strings.HasPrefix(ln, "- [") || len(ln) < 5 || ln[4] != ']' {
			continue
		}
		mark := ln[3]
		total++
		if mark == 'x' || mark == 'X' {
			checked++
		}
	}
	return total, checked
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
