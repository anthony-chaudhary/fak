// Package epicprogress resolves how complete a GitHub epic is from its children,
// via a provenance-honest priority chain. It is the reusable resolver extracted
// from the milestone roadmap dimension (issue #1438) so issue-triage, plan-audit,
// and dogfoodissues can all ask "how complete is epic #N?" without depending on
// internal/milestonereport — this package has NO dependency back on it.
//
// The chain, in order of preference:
//  1. track label  — count the open/closed issues carrying the epic's track label;
//  2. body checklist — count `- [x]` / `- [ ]` task-list items in the epic body;
//  3. errored row    — when neither resolves, EpicCounts.Err is set, NEVER a
//     fabricated {Total: 0}. That seam is what lets a caller tell "0 of N done"
//     from "could not read" — the load-bearing honesty contract.
//
// Discovered fact carried here on purpose: the epics this resolver tracks do NOT
// use GitHub native sub-issues — `gh`'s sub_issues_summary returns total:0 for
// them — which is why completion is measured by the label/checklist chain rather
// than by a native sub-issue count.
package epicprogress

import (
	"encoding/json"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// EpicSpec is one tracked epic: its issue number, a human title, and the optional
// track LABEL whose open/closed child issues measure its completion. When Label is
// empty the resolver falls back to the epic body's task-list checklist; when
// neither resolves, the epic is an honest ERRORED row.
type EpicSpec struct {
	Number int
	Title  string
	Label  string
}

// EpicCounts is the raw child tally for one epic: how many children, how many
// closed, by which Source ("label" | "checklist") — or an Err when no child signal
// could be witnessed. A failed read MUST set Err, never Total 0; downstream folds
// rely on that to tell "0 of N done" from "could not read". Source is the
// provenance label so a fold can report HOW the number was witnessed.
type EpicCounts struct {
	Number int
	Closed int
	Total  int
	Source string
	Err    string
}

// Runner runs a `gh` subprocess and returns its stdout, stderr, and an ok flag
// (true when the process exited 0). It is injectable so the resolver is testable
// without a real gh, mirroring internal/dogfoodissues.Runner.
type Runner func(args []string) (stdout, stderr string, ok bool)

// DefaultRunner shells out to the real `gh` CLI.
func DefaultRunner(args []string) (string, string, bool) {
	cmd := exec.Command("gh", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err == nil
}

// Counts resolves one epic's child completion via the provenance-honest priority
// chain: a track LABEL (open/closed children carrying it), then the epic body's
// task-list CHECKLIST. Whichever answers first wins and stamps Source. When neither
// resolves the result carries Err — never a fabricated {Total: 0}. A nil runner
// uses the real `gh`. repo is "" unless an override is wired.
func Counts(runner Runner, repo string, spec EpicSpec) EpicCounts {
	if runner == nil {
		runner = DefaultRunner
	}
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
	total, checked := CountTaskList(payload.Body)
	if total == 0 {
		return EpicCounts{}, false
	}
	return EpicCounts{Number: spec.Number, Closed: checked, Total: total, Source: "checklist"}, true
}

// CountTaskList counts GitHub markdown task-list items in body: total items and the
// checked subset. A task-list item is a line whose first non-space content is
// "- [ ]" or "- [x]" (case-insensitive on the mark), the same grammar GitHub
// renders as a checkbox.
func CountTaskList(body string) (total, checked int) {
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
