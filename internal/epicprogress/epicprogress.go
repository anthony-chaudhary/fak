// Package epicprogress resolves how complete a GitHub epic is from its children,
// via a provenance-honest priority chain. It is the reusable resolver extracted
// from the milestone roadmap dimension (issue #1438) so issue-triage, plan-audit,
// and dogfoodissues can all ask "how complete is epic #N?" without depending on
// internal/milestonereport — this package has NO dependency back on it.
//
// The chain, in order of preference:
//  1. track label  — count the open/closed issues carrying the epic's track label;
//  2. body checklist — count `- [x]` / `- [ ]` task-list items in the epic body,
//     resolving each row that NAMES a child issue (`- [ ] #1234 …`) against that
//     issue's LIVE state rather than the hand-ticked box (see below);
//  3. errored row    — when neither resolves, EpicCounts.Err is set, NEVER a
//     fabricated {Total: 0}. That seam is what lets a caller tell "0 of N done"
//     from "could not read" — the load-bearing honesty contract.
//
// Why the checklist rung cross-checks the referenced issue (#1315 was the witness):
// a markdown checkbox is a hand-maintained PROXY for completion, and it goes stale in
// both directions. Epic #1315 ("native agent harness") is itself CLOSED with all eight
// children #1316..#1323 CLOSED, yet every box in its body is still unticked — so the
// box-only reading reported it 0/8 done and charged eight units of roadmap debt for
// work that had actually shipped. The mirror hazard is worse: a ticked box for a child
// that is still OPEN would report progress nobody made. So whenever a row names a child
// issue and that issue's state is READABLE, the issue state decides — the checkbox is
// the fallback for rows that name no issue, or whose issue cannot be read. That keeps
// the count cross-checked against reality (editing the epic body cannot move it) rather
// than against a self-report.
//
// Discovered fact carried here on purpose: the epics this resolver tracks do NOT
// use GitHub native sub-issues — `gh`'s sub_issues_summary returns total:0 for
// them — which is why completion is measured by the label/checklist chain rather
// than by a native sub-issue count.
package epicprogress

import (
	"context"
	"encoding/json"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ghRunnerTimeout bounds each real gh subprocess so a stalled network call or a
// blocking auth prompt cannot hang the caller indefinitely (issue #3483). The
// 10s WaitDelay is the straggler backstop for when the context kill leaves a
// grandchild holding the output pipe open.
const ghRunnerTimeout = 60 * time.Second

// maxChildStateLookups bounds how many referenced child issues ONE checklist
// resolution will look up live, so a pathological epic body (a hundred `#N` rows)
// cannot turn one report into a hundred `gh` round-trips. Rows past the budget fall
// back to their checkbox — the same conservative fallback an unreadable child takes.
// The hermetic path (`--epics-from` with pre-resolved counts) makes zero lookups.
const maxChildStateLookups = 64

// childLookupConcurrency bounds how many child-state `gh` round-trips are in flight at
// once. Each one is a fresh subprocess, so serial resolution would add ~2s per child to
// a report the scorecard control pane runs under a per-card timeout; a small fixed
// worker pool keeps the wall clock flat without hammering the API.
const childLookupConcurrency = 8

// SourceChecklist is stamped when the completion came from the epic body's task-list
// alone (no row named a child issue, or none could be read live).
const SourceChecklist = "checklist"

// SourceChecklistIssueState is stamped when at least one task-list row's done-ness was
// decided by its referenced child issue's LIVE state instead of the hand-ticked box.
// The distinct label is the provenance-honesty contract: a reader can tell a
// cross-checked count from a self-reported one.
const SourceChecklistIssueState = "checklist+issue-state"

// EpicSpec is one tracked epic: its issue number, a human title, an optional
// product generation horizon, and the optional track LABEL whose open/closed child
// issues measure its completion. When Label is empty the resolver falls back to the
// epic body's task-list checklist; when neither resolves, the epic is an honest
// ERRORED row.
type EpicSpec struct {
	Number     int
	Title      string
	Generation string
	Label      string
}

// EpicCounts is the raw child tally for one epic: how many children, how many
// closed, by which Source ("label" | "checklist" | "checklist+issue-state") — or an
// Err when no child signal could be witnessed. A failed read MUST set Err, never
// Total 0; downstream folds rely on that to tell "0 of N done" from "could not read".
// Source is the provenance label so a fold can report HOW the number was witnessed —
// in particular whether the checklist count was cross-checked against the referenced
// children's real state or read off the hand-ticked boxes alone.
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
//
// A Runner MUST be safe for concurrent use: Counts resolves a checklist's referenced
// child issues in parallel (childLookupConcurrency at a time) so one epic's children
// cost one round-trip of latency rather than N.
type Runner func(args []string) (stdout, stderr string, ok bool)

// DefaultRunner shells out to the real `gh` CLI.
func DefaultRunner(args []string) (string, string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), ghRunnerTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.WaitDelay = 10 * time.Second
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		if errb.Len() > 0 {
			errb.WriteByte('\n')
		}
		errb.WriteString("gh timed out after " + ghRunnerTimeout.String())
		return out.String(), errb.String(), false
	}
	return out.String(), errb.String(), err == nil
}

// runGH appends --repo when set and invokes runner, returning just stdout and
// the ok flag — the "add the repo override, run gh, surface failure uniformly"
// idiom both countByLabel and countByChecklist share before parsing their own
// JSON shape.
func runGH(runner Runner, repo string, args []string) (string, bool) {
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, _, ok := runner(args)
	return stdout, ok
}

// Counts resolves one epic's child completion via the provenance-honest priority
// chain: a track LABEL (open/closed children carrying it), then the epic body's
// task-list CHECKLIST — whose rows are themselves cross-checked against the live
// state of any child issue they name, so a stale (or optimistic) checkbox cannot
// decide the count. Whichever rung answers first wins and stamps Source. When neither
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
	stdout, ok := runGH(runner, repo, args)
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

// countByChecklist reads the epic issue body, parses its GitHub task-list rows, and
// counts how many are DONE. A row that names a child issue (`- [ ] #1234 …`) is
// decided by that issue's live state when it can be read; every other row falls back
// to its checkbox. Done rows / all rows is the completion. Returns ok=false when the
// body cannot be read or carries no task-list, so the chain ends in an honest errored
// row rather than a fabricated 0%.
func countByChecklist(runner Runner, repo string, spec EpicSpec) (EpicCounts, bool) {
	args := []string{"issue", "view", strconv.Itoa(spec.Number), "--json", "body"}
	stdout, ok := runGH(runner, repo, args)
	if !ok {
		return EpicCounts{}, false
	}
	var payload struct {
		Body string `json:"body"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(stdout)), &payload); err != nil {
		return EpicCounts{}, false
	}
	items := ParseTaskList(payload.Body)
	if len(items) == 0 {
		return EpicCounts{}, false
	}

	// Resolve every referenced child ONCE, up front, then fold sequentially. Splitting
	// the I/O from the count keeps the fold pure and order-independent: the same items +
	// the same child states always produce the same EpicCounts, whatever order the
	// lookups completed in.
	state := childStates(runner, repo, spec.Number, items)

	source := SourceChecklist
	done := 0
	for _, it := range items {
		isDone := it.Checked
		if closed, known := state[it.Ref]; known {
			isDone = closed
			source = SourceChecklistIssueState
		}
		if isDone {
			done++
		}
	}
	return EpicCounts{Number: spec.Number, Closed: done, Total: len(items), Source: source}, true
}

// childStates resolves the live open/closed state of every DISTINCT child issue the
// task-list rows name, keyed by issue number. A child whose state could not be
// witnessed is simply absent from the map, so the caller keeps that row's checkbox
// rather than inventing a state.
//
// self (the epic's own number) is skipped on purpose: an epic that closes itself must
// not silently mark one of its own rows complete. Lookups run CONCURRENTLY under a
// small worker bound because each one is a separate `gh` round-trip — an epic with ten
// children would otherwise add ~20s of serial latency to a report the scorecard control
// pane runs under a per-card timeout. The bound keeps the burst polite to the API.
func childStates(runner Runner, repo string, self int, items []TaskListItem) map[int]bool {
	var refs []int
	seen := map[int]bool{}
	for _, it := range items {
		if it.Ref <= 0 || it.Ref == self || seen[it.Ref] {
			continue
		}
		seen[it.Ref] = true
		refs = append(refs, it.Ref)
		if len(refs) == maxChildStateLookups {
			break
		}
	}
	if len(refs) == 0 {
		return nil
	}

	workers := childLookupConcurrency
	if len(refs) < workers {
		workers = len(refs)
	}
	type result struct {
		ref          int
		closed, know bool
	}
	jobs := make(chan int)
	results := make(chan result, len(refs))
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ref := range jobs {
				closed, know := issueClosed(runner, repo, ref)
				results <- result{ref: ref, closed: closed, know: know}
			}
		}()
	}
	for _, ref := range refs {
		jobs <- ref
	}
	close(jobs)
	wg.Wait()
	close(results)

	out := map[int]bool{}
	for r := range results {
		if r.know {
			out[r.ref] = r.closed
		}
	}
	return out
}

// issueClosed reads one referenced child issue's live state. ok=false means the state
// could not be witnessed (the number is a PR, lives in another repo, or `gh` failed) —
// the caller then keeps the checkbox rather than inventing a state.
func issueClosed(runner Runner, repo string, number int) (closed, ok bool) {
	stdout, runOK := runGH(runner, repo, []string{"issue", "view", strconv.Itoa(number), "--json", "state"})
	if !runOK {
		return false, false
	}
	var payload struct {
		State string `json:"state"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(stdout)), &payload); err != nil {
		return false, false
	}
	switch {
	case strings.EqualFold(payload.State, "closed"):
		return true, true
	case strings.EqualFold(payload.State, "open"):
		return false, true
	default:
		return false, false
	}
}

// TaskListItem is one parsed GitHub task-list row: whether its box is ticked, and the
// child issue the row names (Ref == 0 when it names none). Ref is what lets the
// resolver cross-check a hand-ticked box against the child's real state.
type TaskListItem struct {
	Checked bool `json:"checked"`
	Ref     int  `json:"ref,omitempty"`
}

// issueRefPattern matches the first `#<number>` in a task-list row's text. The leading
// boundary keeps it off intra-word hits (`sha#1` is not a reference) while still
// allowing the markdown decoration these epic bodies use (`**A (keystone)** #1302 — …`).
var issueRefPattern = regexp.MustCompile(`(^|[^0-9A-Za-z_])#([0-9]+)`)

// ParseTaskList parses GitHub markdown task-list rows out of body, in document order.
// A row is a line whose first non-space content is "- [ ]" or "- [x]" (case-insensitive
// on the mark), the same grammar GitHub renders as a checkbox.
func ParseTaskList(body string) []TaskListItem {
	var out []TaskListItem
	for _, raw := range strings.Split(body, "\n") {
		ln := strings.TrimSpace(raw)
		if !strings.HasPrefix(ln, "- [") || len(ln) < 5 || ln[4] != ']' {
			continue
		}
		mark := ln[3]
		item := TaskListItem{Checked: mark == 'x' || mark == 'X'}
		if m := issueRefPattern.FindStringSubmatch(ln[5:]); m != nil {
			if n, err := strconv.Atoi(m[2]); err == nil {
				item.Ref = n
			}
		}
		out = append(out, item)
	}
	return out
}

// CountTaskList counts GitHub markdown task-list items in body: total items and the
// checked subset. It reads the CHECKBOXES only — a caller that wants the reality
// cross-check (a row's referenced child issue deciding its done-ness) goes through
// Counts, which folds ParseTaskList against the live child states.
func CountTaskList(body string) (total, checked int) {
	for _, it := range ParseTaskList(body) {
		total++
		if it.Checked {
			checked++
		}
	}
	return total, checked
}
