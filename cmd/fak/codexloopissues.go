package main

// fak sessions codex-loop --recent --sync-issues — the backlog bridge from the
// recent codex-loop scan to stable, deduplicated GitHub issues. It folds the LOOP
// diagnoses of a recent scan into one dispatchable issue per (tool, output-digest)
// loop class and reuses the internal/dogfoodissues idempotency machinery to create
// or update exactly one issue per class (dedup by an HTML-comment marker key).
//
// Safe by default: without --live it is a dry-run that prints the plan and never
// touches the network; --live is the explicit opt-in that fetches existing issues
// and shells out to `gh`, with the same marker read-back verification the
// dogfood-issues bridge uses. Forward-progress sessions never reach here — a healthy
// distinct-arg update_plan burst is classified OK, not LOOP, so it files nothing.

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// codexLoopGateIssueLabel tags every issue this bridge files so the loop-gate class
// is filterable and reruns can be audited as a cohort.
const codexLoopGateIssueLabel = "codex-loop-gate"

// codexLoopIssueOptions carries the effectful knobs for the issue bridge. They are
// only meaningful alongside --recent --sync-issues.
type codexLoopIssueOptions struct {
	Live               bool
	FetchExisting      bool
	Repo               string
	Milestone          string
	ExistingJSON       string
	Limit              int
	Labels             []string
	ParentIssue        int
	ProjectBaseline    float64
	CompletionStandard string
	TargetEnvelope     string
	WitnessedEnvelope  string
	// Runner injects the gh subprocess for tests; nil uses the real gh CLI. The
	// command wiring always leaves this nil.
	Runner dogfoodissues.Runner
}

// codexLoopGateActionKey is the stable dedup key for one loop class: the repeated
// tool plus a short prefix of its output digest. Reruns and sibling sessions that
// hit the same class map to the same key, so the bridge updates one issue instead
// of opening duplicates. The shape satisfies issuecontract's keyRE
// (`[A-Za-z0-9._:/-]`).
func codexLoopGateActionKey(o codexRepeatedOutcome) string {
	tool := strings.TrimSpace(o.Tool)
	if tool == "" {
		tool = "unknown"
	}
	return "codex-loop-gate/" + tool + "/" + codexLoopShortDigest(o)
}

// codexLoopShortDigest is the marker-safe short form of a repeated outcome's
// output digest: the first 12 hex chars after the "sha256:" tag, or "nodigest"
// when absent. Used in both the dedup key and the witness so they agree.
func codexLoopShortDigest(o codexRepeatedOutcome) string {
	digest := strings.TrimPrefix(strings.TrimSpace(o.OutputDigest), "sha256:")
	if len(digest) > 12 {
		digest = digest[:12]
	}
	if digest == "" {
		return "nodigest"
	}
	return digest
}

// buildCodexLoopGateActionItems folds a recent codex-loop scan into stable issue
// candidates — one per loop class, regardless of how many sessions hit it. Only
// LOOP diagnoses that carry a concrete loop-driving outcome (a repeated tool +
// output digest) file an issue; a diagnosis whose only repeated traffic is
// forward progress is never LOOP and never reaches here, and a LOOP without a
// concrete repeated outcome stays visible in the scan without an auto-issue.
func buildCodexLoopGateActionItems(r codexLoopRecentReport) []dogfoodissues.ActionItem {
	index := map[string]int{} // key -> position in items
	var items []dogfoodissues.ActionItem
	for _, d := range r.Diagnoses {
		if d.Verdict != "LOOP" {
			continue
		}
		outcome, ok := codexTopLoopDrivingOutcome(d.RepeatedOutcomes)
		if !ok {
			continue
		}
		key := codexLoopGateActionKey(outcome)
		note := codexLoopGateSessionNote(d, outcome)
		if pos, dup := index[key]; dup {
			// Another session already contributed this class: fold the recurrence
			// into the existing item rather than opening a sibling issue.
			items[pos].DebtCount++
			if note != "" {
				items[pos].BoundaryNotes = append(items[pos].BoundaryNotes, note)
			}
			continue
		}
		index[key] = len(items)
		items = append(items, codexLoopGateActionItem(key, outcome, d, note))
	}
	return items
}

// codexLoopGateSessionNote is a one-line, argument-free evidence line for one
// session that hit a loop class. It never echoes raw tool arguments — only the
// bounded fold the diagnosis already exposes.
func codexLoopGateSessionNote(d codexLoopDiagnosis, o codexRepeatedOutcome) string {
	id := strings.TrimSpace(d.SessionID)
	if id == "" {
		id = "unknown-session"
	}
	return fmt.Sprintf("Observed in session %s: %s repeated %d× (longest run %d).", id, o.Tool, o.Count, o.LongestRun)
}

// codexLoopGateActionItem builds the dispatchable issue candidate for one loop
// class. It fills every scope field issuecontract requires so the class becomes a
// dispatchable issue rather than a skipped, vague one; the remaining agent-context
// and noise-control fields fall back to dogfoodissues' defaults.
func codexLoopGateActionItem(key string, o codexRepeatedOutcome, d codexLoopDiagnosis, note string) dogfoodissues.ActionItem {
	tool := strings.TrimSpace(o.Tool)
	if tool == "" {
		tool = "unknown"
	}
	excerpt := strings.TrimSpace(o.OutputExcerpt)
	if excerpt == "" {
		excerpt = "(constant output)"
	}
	notes := []string{}
	if note != "" {
		notes = append(notes, note)
	}
	return dogfoodissues.ActionItem{
		Key:          key,
		Title:        fmt.Sprintf("Codex loop-gate: hard-fuse repeated `%s` no-progress outcome", tool),
		SourceProbe:  "fak sessions codex-loop --recent",
		ScoreName:    "codex-loop-gate",
		Grade:        "ACTION",
		DebtName:     "loop-sessions",
		DebtCount:    1,
		EvidencePath: "fak sessions codex-loop --recent --json",
		Finding:      fmt.Sprintf("%s repeats a no-progress outcome", tool),
		NextAction:   fmt.Sprintf("Hard-fuse the `%s` no-progress loop class in the codex loop gate.", tool),
		WorkingSpine: fmt.Sprintf("Halt the run when `%s` returns the same no-progress output across turns, instead of burning tokens until the launch gate refuses the next spawn.", tool),
		WorkUnit:     "leaf",
		InScope: fmt.Sprintf(
			"Hard-fuse the specific no-progress loop where `%s` returns the same output (`%s`) across N turns (digest `%s`); wire the fuse into the existing codex loop gate that `fak sessions codex-loop`/`fak codex`/dispatch already consult.",
			tool, excerpt, o.OutputDigest),
		OutOfScope: "Do not change forward-progress classification — distinct-argument progress tools (e.g. update_plan revising a real plan) must stay OK. Do not touch unrelated tools or rewrite the loop detector wholesale.",
		DoneCondition: fmt.Sprintf(
			"The codex loop gate hard-fuses this `%s` class after the first confirmed repeat, and a fresh `fak sessions codex-loop --recent` reports no LOOP for it.",
			tool),
		Witness: fmt.Sprintf(
			"`fak sessions codex-loop --recent --since-hours 24 --json` shows no LOOP diagnosis with tool=`%s` digest prefix=`%s`.",
			tool, codexLoopShortDigest(o)),
		AcceptanceGate: fmt.Sprintf(
			"A regression test reproduces the `%s` no-progress loop fixture and asserts the gate refuses it; `go test ./cmd/fak/ -run CodexLoop` passes.",
			tool),
		Lane:          "cmd/fak",
		Paths:         []string{"cmd/fak/sessions_codex_loop.go"},
		Labels:        []string{codexLoopGateIssueLabel},
		BoundaryNotes: notes,
	}
}

// runCodexLoopSyncIssues folds the recent scan into the dogfood-issue bridge and
// prints/executes the plan. It mirrors runDogfoodIssues: dry-run by default, gh
// side effects only under --live, existing-issue classification under
// --live/--fetch-existing/--existing-json. Returns the process exit code (0 ok,
// 1 a live gh create/edit failed or an IO/encode error, 2 usage).
func runCodexLoopSyncIssues(stdout, stderr io.Writer, r codexLoopRecentReport, asJSON bool, opt codexLoopIssueOptions) int {
	items := buildCodexLoopGateActionItems(r)

	var existing []dogfoodissues.Issue
	switch {
	case strings.TrimSpace(opt.ExistingJSON) != "":
		if err := readJSONFileInto(opt.ExistingJSON, &existing); err != nil {
			fmt.Fprintf(stderr, "fak sessions codex-loop: --existing-json must contain a JSON list: %v\n", err)
			return 2
		}
	case opt.Live || opt.FetchExisting:
		var err error
		existing, err = dogfoodissues.FetchExistingIssues(opt.Repo, opt.Limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak sessions codex-loop: %v\n", err)
			return 2
		}
	}

	mode := "dry-run"
	if opt.Live {
		mode = "live"
	}
	buildOpt := dogfoodissues.BuildOptions{
		Live:             opt.Live,
		DedupeChecked:    opt.Live || opt.FetchExisting || strings.TrimSpace(opt.ExistingJSON) != "",
		DedupeCap:        opt.Limit,
		DefaultMilestone: strings.TrimSpace(opt.Milestone),
		ParentIssue:      opt.ParentIssue,
		ParentBaseline:   opt.ProjectBaseline, CompletionStandard: opt.CompletionStandard, TargetEnvelope: opt.TargetEnvelope, WitnessedEnvelope: opt.WitnessedEnvelope,
	}
	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, existing, buildOpt)
	result := dogfoodissues.Result{
		Schema:  dogfoodissues.Schema,
		Mode:    mode,
		Report:  "fak sessions codex-loop --recent",
		Planned: plan,
		Synced:  []dogfoodissues.SyncRow{},
		Skipped: skipped,
	}
	if opt.Live && len(plan) > 0 {
		result.Synced = dogfoodissues.Sync(plan, opt.Repo, opt.Labels, opt.Runner)
	}

	if asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak sessions codex-loop: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, dogfoodissues.Render(result))
	}

	if opt.Live {
		for _, row := range result.Synced {
			if !row.OK {
				return 1
			}
		}
	}
	return 0
}
