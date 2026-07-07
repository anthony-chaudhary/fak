package main

// dispatch_status.go — `fak dispatch status`, the native live-loop witness that
// ports the LIVE-WORKER section of tools/dispatch_status.py (#1406) onto the same
// `fak dispatch` surface as tick/wave/sweep/progress. It answers the operator's
// loop-witness question — how many issue-resolution workers are live right now,
// which issues they hold, and which lanes are pinned — as a PURE, hermetic fold
// over the runs directory (resolve-*.log/.pid/.lease-tree.json sidecars a spawned
// worker leaves behind), reusing the already-wired liveResolutionScopes scanner.
//
//	# the human status card
//	fak dispatch status
//	# machine-readable snapshot (schema fleet-dispatch-status/1)
//	fak dispatch status --json
//	# the same card as an operator Markdown block
//	fak dispatch status --markdown
//
// It launches nothing and writes nothing. The gh-backed backlog-by-lane and
// closure-rate sections and the DoS-safe spawn gate of the Python card stay in
// tools/dispatch_status.py until their native folds are wired; this command owns
// the pure-local live-worker witness that needs no network.

import (
	"flag"
	"fmt"
	"io"
	"sort"
	"strings"
)

const dispatchStatusSchema = "fleet-dispatch-status/1"

// dispatchStatusWorker is one live issue-resolution worker in the snapshot.
type dispatchStatusWorker struct {
	Issue   int      `json:"issue"`
	Lane    string   `json:"lane"`
	Worker  string   `json:"worker"`
	PID     int      `json:"pid"`
	LeaseID string   `json:"lease_id"`
	Tree    []string `json:"tree"`
}

// dispatchStatusSnapshot is the fleet-dispatch-status/1 payload.
type dispatchStatusSnapshot struct {
	Schema          string                 `json:"schema"`
	RunsDir         string                 `json:"runs_dir"`
	LiveWorkerCount int                    `json:"live_worker_count"`
	IssuesInFlight  []int                  `json:"issues_in_flight"`
	LanesHeld       []string               `json:"lanes_held"`
	Workers         []dispatchStatusWorker `json:"workers"`
}

func runDispatchStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", dispatchProgressRunsDir, "directory of dispatch worker logs")
	asJSON := fs.Bool("json", false, "emit the fleet-dispatch-status/1 JSON payload")
	asMarkdown := fs.Bool("markdown", false, "render the operator status card as Markdown")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *asJSON && *asMarkdown {
		fmt.Fprintln(stderr, "fak dispatch status: choose at most one of --json or --markdown")
		return 2
	}

	snap := dispatchStatusScan(*runsDir)

	if *asJSON {
		if err := writeIndentedJSON(stdout, snap); err != nil {
			fmt.Fprintf(stderr, "fak dispatch status: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	if *asMarkdown {
		fmt.Fprint(stdout, renderDispatchStatusMarkdown(snap))
		return 0
	}
	fmt.Fprint(stdout, renderDispatchStatus(snap))
	return 0
}

// dispatchStatusScan folds the runs directory into the live-worker snapshot. It is
// pure over the filesystem: the same runs-dir yields the same snapshot, so a test
// drives it hermetically by planting resolve-*.log/.pid sidecars for a live pid.
func dispatchStatusScan(runsDir string) dispatchStatusSnapshot {
	scopes := liveResolutionScopes(runsDir)
	workers := make([]dispatchStatusWorker, 0, len(scopes))
	issueSet := map[int]bool{}
	laneSet := map[string]bool{}
	for _, s := range scopes {
		workers = append(workers, dispatchStatusWorker{
			Issue:   s.Issue,
			Lane:    s.Lane,
			Worker:  s.Worker,
			PID:     s.PID,
			LeaseID: s.LeaseID,
			Tree:    s.Tree,
		})
		issueSet[s.Issue] = true
		if s.Lane != "" {
			laneSet[s.Lane] = true
		}
	}
	issues := make([]int, 0, len(issueSet))
	for issue := range issueSet {
		issues = append(issues, issue)
	}
	sort.Ints(issues)
	lanes := make([]string, 0, len(laneSet))
	for lane := range laneSet {
		lanes = append(lanes, lane)
	}
	sort.Strings(lanes)
	return dispatchStatusSnapshot{
		Schema:          dispatchStatusSchema,
		RunsDir:         runsDir,
		LiveWorkerCount: len(workers),
		IssuesInFlight:  issues,
		LanesHeld:       lanes,
		Workers:         workers,
	}
}

func dispatchStatusLaneField(lanes []string) string {
	if len(lanes) == 0 {
		return "(none)"
	}
	return strings.Join(lanes, ", ")
}

func renderDispatchStatus(snap dispatchStatusSnapshot) string {
	var b strings.Builder
	fmt.Fprintf(&b, "dispatch status — %d live worker(s)\n", snap.LiveWorkerCount)
	fmt.Fprintf(&b, "runs-dir: %s\n", snap.RunsDir)
	fmt.Fprintf(&b, "lanes held: %s\n", dispatchStatusLaneField(snap.LanesHeld))
	if len(snap.Workers) == 0 {
		fmt.Fprint(&b, "no live issue-resolution workers\n")
		return b.String()
	}
	for _, w := range snap.Workers {
		lane := w.Lane
		if lane == "" {
			lane = "(no lane)"
		}
		fmt.Fprintf(&b, "  #%d  lane=%s  pid=%d  worker=%s\n", w.Issue, lane, w.PID, w.Worker)
	}
	return b.String()
}

func renderDispatchStatusMarkdown(snap dispatchStatusSnapshot) string {
	var b strings.Builder
	fmt.Fprintf(&b, "### dispatch status — %d live worker(s)\n\n", snap.LiveWorkerCount)
	fmt.Fprintf(&b, "- runs-dir: `%s`\n", snap.RunsDir)
	fmt.Fprintf(&b, "- lanes held: %s\n\n", dispatchStatusLaneField(snap.LanesHeld))
	if len(snap.Workers) == 0 {
		fmt.Fprint(&b, "_no live issue-resolution workers_\n")
		return b.String()
	}
	fmt.Fprint(&b, "| issue | lane | pid | worker |\n")
	fmt.Fprint(&b, "|---|---|---|---|\n")
	for _, w := range snap.Workers {
		lane := w.Lane
		if lane == "" {
			lane = "(no lane)"
		}
		fmt.Fprintf(&b, "| #%d | %s | %d | %s |\n", w.Issue, lane, w.PID, w.Worker)
	}
	return b.String()
}
