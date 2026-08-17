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
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchdoa"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/focusscore"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
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

// dispatchStatusFocus is the fleet focus WIP-breadth posture (#3223), re-derived from the
// focusscore fold over the trajctl ledger. It surfaces the same signal the tick throttles
// new-objective spawns on, so an operator sees a focus hold/warn DISTINCTLY from the
// rate-limit and collision holds. Present only when the trajctl ledger declared >= 1
// objective (otherwise there is no focus signal to report and the snapshot stays byte-
// identical to today).
type dispatchStatusFocus struct {
	Active    int    `json:"active"`     // active (concurrently live) objectives -- breadth
	WIPCap    int    `json:"wip_cap"`    // the pinned WIP ceiling
	ExcessWIP int    `json:"excess_wip"` // max(0, active-cap): objectives beyond the cap
	Saturated bool   `json:"saturated"`  // active >= cap: a new-objective spawn is throttled
	Posture   string `json:"posture"`    // "warn" | "hold": posture a tick WOULD apply now
}

// dispatchStatusSpawnHealth is the DOA-spawn census (#5868): of the workers that
// FINISHED in the recent window, how many died before they ever began work. It exists
// because the live-worker sections above go quiet in exactly the same way for an idle
// fleet and for a fleet whose every spawn is dying at argv parse — the 2026-07-28..08-03
// outage killed 350 of 382 spawned workers and this card read "0 live worker(s)" for six
// days straight. Present only when the window holds at least one finished run.
type dispatchStatusSpawnHealth struct {
	WindowHours float64        `json:"window_hours"`
	Runs        int            `json:"runs"` // the DENOMINATOR: every finished spawn in the window
	DOA         int            `json:"doa"`
	Rate        float64        `json:"doa_rate"`
	Status      string         `json:"status"` // clear | warn | alarm
	Causes      map[string]int `json:"causes,omitempty"`
	Sample      []string       `json:"sample,omitempty"`
}

// dispatchStatusSnapshot is the fleet-dispatch-status/1 payload.
type dispatchStatusSnapshot struct {
	Schema          string                     `json:"schema"`
	RunsDir         string                     `json:"runs_dir"`
	LiveWorkerCount int                        `json:"live_worker_count"`
	IssuesInFlight  []int                      `json:"issues_in_flight"`
	LanesHeld       []string                   `json:"lanes_held"`
	Workers         []dispatchStatusWorker     `json:"workers"`
	Focus           *dispatchStatusFocus       `json:"focus,omitempty"`
	SpawnHealth     *dispatchStatusSpawnHealth `json:"spawn_health,omitempty"`
	Trajectory      *trajctl.Status            `json:"trajectory,omitempty"`
	Progress        *dispatchProgressSummary   `json:"progress,omitempty"`
}

func runDispatchStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", dispatchProgressRunsDir, "directory of dispatch worker logs")
	workspace := fs.String("workspace", "", "workspace root for the focus WIP-breadth section (default: repo root)")
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

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	snap := dispatchStatusScan(*runsDir, root)
	snap.Progress = dispatchProgressFold(root)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, snap, "fak dispatch status")
	}
	if *asMarkdown {
		fmt.Fprint(stdout, renderDispatchStatusMarkdown(snap))
		return 0
	}
	fmt.Fprint(stdout, renderDispatchStatus(snap))
	return 0
}

// dispatchStatusFold reads the focusscore fold over root's trajctl ledger and returns the
// fleet focus WIP-breadth posture, or nil when there is no ledger signal (so the snapshot
// stays byte-identical to today). Posture reflects what a tick WOULD apply now, resolved
// from FLEET_DISPATCH_FOCUS_HOLD the same way the tick's --focus-hold default does.
func dispatchStatusFold(root string) *dispatchStatusFocus {
	if strings.TrimSpace(root) == "" {
		return nil
	}
	ev := focusscore.Build(focusscore.Options{Root: root}).Evidence
	if !ev.LedgerPresent {
		return nil
	}
	posture := dispatchtick.FocusPostureWarn
	if dispatchBoolValue(os.Getenv("FLEET_DISPATCH_FOCUS_HOLD")) {
		posture = dispatchtick.FocusPostureHold
	}
	return &dispatchStatusFocus{
		Active:    ev.Active,
		WIPCap:    ev.WIPCap,
		ExcessWIP: ev.ExcessWIP,
		Saturated: ev.WIPCap > 0 && ev.Active >= ev.WIPCap,
		Posture:   posture,
	}
}

func dispatchStatusTrajectory(root string) *trajctl.Status {
	if strings.TrimSpace(root) == "" {
		return nil
	}
	status := trajctl.Fold(trajctl.ReadLedgerFile(filepath.Join(root, trajctl.DefaultLedgerRel))).Status()
	if status.Empty() {
		return nil
	}
	return &status
}

// dispatchStatusSpawnHealthFold folds the runs directory's recently-FINISHED workers
// into the DOA-spawn census, or nil when the window held nothing to grade (so a snapshot
// over an empty/absent runs dir stays byte-identical to before #5868).
func dispatchStatusSpawnHealthFold(runsDir string, now time.Time) *dispatchStatusSpawnHealth {
	rep := scanDispatchDOA(runsDir, dispatchDOAWindow, dispatchDOASettle, now)
	if rep.Runs == 0 {
		return nil
	}
	return &dispatchStatusSpawnHealth{
		WindowHours: dispatchDOAWindow.Hours(),
		Runs:        rep.Runs,
		DOA:         rep.DOA,
		Rate:        rep.Rate,
		Status:      rep.Status,
		Causes:      rep.Causes,
		Sample:      rep.Sample,
	}
}

// dispatchStatusSpawnHealthLine renders the census as the operator line, reusing the
// same renderer the JSON payload is folded from so the two can never disagree.
func dispatchStatusSpawnHealthLine(h *dispatchStatusSpawnHealth) string {
	if h == nil {
		return ""
	}
	return dispatchDOALine(dispatchdoa.Report{
		Runs:   h.Runs,
		DOA:    h.DOA,
		Rate:   h.Rate,
		Causes: h.Causes,
		Status: h.Status,
		Sample: h.Sample,
	}, time.Duration(h.WindowHours*float64(time.Hour)))
}

// dispatchStatusScan folds the runs directory into the live-worker snapshot. It is
// pure over the filesystem: the same runs-dir yields the same snapshot, so a test
// drives it hermetically by planting resolve-*.log/.pid sidecars for a live pid. root
// scopes the focus WIP-breadth section (empty disables it).
func dispatchStatusScan(runsDir, root string) dispatchStatusSnapshot {
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
		Focus:           dispatchStatusFold(root),
		SpawnHealth:     dispatchStatusSpawnHealthFold(runsDir, time.Now()),
		Trajectory:      dispatchStatusTrajectory(root),
	}
}

// dispatchStatusFocusLine renders the focus WIP-breadth posture as one operator line,
// labeled "focus:" so it reads DISTINCTLY from a rate-limit or collision hold. Returns ""
// when there is no focus signal.
func dispatchStatusFocusLine(f *dispatchStatusFocus) string {
	if f == nil {
		return ""
	}
	if !f.Saturated {
		return fmt.Sprintf("focus: %d active objective(s) within WIP cap %d — new-objective spawns clear", f.Active, f.WIPCap)
	}
	stance := "WARNED (advisory, still dispatches)"
	if f.Posture == dispatchtick.FocusPostureHold {
		stance = "HELD (--focus-hold: new objectives refused)"
	}
	return fmt.Sprintf("focus: %d active objective(s) at/over WIP cap %d (%d over) — new-objective spawns %s [%s]",
		f.Active, f.WIPCap, f.ExcessWIP, stance, dispatchtick.FocusWIPSaturated)
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
	if line := dispatchStatusSpawnHealthLine(snap.SpawnHealth); line != "" {
		fmt.Fprintf(&b, "%s\n", line)
	}
	if line := dispatchStatusFocusLine(snap.Focus); line != "" {
		fmt.Fprintf(&b, "%s\n", line)
	}
	if line := dispatchProgressLine(snap.Progress); line != "" {
		b.WriteString(line + "\n")
	}
	if snap.Trajectory != nil {
		for _, line := range snap.Trajectory.Lines() {
			fmt.Fprintln(&b, line)
		}
	}
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
	fmt.Fprintf(&b, "- lanes held: %s\n", dispatchStatusLaneField(snap.LanesHeld))
	if line := dispatchStatusSpawnHealthLine(snap.SpawnHealth); line != "" {
		fmt.Fprintf(&b, "- %s\n", line)
	}
	if line := dispatchStatusFocusLine(snap.Focus); line != "" {
		fmt.Fprintf(&b, "- %s\n", line)
	}
	if line := dispatchProgressLine(snap.Progress); line != "" {
		fmt.Fprintf(&b, "- %s\n", line)
	}
	if snap.Trajectory != nil {
		for _, line := range snap.Trajectory.Lines() {
			fmt.Fprintf(&b, "- %s\n", line)
		}
	}
	b.WriteString("\n")
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
