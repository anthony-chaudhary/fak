package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

const (
	dispatchProgressSchema    = "fleet-issue-resolve-progress/1"
	dispatchWeeklySchema      = "fleet-issue-resolve-weekly-retro/1"
	dispatchProgressLoopID    = "issue-resolve-progress"
	dispatchProgressRunsDir   = ".dispatch-runs"
	dispatchProgressLogName   = "progress.jsonl"
	dispatchProgressBaseline  = "progress-baseline.json"
	dispatchProgressTargetIPH = 400.0
)

var dispatchProgressRotateBytes int64 = 8 << 20

type dispatchProgressOptions struct {
	Workspace  string
	Target     int
	MaxCommits int
	AuditJSON  string
	Close      bool
	Live       bool
	LoopLedger string
	RecordLoop bool
	AsJSON     bool
	Weekly     bool
	Since      string
	Until      string
	// FileIssues turns the weekly retro's lane-wedge section into fingerprinted,
	// deduped improvement-ticket candidates (#3338). Dry-run unless Confirm is set;
	// MaxIssues (>0) caps a run worst-first. Weekly-only — see runDispatchProgress.
	FileIssues bool
	Confirm    bool
	MaxIssues  int
	// AgingCandidates points at a ready-set JSON file (the same {id, base_weight,
	// ready_since} array dispatch-aging consumes). When set, the progress readout folds
	// it through dispatchaging so the standing census (starved/aging counts, oldest
	// wait) surfaces alongside the closure metrics (#3590). Empty = no aging fold, and
	// no new keys appear (additive: automation predating aging is unaffected).
	AgingCandidates string
}

var dispatchProgressNow = func() time.Time { return time.Now().UTC() }
var dispatchProgressOpenCount = dispatchProgressOpenCountGH
var dispatchProgressAudit = dispatchProgressAuditDefault

func runDispatchProgress(stdout, stderr io.Writer, argv []string) int {
	opts, code := parseDispatchProgressFlags(stderr, argv)
	if code != 0 {
		return code
	}
	if opts.Close {
		fmt.Fprintln(stderr, "fak dispatch progress: native --close is not implemented yet; use tools/issue_resolve_progress.py --close until #1406 lands")
		return 2
	}
	if opts.Weekly {
		report, err := evaluateDispatchWeeklyReport(opts)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch progress: %v\n", err)
			return 1
		}
		if opts.AsJSON {
			if err := writeIndentedJSON(stdout, report); err != nil {
				fmt.Fprintf(stderr, "fak dispatch progress: encode json: %v\n", err)
				return 1
			}
		} else {
			fmt.Fprint(stdout, renderDispatchWeeklyReport(report))
		}
		if opts.FileIssues {
			// The retro already computed the wedges; --file-issues turns the persistent
			// ones into deduped ticket candidates (#3338). Dry-run unless --confirm.
			return runDispatchWeeklyWedgeAudit(stdout, stderr,
				filepath.Join(opts.Workspace, dispatchProgressRunsDir),
				report, opts.Confirm, opts.MaxIssues)
		}
		return 0
	}
	payload, err := evaluateDispatchProgress(opts, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch progress: %v\n", err)
		return 1
	}
	if opts.AsJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak dispatch progress: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderDispatchProgress(payload))
	}
	if dispatchMapBool(payload, "ok") {
		return 0
	}
	return 1
}

func parseDispatchProgressFlags(stderr io.Writer, argv []string) (dispatchProgressOptions, int) {
	fs := flag.NewFlagSet("dispatch progress", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	target := fs.Int("target", 50, "issue-resolution target")
	maxCommits := fs.Int("max-commits", 2000, "history budget for the closure audit fallback")
	auditJSON := fs.String("audit-json", "", "read an existing issue_closure_audit.py --json payload instead of running the audit")
	closeArm := fs.Bool("close", false, "reserved for the native witnessed close arm")
	live := fs.Bool("live", false, "reserved for --close")
	loopLedger := fs.String("loop-ledger", "", "append this tick to a fak loop ledger (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
	noLoopLedger := fs.Bool("no-loop-ledger", false, "disable loop-ledger append for this tick")
	weekly := fs.Bool("weekly", false, "render a ledger-only weekly throughput retrospective without live mutation")
	since := fs.String("since", "", "weekly report window start (RFC3339 or YYYY-MM-DD; default: now-7d)")
	until := fs.String("until", "", "weekly report window end (RFC3339 or YYYY-MM-DD; default: now)")
	fileIssues := fs.Bool("file-issues", false, "with --weekly: emit a fingerprinted, deduped improvement-ticket candidate per PERSISTENT lane wedge (dry-run unless --confirm)")
	confirm := fs.Bool("confirm", false, "with --weekly --file-issues: actually open a gh issue per NEW wedge (default: dry-run, print only)")
	maxIssues := fs.Int("max-issues", 0, "with --weekly --file-issues --confirm: hard cap on issues filed per run, worst-first (0 = no cap)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	agingCandidates := fs.String("aging-candidates", "", "ready-set JSON ({id, base_weight, ready_since}) to fold through dispatchaging for the starved/oldest-wait readout (#3590)")
	if err := fs.Parse(argv); err != nil {
		return dispatchProgressOptions{}, 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch progress: unexpected argument %q\n", fs.Arg(0))
		return dispatchProgressOptions{}, 2
	}
	root := strings.TrimSpace(*workspace)
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch progress: getwd: %v\n", err)
			return dispatchProgressOptions{}, 1
		}
		root = wd
	}
	if *target < 0 {
		fmt.Fprintln(stderr, "fak dispatch progress: --target must be >= 0")
		return dispatchProgressOptions{}, 2
	}
	if *maxCommits <= 0 {
		fmt.Fprintln(stderr, "fak dispatch progress: --max-commits must be > 0")
		return dispatchProgressOptions{}, 2
	}
	// The wedge lens reads the weekly retro's lane-wedge section, so it has no input
	// outside --weekly. Refuse the combination up front rather than silently no-op.
	if *fileIssues && !*weekly {
		fmt.Fprintln(stderr, "fak dispatch progress: --file-issues requires --weekly")
		return dispatchProgressOptions{}, 2
	}
	return dispatchProgressOptions{
		Workspace:  root,
		Target:     *target,
		MaxCommits: *maxCommits,
		AuditJSON:  strings.TrimSpace(*auditJSON),
		Close:      *closeArm,
		Live:       *live,
		LoopLedger: *loopLedger,
		RecordLoop: !*noLoopLedger,
		AsJSON:     *asJSON,
		Weekly:     *weekly,
		Since:      strings.TrimSpace(*since),
		Until:      strings.TrimSpace(*until),
		FileIssues: *fileIssues,
		Confirm:    *confirm,
		MaxIssues:  *maxIssues,

		AgingCandidates: strings.TrimSpace(*agingCandidates),
	}, 0
}

func evaluateDispatchProgress(opts dispatchProgressOptions, stderr io.Writer) (map[string]any, error) {
	root, err := filepath.Abs(opts.Workspace)
	if err != nil {
		return nil, err
	}
	runsDir := filepath.Join(root, dispatchProgressRunsDir)
	openNow, openErr := dispatchProgressOpenCount(root)
	audit, auditErr := dispatchProgressAudit(root, stderr, opts.MaxCommits, opts.AuditJSON)
	witnessed := []int{}
	auditError := ""
	if auditErr != nil {
		auditError = auditErr.Error()
	} else if msg := dispatchMapString(audit, "_error"); msg != "" {
		auditError = msg
	} else {
		witnessed = dispatchProgressWitnessedOpen(audit)
	}

	baselineOpen, hasBaseline := dispatchProgressLoadBaseline(runsDir)
	if !hasBaseline && openErr == nil {
		baselineOpen = openNow
		hasBaseline = true
		_ = dispatchProgressSaveBaseline(runsDir, baselineOpen)
	}
	closedTotal := dispatchProgressFoldClosedHistory(runsDir)
	closuresToward, closuresRemaining := dispatchProgressClosuresTowardTarget(closedTotal, opts.Target)
	now := dispatchProgressNow().UTC()

	var openAny any
	var baselineAny any
	var resolvedAny any
	var remainingAny any
	ok := openErr == nil
	if ok {
		openAny = openNow
	}
	if hasBaseline {
		baselineAny = baselineOpen
	}
	if ok && hasBaseline {
		resolved := baselineOpen - openNow
		if resolved < 0 {
			resolved = 0
		}
		resolvedAny = resolved
		remaining := opts.Target - resolved
		if remaining < 0 {
			remaining = 0
		}
		remainingAny = remaining
	}

	rec := map[string]any{
		"schema":                 dispatchProgressSchema,
		"utc":                    now.Format("2006-01-02T15:04:05Z"),
		"target":                 opts.Target,
		"ok":                     ok,
		"open_now":               openAny,
		"baseline_open":          baselineAny,
		"resolved_toward_target": resolvedAny,
		"target_remaining":       remainingAny,
		// Backlog-independent close-N accounting (#2639): resolved_toward_target /
		// target_remaining above are net-open reduction (baseline_open - open_now),
		// which new-issue inflow can erase to 0 even while the loop witnesses real
		// closures. closures_toward_target / closures_target_remaining fold gross
		// witnessed loop closures toward the close-N target so the operator readout
		// tracks closures, not backlog drift. Kept alongside the net-open fields for
		// automation that already reads them.
		"closures_toward_target":    closuresToward,
		"closures_target_remaining": closuresRemaining,
		"witnessed_open":            len(witnessed),
		"witnessed_numbers":         dispatchProgressLimitInts(witnessed, 50),
		"closed_now":                0,
		"closed_by_loop_total":      closedTotal,
		"close_live":                nil,
		"close_result":              nil,
		"audit_error":               nil,
	}
	for key, value := range dispatchProgressHourlyProjection(runsDir, now, rec) {
		rec[key] = value
	}
	if aging, agingErr := dispatchProgressFoldAging(root, opts.AgingCandidates, now); agingErr != nil {
		rec["aging_error"] = agingErr.Error()
	} else {
		for key, value := range aging {
			rec[key] = value
		}
	}
	if auditError != "" {
		rec["audit_error"] = auditError
	}
	if openErr != nil {
		rec["open_error"] = openErr.Error()
	}
	_ = dispatchProgressAppend(filepath.Join(runsDir, dispatchProgressLogName), rec)
	if opts.RecordLoop {
		rec["loop_ledger"] = recordDispatchProgressLoop(root, opts.LoopLedger, rec)
	}
	return rec, nil
}

type dispatchWeeklyReport struct {
	Schema                         string                  `json:"schema"`
	WindowStartUTC                 string                  `json:"window_start_utc"`
	WindowEndUTC                   string                  `json:"window_end_utc"`
	WindowHours                    float64                 `json:"window_hours"`
	RowsConsidered                 int                     `json:"rows_considered"`
	TargetIssuesPerHour            float64                 `json:"target_issues_per_hour"`
	WitnessedCloses                int                     `json:"witnessed_closes"`
	AchievedWitnessedClosesPerHour float64                 `json:"achieved_witnessed_closes_per_hour"`
	CapacityLossIssues             float64                 `json:"capacity_loss_issues"`
	TopBlockers                    []dispatchWeeklyBlocker `json:"top_blockers"`
	LaneWedges                     []dispatchLaneWedge     `json:"lane_wedges"`
	NextSafeCapChange              string                  `json:"next_safe_cap_change"`
	ProgressLedger                 string                  `json:"progress_ledger"`
}

type dispatchWeeklyBlocker struct {
	Reason string `json:"reason"`
	Count  int    `json:"count"`
}

type dispatchLaneWedge struct {
	Lane                 string  `json:"lane"`
	Attempts             int     `json:"attempts"`
	WitnessedCloses      int     `json:"witnessed_closes"`
	CloseRate            float64 `json:"close_rate"`
	DominantFailureClass string  `json:"dominant_failure_class"`
	DominantFailureCount int     `json:"dominant_failure_count"`
	NextAction           string  `json:"next_action"`
}

type dispatchLaneWedgeStats struct {
	lane     string
	attempts int
	closes   int
	reasons  map[string]int
}

func evaluateDispatchWeeklyReport(opts dispatchProgressOptions) (dispatchWeeklyReport, error) {
	root, err := filepath.Abs(opts.Workspace)
	if err != nil {
		return dispatchWeeklyReport{}, err
	}
	since, until, err := dispatchWeeklyWindow(opts.Since, opts.Until)
	if err != nil {
		return dispatchWeeklyReport{}, err
	}
	runsDir := filepath.Join(root, dispatchProgressRunsDir)
	return buildDispatchWeeklyReport(runsDir, since, until)
}

func dispatchWeeklyWindow(sinceText, untilText string) (time.Time, time.Time, error) {
	now := dispatchProgressNow().UTC()
	until := now
	var err error
	if untilText != "" {
		until, err = dispatchParseReportTime(untilText, false)
		if err != nil {
			return time.Time{}, time.Time{}, fmt.Errorf("parse --until: %w", err)
		}
	}
	since := until.Add(-7 * 24 * time.Hour)
	if sinceText != "" {
		since, err = dispatchParseReportTime(sinceText, true)
		if err != nil {
			return time.Time{}, time.Time{}, fmt.Errorf("parse --since: %w", err)
		}
	}
	if !until.After(since) {
		return time.Time{}, time.Time{}, fmt.Errorf("--until must be after --since")
	}
	return since.UTC(), until.UTC(), nil
}

func dispatchParseReportTime(s string, startOfDay bool) (time.Time, error) {
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UTC(), nil
	}
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		return time.Time{}, err
	}
	if !startOfDay {
		t = t.Add(24*time.Hour - time.Nanosecond)
	}
	return t.UTC(), nil
}

func buildDispatchWeeklyReport(runsDir string, since, until time.Time) (dispatchWeeklyReport, error) {
	if !until.After(since) {
		return dispatchWeeklyReport{}, fmt.Errorf("weekly report window must be positive")
	}
	rows := dispatchProgressReadRows(runsDir)
	targetIPH := dispatchProgressTargetIPH
	closed := 0
	considered := 0
	blockers := map[string]int{}
	lanes := map[string]*dispatchLaneWedgeStats{}
	for _, row := range rows {
		at, err := time.Parse(time.RFC3339, dispatchMapString(row, "utc"))
		if err != nil || at.Before(since) || at.After(until) {
			continue
		}
		considered++
		closedNow := dispatchMapInt(row, "closed_now")
		closed += closedNow
		if target := dispatchMapFloat(row, "target_issues_per_hour"); target > 0 {
			targetIPH = target
		}
		reasons := dispatchProgressRowBlockers(row)
		for _, reason := range reasons {
			blockers[reason]++
		}
		dispatchWeeklyFoldLane(lanes, row, closedNow, reasons)
	}
	windowHours := until.Sub(since).Hours()
	achieved := 0.0
	if windowHours > 0 {
		achieved = float64(closed) / windowHours
	}
	loss := targetIPH*windowHours - float64(closed)
	if loss < 0 {
		loss = 0
	}
	top := dispatchWeeklyTopBlockers(blockers, 5)
	return dispatchWeeklyReport{
		Schema:                         dispatchWeeklySchema,
		WindowStartUTC:                 dispatchProgressProjectionStamp(since),
		WindowEndUTC:                   dispatchProgressProjectionStamp(until),
		WindowHours:                    dispatchProgressRound2(windowHours),
		RowsConsidered:                 considered,
		TargetIssuesPerHour:            dispatchProgressRound1(targetIPH),
		WitnessedCloses:                closed,
		AchievedWitnessedClosesPerHour: dispatchProgressRound1(achieved),
		CapacityLossIssues:             dispatchProgressRound1(loss),
		TopBlockers:                    top,
		LaneWedges:                     dispatchWeeklyLaneWedges(lanes, 5),
		NextSafeCapChange:              dispatchWeeklyNextCapChange(targetIPH, achieved, top),
		ProgressLedger:                 filepath.Join(runsDir, dispatchProgressLogName),
	}, nil
}

func dispatchProgressReadRows(runsDir string) []map[string]any {
	path := filepath.Join(runsDir, dispatchProgressLogName)
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	rows := []map[string]any{}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var rec map[string]any
		if err := json.Unmarshal([]byte(line), &rec); err == nil {
			rows = append(rows, rec)
		}
	}
	return rows
}

func dispatchProgressRowBlockers(row map[string]any) []string {
	reasons := []string{}
	if reason := dispatchBlockerReason(dispatchMapString(row, "reason")); reason != "" {
		reasons = append(reasons, reason)
	}
	if dispatchMapString(row, "audit_error") != "" {
		reasons = append(reasons, "AUDIT_UNAVAILABLE")
	}
	if dispatchMapString(row, "open_error") != "" {
		reasons = append(reasons, "OPEN_COUNT_UNAVAILABLE")
	}
	if closeResult, ok := row["close_result"].(map[string]any); ok {
		for _, key := range []string{"reason", "blocker_reason", "error"} {
			if reason := dispatchBlockerReason(dispatchMapString(closeResult, key)); reason != "" {
				reasons = append(reasons, reason)
			}
		}
	}
	if _, hasOK := row["ok"]; len(reasons) == 0 && hasOK && !dispatchMapBool(row, "ok") {
		reasons = append(reasons, "PROGRESS_NOT_OK")
	}
	return reasons
}

func dispatchWeeklyFoldLane(lanes map[string]*dispatchLaneWedgeStats, row map[string]any, closedNow int, reasons []string) {
	lane := strings.TrimSpace(dispatchMapString(row, "lane"))
	if lane == "" {
		lane = strings.TrimSpace(dispatchMapString(row, "path_lane"))
	}
	if lane == "" {
		return
	}
	st := lanes[lane]
	if st == nil {
		st = &dispatchLaneWedgeStats{lane: lane, reasons: map[string]int{}}
		lanes[lane] = st
	}
	st.attempts++
	st.closes += closedNow
	for _, reason := range reasons {
		st.reasons[reason]++
	}
}

func dispatchBlockerReason(s string) string {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "ok") {
		return ""
	}
	return strings.ToUpper(strings.Join(strings.Fields(s), "_"))
}

func dispatchWeeklyTopBlockers(counts map[string]int, limit int) []dispatchWeeklyBlocker {
	out := make([]dispatchWeeklyBlocker, 0, len(counts))
	for reason, count := range counts {
		out = append(out, dispatchWeeklyBlocker{Reason: reason, Count: count})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Reason < out[j].Reason
	})
	if limit >= 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func dispatchWeeklyLaneWedges(stats map[string]*dispatchLaneWedgeStats, limit int) []dispatchLaneWedge {
	out := make([]dispatchLaneWedge, 0, len(stats))
	for _, st := range stats {
		if st.attempts == 0 {
			continue
		}
		reason, reasonCount := dispatchWeeklyDominantReason(st.reasons)
		closeRate := float64(st.closes) / float64(st.attempts)
		if !dispatchLaneIsWedged(st.attempts, closeRate, reasonCount) {
			continue
		}
		if reason == "" {
			reason = "LOW_WITNESSED_CLOSE_RATE"
		}
		out = append(out, dispatchLaneWedge{
			Lane:                 st.lane,
			Attempts:             st.attempts,
			WitnessedCloses:      st.closes,
			CloseRate:            dispatchProgressRound2(closeRate),
			DominantFailureClass: reason,
			DominantFailureCount: reasonCount,
			NextAction:           dispatchLaneWedgeNextAction(st.lane, reason),
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].DominantFailureCount != out[j].DominantFailureCount {
			return out[i].DominantFailureCount > out[j].DominantFailureCount
		}
		if out[i].Attempts != out[j].Attempts {
			return out[i].Attempts > out[j].Attempts
		}
		return out[i].Lane < out[j].Lane
	})
	if limit >= 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func dispatchWeeklyDominantReason(counts map[string]int) (string, int) {
	bestReason := ""
	bestCount := 0
	for reason, count := range counts {
		if count > bestCount || (count == bestCount && (bestReason == "" || reason < bestReason)) {
			bestReason = reason
			bestCount = count
		}
	}
	return bestReason, bestCount
}

func dispatchLaneIsWedged(attempts int, closeRate float64, dominantFailures int) bool {
	if attempts < 2 {
		return false
	}
	return closeRate < 0.5 || dominantFailures >= 2
}

func dispatchLaneWedgeNextAction(lane, reason string) string {
	return "inspect " + lane + " lane; clear " + reason + " before adding workers"
}

func dispatchWeeklyNextCapChange(target, achieved float64, blockers []dispatchWeeklyBlocker) string {
	if len(blockers) > 0 {
		return "hold cap; clear " + blockers[0].Reason + " before raising"
	}
	switch {
	case achieved >= target:
		return "raise cap by 10% after one more witnessed green week"
	case achieved >= target*0.8:
		return "hold cap; collect one more near-target witnessed week"
	default:
		return "hold cap; recover witnessed close rate before raising"
	}
}

func dispatchProgressOpenCountGH(root string) (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "api", "repos/{owner}/{repo}", "--jq", ".open_issues_count")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("gh open issue count: %w (%s)", err, strings.TrimSpace(string(out)))
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		return 0, fmt.Errorf("gh open issue count: parse %q: %w", strings.TrimSpace(string(out)), err)
	}
	return n, nil
}

func dispatchProgressAuditDefault(root string, stderr io.Writer, maxCommits int, auditPath string) (map[string]any, error) {
	if auditPath != "" {
		if !filepath.IsAbs(auditPath) {
			auditPath = filepath.Join(root, auditPath)
		}
		doc, err := dispatchReadJSONFile(auditPath)
		if err != nil {
			return nil, fmt.Errorf("read audit json %s: %w", auditPath, err)
		}
		return doc, nil
	}
	return dispatchRunJSON(root, stderr, 300*time.Second,
		filepath.Join("tools", "issue_closure_audit.py"),
		"--json", "--max-commits", strconv.Itoa(maxCommits))
}

func dispatchProgressWitnessedOpen(audit map[string]any) []int {
	raw, _ := audit["issues"].([]any)
	out := make([]int, 0, len(raw))
	for _, item := range raw {
		m, ok := item.(map[string]any)
		if !ok || dispatchMapString(m, "bucket") != "OPEN_WITNESSED" {
			continue
		}
		if n := dispatchMapInt(m, "number"); n != 0 {
			out = append(out, n)
		}
	}
	return out
}

func dispatchProgressLoadBaseline(runsDir string) (int, bool) {
	doc, err := dispatchReadJSONFile(filepath.Join(runsDir, dispatchProgressBaseline))
	if err != nil {
		return 0, false
	}
	n := dispatchMapInt(doc, "baseline_open")
	return n, n != 0 || doc["baseline_open"] != nil
}

func dispatchProgressSaveBaseline(runsDir string, openNow int) error {
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return err
	}
	doc := map[string]any{
		"baseline_open": openNow,
		"recorded_utc":  dispatchProgressNow().UTC().Format("2006-01-02T15:04:05Z"),
	}
	b, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(runsDir, dispatchProgressBaseline), b, 0o644)
}

func dispatchProgressFoldClosedHistory(runsDir string) int {
	total := 0
	for _, rec := range dispatchProgressReadRows(runsDir) {
		total += dispatchMapInt(rec, "closed_now")
	}
	return total
}

// dispatchProgressClosuresTowardTarget folds gross witnessed loop closures into a
// close-N target counter that does NOT drift when new issues are filed. Unlike
// resolved_toward_target (baseline_open - open_now — a net-open reduction that
// backlog inflow can erase back to 0), this counts each witnessed closure toward
// the target: `toward` is clamped to [0, target] for the progress bar, and
// `remaining` is max(0, target - closedTotal). It depends only on the ledger fold
// of closed_now, so it survives a gh open-count outage. #2639.
func dispatchProgressClosuresTowardTarget(closedTotal, target int) (toward, remaining int) {
	if closedTotal < 0 {
		closedTotal = 0
	}
	toward = closedTotal
	if target >= 0 && toward > target {
		toward = target
	}
	remaining = target - closedTotal
	if remaining < 0 {
		remaining = 0
	}
	return toward, remaining
}

func dispatchProgressHourlyProjection(runsDir string, now time.Time, current map[string]any) map[string]any {
	samples := dispatchProgressCloseSamples(runsDir, now.Add(-time.Hour), now)
	if dispatchMapInt(current, "closed_now") > 0 {
		if at, err := time.Parse(time.RFC3339, dispatchMapString(current, "utc")); err == nil {
			samples = append(samples, dispatchProgressCloseSample{At: at, Closed: dispatchMapInt(current, "closed_now")})
		}
	}
	closed := 0
	var first, last time.Time
	for _, sample := range samples {
		closed += sample.Closed
		if first.IsZero() || sample.At.Before(first) {
			first = sample.At
		}
		if last.IsZero() || sample.At.After(last) {
			last = sample.At
		}
	}
	windowHours := 1.0
	if !first.IsZero() && !last.IsZero() && last.After(first) {
		windowHours = last.Sub(first).Hours()
		if windowHours < 1.0/60.0 {
			windowHours = 1.0 / 60.0
		}
	}
	currentIPH := 0.0
	if closed > 0 && windowHours > 0 {
		currentIPH = float64(closed) / windowHours
	}
	gap := dispatchProgressTargetIPH - currentIPH
	if gap < 0 {
		gap = 0
	}
	return map[string]any{
		"current_issues_per_hour":      dispatchProgressRound1(currentIPH),
		"target_issues_per_hour":       dispatchProgressTargetIPH,
		"issues_per_hour_gap":          dispatchProgressRound1(gap),
		"projection_closed_count":      closed,
		"projection_window_hours":      dispatchProgressRound2(windowHours),
		"projection_window_started_at": dispatchProgressProjectionStamp(first),
		"projection_window_ended_at":   dispatchProgressProjectionStamp(last),
	}
}

type dispatchProgressCloseSample struct {
	At     time.Time
	Closed int
}

func dispatchProgressCloseSamples(runsDir string, since, until time.Time) []dispatchProgressCloseSample {
	var out []dispatchProgressCloseSample
	for _, rec := range dispatchProgressReadRows(runsDir) {
		closed := dispatchMapInt(rec, "closed_now")
		if closed <= 0 {
			continue
		}
		at, err := time.Parse(time.RFC3339, dispatchMapString(rec, "utc"))
		if err != nil {
			continue
		}
		if at.Before(since) || at.After(until) {
			continue
		}
		out = append(out, dispatchProgressCloseSample{At: at, Closed: closed})
	}
	return out
}

func dispatchProgressProjectionStamp(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format("2006-01-02T15:04:05Z")
}

func dispatchProgressRound1(v float64) float64 {
	return math.Round(v*10) / 10
}

func dispatchProgressRound2(v float64) float64 {
	return math.Round(v*100) / 100
}

func rotateDispatchProgressIfDue(path string, maxBytes int64) error {
	if maxBytes <= 0 {
		return nil
	}
	fi, err := os.Stat(path)
	if err != nil {
		// Best-effort, matching the append's original inline rotate: a stat hiccup
		// (missing file or any transient error) skips rotation and still lets the
		// record be written — rotation must never fail the progress append.
		return nil
	}
	if fi.Size() < maxBytes {
		return nil
	}
	sealed := path + ".1"
	_ = os.Remove(sealed)
	return os.Rename(path, sealed)
}

// dispatchProgressAppend appends one JSON record to the progress log at logPath,
// sealing the active file to logPath+".1" first when it has reached
// dispatchProgressRotateBytes. Sealing keeps the hot append target bounded (#3475);
// a single .1 backup is retained (any prior .1 is replaced) so growth is capped at
// two segments. logPath is the full log-file path, not its directory.
func dispatchProgressAppend(logPath string, rec map[string]any) error {
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return err
	}
	if err := rotateDispatchProgressIfDue(logPath, dispatchProgressRotateBytes); err != nil {
		return err
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(b, '\n'))
	return err
}

func recordDispatchProgressLoop(root, ledger string, payload map[string]any) map[string]any {
	ledgerPath := dispatchProgressLedgerPath(root, ledger)
	runID := dispatchProgressRunID(payload)
	reason := "OK"
	if !dispatchMapBool(payload, "ok") {
		reason = "OPEN_COUNT_UNAVAILABLE"
	}
	if dispatchMapString(payload, "audit_error") != "" {
		reason = "AUDIT_UNAVAILABLE"
	}
	metrics := dispatchProgressMetrics(payload)
	evidence := []loopmgr.EvidenceRef{{Kind: "progress_log", Ref: filepath.Join(root, dispatchProgressRunsDir, dispatchProgressLogName)}}
	for _, n := range dispatchProgressIntSlice(payload["witnessed_numbers"]) {
		evidence = append(evidence, loopmgr.EvidenceRef{Kind: "open_witnessed_issue", Ref: strconv.Itoa(n)})
	}

	admitStatus := loopmgr.StatusAdmitted
	endStatus := loopmgr.StatusClaimedDone
	if !dispatchMapBool(payload, "ok") {
		admitStatus = loopmgr.StatusRefused
		endStatus = loopmgr.StatusFailed
	}
	events := []loopmgr.Event{
		{LoopID: dispatchProgressLoopID, RunID: runID, Kind: loopmgr.EventFire, Source: "fak dispatch progress", Summary: fmt.Sprintf("issue progress tick target=%d", dispatchMapInt(payload, "target")), Metrics: metrics, EvidenceRefs: evidence},
		{LoopID: dispatchProgressLoopID, RunID: runID, Kind: loopmgr.EventAdmit, Source: "fak dispatch progress", Status: admitStatus, Reason: reason, Summary: fmt.Sprintf("open=%v target_remaining=%v", payload["open_now"], payload["target_remaining"]), Metrics: metrics, EvidenceRefs: evidence},
		{LoopID: dispatchProgressLoopID, RunID: runID, Kind: loopmgr.EventEnd, Source: "fak dispatch progress", Status: endStatus, Reason: reason, Summary: fmt.Sprintf("closed_now=%d witnessed_open=%d", dispatchMapInt(payload, "closed_now"), dispatchMapInt(payload, "witnessed_open")), Metrics: metrics, EvidenceRefs: evidence},
	}
	if dispatchMapBool(payload, "ok") {
		status := loopmgr.StatusWitnessedDone
		verified := "verified_done"
		if dispatchMapString(payload, "audit_error") != "" {
			status = loopmgr.StatusWitnessUnavailable
			verified = "verified_unavailable"
		}
		events = append(events, loopmgr.Event{LoopID: dispatchProgressLoopID, RunID: runID, Kind: loopmgr.EventWitness, Source: "fak dispatch progress", Status: status, Reason: reason, Summary: fmt.Sprintf("open_count=%v audit_error=%s", payload["open_now"], dispatchMapString(payload, "audit_error")), Metrics: metrics, EvidenceRefs: append(evidence, loopmgr.EvidenceRef{Kind: "verified_state", Ref: verified})})
	}

	rows := []map[string]any{}
	ok := true
	for _, ev := range events {
		row, err := loopmgr.Append(ledgerPath, ev)
		if err != nil {
			ok = false
			rows = append(rows, map[string]any{"ok": false, "kind": string(ev.Kind), "error": err.Error()})
			continue
		}
		rows = append(rows, map[string]any{"ok": true, "kind": string(row.Kind), "seq": row.Seq, "hash": row.Hash})
	}
	return map[string]any{"ledger": ledgerPath, "loop_id": dispatchProgressLoopID, "run_id": runID, "events": rows, "ok": ok}
}

func dispatchProgressLedgerPath(root, ledger string) string {
	if strings.TrimSpace(ledger) == "" {
		ledger = defaultLoopLedger()
	}
	if filepath.IsAbs(ledger) {
		return ledger
	}
	return filepath.Join(root, ledger)
}

func dispatchProgressRunID(payload map[string]any) string {
	stamp := dispatchMapString(payload, "utc")
	stamp = strings.NewReplacer("-", "", ":", "", "Z", "").Replace(stamp)
	if stamp == "" {
		stamp = dispatchProgressNow().UTC().Format("20060102T150405")
	}
	return "progress-" + stamp
}

func dispatchProgressMetrics(payload map[string]any) map[string]int64 {
	keys := []string{
		"target", "open_now", "baseline_open", "resolved_toward_target",
		"target_remaining", "closures_toward_target", "closures_target_remaining",
		"witnessed_open", "closed_now", "closed_by_loop_total",
		"starved_count", "aging_count", "oldest_wait_seconds",
	}
	out := map[string]int64{}
	for _, key := range keys {
		if payload[key] == nil {
			continue
		}
		out[key] = int64(dispatchMapInt(payload, key))
	}
	return out
}

func dispatchProgressLimitInts(values []int, limit int) []int {
	if limit >= 0 && len(values) > limit {
		values = values[:limit]
	}
	out := append([]int(nil), values...)
	return out
}

func dispatchProgressIntSlice(v any) []int {
	switch x := v.(type) {
	case []int:
		return append([]int(nil), x...)
	case []any:
		out := make([]int, 0, len(x))
		for _, item := range x {
			switch n := item.(type) {
			case int:
				out = append(out, n)
			case float64:
				out = append(out, int(n))
			case json.Number:
				if i, err := n.Int64(); err == nil {
					out = append(out, int(i))
				}
			}
		}
		return out
	default:
		return nil
	}
}

// dispatchProgressFoldAging folds the ready-set at candidatesPath through dispatchaging so the
// standing readout carries the anti-starvation census (#3590): starved_count, aging_count, and
// oldest_wait_seconds. The candidate JSON is the same {id, base_weight, ready_since} array
// dispatch-aging consumes (so the two surfaces share one input), read via the shared
// decodeCandidates. Before the fold, each candidate inherits its live cooldown window from the
// runs-dir recently-attempted screen (overlayLiveCoolingWindows, #3715), so a cooling unit's
// wait clock is PAUSED over the span it was ineligible instead of counting phantom starvation.
// An empty path returns a nil map (no fold, no new keys) — additive by design,
// so a progress reader predating aging sees no change. A read/parse failure is surfaced by the
// caller as aging_error rather than failing the whole progress tick.
func dispatchProgressFoldAging(root, candidatesPath string, now time.Time) (map[string]any, error) {
	if strings.TrimSpace(candidatesPath) == "" {
		return nil, nil
	}
	path := candidatesPath
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read aging candidates %s: %w", path, err)
	}
	cands, err := decodeCandidates(raw)
	if err != nil {
		return nil, fmt.Errorf("aging candidates %s: %w", path, err)
	}
	overlayLiveCoolingWindows(cands, cooldownIssueRowsAt(filepath.Join(root, dispatchProgressRunsDir), dispatchtick.DefaultCooldownMinutes, now))
	res := dispatchaging.Fold(cands, dispatchaging.DefaultParams(now.Unix()))
	return map[string]any{
		"starved_count":       res.StarvedCount,
		"aging_count":         res.AgingCount,
		"oldest_wait_seconds": res.OldestWaitSeconds,
	}, nil
}

// overlayLiveCoolingWindows stamps each ready candidate's CoolingSince/CoolingUntil from the
// live dispatch path's recently-attempted screen (#3715): a candidate whose ID is an issue
// number with a recorded worker attempt inherits that attempt's latest cooldown window (last
// attempt -> next eligible, exactly the span dispatch_tick's cooldown skip enforced), so the
// aging fold PAUSES its wait clock over the ineligible span instead of accruing phantom
// starvation pressure — and a unit that just cooled never queue-jumps on wait it could not have
// used. A candidate that already declares a window keeps it (the supplied input wins), a
// non-numeric ID or a never-attempted issue is untouched, and with no runs dir the row set is
// empty — so non-cooling candidates fold byte-identically to before (the additive direction).
func overlayLiveCoolingWindows(cands []dispatchaging.Candidate, rows []dispatchCooldownRow) {
	if len(cands) == 0 || len(rows) == 0 {
		return
	}
	byIssue := make(map[int]dispatchCooldownRow, len(rows))
	for _, row := range rows {
		byIssue[row.Issue] = row
	}
	for i, c := range cands {
		if c.CoolingUntil != 0 {
			continue // an explicitly declared window wins over the derived one
		}
		n, err := strconv.Atoi(strings.TrimPrefix(strings.TrimSpace(c.ID), "#"))
		if err != nil {
			continue
		}
		if row, ok := byIssue[n]; ok {
			cands[i].CoolingSince, cands[i].CoolingUntil = row.LastAttemptUnix, row.NextEligibleUnix
		}
	}
}

func renderDispatchProgress(p map[string]any) string {
	target := dispatchMapInt(p, "target")
	// The progress bar tracks witnessed closures toward the close-N target, not
	// net-open reduction — new-issue inflow must not erase closure progress (#2639).
	bar := dispatchProgressBar(dispatchMapInt(p, "closures_toward_target"), target)
	var b strings.Builder
	fmt.Fprintf(&b, "issue-resolve-progress: open=%v (baseline %v)  witnessed closures toward %d: %s\n",
		p["open_now"], p["baseline_open"], target, bar)
	fmt.Fprintf(&b, "  witnessed-open (closeable now): %d  %v\n", dispatchMapInt(p, "witnessed_open"), p["witnessed_numbers"])
	fmt.Fprintf(&b, "  closed this tick: %d  closed-by-loop total: %d  witnessed closures remaining to %d: %v\n",
		dispatchMapInt(p, "closed_now"), dispatchMapInt(p, "closed_by_loop_total"), target, p["closures_target_remaining"])
	fmt.Fprintf(&b, "  net-open reduction (baseline-open, drifts with backlog): %v  net-open remaining: %v\n",
		p["resolved_toward_target"], p["target_remaining"])
	fmt.Fprintf(&b, "  hourly projection: current=%.1f/h target=%.1f/h gap=%.1f/h closes=%d window=%.2fh\n",
		dispatchMapFloat(p, "current_issues_per_hour"),
		dispatchMapFloat(p, "target_issues_per_hour"),
		dispatchMapFloat(p, "issues_per_hour_gap"),
		dispatchMapInt(p, "projection_closed_count"),
		dispatchMapFloat(p, "projection_window_hours"))
	// Anti-starvation census, folded from dispatchaging over the ready set (#3590). Only
	// rendered when an aging fold ran (the --aging-candidates ready set was supplied).
	if _, ok := p["starved_count"]; ok {
		fmt.Fprintf(&b, "  starved: %d  aging: %d  oldest-wait: %s\n",
			dispatchMapInt(p, "starved_count"), dispatchMapInt(p, "aging_count"),
			humanDur(int64(dispatchMapInt(p, "oldest_wait_seconds"))))
	}
	if errText := dispatchMapString(p, "aging_error"); errText != "" {
		fmt.Fprintf(&b, "  ! aging error: %s\n", errText)
	}
	if errText := dispatchMapString(p, "audit_error"); errText != "" {
		fmt.Fprintf(&b, "  ! audit error: %s\n", errText)
	}
	if errText := dispatchMapString(p, "open_error"); errText != "" {
		fmt.Fprintf(&b, "  ! open-count error: %s\n", errText)
	}
	return b.String()
}

func renderDispatchWeeklyReport(r dispatchWeeklyReport) string {
	var b strings.Builder
	fmt.Fprintln(&b, "# Dispatch Weekly Throughput Retrospective")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| metric | value |")
	fmt.Fprintln(&b, "|---|---:|")
	fmt.Fprintf(&b, "| window | %s to %s (%.2fh) |\n", r.WindowStartUTC, r.WindowEndUTC, r.WindowHours)
	fmt.Fprintf(&b, "| progress rows | %d |\n", r.RowsConsidered)
	fmt.Fprintf(&b, "| target witnessed closes/hour | %.1f |\n", r.TargetIssuesPerHour)
	fmt.Fprintf(&b, "| achieved witnessed closes/hour | %.1f |\n", r.AchievedWitnessedClosesPerHour)
	fmt.Fprintf(&b, "| witnessed closes | %d |\n", r.WitnessedCloses)
	fmt.Fprintf(&b, "| capacity loss | %.1f issue(s) |\n", r.CapacityLossIssues)
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Top Blockers")
	if len(r.TopBlockers) == 0 {
		fmt.Fprintln(&b, "- none")
	} else {
		for _, blocker := range r.TopBlockers {
			fmt.Fprintf(&b, "- %s: %d\n", blocker.Reason, blocker.Count)
		}
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Lane Wedges")
	if len(r.LaneWedges) == 0 {
		fmt.Fprintln(&b, "- none")
	} else {
		for _, wedge := range r.LaneWedges {
			fmt.Fprintf(&b, "- %s: attempts=%d closes=%d close_rate=%.2f dominant=%s/%d next=%s\n",
				wedge.Lane, wedge.Attempts, wedge.WitnessedCloses, wedge.CloseRate,
				wedge.DominantFailureClass, wedge.DominantFailureCount, wedge.NextAction)
		}
	}
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Next safe cap change: %s\n", r.NextSafeCapChange)
	return b.String()
}

func dispatchProgressBar(resolved, target int) string {
	if target <= 0 {
		return fmt.Sprintf("%d/%d", resolved, target)
	}
	filled := resolved
	if filled > target {
		filled = target
	}
	width := 30
	n := width * filled / target
	return "[" + strings.Repeat("#", n) + strings.Repeat("-", width-n) + fmt.Sprintf("] %d/%d", filled, target)
}

func dispatchMapFloat(m map[string]any, key string) float64 {
	switch v := m[key].(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case json.Number:
		f, _ := v.Float64()
		return f
	default:
		return 0
	}
}
