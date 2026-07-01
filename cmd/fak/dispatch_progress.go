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
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

const (
	dispatchProgressSchema    = "fleet-issue-resolve-progress/1"
	dispatchProgressLoopID    = "issue-resolve-progress"
	dispatchProgressRunsDir   = ".dispatch-runs"
	dispatchProgressLogName   = "progress.jsonl"
	dispatchProgressBaseline  = "progress-baseline.json"
	dispatchProgressTargetIPH = 400.0
)

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
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
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
		"witnessed_open":         len(witnessed),
		"witnessed_numbers":      dispatchProgressLimitInts(witnessed, 50),
		"closed_now":             0,
		"closed_by_loop_total":   closedTotal,
		"close_live":             nil,
		"close_result":           nil,
		"audit_error":            nil,
	}
	for key, value := range dispatchProgressHourlyProjection(runsDir, now, rec) {
		rec[key] = value
	}
	if auditError != "" {
		rec["audit_error"] = auditError
	}
	if openErr != nil {
		rec["open_error"] = openErr.Error()
	}
	_ = dispatchProgressAppend(runsDir, rec)
	if opts.RecordLoop {
		rec["loop_ledger"] = recordDispatchProgressLoop(root, opts.LoopLedger, rec)
	}
	return rec, nil
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
	path := filepath.Join(runsDir, dispatchProgressLogName)
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	total := 0
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var rec map[string]any
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue
		}
		total += dispatchMapInt(rec, "closed_now")
	}
	return total
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
	path := filepath.Join(runsDir, dispatchProgressLogName)
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []dispatchProgressCloseSample
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var rec map[string]any
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue
		}
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

func dispatchProgressAppend(runsDir string, rec map[string]any) error {
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return err
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(runsDir, dispatchProgressLogName), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
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
		"target_remaining", "witnessed_open", "closed_now", "closed_by_loop_total",
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

func renderDispatchProgress(p map[string]any) string {
	target := dispatchMapInt(p, "target")
	resolved := dispatchMapInt(p, "resolved_toward_target")
	bar := dispatchProgressBar(resolved, target)
	var b strings.Builder
	fmt.Fprintf(&b, "issue-resolve-progress: open=%v (baseline %v)  toward %d: %s\n",
		p["open_now"], p["baseline_open"], target, bar)
	fmt.Fprintf(&b, "  witnessed-open (closeable now): %d  %v\n", dispatchMapInt(p, "witnessed_open"), p["witnessed_numbers"])
	fmt.Fprintf(&b, "  closed this tick: %d  closed-by-loop total: %d  remaining to %d: %v\n",
		dispatchMapInt(p, "closed_now"), dispatchMapInt(p, "closed_by_loop_total"), target, p["target_remaining"])
	fmt.Fprintf(&b, "  hourly projection: current=%.1f/h target=%.1f/h gap=%.1f/h closes=%d window=%.2fh\n",
		dispatchMapFloat(p, "current_issues_per_hour"),
		dispatchMapFloat(p, "target_issues_per_hour"),
		dispatchMapFloat(p, "issues_per_hour_gap"),
		dispatchMapInt(p, "projection_closed_count"),
		dispatchMapFloat(p, "projection_window_hours"))
	if errText := dispatchMapString(p, "audit_error"); errText != "" {
		fmt.Fprintf(&b, "  ! audit error: %s\n", errText)
	}
	if errText := dispatchMapString(p, "open_error"); errText != "" {
		fmt.Fprintf(&b, "  ! open-count error: %s\n", errText)
	}
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
