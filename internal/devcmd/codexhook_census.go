package devcmd

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	codexHookCensusSchema         = "fak/codex-hook-census/v2"
	codexPostToolSettlementWindow = 5 * time.Second
)

type lifecycleCounts struct {
	Denominator int    `json:"denominator"`
	Attempted   int    `json:"attempted"`
	Succeeded   int    `json:"succeeded"`
	Blocked     int    `json:"blocked,omitempty"`
	InvalidJSON int    `json:"invalid_json,omitempty"`
	Failed      int    `json:"failed"`
	Skipped     int    `json:"skipped"`
	Disabled    int    `json:"disabled"`
	Unknown     int    `json:"unknown"`
	Source      string `json:"source"`
}

type hookCensusReport struct {
	Schema                 string             `json:"schema"`
	GeneratedAt            time.Time          `json:"generated_at"`
	Window                 string             `json:"window"`
	CodexHome              string             `json:"codex_home"`
	LogStore               string             `json:"log_store"`
	Workspace              string             `json:"workspace"`
	ObservationStore       string             `json:"observation_store"`
	ProfileMatch           bool               `json:"profile_match"`
	DispatchedCalls        int                `json:"dispatched_calls"`
	DispatchSource         string             `json:"dispatch_source"`
	PreToolUse             lifecycleCounts    `json:"pre_tool_use"`
	PostToolUse            lifecycleCounts    `json:"post_tool_use"`
	PostToolUseRequirement string             `json:"post_tool_use_requirement"`
	PostToolUseStatus      string             `json:"post_tool_use_status"`
	Stop                   lifecycleCounts    `json:"stop"`
	StopFailure            lifecycleCounts    `json:"stop_failure"`
	SubagentStop           lifecycleCounts    `json:"subagent_stop"`
	StopSource             string             `json:"stop_source,omitempty"`
	StopRuns               []stopLifecycleRow `json:"stop_runs,omitempty"`
	TelemetryFresh         bool               `json:"telemetry_fresh"`
	NewestReceipt          time.Time          `json:"newest_receipt,omitempty"`
	Verdict                string             `json:"verdict"`
	PostToolSettlement     string             `json:"post_tool_settlement"`
	Reasons                []string           `json:"reasons,omitempty"`
}

type hookObservation struct {
	CallID     string    `json:"call_id"`
	SessionID  string    `json:"session_id"`
	Workspace  string    `json:"workspace"`
	Profile    string    `json:"profile"`
	PhaseState string    `json:"phase_state"`
	Exit       int       `json:"exit"`
	Verb       string    `json:"verb"`
	TS         time.Time `json:"ts"`
	Outcome    string    `json:"outcome"`
}
type dispatchedCall struct {
	CallID       string
	SessionID    string
	Workspace    string
	ToolName     string
	DispatchedAt time.Time
	Completed    bool
	CompletedAt  time.Time
}

type stopLifecycleRow struct {
	ThreadID    string `json:"thread_id"`
	TurnID      string `json:"turn_id,omitempty"`
	RunID       string `json:"run_id"`
	EventName   string `json:"event_name"`
	HandlerType string `json:"handler_type,omitempty"`
	Source      string `json:"source,omitempty"`
	SourcePath  string `json:"source_path,omitempty"`
	Order       int    `json:"display_order"`
	Status      string `json:"status"`
	StatusText  string `json:"status_message,omitempty"`
	StartedAt   int64  `json:"started_at,omitempty"`
	CompletedAt int64  `json:"completed_at,omitempty"`
}

type hookNotification struct {
	Method string `json:"method"`
	Params struct {
		ThreadID string `json:"threadId"`
		TurnID   string `json:"turnId"`
		Run      struct {
			ID            string `json:"id"`
			EventName     string `json:"eventName"`
			HandlerType   string `json:"handlerType"`
			Source        string `json:"source"`
			SourcePath    string `json:"sourcePath"`
			DisplayOrder  int    `json:"displayOrder"`
			Status        string `json:"status"`
			StatusMessage string `json:"statusMessage"`
			StartedAt     int64  `json:"startedAt"`
			CompletedAt   int64  `json:"completedAt"`
			Entries       []struct {
				Kind string `json:"kind"`
				Text string `json:"text"`
			} `json:"entries"`
		} `json:"run"`
	} `json:"params"`
}

func addStopLifecycle(report *hookCensusReport, path string, after time.Time, threadID string) error {
	abs, _ := filepath.Abs(path)
	report.StopSource = abs
	f, err := os.Open(abs)
	if err != nil {
		return err
	}
	defer f.Close()
	started := map[string]hookNotification{}
	completed := map[string]hookNotification{}
	invalid := map[string]int{"stop": 0, "stopFailure": 0, "subagentStop": 0}
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for s.Scan() {
		line := bytes.TrimSpace(s.Bytes())
		if len(line) == 0 {
			continue
		}
		var n hookNotification
		if err := json.Unmarshal(line, &n); err != nil {
			// Only malformed rows that claim to be hook lifecycle evidence enter the denominator.
			lower := strings.ToLower(string(line))
			if strings.Contains(lower, "hook/") && strings.Contains(lower, "stop") {
				invalid["stop"]++
			}
			continue
		}
		if n.Method != "hook/started" && n.Method != "hook/completed" {
			continue
		}
		event := n.Params.Run.EventName
		if event != "stop" && event != "stopFailure" && event != "subagentStop" {
			continue
		}
		if threadID != "" && n.Params.ThreadID != threadID {
			continue
		}
		when := n.Params.Run.StartedAt
		if n.Params.Run.CompletedAt > 0 {
			when = n.Params.Run.CompletedAt
		}
		if when > 0 && hookRunTime(when).Before(after) {
			continue
		}
		key := n.Params.ThreadID + "\x00" + n.Params.Run.ID
		if n.Params.Run.ID == "" {
			invalid[event]++
			continue
		}
		key = stopRunKey(n)
		if n.Method == "hook/started" {
			started[key] = n
		} else {
			completed[key] = n
		}
	}
	if err := s.Err(); err != nil {
		return err
	}
	for key, n := range started {
		if _, ok := completed[key]; ok {
			continue
		}
		c := stopCounts(report, n.Params.Run.EventName)
		c.Denominator++
		c.Attempted++
		c.Unknown++
		report.StopRuns = append(report.StopRuns, stopRow(n, "unknown"))
	}
	for _, n := range completed {
		c := stopCounts(report, n.Params.Run.EventName)
		c.Denominator++
		c.Attempted++
		status := strings.ToLower(n.Params.Run.Status)
		switch status {
		case "completed":
			c.Succeeded++
		case "blocked":
			c.Blocked++
		case "failed":
			if invalidHookOutput(n) {
				c.InvalidJSON++
			} else {
				c.Failed++
			}
		case "stopped":
			c.Skipped++
		default:
			c.Unknown++
		}
		report.StopRuns = append(report.StopRuns, stopRow(n, status))
	}
	for event, n := range invalid {
		if n == 0 {
			continue
		}
		c := stopCounts(report, event)
		c.Denominator += n
		c.Attempted += n
		c.InvalidJSON += n
	}
	for _, c := range []*lifecycleCounts{&report.Stop, &report.StopFailure, &report.SubagentStop} {
		c.Source = abs
	}
	sort.Slice(report.StopRuns, func(i, j int) bool {
		if report.StopRuns[i].StartedAt != report.StopRuns[j].StartedAt {
			return report.StopRuns[i].StartedAt < report.StopRuns[j].StartedAt
		}
		return report.StopRuns[i].Order < report.StopRuns[j].Order
	})
	return nil
}

func invalidHookOutput(n hookNotification) bool {
	for _, entry := range n.Params.Run.Entries {
		text := strings.ToLower(entry.Text)
		if strings.Contains(text, "invalid json") || strings.Contains(text, "invalid-json") || strings.Contains(text, "json parse") {
			return true
		}
	}
	return false
}

func hookRunTime(v int64) time.Time {
	// Codex HookRunSummary uses Unix seconds despite the historical field comment saying milliseconds.
	if v < 100_000_000_000 {
		return time.Unix(v, 0)
	}
	return time.UnixMilli(v)
}

func stopRunKey(n hookNotification) string {
	return n.Params.ThreadID + "\x00" + n.Params.Run.ID
}

func stopCounts(r *hookCensusReport, event string) *lifecycleCounts {
	switch event {
	case "stopFailure":
		return &r.StopFailure
	case "subagentStop":
		return &r.SubagentStop
	default:
		return &r.Stop
	}
}

func stopRow(n hookNotification, status string) stopLifecycleRow {
	r := n.Params.Run
	return stopLifecycleRow{ThreadID: n.Params.ThreadID, TurnID: n.Params.TurnID, RunID: r.ID, EventName: r.EventName, HandlerType: r.HandlerType, Source: r.Source, SourcePath: r.SourcePath, Order: r.DisplayOrder, Status: status, StatusText: r.StatusMessage, StartedAt: r.StartedAt, CompletedAt: r.CompletedAt}
}

type rolloutRow struct {
	Timestamp time.Time `json:"timestamp"`
	Type      string    `json:"type"`
	Payload   struct {
		Type   string `json:"type"`
		CallID string `json:"call_id"`
		Name   string `json:"name"`
	} `json:"payload"`
}

func RunCodexHookCensus(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("codex-hook-census", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	home := fs.String("codex-home", "", "active Codex home")
	logHome := fs.String("log-home", "", "Codex home whose sessions are counted")
	workspace := fs.String("workspace", ".", "workspace whose calls are counted")
	binary := fs.String("codex-binary", "", "Codex executable used to resolve the effective hook profile")
	threadID := fs.String("thread-id", os.Getenv("CODEX_THREAD_ID"), "Codex thread whose calls are counted (empty counts workspace)")
	observations := fs.String("observations", filepath.Join(".dos", "metrics", "observations.jsonl"), "hook observation JSONL")
	stopEvents := fs.String("stop-events", "", "Codex app-server hook notification JSONL")
	since := fs.Duration("since", 24*time.Hour, "lookback window")
	asJSON := fs.Bool("json", false, "emit JSON")
	nowText := fs.String("now", "", "fixed RFC3339 clock (tests/captures)")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *since <= 0 {
		fmt.Fprintln(stderr, "usage: fak-dev codex-hook-census [--codex-home DIR] [--log-home DIR] [--codex-binary FILE] [--observations FILE] [--stop-events FILE] [--since 24h] [--json]")
		return 2
	}
	now := time.Now().UTC()
	if *nowText != "" {
		var err error
		now, err = time.Parse(time.RFC3339, *nowText)
		if err != nil {
			fmt.Fprintf(stderr, "codex-hook-census: --now: %v\n", err)
			return 2
		}
	}
	if *home == "" {
		*home = os.Getenv("CODEX_HOME")
	}
	if *home == "" {
		h, err := os.UserHomeDir()
		if err != nil {
			return 1
		}
		*home = filepath.Join(h, ".codex")
	}
	if *logHome == "" {
		*logHome = *home
	}
	profile, err := inspectCodexHookProfile(*home, *workspace, *binary)
	if err != nil {
		fmt.Fprintf(stderr, "codex-hook-census: profile: %v\n", err)
		return 1
	}
	report, err := buildHookCensus(*home, *logHome, *workspace, *threadID, *observations, profile, *since, now)
	if err != nil {
		fmt.Fprintf(stderr, "codex-hook-census: %v\n", err)
		return 1
	}
	if *stopEvents != "" {
		if err := addStopLifecycle(&report, *stopEvents, now.Add(-*since), *threadID); err != nil {
			fmt.Fprintf(stderr, "codex-hook-census: stop events: %v\n", err)
			return 1
		}
	}
	finalizeCensusVerdict(&report)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(report)
	} else {
		writeHookCensus(stdout, report)
	}
	if report.Verdict != "HEALTHY" {
		return 1
	}
	return 0
}

func buildHookCensus(home, logHome, workspace, threadID, observations string, profile hookProfileReport, since time.Duration, now time.Time) (hookCensusReport, error) {
	absHome, _ := filepath.Abs(home)
	absLog, _ := filepath.Abs(logHome)
	absObs, _ := filepath.Abs(observations)
	absWork, _ := filepath.Abs(workspace)
	calls, err := readCodexDispatches(filepath.Join(absLog, "sessions"), absWork, threadID, now.Add(-since))
	if err != nil {
		return hookCensusReport{}, err
	}
	receipts, newest, err := readHookObservations(absObs, now.Add(-since))
	if err != nil {
		return hookCensusReport{}, err
	}
	r := hookCensusReport{Schema: codexHookCensusSchema, GeneratedAt: now, Window: since.String(), PostToolSettlement: codexPostToolSettlementWindow.String(), CodexHome: absHome, LogStore: absLog, Workspace: absWork, ObservationStore: absObs, ProfileMatch: samePath(absHome, absLog), DispatchedCalls: len(calls), DispatchSource: filepath.Join(absLog, "sessions"), NewestReceipt: newest, TelemetryFresh: !newest.IsZero() && now.Sub(newest) <= 5*time.Minute, PostToolUseRequirement: "optional"}
	r.PreToolUse = phaseCountsForCalls(calls, receipts["pretool"], absHome, absWork, absObs)
	r.PostToolUse = postToolPhaseCounts(postToolEligibleDispatches(calls, now.Add(-codexPostToolSettlementWindow)), receipts["posttool"], absHome, absWork, absObs)
	applyHookProfileToCensus(&r, profile)
	if !r.TelemetryFresh {
		r.Reasons = append(r.Reasons, "STALE_TELEMETRY")
	}
	if !r.ProfileMatch {
		r.Reasons = append(r.Reasons, "PROFILE_LOG_STORE_MISMATCH")
	}
	if len(calls) == 0 {
		r.Reasons = append(r.Reasons, "DISPATCH_DENOMINATOR_UNOBSERVED")
	}
	finalizeCensusVerdict(&r)
	return r, nil
}

func applyHookProfileToCensus(r *hookCensusReport, profile hookProfileReport) {
	present := false
	for _, h := range profile.Hooks {
		if normalizeHookEvent(h.EventName) == "post_tool_use" {
			present = true
			break
		}
	}
	r.PostToolUseRequirement = "optional"
	if !present {
		source := r.PostToolUse.Source
		r.PostToolUse = lifecycleCounts{Source: source}
		r.PostToolUseStatus = "intentionally_disabled"
		return
	}
	r.PostToolUseStatus = "observed"
	if r.PostToolUse.Disabled > 0 {
		r.PostToolUseStatus = "disabled"
	} else if r.PostToolUse.Failed > 0 {
		r.PostToolUseStatus = "failing"
	} else if r.PostToolUse.Unknown > 0 {
		r.PostToolUseStatus = "incomplete"
	}
}

func finalizeCensusVerdict(r *hookCensusReport) {
	reasons := append([]string(nil), r.Reasons...)
	if r.PreToolUse.Unknown > 0 {
		reasons = append(reasons, "PRE_TOOL_USE_UNKNOWN")
	}
	if r.PostToolUseStatus != "intentionally_disabled" && r.PostToolUse.Unknown > 0 {
		reasons = append(reasons, "POST_TOOL_USE_UNKNOWN")
	}
	if r.PreToolUse.Failed > 0 || r.PostToolUse.Failed > 0 {
		reasons = append(reasons, "HOOK_FAILURES_PRESENT")
	}
	if r.PostToolUse.Disabled > 0 {
		reasons = append(reasons, "POST_TOOL_USE_DISABLED")
	}
	for _, stop := range []struct {
		name string
		c    lifecycleCounts
	}{
		{"STOP", r.Stop}, {"STOP_FAILURE", r.StopFailure}, {"SUBAGENT_STOP", r.SubagentStop},
	} {
		if stop.c.InvalidJSON > 0 {
			reasons = append(reasons, stop.name+"_INVALID_JSON")
		}
		if stop.c.Failed > 0 {
			reasons = append(reasons, stop.name+"_FAILED")
		}
		if stop.c.Unknown > 0 {
			reasons = append(reasons, stop.name+"_UNKNOWN")
		}
	}
	r.Reasons = uniqueStrings(reasons)
	r.Verdict = "HEALTHY"
	if len(r.Reasons) > 0 {
		r.Verdict = "UNHEALTHY"
	}
}

func uniqueStrings(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

func samePath(a, b string) bool { return strings.EqualFold(filepath.Clean(a), filepath.Clean(b)) }
func phaseCounts(den int, obs []hookObservation, source string) lifecycleCounts {
	c := lifecycleCounts{Denominator: den, Source: source}
	for _, o := range obs {
		classifyPhase(&c, o)
	}
	accounted := c.Succeeded + c.Failed + c.Skipped + c.Disabled
	if accounted < den {
		c.Unknown = den - accounted
	}
	return c
}

func postToolEligibleDispatches(calls []dispatchedCall, settledBefore time.Time) []dispatchedCall {
	out := make([]dispatchedCall, 0, len(calls))
	for _, call := range calls {
		if call.Completed && !call.CompletedAt.After(settledBefore) && codexPostToolMatcher(call.ToolName) {
			out = append(out, call)
		}
	}
	return out
}
func codexPostToolMatcher(toolName string) bool {
	switch strings.ToLower(toolName) {
	case "read", "bash", "grep", "glob", "exec", "exec_command":
		return true
	default:
		return false
	}
}

func postToolPhaseCounts(calls []dispatchedCall, obs []hookObservation, profile, workspace, source string) lifecycleCounts {
	c := lifecycleCounts{Denominator: len(calls), Source: source}
	filtered := make([]hookObservation, 0, len(obs))
	byExact := make(map[string]hookObservation, len(obs))
	for _, o := range obs {
		if !samePath(o.Workspace, workspace) || !samePath(o.Profile, profile) {
			continue
		}
		filtered = append(filtered, o)
		byExact[o.SessionID+"\x00"+o.CallID] = o
	}

	// Reserve stable-ID receipts before considering host-generated IDs. Otherwise an
	// earlier missing call can consume a later call's exact receipt and shift the queue.
	used := make(map[string]bool, len(filtered))
	unmatched := make([]dispatchedCall, 0, len(calls))
	for _, call := range calls {
		key := call.SessionID + "\x00" + call.CallID
		if o, ok := byExact[key]; ok {
			used[key] = true
			classifyPhase(&c, o)
			continue
		}
		unmatched = append(unmatched, call)
	}
	sort.Slice(unmatched, func(i, j int) bool {
		if unmatched[i].SessionID != unmatched[j].SessionID {
			return unmatched[i].SessionID < unmatched[j].SessionID
		}
		return unmatched[i].CompletedAt.Before(unmatched[j].CompletedAt)
	})
	sort.Slice(filtered, func(i, j int) bool {
		if filtered[i].SessionID != filtered[j].SessionID {
			return filtered[i].SessionID < filtered[j].SessionID
		}
		return filtered[i].TS.Before(filtered[j].TS)
	})
	for _, call := range unmatched {
		for _, o := range filtered {
			observationKey := o.SessionID + "\x00" + o.CallID
			if used[observationKey] || !postObservationCorrelates(call, o) {
				continue
			}
			used[observationKey] = true
			classifyPhase(&c, o)
			break
		}
	}
	accounted := c.Succeeded + c.Failed + c.Skipped + c.Disabled
	c.Unknown = len(calls) - accounted
	return c
}

func postObservationCorrelates(call dispatchedCall, observation hookObservation) bool {
	if call.SessionID != observation.SessionID || !strings.HasPrefix(call.CallID, "call_") {
		return false
	}
	if !strings.EqualFold(call.ToolName, "exec") && !strings.EqualFold(call.ToolName, "exec_command") {
		return false
	}
	if !strings.HasPrefix(strings.ToLower(observation.CallID), "exec-") {
		return false
	}
	// DOS timestamps have second precision while rollout completion timestamps have
	// milliseconds. A Post receipt belongs at completion, never merely after dispatch.
	earliest := call.CompletedAt.Truncate(time.Second)
	latest := call.CompletedAt.Add(codexPostToolSettlementWindow)
	return !observation.TS.Before(earliest) && !observation.TS.After(latest)
}

func phaseCountsForCalls(calls []dispatchedCall, obs []hookObservation, profile, workspace, source string) lifecycleCounts {
	c := lifecycleCounts{Denominator: len(calls), Source: source}
	wanted := make(map[string]dispatchedCall, len(calls))
	for _, call := range calls {
		wanted[call.SessionID+"\x00"+call.CallID] = call
	}
	matched := map[string]bool{}
	for _, o := range obs {
		key := o.SessionID + "\x00" + o.CallID
		call, ok := wanted[key]
		if !ok || matched[key] || !samePath(o.Workspace, call.Workspace) || !samePath(o.Workspace, workspace) || !samePath(o.Profile, profile) {
			continue
		}
		matched[key] = true
		classifyPhase(&c, o)
	}
	c.Unknown = len(calls) - len(matched)
	return c
}

func classifyPhase(c *lifecycleCounts, o hookObservation) {
	switch strings.ToLower(o.PhaseState) {
	case "skipped":
		c.Skipped++
	case "disabled":
		c.Disabled++
	case "failed":
		c.Attempted++
		c.Failed++
	case "succeeded":
		c.Attempted++
		c.Succeeded++
	default:
		switch strings.ToLower(o.Outcome) {
		case "skipped":
			c.Skipped++
		case "disabled":
			c.Disabled++
		default:
			c.Attempted++
			if o.Exit == 0 {
				c.Succeeded++
			} else {
				c.Failed++
			}
		}
	}
}
func readCodexDispatches(root, workspace, threadID string, after time.Time) ([]dispatchedCall, error) {
	seen := map[string]dispatchedCall{}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".jsonl") {
			return nil
		}
		if threadID != "" && !strings.Contains(filepath.Base(path), threadID) {
			return nil
		}
		f, e := os.Open(path)
		if e != nil {
			return nil
		}
		defer f.Close()
		var rows []rolloutRow
		rowWorkspace := ""
		sessionID := ""
		s := bufio.NewScanner(f)
		s.Buffer(make([]byte, 64*1024), 4*1024*1024)
		for s.Scan() {
			var raw struct {
				Timestamp time.Time       `json:"timestamp"`
				Type      string          `json:"type"`
				Payload   json.RawMessage `json:"payload"`
			}
			if json.Unmarshal(s.Bytes(), &raw) != nil {
				continue
			}
			if raw.Type == "session_meta" {
				var meta struct {
					CWD       string `json:"cwd"`
					ID        string `json:"id"`
					SessionID string `json:"session_id"`
				}
				_ = json.Unmarshal(raw.Payload, &meta)
				rowWorkspace = meta.CWD
				sessionID = meta.ID
				if sessionID == "" {
					sessionID = meta.SessionID
				}
			}
			if raw.Timestamp.Before(after) || raw.Type != "response_item" {
				continue
			}
			var row rolloutRow
			row.Timestamp, row.Type = raw.Timestamp, raw.Type
			_ = json.Unmarshal(raw.Payload, &row.Payload)
			rows = append(rows, row)
		}
		if !samePath(rowWorkspace, workspace) {
			return s.Err()
		}
		for _, row := range rows {
			id := row.Payload.CallID
			switch row.Payload.Type {
			case "function_call", "custom_tool_call":
				if id == "" {
					id = path + fmt.Sprint(len(seen))
				}
				seen[sessionID+"\x00"+id] = dispatchedCall{CallID: id, SessionID: sessionID, Workspace: rowWorkspace, ToolName: row.Payload.Name, DispatchedAt: row.Timestamp}
			case "function_call_output", "custom_tool_call_output":
				key := sessionID + "\x00" + id
				if call, ok := seen[key]; ok {
					call.Completed = true
					call.CompletedAt = row.Timestamp
					seen[key] = call
				}
			}
		}
		return s.Err()
	})
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("session log store missing: %s", root)
	}
	out := make([]dispatchedCall, 0, len(seen))
	for _, call := range seen {
		out = append(out, call)
	}
	return out, err
}

func readHookObservations(path string, after time.Time) (map[string][]hookObservation, time.Time, error) {
	out := map[string][]hookObservation{}
	var newest time.Time
	f, err := os.Open(path)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("observation store: %w", err)
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 64*1024), 2*1024*1024)
	for s.Scan() {
		var o hookObservation
		if json.Unmarshal(s.Bytes(), &o) == nil && !o.TS.Before(after) && (o.Verb == "pretool" || o.Verb == "posttool") {
			out[o.Verb] = append(out[o.Verb], o)
			if o.TS.After(newest) {
				newest = o.TS
			}
		}
	}
	return out, newest, s.Err()
}
func writeHookCensus(w io.Writer, r hookCensusReport) {
	fmt.Fprintf(w, "Codex hook lifecycle census (%s)\nprofile: %s\nlog store: %s (match=%t)\ndispatched calls: %d [source=%s]\n", r.Window, r.CodexHome, r.LogStore, r.ProfileMatch, r.DispatchedCalls, r.DispatchSource)
	fmt.Fprintf(w, "PreToolUse: denominator=%d attempted=%d succeeded=%d failed=%d skipped=%d disabled=%d unknown=%d [source=%s]\n", r.PreToolUse.Denominator, r.PreToolUse.Attempted, r.PreToolUse.Succeeded, r.PreToolUse.Failed, r.PreToolUse.Skipped, r.PreToolUse.Disabled, r.PreToolUse.Unknown, r.PreToolUse.Source)
	fmt.Fprintf(w, "PostToolUse: requirement=%s status=%s denominator=%d attempted=%d succeeded=%d failed=%d skipped=%d disabled=%d unknown=%d [source=%s]\n", r.PostToolUseRequirement, r.PostToolUseStatus, r.PostToolUse.Denominator, r.PostToolUse.Attempted, r.PostToolUse.Succeeded, r.PostToolUse.Failed, r.PostToolUse.Skipped, r.PostToolUse.Disabled, r.PostToolUse.Unknown, r.PostToolUse.Source)
	sort.Strings(r.Reasons)
	fmt.Fprintf(w, "verdict: %s", r.Verdict)
	if len(r.Reasons) > 0 {
		fmt.Fprintf(w, " (%s)", strings.Join(r.Reasons, ", "))
	}
	fmt.Fprintln(w)
}
