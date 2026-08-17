package devcmd

import (
	"bufio"
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

const codexHookCensusSchema = "fak/codex-hook-census/v1"

type lifecycleCounts struct {
	Denominator int    `json:"denominator"`
	Attempted   int    `json:"attempted"`
	Succeeded   int    `json:"succeeded"`
	Failed      int    `json:"failed"`
	Skipped     int    `json:"skipped"`
	Disabled    int    `json:"disabled"`
	Unknown     int    `json:"unknown"`
	Source      string `json:"source"`
}

type hookCensusReport struct {
	Schema           string          `json:"schema"`
	GeneratedAt      time.Time       `json:"generated_at"`
	Window           string          `json:"window"`
	CodexHome        string          `json:"codex_home"`
	LogStore         string          `json:"log_store"`
	ObservationStore string          `json:"observation_store"`
	ProfileMatch     bool            `json:"profile_match"`
	DispatchedCalls  int             `json:"dispatched_calls"`
	DispatchSource   string          `json:"dispatch_source"`
	PreToolUse       lifecycleCounts `json:"pre_tool_use"`
	PostToolUse      lifecycleCounts `json:"post_tool_use"`
	TelemetryFresh   bool            `json:"telemetry_fresh"`
	NewestReceipt    time.Time       `json:"newest_receipt,omitempty"`
	Verdict          string          `json:"verdict"`
	Reasons          []string        `json:"reasons,omitempty"`
}

type hookObservation struct {
	Exit    int       `json:"exit"`
	Verb    string    `json:"verb"`
	TS      time.Time `json:"ts"`
	Outcome string    `json:"outcome"`
}
type rolloutRow struct {
	Timestamp time.Time `json:"timestamp"`
	Type      string    `json:"type"`
	Payload   struct {
		Type   string `json:"type"`
		CallID string `json:"call_id"`
	} `json:"payload"`
}

func RunCodexHookCensus(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("codex-hook-census", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	home := fs.String("codex-home", "", "active Codex home")
	logHome := fs.String("log-home", "", "Codex home whose sessions are counted")
	workspace := fs.String("workspace", ".", "workspace whose calls are counted")
	threadID := fs.String("thread-id", os.Getenv("CODEX_THREAD_ID"), "Codex thread whose calls are counted (empty counts workspace)")
	observations := fs.String("observations", filepath.Join(".dos", "metrics", "observations.jsonl"), "hook observation JSONL")
	since := fs.Duration("since", 24*time.Hour, "lookback window")
	asJSON := fs.Bool("json", false, "emit JSON")
	nowText := fs.String("now", "", "fixed RFC3339 clock (tests/captures)")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *since <= 0 {
		fmt.Fprintln(stderr, "usage: fak-dev codex-hook-census [--codex-home DIR] [--log-home DIR] [--observations FILE] [--since 24h] [--json]")
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
	report, err := buildHookCensus(*home, *logHome, *workspace, *threadID, *observations, *since, now)
	if err != nil {
		fmt.Fprintf(stderr, "codex-hook-census: %v\n", err)
		return 1
	}
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

func buildHookCensus(home, logHome, workspace, threadID, observations string, since time.Duration, now time.Time) (hookCensusReport, error) {
	absHome, _ := filepath.Abs(home)
	absLog, _ := filepath.Abs(logHome)
	absObs, _ := filepath.Abs(observations)
	absWork, _ := filepath.Abs(workspace)
	calls, err := countCodexDispatches(filepath.Join(absLog, "sessions"), absWork, threadID, now.Add(-since))
	if err != nil {
		return hookCensusReport{}, err
	}
	receipts, newest, err := readHookObservations(absObs, now.Add(-since))
	if err != nil {
		return hookCensusReport{}, err
	}
	r := hookCensusReport{Schema: codexHookCensusSchema, GeneratedAt: now, Window: since.String(), CodexHome: absHome, LogStore: absLog, ObservationStore: absObs, ProfileMatch: samePath(absHome, absLog), DispatchedCalls: calls, DispatchSource: filepath.Join(absLog, "sessions"), NewestReceipt: newest, TelemetryFresh: !newest.IsZero() && now.Sub(newest) <= 5*time.Minute}
	r.PreToolUse = phaseCounts(calls, receipts["pretool"], absObs)
	r.PostToolUse = phaseCounts(calls, receipts["posttool"], absObs)
	if !r.TelemetryFresh {
		r.Reasons = append(r.Reasons, "STALE_TELEMETRY")
	}
	if !r.ProfileMatch {
		r.Reasons = append(r.Reasons, "PROFILE_LOG_STORE_MISMATCH")
	}
	if calls == 0 {
		r.Reasons = append(r.Reasons, "DISPATCH_DENOMINATOR_UNOBSERVED")
	}
	if r.PreToolUse.Unknown > 0 {
		r.Reasons = append(r.Reasons, "PRE_TOOL_USE_UNKNOWN")
	}
	if r.PostToolUse.Unknown > 0 {
		r.Reasons = append(r.Reasons, "POST_TOOL_USE_UNKNOWN")
	}
	if r.PreToolUse.Failed > 0 || r.PostToolUse.Failed > 0 {
		r.Reasons = append(r.Reasons, "HOOK_FAILURES_PRESENT")
	}
	if len(r.Reasons) == 0 {
		r.Verdict = "HEALTHY"
	} else {
		r.Verdict = "UNHEALTHY"
	}
	return r, nil
}
func samePath(a, b string) bool { return strings.EqualFold(filepath.Clean(a), filepath.Clean(b)) }
func phaseCounts(den int, obs []hookObservation, source string) lifecycleCounts {
	c := lifecycleCounts{Denominator: den, Source: source}
	for _, o := range obs {
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
	accounted := c.Succeeded + c.Failed + c.Skipped + c.Disabled
	if accounted < den {
		c.Unknown = den - accounted
	}
	return c
}
func countCodexDispatches(root, workspace, threadID string, after time.Time) (int, error) {
	seen := map[string]bool{}
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
					CWD string `json:"cwd"`
				}
				_ = json.Unmarshal(raw.Payload, &meta)
				rowWorkspace = meta.CWD
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
			if row.Payload.Type == "function_call" || row.Payload.Type == "custom_tool_call" {
				id := row.Payload.CallID
				if id == "" {
					id = path + fmt.Sprint(len(seen))
				}
				seen[id] = true
			}
		}
		return s.Err()
	})
	if os.IsNotExist(err) {
		return 0, fmt.Errorf("session log store missing: %s", root)
	}
	return len(seen), err
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
	for _, p := range []struct {
		name string
		c    lifecycleCounts
	}{{"PreToolUse", r.PreToolUse}, {"PostToolUse", r.PostToolUse}} {
		fmt.Fprintf(w, "%s: denominator=%d attempted=%d succeeded=%d failed=%d skipped=%d disabled=%d unknown=%d [source=%s]\n", p.name, p.c.Denominator, p.c.Attempted, p.c.Succeeded, p.c.Failed, p.c.Skipped, p.c.Disabled, p.c.Unknown, p.c.Source)
	}
	sort.Strings(r.Reasons)
	fmt.Fprintf(w, "verdict: %s", r.Verdict)
	if len(r.Reasons) > 0 {
		fmt.Fprintf(w, " (%s)", strings.Join(r.Reasons, ", "))
	}
	fmt.Fprintln(w)
}
