package main

// dispatch_sessions.go — `fak dispatch sessions`, the unified per-SESSION view.
//
// A dispatch worker is not a bare agent CLI: the tick wraps it in `fak guard`
// (dispatch_tick.go → dispatchtick.GuardedLaunchCommand), which runs an in-process
// adjudicating gateway. That produces TWO disjoint observability planes that no
// command has ever cross-referenced:
//
//   - the .dispatch-runs/ sidecar plane (resolve-<issue>-<stamp>.{log,pid,backend,
//     lease-*}), keyed by issue+timestamp with PID-derived liveness — read by
//     `fak dispatch status` / `audit` / `evidence`; and
//   - the guard_sessions.jsonl plane (handle, trace-id, agent, pid, audit-path)
//     under the fleet registry dir — read ONLY by `fak guard sessions`.
//
// `fak dispatch status` answers "which workers are live" but never resolves a worker
// to its outcome, its age/last-activity, or its guard session. This verb folds BOTH
// planes plus the audit outcome into one per-session row for live AND recently-
// completed workers, so an operator sees the actual sessions themselves:
//
//	# the human session card
//	fak dispatch sessions
//	# machine-readable snapshot (schema fleet-dispatch-sessions/1)
//	fak dispatch sessions --json
//	# the same card as an operator Markdown block
//	fak dispatch sessions --markdown
//
// It is a PURE fold over the filesystem (same runs-dir + reg-dir + clock → same
// snapshot), so a test drives it hermetically by planting resolve-* sidecars and a
// guard_sessions.jsonl row. It launches nothing and writes nothing — the join key is
// the worker's .pid sidecar (the `fak guard` process pid) against guardsessions.Row.PID
// (os.Getpid() recorded inside guard), so a live worker resolves to its guard handle,
// trace id, and audit-journal path. Only STRUCTURED evidence is surfaced (the audit
// EvidenceSummary), never the raw transcript.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
)

const (
	dispatchSessionsSchema = "fleet-dispatch-sessions/1"
	dispatchSessionSchema  = "fleet-dispatch-session/1"
)

// dispatchSessionStampRE pulls the UTC spawn stamp out of a resolve log name
// (resolve-<issue>-<YYYYMMDD-HHMMSS>...) so the row can report when the session
// started without reading the log body.
var dispatchSessionStampRE = regexp.MustCompile(`^resolve-\d+-(\d{8}-\d{6})`)

// dispatchSessionGuard is the cross-plane join: the guard session a worker's pid
// resolves to. Absent (nil) when no guard_sessions.jsonl row shares the pid.
type dispatchSessionGuard struct {
	Handle    string `json:"handle"`
	TraceID   string `json:"trace_id,omitempty"`
	AuditPath string `json:"audit_path,omitempty"`
}

// dispatchSessionRow is one dispatch worker session, unifying the runs-dir scope,
// the audit outcome, and the guard session.
type dispatchSessionRow struct {
	Schema     string                `json:"schema"`
	Issue      string                `json:"issue,omitempty"`
	Lane       string                `json:"lane,omitempty"`
	Backend    string                `json:"backend"`
	Worker     string                `json:"worker"`
	PID        int                   `json:"pid,omitempty"`
	PIDAlive   bool                  `json:"pid_alive"`
	Live       bool                  `json:"live"`
	Outcome    string                `json:"outcome"`
	Reason     string                `json:"reason,omitempty"`
	Evidence   string                `json:"evidence,omitempty"`
	AgeSeconds int64                 `json:"age_seconds"`
	Started    string                `json:"started,omitempty"`
	LeaseID    string                `json:"lease_id,omitempty"`
	Tree       []string              `json:"tree,omitempty"`
	Guard      *dispatchSessionGuard `json:"guard,omitempty"`
}

// dispatchSessionsSnapshot is the fleet-dispatch-sessions/1 payload.
type dispatchSessionsSnapshot struct {
	Schema       string               `json:"schema"`
	RunsDir      string               `json:"runs_dir"`
	RegDir       string               `json:"reg_dir"`
	GeneratedUTC string               `json:"generated_utc"`
	SessionCount int                  `json:"session_count"`
	LiveCount    int                  `json:"live_count"`
	Sessions     []dispatchSessionRow `json:"sessions"`
}

func runDispatchSessions(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch sessions", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", dispatchProgressRunsDir, "directory of dispatch worker logs")
	regDir := fs.String("reg-dir", "", "registry dir holding guard_sessions.jsonl (default: $FLEET_REG_DIR, else the host Fleet registry, else <repo>/tools/_registry)")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for age math (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the fleet-dispatch-sessions/1 JSON payload")
	asMarkdown := fs.Bool("markdown", false, "render the operator session card as Markdown")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch sessions: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *asJSON && *asMarkdown {
		fmt.Fprintln(stderr, "fak dispatch sessions: choose at most one of --json or --markdown")
		return 2
	}

	now := time.Now().UTC()
	if *nowUnix > 0 {
		now = time.Unix(*nowUnix, 0).UTC()
	}
	resolvedReg := resolveSweepRegDir(*regDir)
	snap := dispatchSessionsScan(*runsDir, resolvedReg, now)

	if *asJSON {
		if err := writeIndentedJSON(stdout, snap); err != nil {
			fmt.Fprintf(stderr, "fak dispatch sessions: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	if *asMarkdown {
		fmt.Fprint(stdout, renderDispatchSessionsMarkdown(snap))
		return 0
	}
	fmt.Fprint(stdout, renderDispatchSessions(snap))
	return 0
}

// dispatchSessionsScan folds the runs directory and the guard-session index into the
// per-session snapshot. PURE over its inputs: the same (runsDir, regDir, now) yields
// the same snapshot, so a test plants sidecars + a guard row + a fixed clock. A missing
// runs dir fails soft to an empty snapshot (nothing running), matching `dispatch status`.
func dispatchSessionsScan(runsDir, regDir string, now time.Time) dispatchSessionsSnapshot {
	workers, _ := dispatchaudit.ScanDir(runsDir)

	// Live scopes carry the lease + tree the audit Worker does not; key them by the
	// worker stem (the base name without .log) so a completed worker simply misses.
	liveByWorker := map[string]dispatchLiveScope{}
	for _, s := range liveResolutionScopes(runsDir) {
		if _, ok := liveByWorker[s.Worker]; !ok {
			liveByWorker[s.Worker] = s
		}
	}

	// Guard sessions join by pid (the .pid sidecar is the `fak guard` process pid,
	// which guardsessions records via os.Getpid()). First row per pid wins; Load is
	// already newest-start-first.
	guardByPID := map[int]guardsessions.Row{}
	for _, r := range guardsessions.Load(regDir) {
		if r.PID <= 0 {
			continue
		}
		if _, ok := guardByPID[r.PID]; !ok {
			guardByPID[r.PID] = r
		}
	}

	rows := make([]dispatchSessionRow, 0, len(workers))
	live := 0
	for _, w := range workers {
		c := dispatchaudit.Classify(w, dispatchaudit.DefaultThresholds())
		stem := strings.TrimSuffix(w.Log, filepath.Ext(w.Log))

		row := dispatchSessionRow{
			Schema:     dispatchSessionSchema,
			Issue:      w.Issue,
			Lane:       w.Lane,
			Backend:    string(c.Backend),
			Worker:     stem,
			PID:        w.PID,
			PIDAlive:   w.PIDAlive,
			Outcome:    string(c.Outcome),
			Reason:     c.Reason,
			Evidence:   c.EvidenceSummary,
			AgeSeconds: dispatchSessionAgeSeconds(filepath.Join(runsDir, w.Log), now),
			Started:    dispatchSessionStarted(w.Log),
		}
		if scope, ok := liveByWorker[stem]; ok {
			row.Live = true
			if row.Lane == "" {
				row.Lane = scope.Lane
			}
			row.LeaseID = scope.LeaseID
			row.Tree = scope.Tree
			live++
		}
		if w.PID > 0 {
			if g, ok := guardByPID[w.PID]; ok {
				row.Guard = &dispatchSessionGuard{Handle: g.Handle, TraceID: g.TraceID, AuditPath: g.AuditPath}
			}
		}
		rows = append(rows, row)
	}

	// Deterministic operator order: live sessions first, then by worker name.
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Live != rows[j].Live {
			return rows[i].Live
		}
		return rows[i].Worker < rows[j].Worker
	})

	return dispatchSessionsSnapshot{
		Schema:       dispatchSessionsSchema,
		RunsDir:      runsDir,
		RegDir:       regDir,
		GeneratedUTC: now.UTC().Format("2006-01-02T15:04:05Z"),
		SessionCount: len(rows),
		LiveCount:    live,
		Sessions:     rows,
	}
}

// dispatchSessionAgeSeconds is the seconds since the worker log last changed (its
// last activity), clamped to >= 0. A log we cannot stat reports 0.
func dispatchSessionAgeSeconds(logPath string, now time.Time) int64 {
	info, err := os.Stat(logPath)
	if err != nil {
		return 0
	}
	age := int64(now.Sub(info.ModTime()).Seconds())
	if age < 0 {
		return 0
	}
	return age
}

// dispatchSessionStarted parses the UTC spawn stamp out of the log name into an
// RFC3339 stamp, or "" when the name carries none.
func dispatchSessionStarted(logName string) string {
	m := dispatchSessionStampRE.FindStringSubmatch(logName)
	if m == nil {
		return ""
	}
	t, err := time.Parse("20060102-150405", m[1])
	if err != nil {
		return ""
	}
	return t.UTC().Format("2006-01-02T15:04:05Z")
}

func dispatchSessionLaneField(lane string) string {
	if strings.TrimSpace(lane) == "" {
		return "(no lane)"
	}
	return lane
}

func dispatchSessionGuardField(g *dispatchSessionGuard) string {
	if g == nil {
		return "(no guard session)"
	}
	if g.Handle == "" {
		return "(guard)"
	}
	return g.Handle
}

func renderDispatchSessions(snap dispatchSessionsSnapshot) string {
	var b strings.Builder
	fmt.Fprintf(&b, "dispatch sessions — %d session(s), %d live\n", snap.SessionCount, snap.LiveCount)
	fmt.Fprintf(&b, "runs-dir: %s\n", snap.RunsDir)
	fmt.Fprintf(&b, "reg-dir:  %s\n", snap.RegDir)
	if len(snap.Sessions) == 0 {
		fmt.Fprint(&b, "no dispatch worker sessions\n")
		return b.String()
	}
	for _, s := range snap.Sessions {
		flag := "done"
		if s.Live {
			flag = "LIVE"
		}
		fmt.Fprintf(&b, "  [%s] #%s  lane=%s  backend=%s  %s  age=%s  guard=%s\n",
			flag, dispatchSessionIssueField(s.Issue), dispatchSessionLaneField(s.Lane),
			dispatchSessionBackendField(s.Backend), s.Outcome,
			compactDuration(s.AgeSeconds), dispatchSessionGuardField(s.Guard))
		fmt.Fprintf(&b, "         worker=%s  pid=%d  %s\n", s.Worker, s.PID, s.Reason)
	}
	return b.String()
}

func renderDispatchSessionsMarkdown(snap dispatchSessionsSnapshot) string {
	var b strings.Builder
	fmt.Fprintf(&b, "### dispatch sessions — %d session(s), %d live\n\n", snap.SessionCount, snap.LiveCount)
	fmt.Fprintf(&b, "- runs-dir: `%s`\n", snap.RunsDir)
	fmt.Fprintf(&b, "- reg-dir: `%s`\n\n", snap.RegDir)
	if len(snap.Sessions) == 0 {
		fmt.Fprint(&b, "_no dispatch worker sessions_\n")
		return b.String()
	}
	fmt.Fprint(&b, "| live | issue | lane | backend | outcome | age | guard | worker |\n")
	fmt.Fprint(&b, "|---|---|---|---|---|---|---|---|\n")
	for _, s := range snap.Sessions {
		live := ""
		if s.Live {
			live = "●"
		}
		fmt.Fprintf(&b, "| %s | #%s | %s | %s | %s | %s | %s | %s |\n",
			live, dispatchSessionIssueField(s.Issue), dispatchSessionLaneField(s.Lane),
			dispatchSessionBackendField(s.Backend), s.Outcome, compactDuration(s.AgeSeconds),
			dispatchSessionGuardField(s.Guard), s.Worker)
	}
	return b.String()
}

func dispatchSessionIssueField(issue string) string {
	if strings.TrimSpace(issue) == "" {
		return "?"
	}
	return issue
}

func dispatchSessionBackendField(backend string) string {
	if strings.TrimSpace(backend) == "" {
		return "unknown"
	}
	return backend
}
