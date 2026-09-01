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
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/processstart"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const (
	dispatchSessionsSchema = "fleet-dispatch-sessions/1"
	dispatchSessionSchema  = "fleet-dispatch-session/1"
)

// dispatchSessionStampRE pulls the UTC spawn stamp out of a resolve log name
// (resolve-<issue>-<YYYYMMDD-HHMMSS>...) so the row can report when the session
// started without reading the log body.
var dispatchSessionStampRE = regexp.MustCompile(`^resolve-\d+-(\d{8}-\d{6})`)

var dispatchProcessStart = processstart.Start

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
	Schema         string                           `json:"schema"`
	Issue          string                           `json:"issue,omitempty"`
	Lane           string                           `json:"lane,omitempty"`
	Backend        string                           `json:"backend"`
	Worker         string                           `json:"worker"`
	PID            int                              `json:"pid,omitempty"`
	PIDAlive       bool                             `json:"pid_alive"`
	PIDIdentity    string                           `json:"pid_identity,omitempty"`
	PIDReason      string                           `json:"pid_reason,omitempty"`
	Live           bool                             `json:"live"`
	Outcome        string                           `json:"outcome"`
	Reason         string                           `json:"reason,omitempty"`
	Evidence       string                           `json:"evidence,omitempty"`
	AgeSeconds     int64                            `json:"age_seconds"`
	Started        string                           `json:"started,omitempty"`
	LeaseID        string                           `json:"lease_id,omitempty"`
	Tree           []string                         `json:"tree,omitempty"`
	Guard          *dispatchSessionGuard            `json:"guard,omitempty"`
	WorkerWorktree *workerworktree.StatusProjection `json:"worker_worktree,omitempty"`
	// Token/cost accounting (#3329), folded from the gateway-usage ledger by the
	// worker's guard trace-id. All three are omitempty: a session whose trace has no
	// usage row (or a fleet with no ledger yet) carries none of them, so the snapshot
	// stays byte-identical to the pre-#3329 shape. Tokens is the total token volume
	// the session moved; Cost is an OBSERVED input-token-equivalent (see
	// dispatchSessionCostEquiv); CacheReadShare is cache_read / total prompt tokens.
	Tokens         uint64  `json:"tokens,omitempty"`
	Cost           float64 `json:"cost,omitempty"`
	CacheReadShare float64 `json:"cache_read_share,omitempty"`
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
	usageLedger := fs.String("usage-ledger", "", "gateway-usage ledger for per-session token/cost accounting (default: .fak/nightrun/gateway-usage.jsonl under the repo root)")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for age math (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the fleet-dispatch-sessions/1 JSON payload")
	asMarkdown := fs.Bool("markdown", false, "render the operator session card as Markdown")
	fileIssues := fs.Bool("file-issues", false, "run the systemic-waste lens over the sessions and print fingerprinted improvement-ticket candidates (dry-run unless --confirm)")
	confirm := fs.Bool("confirm", false, "with --file-issues: actually open a gh issue per NEW finding (default: dry-run, print only)")
	maxIssues := fs.Int("max-issues", 0, "with --file-issues --confirm: hard cap on issues filed per run, worst-first (0 = no cap)")
	watch := fs.Bool("watch", false, "re-render the session card on an interval (bounded by --watch-iterations)")
	watchInterval := fs.Duration("watch-interval", 2*time.Second, "with --watch: delay between re-renders")
	watchIterations := fs.Int("watch-iterations", 0, "with --watch: number of renders before returning (0 = run until interrupted)")
	tail := fs.String("tail", "", "print the secret-scrubbed transcript tail of the worker whose guard handle or trace-id has this prefix")
	demo := fs.Bool("demo", false, "render a deterministic unified-session fixture without reading host state")
	selfcheck := fs.Bool("selfcheck", false, "with --demo: verify human and JSON projections before rendering")
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
	if *selfcheck && !*demo {
		fmt.Fprintln(stderr, "fak dispatch sessions: --selfcheck requires --demo")
		return 2
	}
	if *demo {
		snap := dispatchSessionsSnapshot{
			Schema: dispatchSessionsSchema, RunsDir: "<demo>", RegDir: "<demo>",
			SessionCount: 1, LiveCount: 1,
			Sessions: []dispatchSessionRow{{Worker: "worker-demo", Issue: "3330", Lane: "cmd", Backend: "codex", PID: 3330, PIDAlive: true, Live: true, Outcome: "running", Reason: "launch-confirmed", Guard: &dispatchSessionGuard{Handle: "demo-guard", TraceID: "demo-trace"}}},
		}
		if *selfcheck {
			if snap.Schema != dispatchSessionsSchema || snap.SessionCount != 1 || snap.LiveCount != 1 || !strings.Contains(renderDispatchSessions(snap), "#3330") {
				fmt.Fprintln(stderr, "fak dispatch sessions: selfcheck failed")
				return 1
			}
			fmt.Fprintln(stderr, "dispatch sessions selfcheck OK")
		}
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, snap, "fak dispatch sessions")
		}
		if *asMarkdown {
			fmt.Fprint(stdout, renderDispatchSessionsMarkdown(snap))
			return 0
		}
		fmt.Fprint(stdout, renderDispatchSessions(snap))
		return 0
	}

	now := time.Now().UTC()
	if *nowUnix > 0 {
		now = time.Unix(*nowUnix, 0).UTC()
	}
	resolvedReg := resolveSweepRegDir(*regDir)
	usagePath := resolveDispatchSessionsUsageLedger(*usageLedger)

	// --tail resolves a single session by handle/trace prefix and prints its scrubbed
	// transcript tail; it does not fold the whole snapshot.
	if strings.TrimSpace(*tail) != "" {
		return runDispatchSessionsTail(stdout, stderr, *runsDir, resolvedReg, *tail, now)
	}

	// --watch loops a bounded, interruptible re-render of the human card.
	if *watch {
		return runDispatchSessionsWatch(stdout, *runsDir, resolvedReg, usagePath, *nowUnix, *watchInterval, *watchIterations)
	}

	snap := dispatchSessionsScan(*runsDir, resolvedReg, usagePath, now)

	// --file-issues turns the read-only view into the systemic-waste lens.
	if *fileIssues {
		return runDispatchSessionsAudit(stdout, stderr, *runsDir, snap, *confirm, *maxIssues)
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, snap, "fak dispatch sessions")
	}
	if *asMarkdown {
		fmt.Fprint(stdout, renderDispatchSessionsMarkdown(snap))
		return 0
	}
	fmt.Fprint(stdout, renderDispatchSessions(snap))
	return 0
}

// resolveDispatchSessionsUsageLedger resolves the gateway-usage ledger path for the
// token/cost fold: the explicit --usage-ledger when given, else the default
// .fak/nightrun/gateway-usage.jsonl under the repo root. An absent ledger simply
// yields no fold (the fields stay omitted), so this never needs to exist.
func resolveDispatchSessionsUsageLedger(flagVal string) string {
	if strings.TrimSpace(flagVal) != "" {
		return flagVal
	}
	cwd, _ := os.Getwd()
	return filepath.Join(findRepoRoot(cwd), filepath.FromSlash(gatewayusageledger.DefaultLedgerRel))
}

// runDispatchSessionsWatch re-renders the human session card up to iterations times,
// sleeping interval between renders. iterations<=0 runs until the process is
// interrupted; a test passes a bounded count so the loop always terminates. The
// clock advances with wall time unless nowUnix pins it (hermetic single renders).
func runDispatchSessionsWatch(stdout io.Writer, runsDir, regDir, usagePath string, nowUnix int64, interval time.Duration, iterations int) int {
	for i := 0; iterations <= 0 || i < iterations; i++ {
		now := time.Now().UTC()
		if nowUnix > 0 {
			now = time.Unix(nowUnix, 0).UTC()
		}
		snap := dispatchSessionsScan(runsDir, regDir, usagePath, now)
		fmt.Fprintf(stdout, "\x1b[2J\x1b[H")
		fmt.Fprint(stdout, renderDispatchSessions(snap))
		fmt.Fprintf(stdout, "\n(watch %d — %s interval; Ctrl-C to stop)\n", i+1, interval)
		last := iterations > 0 && i == iterations-1
		if !last {
			time.Sleep(interval)
		}
	}
	return 0
}

// dispatchSessionsScan folds the runs directory and the guard-session index into the
// per-session snapshot. PURE over its inputs: the same (runsDir, regDir, now) yields
// the same snapshot, so a test plants sidecars + a guard row + a fixed clock. A missing
// runs dir fails soft to an empty snapshot (nothing running), matching `dispatch status`.
func dispatchSessionsScan(runsDir, regDir, usageLedgerPath string, now time.Time) dispatchSessionsSnapshot {
	workers, _ := dispatchaudit.ScanDir(runsDir)
	worktreeInputs := dispatchWorkerWorktreeInputs(findRepoRoot(runsDir))

	// Per-trace token/cost economy (#3329): the gateway-usage ledger keyed by the
	// served trace-id. Empty (nil) when no ledger is present, so the fold is skipped
	// and every row stays byte-identical to the pre-accounting shape.
	usageByTrace := dispatchSessionUsageByTrace(usageLedgerPath)

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

		pidIdentity, pidReason := dispatchSessionPIDIdentity(w.PID, w.PIDAlive, dispatchSessionStartedTime(w.Log))
		row := dispatchSessionRow{
			Schema:      dispatchSessionSchema,
			Issue:       w.Issue,
			Lane:        w.Lane,
			Backend:     string(c.Backend),
			Worker:      stem,
			PID:         w.PID,
			PIDAlive:    w.PIDAlive,
			PIDIdentity: pidIdentity,
			PIDReason:   pidReason,
			Outcome:     string(c.Outcome),
			Reason:      c.Reason,
			Evidence:    c.EvidenceSummary,
			AgeSeconds:  dispatchSessionAgeSeconds(filepath.Join(runsDir, w.Log), now),
			Started:     dispatchSessionStarted(w.Log),
		}
		if scope, ok := liveByWorker[stem]; ok {
			row.Live = pidIdentity != "stale"
			if row.Lane == "" {
				row.Lane = scope.Lane
			}
			row.LeaseID = scope.LeaseID
			row.Tree = scope.Tree
			if row.Live {
				live++
			}
		}
		row.WorkerWorktree = matchDispatchWorkerWorktree(row, worktreeInputs)
		if w.PID > 0 && pidIdentity != "stale" {
			if g, ok := guardByPID[w.PID]; ok {
				row.Guard = &dispatchSessionGuard{Handle: g.Handle, TraceID: g.TraceID, AuditPath: g.AuditPath}
			}
		}
		if row.Guard != nil && row.Guard.TraceID != "" {
			if c, ok := usageByTrace[row.Guard.TraceID]; ok {
				applyDispatchSessionUsage(&row, c)
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

// dispatchSessionUsageByTrace indexes the LATEST gateway-usage counters per served
// trace-id, so a session row can fold its own token/cost economy in by its guard
// trace-id. The ledger is keyed by SessionID (the served trace when a caller has
// one); an empty session id is skipped so a trace-less serve row can never join an
// empty guard trace-id. A missing/unreadable/empty ledger yields nil (no fold),
// keeping the snapshot byte-identical to a fleet with no usage ledger yet. Rows are
// cumulative counter snapshots, so the newest row per session already holds that
// task's running total (mirrors budget.go's latestTaskRow).

func dispatchSessionStartedTime(logName string) time.Time {
	m := dispatchSessionStampRE.FindStringSubmatch(filepath.Base(logName))
	if len(m) != 2 {
		return time.Time{}
	}
	started, _ := time.Parse("20060102-150405", m[1])
	return started.UTC()
}

func dispatchSessionPIDIdentity(pid int, alive bool, launched time.Time) (string, string) {
	if pid <= 0 || !alive {
		return "ended", "process is not alive"
	}
	started, ok := dispatchProcessStart(pid)
	if !ok || launched.IsZero() {
		return "unknown", "process-start identity is unavailable"
	}
	if started.UTC().After(launched.Add(2 * time.Minute)) {
		return "stale", "live PID started after the recorded dispatch launch"
	}
	return "launch-confirmed", "process start is compatible with the dispatch launch"
}

func dispatchSessionUsageByTrace(ledgerPath string) map[string]gatewayusageledger.Counters {
	if strings.TrimSpace(ledgerPath) == "" {
		return nil
	}
	best := map[string]gatewayusageledger.Row{}
	for _, r := range gatewayusageledger.ReadLedgerFile(ledgerPath) {
		sid := strings.TrimSpace(r.SessionID)
		if sid == "" {
			continue
		}
		if cur, ok := best[sid]; !ok || r.UnixMillis > cur.UnixMillis {
			best[sid] = r
		}
	}
	if len(best) == 0 {
		return nil
	}
	out := make(map[string]gatewayusageledger.Counters, len(best))
	for sid, r := range best {
		out[sid] = r.Counters
	}
	return out
}

// applyDispatchSessionUsage folds one session's OBSERVED gateway-usage counters onto
// its row: the total token volume moved, an input-token-equivalent cost, and the
// cache-read share. All three are omitempty, so a zero-usage row stays byte-identical
// to the pre-#3329 shape.
func applyDispatchSessionUsage(row *dispatchSessionRow, c gatewayusageledger.Counters) {
	row.Tokens = c.InputTokens + c.OutputTokens + c.CachedPromptTokens + c.CacheCreationTokens
	row.Cost = dispatchSessionCostEquiv(c)
	row.CacheReadShare = dispatchSessionCacheReadShare(c)
}

// dispatchSessionCostEquiv prices a session's counters into an OBSERVED
// input-token-equivalent cost using the canonical cacheprice multipliers (the ONE
// source of the 0.1x read / 1.25x write economics): uncached input and output bill
// at 1.0x, a cache read at ReadMultiplier (0.1x), a 5-minute cache write at
// Write5mMultiplier (1.25x). It is a cost BASIS in input-token-equivalents —
// deliberately not dollars, since the per-model price is a worker property this fold
// does not carry — so the waste analyzer can rank sessions by relative spend without
// a price table.
func dispatchSessionCostEquiv(c gatewayusageledger.Counters) float64 {
	return float64(c.InputTokens) +
		float64(c.OutputTokens) +
		float64(c.CachedPromptTokens)*cacheprice.ReadMultiplier +
		float64(c.CacheCreationTokens)*cacheprice.Write5mMultiplier
}

// dispatchSessionCacheReadShare is the fraction of a session's PROMPT tokens the
// provider served from its cache: cache_read / (input + cache_read + cache_creation)
// — the same denominator the cache-value report scores hit-rate on. A collapse
// toward zero is the waste signal the audit lens keys on. Returns 0 (omitted) when
// there were no prompt tokens.
func dispatchSessionCacheReadShare(c gatewayusageledger.Counters) float64 {
	prompt := c.InputTokens + c.CachedPromptTokens + c.CacheCreationTokens
	if prompt == 0 {
		return 0
	}
	return float64(c.CachedPromptTokens) / float64(prompt)
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

func dispatchWorkerWorktreeInputs(repoRoot string) []workerworktree.StatusEvidence {
	_, paths := workerworktree.Count(repoRoot, nil)
	rows := worktreeWorkerLifecycleInventory(repoRoot, paths, worktreeWorkerLifecycleProbes{})
	inputs := make([]workerworktree.StatusEvidence, 0, len(rows))
	for _, row := range rows {
		in, err := workerworktree.LoadIntent(row.Path)
		issue := 0
		if err == nil {
			issue = in.IssueNumber
		}
		inputs = append(inputs, workerworktree.StatusEvidence{
			IssueNumber: issue, Lane: row.Association.Lane, Session: row.Association.LeaseID,
			HeadSHA: row.HeadSHA, BaseSHA: row.BaseSHA,
			AssociationKnown: row.Association.State == worktreeEvidenceAssociated,
			OwnerLive:        row.Liveness.Owner == worktreeEvidenceLive, LeaseLive: row.Liveness.Lease == worktreeEvidenceLive,
			Dirty: row.Cleanliness.State == worktreeEvidenceDirty, CleanupReady: row.ReapReadiness.Reapable,
		})
	}
	return inputs
}

func matchDispatchWorkerWorktree(row dispatchSessionRow, inputs []workerworktree.StatusEvidence) *workerworktree.StatusProjection {
	issue, _ := strconv.Atoi(strings.TrimSpace(row.Issue))
	matched := -1
	for i, in := range inputs {
		issueMatch := issue > 0 && in.IssueNumber == issue
		laneMatch := row.Lane != "" && in.Lane == row.Lane
		leaseMatch := row.LeaseID != "" && in.Session == row.LeaseID
		if !issueMatch && !laneMatch && !leaseMatch {
			continue
		}
		if matched >= 0 {
			return nil // ambiguous association fails closed
		}
		matched = i
	}
	if matched < 0 {
		return nil
	}
	projection := workerworktree.ProjectStatus(inputs[matched])
	return &projection
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
		if s.WorkerWorktree != nil {
			fmt.Fprintf(&b, "         worker-worktree=%s  complete=%t", s.WorkerWorktree.State, s.WorkerWorktree.Complete)
			if s.WorkerWorktree.Commit != "" {
				fmt.Fprintf(&b, "  commit=%s", s.WorkerWorktree.Commit)
			}
			fmt.Fprintln(&b)
		}
		if s.Tokens > 0 {
			fmt.Fprintf(&b, "         tokens=%d  cost~=%.0f  cache-read=%.0f%%\n", s.Tokens, s.Cost, s.CacheReadShare*100)
		}
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
	fmt.Fprint(&b, "| live | issue | lane | backend | outcome | worker worktree | age | guard | worker |\n")
	fmt.Fprint(&b, "|---|---|---|---|---|---|---|---|---|\n")
	for _, s := range snap.Sessions {
		live := ""
		if s.Live {
			live = "●"
		}
		worktreeState := ""
		if s.WorkerWorktree != nil {
			worktreeState = string(s.WorkerWorktree.State)
		}
		fmt.Fprintf(&b, "| %s | #%s | %s | %s | %s | %s | %s | %s | %s |\n",
			live, dispatchSessionIssueField(s.Issue), dispatchSessionLaneField(s.Lane),
			dispatchSessionBackendField(s.Backend), s.Outcome, worktreeState, compactDuration(s.AgeSeconds),
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

// runDispatchSessionsTail resolves a single session by a guard handle or trace-id
// prefix (guardsessions.Resolve), joins it to its worker log by PID, and prints the
// secret-scrubbed transcript tail. It reuses the same canon matcher + line-aligned
// tail cut `fak dispatch evidence` uses, so redaction never drifts between the two:
// a tail carrying an obfuscated (non-raw-locatable) secret is SEALED rather than
// risk a leak. Ambiguous/absent prefixes fail soft with a non-zero exit.
func runDispatchSessionsTail(stdout, stderr io.Writer, runsDir, regDir, query string, now time.Time) int {
	res := guardsessions.Resolve(guardsessions.Load(regDir), query)
	switch res.Matched {
	case 0:
		fmt.Fprintf(stderr, "fak dispatch sessions --tail: no guard session matches %q\n", query)
		return 1
	case 1:
		// unambiguous — fall through
	default:
		fmt.Fprintf(stderr, "fak dispatch sessions --tail: %q is ambiguous (%d sessions); use a longer prefix\n", query, res.Matched)
		for _, c := range res.Candidates {
			fmt.Fprintf(stderr, "  handle=%s trace=%s pid=%d\n", c.Handle, c.TraceID, c.PID)
		}
		return 1
	}
	g := res.Row

	// Join the resolved guard session to its worker log by PID (the same key the
	// snapshot join uses). A guard session with no live worker log has nothing to tail.
	workers, _ := dispatchaudit.ScanDir(runsDir)
	logPath := ""
	for _, w := range workers {
		if w.PID != 0 && w.PID == g.PID {
			logPath = filepath.Join(runsDir, w.Log)
			break
		}
	}
	if logPath == "" {
		fmt.Fprintf(stderr, "fak dispatch sessions --tail: session %s (pid %d) has no worker log in %s\n", g.Handle, g.PID, runsDir)
		return 1
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch sessions --tail: read %s: %v\n", logPath, err)
		return 1
	}

	// Line-aligned tail so the cut never splits a credential mid-token.
	tail := lineAlignedTranscriptTail(raw)

	fmt.Fprintf(stdout, "dispatch session tail — handle=%s trace=%s pid=%d\n", g.Handle, g.TraceID, g.PID)
	fmt.Fprintf(stdout, "worker=%s\n\n", logPath)
	if canon.RawSecretComplete(tail) {
		scrubbed, masked := canon.RedactSecrets(tail)
		fmt.Fprint(stdout, string(scrubbed))
		if len(scrubbed) == 0 || scrubbed[len(scrubbed)-1] != '\n' {
			fmt.Fprintln(stdout)
		}
		if masked > 0 {
			fmt.Fprintf(stdout, "\n(%d secret span(s) masked)\n", masked)
		}
		return 0
	}
	fmt.Fprint(stdout, "(transcript tail sealed — obfuscated secret not raw-maskable)\n")
	return 0
}
