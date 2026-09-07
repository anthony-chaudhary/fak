package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// sync_drain.go implements `fak sync drain` (#3617): the deterministic release valve for
// commits stranded by a red-trunk push refusal. When origin/main is red-from-committed-state
// (a peer's half-committed feature) the pre-push gate refuses EVERY commit — including a
// hermetic, unrelated fix — so each worker retries blindly and a green window is raced by N
// uncoordinated pushers (memory: chronic-red-trunk-strands-unrelated-commits). This verb
// instead records the stranded commits into a local queue, polls for a trunk-green quiescent
// window by REUSING the existing pre-push build witness (evaluatePrePushBuild) plus the
// safesync peer-state assessment, flushes the queue in a single push when the window opens,
// and backs off (not blind-retries) while red. (The name is prefixed syncDrain* to avoid the
// unrelated doomloop "drain" namespace in this package.)

const syncDrainSchema = "fak.sync_drain.v1"

// Backoff bounds for the red window: exponential from a 15s base, capped at 15 minutes. The
// growing gap is what makes this a release valve rather than a thundering herd — a scheduler
// that invokes drain on a cadence paces itself off NextRetryUnix instead of hammering the gate.
const (
	syncDrainBackoffBaseSec int64 = 15
	syncDrainBackoffCapSec  int64 = 900
)

// syncDrainEntry is one commit stranded by a push refusal, awaiting a green flush.
type syncDrainEntry struct {
	SHA           string `json:"sha"`
	Subject       string `json:"subject"`
	RefusalReason string `json:"refusal_reason"`
	QueuedAtUnix  int64  `json:"queued_at_unix"`
}

// syncDrainQueue is the persisted stranded-commit queue (<repo>/.fak/sync-drain-queue.json).
type syncDrainQueue struct {
	Schema        string           `json:"schema"`
	Entries       []syncDrainEntry `json:"entries"`
	Attempts      int              `json:"attempts"`        // consecutive red/refused checks since the last flush
	NextRetryUnix int64            `json:"next_retry_unix"` // earliest sensible next check (backoff floor)
}

// syncDrainWindowVerdict is the quiescent-window read: build-green (pre-push witness) AND no peer
// merge in flight (safesync peer state). Green iff BOTH halves are clear.
type syncDrainWindowVerdict struct {
	Green        bool   `json:"green"`
	Reason       string `json:"reason,omitempty"` // why red (empty when green)
	BuildVerdict string `json:"build_verdict"`    // trunkBuildResult.Verdict (OK|NOOP|TRUNK_WOULD_NOT_COMPILE|…)
	PeerState    string `json:"peer_state"`       // safesync state (ahead|in-sync|behind|diverged|…)
}

// syncDrainReport is the --json payload and the source the human view renders from.
type syncDrainReport struct {
	Schema         string                 `json:"schema"`
	Verdict        string                 `json:"verdict"` // FLUSHED|IDLE|QUEUED|FLUSH_REFUSED|FLUSH_ERROR
	Window         syncDrainWindowVerdict `json:"window"`
	Queued         []syncDrainEntry       `json:"queued,omitempty"`
	Flushed        []syncDrainEntry       `json:"flushed,omitempty"`
	Attempts       int                    `json:"attempts"`
	BackoffSeconds int64                  `json:"backoff_seconds,omitempty"`
	NextRetryUnix  int64                  `json:"next_retry_unix,omitempty"`
	Detail         string                 `json:"detail,omitempty"`
}

type syncDrainConfig struct {
	repo      string
	remote    string
	branch    string
	queuePath string
	asJSON    bool
	budget    time.Duration
	sourceSHA string
}

// Seams (overridden in tests): the window read, the stranded-commit enumeration, the flush push,
// and the clock. Package-level vars mirror sync.go's syncAheadAudit / syncWorktree convention.
var (
	syncDrainWindow   = defaultSyncDrainWindow
	syncDrainStranded = defaultSyncDrainStranded
	syncDrainFlush    = defaultSyncDrainFlush
	syncDrainNow      = func() int64 { return time.Now().Unix() }
)

// runSyncDrain is the pure core: load the queue, read the window, and either flush (green) or
// queue-and-back-off (red). It never pushes while the window is red — that is the invariant the
// backoff test pins. Returns syncExitOK on a clean flush/idle, syncExitRefused when commits are
// held, syncExitInternal on a queue I/O fault.
func runSyncDrain(stdout, stderr io.Writer, cfg syncDrainConfig) int {
	ctx := context.Background()
	queue, err := loadSyncDrainQueue(cfg.queuePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak sync drain: %v\n", err)
		return syncExitInternal
	}
	stranded, err := syncDrainStranded(ctx, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "fak sync drain: %v\n", err)
		return syncExitInternal
	}
	now := syncDrainNow()
	report := syncDrainReport{Schema: syncDrainSchema}

	// Nothing stranded and nothing already queued — a clean idle tick. Short-circuit BEFORE the
	// (expensive) window read: reading the trunk build witness when there is nothing to flush is
	// pure waste. Reset any stale backoff and report nothing to drain.
	if len(queue.Entries) == 0 && len(stranded) == 0 {
		if serr := saveSyncDrainQueue(cfg.queuePath, syncDrainQueue{Schema: syncDrainSchema}); serr != nil {
			fmt.Fprintf(stderr, "fak sync drain: %v\n", serr)
			return syncExitInternal
		}
		report.Verdict = "IDLE"
		return emitSyncDrainReport(stdout, stderr, report, cfg.asJSON, syncExitOK)
	}

	if strings.TrimSpace(cfg.sourceSHA) == "" {
		sha, captureErr := syncCaptureSource(cfg.repo)
		if captureErr != nil || strings.TrimSpace(sha) == "" {
			fmt.Fprintf(stderr, "fak sync drain: capture push source: %v\n", captureErr)
			return syncExitInternal
		}
		cfg.sourceSHA = strings.TrimSpace(sha)
	}

	window := syncDrainWindow(ctx, cfg)
	report.Window = window

	// hold records the stranded commits (dedup by SHA) and backs off — the "never hammer the
	// gate" path shared by a red window and a green-but-still-refused push.
	hold := func(reason, verdict, detail string) {
		queue.Entries = mergeSyncDrainEntries(queue.Entries, stranded, reason, now)
		queue.Attempts++
		queue.NextRetryUnix = now + syncDrainBackoffSeconds(queue.Attempts)
		report.Verdict, report.Detail = verdict, detail
	}

	switch {
	case !window.Green:
		// Red window: queue the stranded commits with the refusal reason and back off.
		hold(window.Reason, "QUEUED", "")
	default:
		res, ferr := syncDrainFlush(ctx, cfg)
		switch {
		case ferr != nil:
			// Infra fault (e.g. git missing) — hold and back off rather than spin on it.
			hold("flush error", "FLUSH_ERROR", ferr.Error())
		case res.Pushed:
			// One push drained the whole queue. Clear it and the backoff.
			report.Flushed = mergeSyncDrainEntries(queue.Entries, stranded, "", now)
			queue = syncDrainQueue{Schema: syncDrainSchema}
			report.Verdict = "FLUSHED"
		default:
			// Build was green but the push was still refused (a transient race or a peer landing
			// between the window read and the push). Hold and back off — do not spin.
			detail := strings.TrimSpace(res.Reason + ": " + res.Detail)
			hold(detail, "FLUSH_REFUSED", detail)
		}
	}

	if serr := saveSyncDrainQueue(cfg.queuePath, queue); serr != nil {
		fmt.Fprintf(stderr, "fak sync drain: %v\n", serr)
		return syncExitInternal
	}

	report.Attempts = queue.Attempts
	report.NextRetryUnix = queue.NextRetryUnix
	if queue.NextRetryUnix > now {
		report.BackoffSeconds = queue.NextRetryUnix - now
	}
	if report.Verdict != "FLUSHED" {
		report.Queued = queue.Entries
	}

	code := syncExitRefused
	if report.Verdict == "FLUSHED" {
		code = syncExitOK
	}
	return emitSyncDrainReport(stdout, stderr, report, cfg.asJSON, code)
}

// emitSyncDrainReport writes the report (JSON or human) and returns okCode, or syncExitInternal on
// a JSON encode fault.
func emitSyncDrainReport(stdout, stderr io.Writer, report syncDrainReport, asJSON bool, okCode int) int {
	if asJSON {
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak sync drain: %v\n", err)
			return syncExitInternal
		}
	} else {
		renderSyncDrain(stdout, report)
	}
	return okCode
}

// mergeSyncDrainEntries appends the newly-stranded commits to the existing queue, deduping by SHA
// so a re-run does not double-count a commit already held. New entries are stamped with now and the
// current refusal reason; existing entries keep their original stamp and reason.
func mergeSyncDrainEntries(existing, stranded []syncDrainEntry, reason string, now int64) []syncDrainEntry {
	seen := make(map[string]bool, len(existing)+len(stranded))
	out := make([]syncDrainEntry, 0, len(existing)+len(stranded))
	for _, e := range existing {
		if e.SHA == "" || seen[e.SHA] {
			continue
		}
		seen[e.SHA] = true
		out = append(out, e)
	}
	for _, e := range stranded {
		if e.SHA == "" || seen[e.SHA] {
			continue
		}
		seen[e.SHA] = true
		if e.QueuedAtUnix == 0 {
			e.QueuedAtUnix = now
		}
		if e.RefusalReason == "" {
			e.RefusalReason = reason
		}
		out = append(out, e)
	}
	return out
}

// syncDrainBackoffSeconds returns the backoff floor for the Nth consecutive red/refused check:
// 15, 30, 60, … doubling from a 15s base, capped at 900s (15 min).
func syncDrainBackoffSeconds(attempts int) int64 {
	if attempts <= 1 {
		return syncDrainBackoffBaseSec
	}
	b := syncDrainBackoffBaseSec
	for i := 1; i < attempts; i++ {
		b *= 2
		if b >= syncDrainBackoffCapSec {
			return syncDrainBackoffCapSec
		}
	}
	return b
}

func loadSyncDrainQueue(path string) (syncDrainQueue, error) {
	if strings.TrimSpace(path) == "" {
		return syncDrainQueue{Schema: syncDrainSchema}, nil
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return syncDrainQueue{Schema: syncDrainSchema}, nil
	}
	if err != nil {
		return syncDrainQueue{}, fmt.Errorf("read queue %s: %w", path, err)
	}
	var q syncDrainQueue
	if err := json.Unmarshal(b, &q); err != nil {
		return syncDrainQueue{}, fmt.Errorf("parse queue %s: %w", path, err)
	}
	if q.Schema == "" {
		q.Schema = syncDrainSchema
	}
	return q, nil
}

func saveSyncDrainQueue(path string, q syncDrainQueue) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	q.Schema = syncDrainSchema
	if err := writeIndentedJSONFile(path, q); err != nil {
		return fmt.Errorf("write queue %s: %w", path, err)
	}
	return nil
}

// defaultSyncDrainWindow reads the quiescent window: build-green from the pre-push witness
// (advisory=true so a drainer single-flights under host contention instead of piling on a
// redundant concurrent build) AND no peer merge in flight from the safesync assessment.
func defaultSyncDrainWindow(ctx context.Context, cfg syncDrainConfig) syncDrainWindowVerdict {
	budget := cfg.budget
	if budget <= 0 {
		budget = 60 * time.Second
	}
	build, _ := evaluatePrePushBuild(cfg.repo, "", budget, true)
	v := syncDrainWindowVerdict{BuildVerdict: build.Verdict}

	info, err := safesync.Assess(ctx, safesync.Options{
		Repo:   cfg.repo,
		Remote: cfg.remote,
		Branch: cfg.branch,
		Fetch:  true,
	})
	if err != nil {
		v.PeerState = "unknown"
		v.Reason = "cannot assess remote: " + err.Error()
		return v
	}
	v.PeerState = info.State

	switch {
	case !build.OK:
		v.Reason = "trunk build not green: " + build.Verdict
	case info.State == safesync.StateBehind || info.State == safesync.StateDiverged:
		v.Reason = "peer merge in flight: " + info.State
	default:
		v.Green = true
	}
	return v
}

// defaultSyncDrainStranded lists the local commits ahead of the remote branch (origin/<branch>..HEAD),
// oldest first — the commits a push would carry and that a refusal strands. On a missing remote ref
// or a git error it returns no entries (best-effort: nothing knowable to queue), not a hard failure.
func defaultSyncDrainStranded(ctx context.Context, cfg syncDrainConfig) ([]syncDrainEntry, error) {
	remote := cfg.remote
	if remote == "" {
		remote = "origin"
	}
	branch := cfg.branch
	if branch == "" {
		branch = syncDrainCurrentBranch(ctx, cfg.repo)
	}
	if branch == "" {
		return nil, nil
	}
	rangeSpec := fmt.Sprintf("%s/%s..HEAD", remote, branch)
	cmd := exec.CommandContext(ctx, "git", "log", "--reverse", "--format=%H%x1f%s", rangeSpec)
	cmd.Dir = cfg.repo
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, nil
	}
	var entries []syncDrainEntry
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		sha, subject, _ := strings.Cut(line, "\x1f")
		if strings.TrimSpace(sha) == "" {
			continue
		}
		entries = append(entries, syncDrainEntry{SHA: sha, Subject: strings.TrimSpace(subject)})
	}
	return entries, nil
}

func syncDrainCurrentBranch(ctx context.Context, repo string) string {
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--abbrev-ref", "HEAD")
	cmd.Dir = repo
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	branch := strings.TrimSpace(string(out))
	if branch == "HEAD" { // detached
		return ""
	}
	return branch
}

// defaultSyncDrainFlush drains the queue in a single push via the shared SafePush helper, which
// already retries a transient non-fast-forward race and stops cleanly on a genuine behind/diverge.
func defaultSyncDrainFlush(ctx context.Context, cfg syncDrainConfig) (safesync.PushResult, error) {
	return safesync.SafePush(ctx, safesync.PushOptions{
		Repo:       cfg.repo,
		Remote:     cfg.remote,
		Branch:     cfg.branch,
		SourceRef:  cfg.sourceSHA,
		TargetRef:  syncTargetRef(cfg.branch),
		MaxRetries: 3,
	})
}

func renderSyncDrain(w io.Writer, r syncDrainReport) {
	switch r.Verdict {
	case "FLUSHED":
		fmt.Fprintf(w, "flushed %d stranded commit(s) in one push (window green: build=%s peer=%s)\n",
			len(r.Flushed), r.Window.BuildVerdict, r.Window.PeerState)
		for _, e := range r.Flushed {
			fmt.Fprintf(w, "  FLUSHED  %s  %s\n", short(e.SHA), e.Subject)
		}
	case "IDLE":
		fmt.Fprintf(w, "drain idle: window green, nothing queued or stranded (build=%s peer=%s)\n",
			r.Window.BuildVerdict, r.Window.PeerState)
	case "QUEUED":
		fmt.Fprintf(w, "[HELD] window red (%s); %d commit(s) queued, backing off %ds (attempt %d)\n",
			r.Window.Reason, len(r.Queued), r.BackoffSeconds, r.Attempts)
		for _, e := range r.Queued {
			fmt.Fprintf(w, "  QUEUED  %s  %s  (%s)\n", short(e.SHA), e.Subject, e.RefusalReason)
		}
		if r.Attempts >= 2 {
			fmt.Fprintln(w, "  hint: trunk is red; invoke /ci-repair or run 'fak-dev ci-preflight' to diagnose and auto-heal")
		}
	case "FLUSH_REFUSED":
		fmt.Fprintf(w, "[HELD] window green but push refused (%s); %d commit(s) held, backing off %ds (attempt %d)\n",
			r.Detail, len(r.Queued), r.BackoffSeconds, r.Attempts)
		for _, e := range r.Queued {
			fmt.Fprintf(w, "  QUEUED  %s  %s\n", short(e.SHA), e.Subject)
		}
	default:
		fmt.Fprintf(w, "[%s] %s\n", r.Verdict, r.Detail)
		for _, e := range r.Queued {
			fmt.Fprintf(w, "  QUEUED  %s  %s\n", short(e.SHA), e.Subject)
		}
	}
}
