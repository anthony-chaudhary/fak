package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchloop"
	"github.com/anthony-chaudhary/fak/internal/nightrun"
)

func cmdBenchLoop(argv []string) { os.Exit(runBenchLoop(os.Stdout, os.Stderr, argv)) }

func runBenchLoop(stdout, stderr io.Writer, argv []string) int {
	// The --no-reuse escape hatch is bench-loop-wide (status|next|run share no flag
	// set), so it is intercepted here and exported as FAK_BENCH_NO_REUSE=1 BEFORE any
	// benchloop.Load, flipping the launch gate from reuse_run (skip) to collect_local
	// (force re-run). Follow-up to #4600's env-only escape hatch.
	argv, force := stripNoReuseFlag(argv)
	if force {
		_ = os.Setenv(benchloop.NoReuseEnv, "1")
	}
	if len(argv) == 0 {
		return benchLoopStatus(stdout, stderr, nil)
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "status":
		return benchLoopStatus(stdout, stderr, rest)
	case "next", "enter":
		return benchLoopNext(stdout, stderr, rest)
	case "walk":
		return benchLoopWalk(stdout, stderr, rest)
	case "run", "collect":
		return nightrunRun(stdout, stderr, rest)
	case "fleet":
		return runBenchFleet(stdout, stderr, rest)
	case "install":
		return runBenchFleetInstall(stdout, stderr, rest)
	case "-h", "--help", "help":
		benchLoopUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak bench-loop: unknown subcommand %q\n", sub)
		benchLoopUsage(stderr)
		return 2
	}
}

// stripNoReuseFlag pulls the bench-loop-wide --no-reuse escape hatch out of argv
// before any per-subcommand flag set sees it, returning the remaining args and
// whether the flag was present and truthy. Both --no-reuse and -no-reuse are
// accepted, in either order relative to the subcommand token; an explicit
// --no-reuse=false/0/off/no leaves the gate on (the hatch is opt-in force-rerun).
func stripNoReuseFlag(argv []string) ([]string, bool) {
	rest := make([]string, 0, len(argv))
	force := false
	for _, a := range argv {
		switch {
		case a == "--no-reuse" || a == "-no-reuse":
			force = true
		case strings.HasPrefix(a, "--no-reuse=") || strings.HasPrefix(a, "-no-reuse="):
			force = noReuseFlagTruthy(a[strings.IndexByte(a, '=')+1:])
		default:
			rest = append(rest, a)
		}
	}
	return rest, force
}

// noReuseFlagTruthy mirrors the reuse gate's own truthiness: an explicit
// 0/false/no/off (or empty) value leaves reuse enabled; anything else forces a
// re-run. Bare --no-reuse (no value) is handled by the caller as force=true.
func noReuseFlagTruthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

type benchLoopFlags struct {
	workspace string
	overlay   string
	ledger    string
	now       string
	asJSON    bool
}

func parseBenchLoopFlags(name string, stderr io.Writer, argv []string) (*benchLoopFlags, int, bool) {
	fs := flag.NewFlagSet("fak bench-loop "+name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	f := &benchLoopFlags{}
	fs.StringVar(&f.workspace, "workspace", "", "workspace root (default: repo root)")
	fs.StringVar(&f.overlay, "overlay", "", "nightrun overlay file (default: <root>/"+nightrun.DefaultOverlayRel+" if present)")
	fs.StringVar(&f.ledger, "ledger", "", "nightrun ledger path (default: <root>/"+nightrun.DefaultLedgerRel+")")
	fs.StringVar(&f.now, "now", "", "evaluate as-of RFC3339, compact 20060102T150405Z, or YYYY-MM-DD")
	fs.BoolVar(&f.asJSON, "json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return nil, 2, false
	}
	return f, 0, true
}

// workspaceOrRepoRoot resolves an optional --workspace value: empty means the
// repo root, anything else is absolutized when possible.
func workspaceOrRepoRoot(workspace string) string {
	if workspace == "" {
		return repoRoot()
	}
	if abs, err := filepath.Abs(workspace); err == nil {
		return abs
	}
	return workspace
}

// parsePinnedTimeFlag parses a non-empty pinned-time flag value in RFC3339,
// compact 20060102T150405Z, or YYYY-MM-DD form, normalized to UTC. The caller
// owns the empty-value default (zero time vs wall clock).
func parsePinnedTimeFlag(flagName, value string) (time.Time, error) {
	for _, layout := range []string{time.RFC3339, "20060102T150405Z", "2006-01-02"} {
		if t, err := time.Parse(layout, value); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("%s %q is not RFC3339, compact (20060102T150405Z), or YYYY-MM-DD", flagName, value)
}

func (f *benchLoopFlags) root() string {
	return workspaceOrRepoRoot(f.workspace)
}

func (f *benchLoopFlags) nowOrZero() (time.Time, error) {
	if f.now == "" {
		return time.Time{}, nil
	}
	return parsePinnedTimeFlag("--now", f.now)
}

func benchLoopReport(stderr io.Writer, argv []string, name string) (benchloop.Report, *benchLoopFlags, int, bool) {
	f, rc, ok := parseBenchLoopFlags(name, stderr, argv)
	if !ok {
		return benchloop.Report{}, nil, rc, false
	}
	now, err := f.nowOrZero()
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-loop: %v\n", err)
		return benchloop.Report{}, nil, 2, false
	}
	rep := benchloop.Load(benchloop.Options{
		Root:        f.root(),
		OverlayPath: f.overlay,
		LedgerPath:  f.ledger,
		Now:         now,
	})
	return rep, f, 0, true
}

func benchLoopStatus(stdout, stderr io.Writer, argv []string) int {
	rep, f, rc, ok := benchLoopReport(stderr, argv, "status")
	if !ok {
		return rc
	}
	if f.asJSON {
		_ = writeIndentedJSONNoEscape(stdout, rep)
		return 0
	}
	fmt.Fprint(stdout, benchloop.RenderStatus(rep))
	return 0
}

func benchLoopNext(stdout, stderr io.Writer, argv []string) int {
	rep, f, rc, ok := benchLoopReport(stderr, argv, "next")
	if !ok {
		return rc
	}
	if f.asJSON {
		_ = writeIndentedJSONNoEscape(stdout, rep.NextAction)
		return 0
	}
	fmt.Fprint(stdout, benchloop.RenderNext(rep.NextAction))
	return 0
}

func benchLoopWalk(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-loop walk", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	walk := benchloop.Walk()
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, walk)
		return 0
	}
	fmt.Fprint(stdout, benchloop.RenderWalk(walk))
	return 0
}

func benchLoopUsage(w io.Writer) {
	fmt.Fprint(w, `fak bench-loop - benchmark super-loop manager

usage:
  fak bench-loop status [--json] [--no-reuse] [--workspace DIR] [--now STAMP]
  fak bench-loop next   [--json] [--no-reuse] [--workspace DIR] [--now STAMP]
  fak bench-loop walk   [--json]
  fak bench-loop run    [--no-reuse] [nightrun run flags...]
  fak bench-loop fleet  [status|dispatch] [--apply] [--json] [--now STAMP] [--plan-json FILE]
  fak bench-loop install [--interval MINUTES] [--remove]

status folds the benchmark registry, recorded run catalog, nightrun ledger,
local capability-aware next selection, and benchmark-authority gap. --no-reuse
exports FAK_BENCH_NO_REUSE=1 so the launch gate forces a re-run (collect_local)
instead of skipping a redundant same-lineage run (reuse_run). run delegates
to fak nightrun run, which is dry-run unless --apply is passed. fleet walks the
benchmark machine plan and queues one idempotent request per node; fleet dispatch atomically claims and witnesses requests; install keeps both
tick always on via the Windows Task Scheduler (use cron elsewhere).
`)
}

// Stop conditions for a benchmark send loop.
//
// (StopCondition and its five subclasses, _STOP_CONDITION_CLASSES,
// StopConditionChecker), pinned at
// in idiomatic Go as a closed table over value snapshots; no upstream code is
// copied. Study note: fak-private docs/research/STUDY-aiperf-2026-11-09.md,
// candidate aiperf-lifecycle-9.
//
// Why the set is closed and ordered: a loop that checks "may I send?" before
// every request needs one answer that cannot drift between call sites. The
// table below is the only place a condition is declared, its position is the
// order it is evaluated in, and a condition whose limit is unset is dropped
// when the checker is built rather than re-tested per request.
//
// Why DAG children get their own gate: a child request is fanned out when its
// parent returns, not by the send loop, and belongs to its parent's session.
// A wire-request cap, a deadline and cancellation are promises about every
// request on the wire, so they bind children. A session target counts planned
// conversations, and "sending complete" only says the loop stopped issuing
// roots; honoring either for children would cut a conversation tree in half.

// benchStopLimits holds the configured limits. A nil limit is unset: its
// condition is left out of the checker.
type benchStopLimits struct {
	MaxRequests *int
	MaxSessions *int
	Duration    *time.Duration
}

// benchStopState is a read-only snapshot of the send loop. Conditions never
// mutate it.
type benchStopState struct {
	Cancelled       bool
	SendingComplete bool
	// RequestsSent counts every request on the wire, children included.
	RequestsSent int
	// RootRequestsSent counts only requests the send loop issued. The
	// session condition compares it to TotalSessionTurns, which is also
	// root-only; child requests arriving in parallel would otherwise make a
	// session look finished early.
	RootRequestsSent  int
	SessionsSent      int
	TotalSessionTurns int
	Elapsed           time.Duration
}

type benchStopCondition struct {
	Name string
	// AppliesToDAGChildren reports whether the condition gates child
	// dispatch as well as the send loop.
	AppliesToDAGChildren bool
	ShouldUse            func(benchStopLimits) bool
	CanSendAnyTurn       func(benchStopLimits, benchStopState) bool
	// CanStartNewSession, when set, is an extra restriction on first turns
	// only; a nil func adds none.
	CanStartNewSession func(benchStopLimits, benchStopState) bool
}

const (
	benchStopLifecycle    = "lifecycle"
	benchStopDAGLifecycle = "dag_lifecycle"
	benchStopRequestCount = "request_count"
	benchStopSessionCount = "session_count"
	benchStopDuration     = "duration"
)

func benchStopAlways(benchStopLimits) bool { return true }

// benchStopConditions is the closed set, in evaluation order.
var benchStopConditions = [...]benchStopCondition{
	{
		Name:      benchStopLifecycle,
		ShouldUse: benchStopAlways,
		CanSendAnyTurn: func(_ benchStopLimits, s benchStopState) bool {
			return !s.Cancelled && !s.SendingComplete
		},
	},
	{
		// The send loop's lifecycle gate skips children because of
		// SendingComplete; this entry restores cancellation for them.
		Name:                 benchStopDAGLifecycle,
		AppliesToDAGChildren: true,
		ShouldUse:            benchStopAlways,
		CanSendAnyTurn: func(_ benchStopLimits, s benchStopState) bool {
			return !s.Cancelled
		},
	},
	{
		Name:                 benchStopRequestCount,
		AppliesToDAGChildren: true,
		ShouldUse:            func(l benchStopLimits) bool { return l.MaxRequests != nil },
		CanSendAnyTurn: func(l benchStopLimits, s benchStopState) bool {
			return s.RequestsSent < *l.MaxRequests
		},
	},
	{
		Name:      benchStopSessionCount,
		ShouldUse: func(l benchStopLimits) bool { return l.MaxSessions != nil },
		CanSendAnyTurn: func(l benchStopLimits, s benchStopState) bool {
			return s.SessionsSent < *l.MaxSessions || s.RootRequestsSent < s.TotalSessionTurns
		},
		CanStartNewSession: func(l benchStopLimits, s benchStopState) bool {
			return s.SessionsSent < *l.MaxSessions
		},
	},
	{
		Name:                 benchStopDuration,
		AppliesToDAGChildren: true,
		ShouldUse:            func(l benchStopLimits) bool { return l.Duration != nil },
		CanSendAnyTurn: func(l benchStopLimits, s benchStopState) bool {
			return s.Elapsed < *l.Duration
		},
	},
}

// benchStopChecker answers the send loop's three questions from the
// conditions that apply to one set of limits.
type benchStopChecker struct {
	limits benchStopLimits
	active []benchStopCondition
}

func newBenchStopChecker(limits benchStopLimits) benchStopChecker {
	c := benchStopChecker{limits: limits}
	for _, cond := range benchStopConditions {
		if cond.ShouldUse(limits) {
			c.active = append(c.active, cond)
		}
	}
	return c
}

// Active lists the conditions in force, in evaluation order.
func (c benchStopChecker) Active() []string {
	names := make([]string, len(c.active))
	for i, cond := range c.active {
		names[i] = cond.Name
	}
	return names
}

// FirstStop names the earliest condition refusing a turn, or "" when the
// loop may send. forDAGChild restricts the walk to the child gate.
func (c benchStopChecker) FirstStop(s benchStopState, forDAGChild bool) string {
	for _, cond := range c.active {
		if forDAGChild && !cond.AppliesToDAGChildren {
			continue
		}
		if !cond.CanSendAnyTurn(c.limits, s) {
			return cond.Name
		}
	}
	return ""
}

// CanSendAnyTurn reports whether the send loop may issue any turn, first or
// continuation.
func (c benchStopChecker) CanSendAnyTurn(s benchStopState) bool {
	return c.FirstStop(s, false) == ""
}

// CanSendDAGChildTurn reports whether a child turn may still be dispatched.
func (c benchStopChecker) CanSendDAGChildTurn(s benchStopState) bool {
	return c.FirstStop(s, true) == ""
}

// CanStartNewSession reports whether a first turn may be issued. It implies
// CanSendAnyTurn and adds the per-condition session restrictions, so a
// reached session target stops new sessions while open ones keep sending.
func (c benchStopChecker) CanStartNewSession(s benchStopState) bool {
	if !c.CanSendAnyTurn(s) {
		return false
	}
	for _, cond := range c.active {
		if cond.CanStartNewSession != nil && !cond.CanStartNewSession(c.limits, s) {
			return false
		}
	}
	return true
}

// per-phase progress-stall watchdog.
type phaseProgressWatchdog struct {
	phase        string
	progress     int64
	lastAdvance  time.Time
	stallTimeout time.Duration
}

func newPhaseProgressWatchdog(phase string, now time.Time, timeout time.Duration) *phaseProgressWatchdog {
	return &phaseProgressWatchdog{phase: phase, lastAdvance: now, stallTimeout: timeout}
}

func (w *phaseProgressWatchdog) Update(phase string, progress int64, now time.Time) bool {
	if phase != w.phase || progress > w.progress {
		w.phase = phase
		w.progress = progress
		w.lastAdvance = now
		return false
	}
	return now.Sub(w.lastAdvance) > w.stallTimeout
}

// split CoT/thinking block from final answer and grade exact match on answer only.
func splitThinkingAndAnswer(raw string) (thinking, answer string) {
	const openTag, closeTag = "<think>", "</think>"
	start := strings.Index(raw, openTag)
	end := strings.Index(raw, closeTag)
	if start >= 0 && end > start {
		thinking = strings.TrimSpace(raw[start+len(openTag) : end])
		answer = strings.TrimSpace(raw[end+len(closeTag):])
		return thinking, answer
	}
	return "", strings.TrimSpace(raw)
}

func gradeExactMatch(raw, expected string) bool {
	_, ans := splitThinkingAndAnswer(raw)
	return ans == strings.TrimSpace(expected)
}
