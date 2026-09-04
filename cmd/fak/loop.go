package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchpost"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/perfrsiscore"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/repoguard"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

func cmdLoop(argv []string) { os.Exit(runLoop(os.Stdout, os.Stderr, argv)) }

func runLoop(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		loopUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "append":
		return runLoopAppend(stdout, stderr, argv[1:])
	case "run":
		return runLoopRun(stdout, stderr, argv[1:])
	case "status":
		return runLoopStatus(stdout, stderr, argv[1:])
	case "health":
		return runLoopHealth(stdout, stderr, argv[1:])
	case "rollup":
		return runLoopRollup(stdout, stderr, argv[1:])
	case "economics":
		return runLoopEconomics(stdout, stderr, argv[1:])
	case "admit":
		return runLoopAdmit(stdout, stderr, argv[1:])
	case "policy":
		return runLoopPolicy(stdout, stderr, argv[1:])
	case "region":
		return runLoopRegion(stdout, stderr, argv[1:])
	case "recover":
		return runLoopRecover(stdout, stderr, argv[1:])
	case "repair":
		return runLoopRepair(stdout, stderr, argv[1:])
	case "drive":
		return runLoopDrive(stdout, stderr, argv[1:])
	case "reap":
		return runLoopReap(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		loopUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak loop: unknown subcommand %q\n", argv[0])
		loopUsage(stderr)
		return 2
	}
}

type loopCommand interface {
	Start() error
	Wait() error
	PID() int
	Kill() error
}

type execLoopCommand struct {
	cmd *exec.Cmd
}

func (c execLoopCommand) Start() error { return c.cmd.Start() }
func (c execLoopCommand) Wait() error  { return c.cmd.Wait() }
func (c execLoopCommand) PID() int {
	if c.cmd == nil || c.cmd.Process == nil {
		return 0
	}
	return c.cmd.Process.Pid
}
func (c execLoopCommand) Kill() error {
	if c.cmd == nil || c.cmd.Process == nil {
		return nil
	}
	// Tree kill, not a bare single-PID kill (#3101, sibling of #2989). The wrapped child is a
	// user-supplied agent CLI after `--` (or `fak guard -- <agent>`) that spawns a descendant
	// subtree (node runtime + MCP/tool subprocesses). c.cmd.Process.Kill() reaps only the
	// immediate PID and orphans that subtree; on a `fak loop drive --deadline` loop that
	// re-spawns the agent every turn, the orphans accumulate and poison dispatch preflight
	// (unattributed_live -> REFUSE_NO_SEAT), exactly as #2989 did. procguard.KillPID reaps the
	// whole tree (taskkill /T /F on Windows, process-group/descendant SIGKILL on POSIX).
	loopTreeKill(c.cmd.Process.Pid)
	return nil
}

// loopTreeKill is the destructive tree reaper for the loop supervisor's kill sink. Fixing this
// one seam fixes every loopCommand.Kill() call site at once (the START-ledger-append failure in
// runLoop, the onStart ledger-callback error in loop drive, and both deadline-expiry hot paths
// in waitLoopDriveCommand). Injectable for tests. Mirrors fleetKillPID (fleet.go) and
// guardChildTreeKill (guard_child.go).
var loopTreeKill = procguard.KillPID

var loopExecutable = os.Executable

var loopNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return execLoopCommand{cmd: cmd}
}

const (
	loopPerformanceRSIOutputEnv = "FAK_PERFORMANCE_RSI_OUTPUT"
	loopIDEnv                   = "FAK_LOOP_ID"
	loopRunIDEnv                = "FAK_LOOP_RUN_ID"
	loopSandboxEnvAllow         = "FAK_SANDBOX_ENV_ALLOW"
	loopPerformanceRSIMaxBytes  = 1 << 20
)

func runLoopAppend(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop append", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	loopID := fs.String("loop", "", "loop id")
	kind := fs.String("kind", "", "event kind: armed|fire|admit|start|heartbeat|end|witness|notify")
	runID := fs.String("run", "", "run id")
	source := fs.String("source", "", "event source, such as schedule|github|slack|task-scheduler")
	principal := fs.String("principal", "", "authenticated principal or producer id")
	state := fs.String("state", "", "loop state")
	status := fs.String("status", "", "run/admission/witness status")
	reason := fs.String("reason", "", "bounded reason token or short refusal code")
	summary := fs.String("summary", "", "bounded human summary")
	asJSON := fs.Bool("json", false, "emit the appended event as JSON")
	var evidence loopKVList
	var metrics loopKVList
	fs.Var(&evidence, "evidence", "repeatable KIND=REF evidence ref")
	fs.Var(&metrics, "metric", "repeatable NAME=INT64 metric")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop append: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	ev := loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventKind(*kind),
		Source:       *source,
		Principal:    *principal,
		State:        loopmgr.LoopState(*state),
		Status:       loopmgr.RunStatus(*status),
		Reason:       *reason,
		Summary:      *summary,
		EvidenceRefs: parseLoopEvidence(evidence),
	}
	if len(metrics) > 0 {
		ev.Metrics = map[string]int64{}
		for _, item := range metrics {
			k, v, ok := strings.Cut(item, "=")
			if !ok || strings.TrimSpace(k) == "" {
				fmt.Fprintf(stderr, "fak loop append: --metric must be NAME=INT64, got %q\n", item)
				return 2
			}
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err != nil {
				fmt.Fprintf(stderr, "fak loop append: --metric %q has invalid value: %v\n", item, err)
				return 2
			}
			ev.Metrics[strings.TrimSpace(k)] = n
		}
	}

	appended, err := loopmgr.Append(*ledger, ev)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop append: %v\n", err)
		return 2
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, appended, "fak loop append")
	}
	fmt.Fprintf(stdout, "appended loop event seq=%d kind=%s loop=%s ledger=%s\n",
		appended.Seq, appended.Kind, appended.LoopID, *ledger)
	return 0
}

type loopRunOptions struct {
	ledger          string
	loopID          string
	runID           string
	source          string
	principal       string
	asJSON          bool
	notifySlack     bool
	dispatchChannel string
	dispatchToken   string
	noGuard         bool
	cmdArgs         []string
}

func parseLoopRunOptions(stderr io.Writer, argv []string) (loopRunOptions, int) {
	fs := flag.NewFlagSet("loop run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	loopID := fs.String("loop", "", "loop id")
	runID := fs.String("run", "", "run id")
	source := fs.String("source", "manual", "trigger source, such as cron|launchd|task-scheduler|manual")
	principal := fs.String("principal", "", "authenticated principal or producer id")
	asJSON := fs.Bool("json", false, "emit a JSON run report")
	notifySlack := fs.Bool("notify-slack", false, "opt in to one per-run dispatch Slack card (start, final edit, threaded witness)")
	dispatchChannel := fs.String("dispatch-channel", "", "override dispatch channel id (default: $FAK_DISPATCH_CHANNEL / .env.slack.local)")
	dispatchToken := fs.String("dispatch-token", "", "override dispatch bot token (default: $FAK_DISPATCH_TOKEN, then scoreboard token)")
	noGuard := fs.Bool("no-guard", false, "explicitly disable the default fak guard containment wrapper for this run (logged in the loop ledger)")
	if !parseFlags(fs, argv) {
		return loopRunOptions{}, 2
	}
	cmdArgs := fs.Args()
	if strings.TrimSpace(*loopID) == "" {
		fmt.Fprintln(stderr, "fak loop run: --loop is required")
		return loopRunOptions{}, 2
	}
	if len(cmdArgs) == 0 {
		fmt.Fprintln(stderr, "fak loop run: command is required after --")
		return loopRunOptions{}, 2
	}
	if *runID == "" {
		*runID = defaultLoopRunID(*loopID)
	}
	return loopRunOptions{
		ledger:          *ledger,
		loopID:          *loopID,
		runID:           *runID,
		source:          *source,
		principal:       *principal,
		asJSON:          *asJSON,
		notifySlack:     *notifySlack,
		dispatchChannel: *dispatchChannel,
		dispatchToken:   *dispatchToken,
		noGuard:         *noGuard,
		cmdArgs:         cmdArgs,
	}, 0
}

func handleLoopPerformanceRSI(stderr io.Writer, runID, performanceRSIInput, performanceRSIOutput string, performanceRSIPrepErr error) {
	performanceRSIReceipt := perfrsiscore.ScoreLoopTurnFromEnvironment()
	if performanceRSIInput == "" {
		performanceRSIReceipt = scoreAutomaticLoopPerformanceRSI(performanceRSIOutput, runID, performanceRSIPrepErr)
	}
	if err := perfrsiscore.RecordLoopTurnUsage(performanceRSIReceipt); err != nil {
		fmt.Fprintf(stderr, "fak loop run: record performance-rsi usage: %v\n", err)
	}
	fmt.Fprintf(stderr, "fak loop run: performance-rsi loop-turn %s\n", perfrsiscore.FormatLoopTurnReceipt(performanceRSIReceipt))
}

func runLoopRun(stdout, stderr io.Writer, argv []string) int {
	opts, exitCode := parseLoopRunOptions(stderr, argv)
	if exitCode != 0 {
		return exitCode
	}
	ledger := &opts.ledger
	loopID := &opts.loopID
	runID := &opts.runID
	source := &opts.source
	principal := &opts.principal
	asJSON := &opts.asJSON
	notifySlack := &opts.notifySlack
	dispatchChannel := &opts.dispatchChannel
	dispatchToken := &opts.dispatchToken
	cmdArgs := opts.cmdArgs
	guardEnabled := !opts.noGuard
	headBefore := dispatchpost.HeadSHA(ctx(), "")
	baseEvidence := []loopmgr.EvidenceRef{{Kind: "command", Ref: filepath.Base(cmdArgs[0])}}
	baseMetrics := map[string]int64{"argc": int64(len(cmdArgs))}
	if guardEnabled {
		baseEvidence = append(baseEvidence, loopmgr.EvidenceRef{Kind: "guard", Ref: "fak guard"})
		baseMetrics["guard_enabled"] = 1
	} else {
		baseMetrics["guard_enabled"] = 0
	}
	// Every ledger event appended HERE identifies the SAME run: one loop id, one run id, one
	// trigger source, one principal, one evidence set. Stamping that identity in one place
	// means a refusal, a guard-unavailable end and a normal admit cannot disagree about whose
	// run they belong to; each caller below supplies only what makes its event distinct. (The
	// child's own START/END pair is stamped from the same values by loopRunChildCtx, which
	// carries them across into loopRunChild.)
	loopEvent := func(ev loopmgr.Event) loopmgr.Event {
		ev.LoopID = *loopID
		ev.RunID = *runID
		ev.Source = *source
		ev.Principal = *principal
		ev.EvidenceRefs = baseEvidence
		return ev
	}
	if err := appendLoopRunEvent(*ledger, loopEvent(loopmgr.Event{
		Kind:    loopmgr.EventFire,
		Summary: "loop run requested",
		Metrics: cloneLoopMetrics(baseMetrics),
	})); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 1
	}

	childArgv := append([]string(nil), cmdArgs...)
	admitReason := "GUARD_ADMITTED"
	admitSummary := "loop wrapper admitted command under fak guard"
	if guardEnabled {
		if violations := loopContainmentViolations(cmdArgs); len(violations) > 0 {
			m := cloneLoopMetrics(baseMetrics)
			m["violations"] = int64(len(violations))
			summary := repoguard.RenderReason(violations)
			if err := appendLoopRunEvent(*ledger, loopEvent(loopmgr.Event{
				Kind:    loopmgr.EventAdmit,
				Status:  loopmgr.StatusRefused,
				Reason:  repoguard.Reason,
				Summary: summary,
				Metrics: m,
			})); err != nil {
				fmt.Fprintf(stderr, "fak loop run: %v\n", err)
				return 1
			}
			fmt.Fprintf(stderr, "fak loop run: containment refused command: %s\n", summary)
			if *asJSON && !writeLoopRunReport(stdout, stderr, *ledger, *loopID, *runID, map[string]any{
				"status":    "refused",
				"reason":    repoguard.Reason,
				"exit_code": 3,
			}) {
				return 1
			}
			return 3
		}
		fakBin, err := loopExecutable()
		if err != nil {
			m := cloneLoopMetrics(baseMetrics)
			m["exit_code"] = 127
			_ = appendLoopRunEvent(*ledger, loopEvent(loopmgr.Event{
				Kind:    loopmgr.EventEnd,
				Status:  loopmgr.StatusFailed,
				Reason:  "GUARD_UNAVAILABLE",
				Summary: err.Error(),
				Metrics: m,
			}))
			fmt.Fprintf(stderr, "fak loop run: resolve fak guard binary: %v\n", err)
			return 127
		}
		childArgv = loopGuardArgv(fakBin, cmdArgs)
	} else {
		admitReason = "GUARD_DISABLED"
		admitSummary = "--no-guard disabled fak guard containment"
		fmt.Fprintln(stderr, "fak loop run: WARNING --no-guard disables fak guard containment for this run")
	}
	if err := appendLoopRunEvent(*ledger, loopEvent(loopmgr.Event{
		Kind:    loopmgr.EventAdmit,
		Status:  loopmgr.StatusAdmitted,
		Reason:  admitReason,
		Summary: admitSummary,
		Metrics: cloneLoopMetrics(baseMetrics),
	})); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 1
	}

	// Per-run Slack is EXPLICITLY armed: --notify-slack uses the resolved channel,
	// while --dispatch-channel is itself an explicit destination. Merely inheriting
	// the machine-wide FAK_DISPATCH_CHANNEL must not turn every scheduler/test/scout
	// loop into top-level operator-channel traffic (#4677, parent #4652).
	card := openDispatchRunCardIfRequested(stderr, *notifySlack, *dispatchChannel, *dispatchToken, dispatchpost.Result{
		LoopID:  *loopID,
		RunID:   *runID,
		Command: filepath.Base(cmdArgs[0]),
	})

	performanceRSIInput := strings.TrimSpace(os.Getenv(perfrsiscore.LoopTurnInputEnv))
	var performanceRSIOutput string
	var performanceRSIPrepErr error
	var childEnv []string
	if performanceRSIInput == "" {
		performanceRSIOutput, performanceRSIPrepErr = reserveLoopPerformanceRSIOutput(*ledger)
		env := envMap(os.Environ())
		env[loopIDEnv] = *loopID
		env[loopRunIDEnv] = *runID
		if performanceRSIOutput != "" {
			env[loopPerformanceRSIOutputEnv] = performanceRSIOutput
		} else {
			delete(env, loopPerformanceRSIOutputEnv)
		}
		env[loopSandboxEnvAllow] = appendLoopEnvAllow(env[loopSandboxEnvAllow],
			loopPerformanceRSIOutputEnv, loopIDEnv, loopRunIDEnv)
		childEnv = envSliceFromMap(env)
	}

	exitCode, durationMS, fatal := loopRunChild(stdout, stderr, childArgv, loopRunChildCtx{
		ledger:    *ledger,
		loopID:    *loopID,
		runID:     *runID,
		source:    *source,
		principal: *principal,
		evidence:  baseEvidence,
		metrics:   baseMetrics,
		env:       childEnv,
	})
	if fatal != 0 {
		return fatal
	}

	// Post a witnessed dispatch-result card to Slack so a slow background dispatch
	// reports its outcome without anyone tailing the ledger. Gated and best-effort:
	// a resolved channel (or --notify-slack) arms it, and any failure is reported to
	// stderr without changing the run's exit code — the dispatch's result must stand
	// on its own even if Slack is unreachable.
	postDispatchResult(stderr, *notifySlack, *dispatchChannel, *dispatchToken, card,
		dispatchpost.Result{
			LoopID:     *loopID,
			RunID:      *runID,
			ExitCode:   exitCode,
			DurationMS: durationMS,
			Command:    filepath.Base(cmdArgs[0]),
			HeadBefore: headBefore,
			HeadAfter:  dispatchpost.HeadSHA(ctx(), ""),
		})

	handleLoopPerformanceRSI(stderr, *runID, performanceRSIInput, performanceRSIOutput, performanceRSIPrepErr)

	if *asJSON {
		if !writeLoopRunReport(stdout, stderr, *ledger, *loopID, *runID, map[string]any{
			"exit_code":   exitCode,
			"duration_ms": durationMS,
		}) {
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "loop run %s exit=%d ledger=%s\n", *runID, exitCode, *ledger)
	}
	return exitCode
}

// loopRunChildCtx carries the ledger identity + base evidence/metrics threaded through the
// START and END loop events that loopRunChild records around the child process.
type loopRunChildCtx struct {
	ledger    string
	loopID    string
	runID     string
	source    string
	principal string
	evidence  []loopmgr.EvidenceRef
	metrics   map[string]int64
	env       []string
}

// loopRunChild starts the child process, records its START and END loop events, and returns
// the child's exit code + wall-clock duration. fatal != 0 is a terminal code runLoopRun must
// return directly: 127 when the child fails to start (an END(failed) event is still recorded),
// or 1 when a ledger append fails in a way that must not be reported as success (the start
// event, or the end event on an otherwise-clean exit). A non-zero child exit with a failed
// end-append still returns fatal 0 so the real exit code reaches the caller's report.
func loopRunChild(stdout, stderr io.Writer, childArgv []string, rc loopRunChildCtx) (exitCode int, durationMS int64, fatal int) {
	cmd := loopNewCommand(childArgv, rc.env, stdout, stderr)
	started := time.Now()
	if err := cmd.Start(); err != nil {
		m := cloneLoopMetrics(rc.metrics)
		m["exit_code"] = 127
		_ = appendLoopRunEvent(rc.ledger, loopmgr.Event{
			LoopID:       rc.loopID,
			RunID:        rc.runID,
			Kind:         loopmgr.EventEnd,
			Source:       rc.source,
			Principal:    rc.principal,
			Status:       loopmgr.StatusFailed,
			Reason:       "START_FAILED",
			Summary:      err.Error(),
			EvidenceRefs: rc.evidence,
			Metrics:      m,
		})
		fmt.Fprintf(stderr, "fak loop run: start command: %v\n", err)
		return 0, 0, 127
	}
	mStart := cloneLoopMetrics(rc.metrics)
	mStart["pid"] = int64(cmd.PID())
	if err := appendLoopRunEvent(rc.ledger, loopmgr.Event{
		LoopID:       rc.loopID,
		RunID:        rc.runID,
		Kind:         loopmgr.EventStart,
		Source:       rc.source,
		Principal:    rc.principal,
		Status:       loopmgr.StatusRunning,
		Reason:       "STARTED",
		Summary:      "child process started",
		EvidenceRefs: rc.evidence,
		Metrics:      mStart,
	}); err != nil {
		_ = cmd.Kill()
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 0, 0, 1
	}

	waitErr := cmd.Wait()
	durationMS = time.Since(started).Milliseconds()
	status := loopmgr.StatusClaimedDone
	reason := "EXIT_0"
	if waitErr != nil {
		status = loopmgr.StatusFailed
		reason = "EXIT_NONZERO"
		if ee, ok := waitErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = 1
			reason = "WAIT_FAILED"
		}
	}
	mEnd := cloneLoopMetrics(rc.metrics)
	mEnd["pid"] = int64(cmd.PID())
	mEnd["exit_code"] = int64(exitCode)
	mEnd["duration_ms"] = durationMS
	if err := appendLoopRunEvent(rc.ledger, loopmgr.Event{
		LoopID:       rc.loopID,
		RunID:        rc.runID,
		Kind:         loopmgr.EventEnd,
		Source:       rc.source,
		Principal:    rc.principal,
		Status:       status,
		Reason:       reason,
		Summary:      fmt.Sprintf("child exited with code %d", exitCode),
		EvidenceRefs: rc.evidence,
		Metrics:      mEnd,
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		if exitCode == 0 {
			return exitCode, durationMS, 1
		}
	}
	return exitCode, durationMS, 0
}

func runLoopStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	asJSON := fs.Bool("json", false, "emit the full JSON status")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	st, integ, err := loopmgr.SnapshotFilePartial(*ledger, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak loop status: %v\n", err)
		return 1
	}
	if integ.Broken {
		fmt.Fprintf(stderr, "fak loop status: ledger integrity break at line %d: %s (recovered %d event(s))\n",
			integ.AtLine, integ.Reason, integ.Recovered)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, st, "fak loop status")
	}
	renderLoopStatus(stdout, st)
	return 0
}

type loopRepairReport struct {
	Schema          string            `json:"schema"`
	LedgerPath      string            `json:"ledger_path"`
	ArchivePath     string            `json:"archive_path,omitempty"`
	Repaired        bool              `json:"repaired"`
	RecoveredEvents int               `json:"recovered_events"`
	ArchivedEvents  int               `json:"archived_events,omitempty"`
	Integrity       loopmgr.Integrity `json:"integrity"`
}

func runLoopRepair(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop repair", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	confirm := fs.Bool("confirm", false, "confirm rewriting the ledger to its recovered valid prefix and archiving the broken tail")
	asJSON := fs.Bool("json", false, "emit the repair report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop repair: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if !*confirm {
		fmt.Fprintln(stderr, "fak loop repair: refusing to mutate the audit ledger without --confirm")
		return 2
	}

	rep, err := repairLoopLedger(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop repair: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak loop repair")
	}
	if !rep.Repaired {
		fmt.Fprintf(stdout, "loop repair: ledger %s is already strict-clean (%d event(s))\n",
			rep.LedgerPath, rep.RecoveredEvents)
		return 0
	}
	fmt.Fprintf(stdout, "loop repair: repaired ledger=%s recovered=%d archived=%d archive=%s\n",
		rep.LedgerPath, rep.RecoveredEvents, rep.ArchivedEvents, rep.ArchivePath)
	return 0
}

func repairLoopLedger(path string) (loopRepairReport, error) {
	events, integ, err := loopmgr.LoadPrefix(path)
	if err != nil {
		return loopRepairReport{}, err
	}
	rep := loopRepairReport{
		Schema:          "fak.loop-repair.v1",
		LedgerPath:      path,
		RecoveredEvents: len(events),
		Integrity:       integ,
	}
	if !integ.Broken {
		return rep, nil
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return loopRepairReport{}, err
	}
	prefix, tail, archived, err := splitLoopLedgerAtRecovered(raw, len(events))
	if err != nil {
		return loopRepairReport{}, err
	}
	if archived == 0 {
		return loopRepairReport{}, fmt.Errorf("integrity break reported at line %d but no broken tail was found", integ.AtLine)
	}

	archive := loopRepairArchivePath(path, integ)
	if err := writeFileExclusive(archive, tail, 0o644); err != nil {
		return loopRepairReport{}, fmt.Errorf("archive broken tail: %w", err)
	}
	if err := os.WriteFile(path, prefix, 0o644); err != nil {
		return loopRepairReport{}, fmt.Errorf("rewrite recovered prefix: %w", err)
	}
	rep.ArchivePath = archive
	rep.Repaired = true
	rep.ArchivedEvents = archived
	return rep, nil
}

func splitLoopLedgerAtRecovered(raw []byte, recovered int) (prefix []byte, tail []byte, archivedEvents int, err error) {
	if recovered < 0 {
		return nil, nil, 0, fmt.Errorf("recovered event count %d is invalid", recovered)
	}
	seen := 0
	for _, chunk := range bytes.SplitAfter(raw, []byte("\n")) {
		if len(chunk) == 0 {
			continue
		}
		if seen < recovered {
			prefix = append(prefix, chunk...)
			if len(bytes.TrimSpace(chunk)) > 0 {
				seen++
			}
			continue
		}
		tail = append(tail, chunk...)
		if len(bytes.TrimSpace(chunk)) > 0 {
			archivedEvents++
		}
	}
	if seen != recovered {
		return nil, nil, 0, fmt.Errorf("ledger contains %d valid event line(s), want recovered prefix %d", seen, recovered)
	}
	return prefix, tail, archivedEvents, nil
}

func loopRepairArchivePath(path string, integ loopmgr.Integrity) string {
	n := integ.AtSeq
	if n == 0 {
		n = uint64(integ.AtLine)
	}
	return fmt.Sprintf("%s.broken-%d", path, n)
}

func writeFileExclusive(path string, data []byte, perm os.FileMode) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err != nil {
		return err
	}
	defer f.Close()
	n, err := f.Write(data)
	if err != nil {
		return err
	}
	if n != len(data) {
		return io.ErrShortWrite
	}
	return nil
}

func runLoopHealth(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop health", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	registryPath := fs.String("registry", defaultLoopRegistry(), "loop registry JSON path")
	asJSON := fs.Bool("json", false, "emit the loop-health report as JSON")
	check := fs.Bool("check", false, "exit 3 when any loop is dark")
	schedFrom := fs.String("sched-from", "", "corroborate DARK loops against a captured schedscan snapshot (JSON); enables the #4989 OS-scheduler rung off-Windows")
	osTasks := fs.Bool("os-tasks", false, "corroborate DARK loops against a live Windows Task Scheduler query (#4989 OS-scheduler rung)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop health: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	now := time.Now()
	st, integ, err := loopmgr.SnapshotFilePartial(*ledger, now)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop health: %v\n", err)
		return 1
	}
	if integ.Broken {
		fmt.Fprintf(stderr, "fak loop health: ledger integrity break at line %d: %s (recovered %d event(s))\n",
			integ.AtLine, integ.Reason, integ.Recovered)
	}
	reg, err := loopmgr.LoadRegistry(*registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop health: %v\n", err)
		return 2
	}
	osWitness := loopOSWitnesses(loopOSTaskRows(stderr, *schedFrom, *osTasks), loopOSTaskMap())
	rep := loopmgr.FoldHealthWithOS(st, reg, now, loopmgr.HealthThresholds{}, osWitness)
	attachLearningDocsDebt(&rep)
	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak loop health: encode json: %v\n", err)
			return 1
		}
	} else {
		renderLoopHealth(stdout, rep, *ledger, *registryPath)
	}
	if *check && rep.Rollup.Dark > 0 {
		return 3
	}
	return 0
}

var loopLearningDebt = learningDocsDebtFromScorecard

func attachLearningDocsDebt(rep *loopmgr.HealthReport) {
	idx := -1
	for i := range rep.Rows {
		if rep.Rows[i].LoopID == "learning-docs-freshness" {
			idx = i
			break
		}
	}
	if idx < 0 {
		return
	}
	debt, ok := loopLearningDebt(repoRoot())
	if !ok {
		return
	}
	v := debt
	rep.Rows[idx].LearningDebt = &v
}

func learningDocsDebtFromScorecard(root string) (int64, bool) {
	py, ok := loopPython()
	if !ok {
		return 0, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, py, filepath.Join(root, "tools", "learning_scorecard.py"), "--json")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil && len(bytes.TrimSpace(out)) == 0 {
		return 0, false
	}
	return learningDocsDebtFromJSON(out)
}

func learningDocsDebtFromJSON(out []byte) (int64, bool) {
	var payload struct {
		Corpus struct {
			LearningDebt int64 `json:"learning_debt"`
		} `json:"corpus"`
	}
	if err := json.Unmarshal(out, &payload); err != nil {
		return 0, false
	}
	return payload.Corpus.LearningDebt, true
}

func loopPython() (string, bool) {
	if v := strings.TrimSpace(os.Getenv("FAK_PYTHON")); v != "" {
		if p, err := exec.LookPath(v); err == nil {
			return p, true
		}
	}
	for _, name := range []string{"python3", "python"} {
		if p, err := exec.LookPath(name); err == nil {
			return p, true
		}
	}
	return "", false
}

// runLoopRollup is the cross-node read-only fold (#769, Pillar 7): it ingests N
// nodes' loop ledgers and renders ONE fleet-wide "how often did every loop run"
// view — per-loop run counts, cadence, and last-run, the per-node columns reusing
// the aligned `fak ps` table idiom. It only reads the journals: it appends no
// event and issues no control verb, so adding a node's journal changes only this
// rollup, never any node's behavior. This is aggregation, explicitly NOT consensus
// (epic §5) — it has no write path that could influence another node's admission.
//
//	fak loop rollup --ledger node-a.jsonl --ledger node-b.jsonl   explicit per-node ledgers
//	fak loop rollup --ledger mac=/path/loops.jsonl                NODE=PATH labels the node
//	fak loop rollup --dir .fleet-journals [--glob '*.jsonl']      every match is one node
//	fak loop rollup ... --json                                    machine-readable fold
func runLoopRollup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop rollup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var ledgers loopKVList
	fs.Var(&ledgers, "ledger", "repeatable node ledger: PATH or NODE=PATH (node id defaults to the file basename)")
	dir := fs.String("dir", "", "directory of per-node ledgers (each file matching --glob is one node)")
	glob := fs.String("glob", "*.jsonl", "filename glob used with --dir")
	asJSON := fs.Bool("json", false, "emit the fleet rollup as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop rollup: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)

	nodes, err := loopRollupNodes(ledgers, *dir, *glob)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop rollup: %v\n", err)
		return 2
	}
	if len(nodes) == 0 {
		fmt.Fprintln(stderr, "fak loop rollup: no node ledgers given (--ledger PATH ... or --dir DIR)")
		return 2
	}

	rep := foldLoopRollup(nodes, time.Now())
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak loop rollup")
	}
	renderLoopRollup(stdout, rep)
	return 0
}

// runLoopAdmit applies the tunable loop-admission policy to the folded ledger
// and reports an admit/refuse verdict per loop. This is the governor surface
// that makes the always-on loop tunable: a scheduler line gates work on
// `fak loop admit --loop ID` (exit 0 admit, exit 3 refused), and the operator
// retunes the policy file — pause, cadence floor, refusal-storm backoff,
// witness-collapse hold — without re-registering the OS task. It only reads:
// it appends no event, so a refusal here is not itself a recorded refusal.
func runLoopAdmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop admit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	policyPath := fs.String("policy", defaultLoopPolicy(), "loop admission policy JSON path")
	loopID := fs.String("loop", "", "evaluate one loop id (default: every loop in the ledger)")
	asJSON := fs.Bool("json", false, "emit the decisions as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop admit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	// A scheduler gates a background loop on this verb; with no operator policy it
	// inherits the embedded sane default (cadence floor + refusal-storm cap) rather
	// than admit-always, so an unconfigured always-on loop is still braked.
	policies, err := loopmgr.LoadPoliciesOrDefault(*policyPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop admit: %v\n", err)
		return 2
	}
	now := time.Now()
	st, err := loopmgr.SnapshotFile(*ledger, now)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop admit: %v\n", err)
		return 1
	}

	var decisions []loopmgr.Decision
	if id := strings.TrimSpace(*loopID); id != "" {
		// A named loop the ledger has never seen still gets a verdict: an empty
		// snapshot under its policy, so an operator can pre-pause a loop that has
		// not fired yet, and a first-ever fire is evaluated rather than skipped.
		decisions = []loopmgr.Decision{loopmgr.Admit(loopSnapshotForID(st, id), policies.PolicyFor(id), now)}
	} else {
		decisions = loopmgr.AdmitAll(st, policies, now)
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, map[string]any{
			"schema":      "fak.loop-admit.v1",
			"ledger_path": *ledger,
			"policy_path": *policyPath,
			"decisions":   decisions,
		}); err != nil {
			fmt.Fprintf(stderr, "fak loop admit: encode json: %v\n", err)
			return 1
		}
	} else {
		for _, d := range decisions {
			verdict := "ADMIT"
			if !d.Admit {
				verdict = "REFUSE"
			}
			fmt.Fprintf(stdout, "%-6s %-30s %-18s %s\n", verdict, d.LoopID, d.Reason, d.Summary)
		}
		if len(decisions) == 0 {
			fmt.Fprintf(stdout, "no loops to admit (ledger %s)\n", *ledger)
		}
	}

	// Exit 3 when any evaluated loop is refused, so a scheduler can gate on it:
	//   fak loop admit --loop ID && python tick.py ...
	for _, d := range decisions {
		if !d.Admit {
			return 3
		}
	}
	return 0
}

// loopSnapshotForID returns the folded snapshot for a loop id, or an empty
// snapshot bearing just that id when the ledger has never seen it — so a policy
// can still be evaluated against a not-yet-fired loop.
func loopSnapshotForID(st loopmgr.Status, id string) loopmgr.LoopSnapshot {
	for _, l := range st.Loops {
		if l.LoopID == id {
			return l
		}
	}
	return loopmgr.LoopSnapshot{LoopID: id}
}

// runLoopPolicy dispatches the read-only governor-policy verbs. Today it carries
// only the propose-only self-tuning readout (#3976); every verb under it is strictly
// read-only — the operator lands any policy edit, this surface never writes it.
func runLoopPolicy(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak loop policy: expected a subcommand (propose)")
		return 2
	}
	switch argv[0] {
	case "propose":
		return runLoopPolicyPropose(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, "fak loop policy propose [--ledger FILE] [--policy FILE] [--json]")
		return 0
	default:
		fmt.Fprintf(stderr, "fak loop policy: unknown subcommand %q\n", argv[0])
		return 2
	}
}

// runLoopPolicyPropose prints the propose-only governor-policy fold: bounded,
// human-gated knob nudges for loops whose ledger signal has tripped a governor gate
// (a refusal-storming loop -> a bounded cadence-floor raise; a witness-collapsed loop
// -> a pause), each with a one-line rationale. It is strictly READ-ONLY: it folds the
// ledger under the current policy and prints proposals; it NEVER writes the policy
// file — the operator lands the edit (the human gate is the edit itself), and
// `fak loop admit` re-reads the policy per tick so a landed edit needs no reload.
// Exits 0 always: an empty proposal set is a healthy fleet, not an error.
func runLoopPolicyPropose(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop policy propose", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	policyPath := fs.String("policy", defaultLoopPolicy(), "loop admission policy JSON path")
	asJSON := fs.Bool("json", false, "emit the proposals as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop policy propose: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	// Fold against the SAME effective policy the governor admits under, so a proposal
	// triggers on exactly the gate a live `fak loop admit` would refuse on.
	policies, err := loopmgr.LoadPoliciesOrDefault(*policyPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop policy propose: %v\n", err)
		return 2
	}
	st, err := loopmgr.SnapshotFile(*ledger, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak loop policy propose: %v\n", err)
		return 1
	}

	proposals := loopmgr.ProposePolicies(st, policies, loopmgr.DefaultProposeConfig())

	if *asJSON {
		if proposals == nil {
			proposals = []loopmgr.PolicyProposal{}
		}
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":      "fak.loop-policy-propose.v1",
			"ledger_path": *ledger,
			"policy_path": *policyPath,
			"proposals":   proposals,
		}, "fak loop policy propose")
	}

	if len(proposals) == 0 {
		fmt.Fprintf(stdout, "no policy proposals: every loop is within its governor gates (ledger %s)\n", *ledger)
		return 0
	}
	for _, p := range proposals {
		fmt.Fprintf(stdout, "%-30s %-20s %-16s %s->%s  %s\n", p.LoopID, p.Field, p.Reason, p.Before, p.After, p.Rationale)
	}
	fmt.Fprintf(stdout, "\npropose-only: land an edit in %s yourself; nothing here writes it.\n", *policyPath)
	return 0
}

func defaultLoopLedger() string {
	if v := os.Getenv("FAK_LOOP_LEDGER"); v != "" {
		return v
	}
	return filepath.Join(".fak", "loops.jsonl")
}

func defaultLoopPolicy() string {
	if v := os.Getenv("FAK_LOOP_POLICY"); v != "" {
		return v
	}
	return filepath.Join(".fak", "loop-policy.json")
}

func defaultLoopRegistry() string {
	if v := os.Getenv("FAK_LOOP_REGISTRY"); v != "" {
		return v
	}
	return filepath.Join("tools", "loop-registry.json")
}

func appendLoopRunEvent(ledger string, ev loopmgr.Event) error {
	_, err := loopmgr.Append(ledger, ev)
	return err
}

func cloneLoopMetrics(in map[string]int64) map[string]int64 {
	out := make(map[string]int64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func defaultLoopRunID(loopID string) string {
	replacer := strings.NewReplacer("/", "-", "\\", "-", ":", "-", " ", "-")
	name := strings.Trim(replacer.Replace(loopID), "-")
	if name == "" {
		name = "loop"
	}
	return fmt.Sprintf("%s-%s-%d", name, time.Now().UTC().Format("20060102T150405Z"), os.Getpid())
}

func loopGuardArgv(fakBin string, cmdArgs []string) []string {
	out := []string{fakBin, "guard", "--"}
	out = append(out, cmdArgs...)
	return out
}

func loopContainmentViolations(cmdArgs []string) []repoguard.Violation {
	command := loopRepoguardCommand(cmdArgs)
	if strings.TrimSpace(command) == "" {
		return nil
	}
	cwd, _ := os.Getwd()
	workspaceRoot := repoguard.FindRepoRoot(cwd)
	return repoguard.ClassifyCommand(command, workspaceRoot, repoguard.SafeRootsForWorkspace(workspaceRoot))
}

func loopRepoguardCommand(cmdArgs []string) string {
	if len(cmdArgs) == 0 {
		return ""
	}
	if command, ok := loopShellCCommand(cmdArgs); ok {
		return command
	}
	parts := make([]string, 0, len(cmdArgs))
	for _, arg := range cmdArgs {
		parts = append(parts, loopShellQuote(arg))
	}
	return strings.Join(parts, " ")
}

func loopShellCCommand(cmdArgs []string) (string, bool) {
	if len(cmdArgs) < 3 {
		return "", false
	}
	base := strings.ToLower(strings.TrimSuffix(filepath.Base(cmdArgs[0]), ".exe"))
	switch base {
	case "bash", "sh", "zsh", "dash", "ksh":
	default:
		return "", false
	}
	for i := 1; i < len(cmdArgs)-1; i++ {
		arg := cmdArgs[i]
		if arg == "--" {
			return "", false
		}
		if strings.HasPrefix(arg, "--") {
			continue
		}
		if arg == "-c" || (strings.HasPrefix(arg, "-") && strings.Contains(arg[1:], "c")) {
			return cmdArgs[i+1], true
		}
	}
	return "", false
}

func loopShellQuote(arg string) string {
	if arg == "" {
		return "''"
	}
	if strings.IndexFunc(arg, func(r rune) bool {
		return r <= ' ' || strings.ContainsRune(`'"$`+"\\"+`;|&<>(){}[]*?~!`, r)
	}) < 0 {
		return arg
	}
	return "'" + strings.ReplaceAll(arg, "'", "'\\''") + "'"
}

type loopKVList []string

func (l *loopKVList) String() string {
	if l == nil {
		return ""
	}
	return strings.Join(*l, ",")
}

func (l *loopKVList) Set(v string) error {
	*l = append(*l, v)
	return nil
}

func parseLoopEvidence(items []string) []loopmgr.EvidenceRef {
	out := make([]loopmgr.EvidenceRef, 0, len(items))
	for _, item := range items {
		kind, ref, ok := strings.Cut(item, "=")
		if !ok {
			continue
		}
		kind = strings.TrimSpace(kind)
		ref = strings.TrimSpace(ref)
		if kind == "" || ref == "" {
			continue
		}
		out = append(out, loopmgr.EvidenceRef{Kind: kind, Ref: ref})
	}
	return out
}

func renderLoopStatus(w io.Writer, st loopmgr.Status) {
	if len(st.Loops) == 0 {
		fmt.Fprintf(w, "no loops found (ledger %s)\n", st.LedgerPath)
		return
	}
	fmt.Fprintf(w, "loop ledger=%s loops=%d\n", st.LedgerPath, len(st.Loops))
	for _, loop := range st.Loops {
		state := loop.State
		if state == "" {
			state = "-"
		}
		lastRun := "-"
		if loop.LastRun != nil {
			lastRun = string(loop.LastRun.Status)
			if loop.LastRun.RunID != "" {
				lastRun = loop.LastRun.RunID + ":" + lastRun
			}
		}
		fmt.Fprintf(w, "loop %-28s state=%-20s fires=%d admitted=%d refused=%d started=%d ended=%d witnessed=%d notify=%d last=%s last_run=%s\n",
			loop.LoopID,
			state,
			loop.Fires,
			loop.Admitted,
			loop.Refused,
			loop.Started,
			loop.Ended,
			loop.Witnessed,
			loop.Notifications,
			formatLoopTime(loop.LastEventUnixNano),
			lastRun,
		)
	}
}

func renderLoopHealth(w io.Writer, rep loopmgr.HealthReport, ledger, registry string) {
	if len(rep.Rows) == 0 {
		fmt.Fprintf(w, "no loops found (ledger %s registry %s)\n", ledger, registry)
		return
	}
	fmt.Fprintf(w, "fak loop health: loops=%d live=%d stale=%d dark=%d unknown=%d registered=%d ledgered=%d",
		rep.Rollup.Loops, rep.Rollup.Live, rep.Rollup.Stale, rep.Rollup.Dark,
		rep.Rollup.Unknown, rep.Rollup.Registered, rep.Rollup.Ledgered)
	// The #4989 OS-scheduler tally is a SUBSET of dark, printed only when it is
	// non-zero: a bare "os-live-ledger-dark=0" would be noise, and the fail-closed
	// tests read its absence as proof no loop was fabricated live.
	if rep.Rollup.OSFiredNoLedgerRow > 0 {
		fmt.Fprintf(w, " os-live-ledger-dark=%d", rep.Rollup.OSFiredNoLedgerRow)
	}
	fmt.Fprint(w, "\n")
	// #6497: the liveness line above answers "is it ticking"; this one answers "is it
	// working". A loop can be perfectly live on cadence and have failed every single
	// recorded run, which is exactly how the quarantined scout/logvault loops looked.
	fmt.Fprintf(w, "utility: runs=%d effects=%d no-fuel=%d unattributed=%d failed=%d alerting=%d never-succeeded=%d cost=%s\n",
		rep.Rollup.Runs, rep.Rollup.Effects, rep.Rollup.NoFuel, rep.Rollup.Unattributed,
		rep.Rollup.Failed, rep.Rollup.FailureAlert, rep.Rollup.NeverSucceeded,
		loopHealthCost(rep.Rollup.CostMilliUSD, rep.Rollup.CostedRuns))
	fmt.Fprint(w, "\n")
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "LOOP\tSTATE\tLAST\tAGE\tCADENCE\tRUNS\tFAILED\tALERT\tWITNESSED\tKEEP\tDEBT")
	for _, row := range rep.Rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%d\t%s\t%s\n",
			row.LoopID,
			loopHealthState(row),
			formatLoopTime(row.LastTickUnixNano),
			loopHealthAge(row),
			humanCadence(float64(row.CadenceSeconds)),
			row.Runs,
			row.Failed,
			loopHealthAlert(row),
			row.Witnessed,
			loopHealthKeepRate(row.KeepRate),
			loopHealthDebt(row.LearningDebt),
		)
	}
	_ = tw.Flush()
}

// loopHealthAlert names the loudest #6497 alarm on a row. never-succeeded outranks a
// mere repeated failure: a loop with zero successful recorded runs has never been
// shown to work at all, which is a different conversation from one that broke today.
func loopHealthAlert(row loopmgr.HealthRow) string {
	switch {
	case row.NeverSucceeded:
		return fmt.Sprintf("never-succeeded(%d)", row.ConsecutiveFailures)
	case row.FailureAlert:
		return fmt.Sprintf("repeated-failure(%d)", row.ConsecutiveFailures)
	}
	return "-"
}

// loopHealthCost renders the reported cost, keeping "nobody measured it" distinct
// from "it was free" — a zero over zero runs is the former.
func loopHealthCost(milliUSD int64, runs int) string {
	if runs == 0 {
		return "unmeasured"
	}
	return fmt.Sprintf("$%.3f/%drun", float64(milliUSD)/1000, runs)
}

func loopHealthState(row loopmgr.HealthRow) string {
	if row.Dark {
		// #4989: a ledger-dark loop whose mapped OS task fired within cadence is alive
		// at the OS layer, just not writing a ledger row — say so, don't call it dark-loop.
		if row.OSFiredNoLedgerRow {
			return "os-live-ledger-dark"
		}
		return "dark-loop"
	}
	return string(row.State)
}

func loopHealthAge(row loopmgr.HealthRow) string {
	if row.LastTickUnixNano == 0 {
		return "-"
	}
	return humanCadence(float64(row.AgeSeconds))
}

func loopHealthKeepRate(rate float64) string {
	if rate < 0 {
		return "-"
	}
	return fmt.Sprintf("%.3f", rate)
}

func loopHealthDebt(debt *int64) string {
	if debt == nil {
		return "-"
	}
	return strconv.FormatInt(*debt, 10)
}

func formatLoopTime(ts int64) string {
	if ts == 0 {
		return "-"
	}
	return time.Unix(0, ts).UTC().Format(time.RFC3339)
}

// dispatchAPIBase overrides the Slack API base URL for the run-card wire —
// the test seam that points the drain at an httptest fake. "" = live Slack.
var dispatchAPIBase = ""

// dispatchCardWire builds the run-card drain transport on the dispatch token
// (falling back to the shared scoreboard token, like the legacy post).
func dispatchCardWire(tokenOverride string) (*slackwire.Client, error) {
	tok := tokenOverride
	if tok == "" {
		tok = dispatchpost.ResolveToken()
	}
	if tok == "" {
		return nil, fmt.Errorf("no dispatch bot token: set FAK_DISPATCH_TOKEN or FAK_SCOREBOARD_TOKEN")
	}
	var opts []slackwire.Option
	if dispatchAPIBase != "" {
		opts = append(opts, slackwire.WithAPIBase(dispatchAPIBase))
	}
	return slackwire.New(tok, opts...), nil
}

// drainDispatchCard runs one best-effort outbox drain so the card's enqueued
// intents reach Slack now. A failure leaves the rows durably spooled for the
// next `fak slack outbox drain` — reported, never fatal.
func drainDispatchCard(stderr io.Writer, card *dispatchpost.RunCard, tokenOverride string) {
	wire, err := dispatchCardWire(tokenOverride)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop run: card drain skipped (rows stay spooled): %v\n", err)
		return
	}
	if _, err := card.Outbox.Drain(ctx(), wire, stdDrainOpts()); err != nil && err != slackoutbox.ErrDrainBusy {
		fmt.Fprintf(stderr, "fak loop run: card drain: %v\n", err)
	}
}

func dispatchSlackRequested(notify bool, channelOverride string) bool {
	return notify || strings.TrimSpace(channelOverride) != ""
}

// openDispatchRunCardIfRequested keeps ambient machine-wide channel configuration from
// arming per-run traffic. Explicit --notify-slack may resolve that ambient destination;
// explicit --dispatch-channel is also sufficient intent.
func openDispatchRunCardIfRequested(stderr io.Writer, notify bool, channelOverride, tokenOverride string, res dispatchpost.Result) *dispatchpost.RunCard {
	if !dispatchSlackRequested(notify, channelOverride) {
		return nil
	}
	return openDispatchRunCard(stderr, channelOverride, tokenOverride, res)
}

// openDispatchRunCard arms the live run card (#2263) after the caller has established
// explicit notification intent: it opens (or, after a restart, resumes) the run's card
// over the durable outbox spool, enqueues the start post, and drains once so the channel
// sees the run begin. nil means unarmed (no channel) or unavailable (reported); the
// dispatch itself is never affected.
func openDispatchRunCard(stderr io.Writer, channelOverride, tokenOverride string, res dispatchpost.Result) *dispatchpost.RunCard {
	ch := channelOverride
	if ch == "" {
		ch = dispatchpost.ResolveChannel()
	}
	if ch == "" {
		return nil
	}
	card, err := dispatchpost.OpenRunCard(resolveOutboxDir(), res.LoopID, res.RunID)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop run: run card unavailable: %v\n", err)
		return nil
	}
	if err := card.Start(ch, res); err != nil {
		fmt.Fprintf(stderr, "fak loop run: run card start: %v\n", err)
		return nil
	}
	drainDispatchCard(stderr, card, tokenOverride)
	return card
}

// postDispatchResult reports a finished dispatch to the dispatch Slack channel.
// With an armed run card (#2263) it FINALIZES the card: the start message is
// edited in place into the witnessed verdict (commit SHA + ship-stamp grepped
// from the HEAD delta, verify source, exit code) and the full result body rides
// in the card's thread — one channel line per run. Without a card (or when the
// card path fails) it falls back to the legacy terminal post.
//
// It is gated and best-effort. The post is attempted only when --notify-slack is set or
// an explicit --dispatch-channel was supplied; an ambient channel alone is a silent no-op.
// Any error (no channel under --notify-slack, no token, a Slack API failure) is reported
// to stderr and NEVER changes the run's exit code — the dispatch result stands on its own.
func postDispatchResult(stderr io.Writer, notify bool, channelOverride, tokenOverride string, card *dispatchpost.RunCard, res dispatchpost.Result) {
	if !dispatchSlackRequested(notify, channelOverride) {
		return
	}
	ch := channelOverride
	if ch == "" {
		ch = dispatchpost.ResolveChannel()
	}
	if ch == "" {
		// No channel: skip silently unless the operator explicitly asked to notify,
		// in which case surface the misconfiguration (but still don't fail the run).
		if notify {
			fmt.Fprintln(stderr, "fak loop run: --notify-slack set but no dispatch channel: set FAK_DISPATCH_CHANNEL or pass --dispatch-channel")
		}
		return
	}

	// Fill the witness: the commits the dispatch landed between the captured HEADs.
	res.Commits = dispatchpost.CommitsBetween(ctx(), "", res.HeadBefore, res.HeadAfter)
	if res.Source == "" {
		res.Source = defaultSource()
	}

	if card != nil {
		err := card.Finalize(res)
		if errors.Is(err, slackoutbox.ErrCardNotPosted) {
			// The start post never drained (e.g. Slack was down at run start):
			// deliver it now, then finalize against the resolved ts.
			drainDispatchCard(stderr, card, tokenOverride)
			err = card.Finalize(res)
		}
		if err == nil {
			drainDispatchCard(stderr, card, tokenOverride)
			fmt.Fprintf(stderr, "fak loop run: dispatch run card finalized in %s\n", ch)
			return
		}
		fmt.Fprintf(stderr, "fak loop run: run card finalize failed (%v); falling back to direct post\n", err)
	}

	tok := tokenOverride
	if tok == "" {
		tok = dispatchpost.ResolveToken()
	}
	client, err := scoreboard.NewClient(tok)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop run: dispatch post skipped: %v\n", err)
		return
	}
	if _, err := client.Post(ctx(), ch, res.Text(), res.Blocks()); err != nil {
		fmt.Fprintf(stderr, "fak loop run: dispatch post failed: %v\n", err)
		return
	}
	fmt.Fprintf(stderr, "fak loop run: dispatch result posted to %s\n", ch)
}
