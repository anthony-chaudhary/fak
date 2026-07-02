package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchpost"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/maputil"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/repoguard"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
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
	case "admit":
		return runLoopAdmit(stdout, stderr, argv[1:])
	case "region":
		return runLoopRegion(stdout, stderr, argv[1:])
	case "recover":
		return runLoopRecover(stdout, stderr, argv[1:])
	case "repair":
		return runLoopRepair(stdout, stderr, argv[1:])
	case "drive":
		return runLoopDrive(stdout, stderr, argv[1:])
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
	return c.cmd.Process.Kill()
}

var loopExecutable = os.Executable

var loopNewCommand = func(argv []string, stdout, stderr io.Writer) loopCommand {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return execLoopCommand{cmd: cmd}
}

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
	if err := fs.Parse(argv); err != nil {
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

func runLoopRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	loopID := fs.String("loop", "", "loop id")
	runID := fs.String("run", "", "run id")
	source := fs.String("source", "manual", "trigger source, such as cron|launchd|task-scheduler|manual")
	principal := fs.String("principal", "", "authenticated principal or producer id")
	asJSON := fs.Bool("json", false, "emit a JSON run report")
	notifySlack := fs.Bool("notify-slack", false, "post a witnessed dispatch-result card to the dispatch Slack channel when the run ends")
	dispatchChannel := fs.String("dispatch-channel", "", "override dispatch channel id (default: $FAK_DISPATCH_CHANNEL / .env.slack.local)")
	dispatchToken := fs.String("dispatch-token", "", "override dispatch bot token (default: $FAK_DISPATCH_TOKEN, then scoreboard token)")
	noGuard := fs.Bool("no-guard", false, "explicitly disable the default fak guard containment wrapper for this run (logged in the loop ledger)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	cmdArgs := fs.Args()
	if strings.TrimSpace(*loopID) == "" {
		fmt.Fprintln(stderr, "fak loop run: --loop is required")
		return 2
	}
	if len(cmdArgs) == 0 {
		fmt.Fprintln(stderr, "fak loop run: command is required after --")
		return 2
	}
	if *runID == "" {
		*runID = defaultLoopRunID(*loopID)
	}

	// Capture HEAD before the dispatch so the result card can WITNESS what it landed
	// (HeadBefore..HeadAfter), not trust the child's self-report. "" if not a git repo.
	headBefore := dispatchpost.HeadSHA(ctx(), "")

	guardEnabled := !*noGuard
	baseEvidence := []loopmgr.EvidenceRef{{Kind: "command", Ref: filepath.Base(cmdArgs[0])}}
	baseMetrics := map[string]int64{"argc": int64(len(cmdArgs))}
	if guardEnabled {
		baseEvidence = append(baseEvidence, loopmgr.EvidenceRef{Kind: "guard", Ref: "fak guard"})
		baseMetrics["guard_enabled"] = 1
	} else {
		baseMetrics["guard_enabled"] = 0
	}
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventFire,
		Source:       *source,
		Principal:    *principal,
		Summary:      "loop run requested",
		EvidenceRefs: baseEvidence,
		Metrics:      cloneLoopMetrics(baseMetrics),
	}); err != nil {
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
			if err := appendLoopRunEvent(*ledger, loopmgr.Event{
				LoopID:       *loopID,
				RunID:        *runID,
				Kind:         loopmgr.EventAdmit,
				Source:       *source,
				Principal:    *principal,
				Status:       loopmgr.StatusRefused,
				Reason:       repoguard.Reason,
				Summary:      summary,
				EvidenceRefs: baseEvidence,
				Metrics:      m,
			}); err != nil {
				fmt.Fprintf(stderr, "fak loop run: %v\n", err)
				return 1
			}
			fmt.Fprintf(stderr, "fak loop run: containment refused command: %s\n", summary)
			if *asJSON {
				rep := map[string]any{
					"schema":      "fak.loop-run-report.v1",
					"ledger_path": *ledger,
					"loop_id":     *loopID,
					"run_id":      *runID,
					"status":      "refused",
					"reason":      repoguard.Reason,
					"exit_code":   3,
				}
				if err := writeIndentedJSON(stdout, rep); err != nil {
					fmt.Fprintf(stderr, "fak loop run: encode json: %v\n", err)
					return 1
				}
			}
			return 3
		}
		fakBin, err := loopExecutable()
		if err != nil {
			m := cloneLoopMetrics(baseMetrics)
			m["exit_code"] = 127
			_ = appendLoopRunEvent(*ledger, loopmgr.Event{
				LoopID:       *loopID,
				RunID:        *runID,
				Kind:         loopmgr.EventEnd,
				Source:       *source,
				Principal:    *principal,
				Status:       loopmgr.StatusFailed,
				Reason:       "GUARD_UNAVAILABLE",
				Summary:      err.Error(),
				EvidenceRefs: baseEvidence,
				Metrics:      m,
			})
			fmt.Fprintf(stderr, "fak loop run: resolve fak guard binary: %v\n", err)
			return 127
		}
		childArgv = loopGuardArgv(fakBin, cmdArgs)
	} else {
		admitReason = "GUARD_DISABLED"
		admitSummary = "--no-guard disabled fak guard containment"
		fmt.Fprintln(stderr, "fak loop run: WARNING --no-guard disables fak guard containment for this run")
	}
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventAdmit,
		Source:       *source,
		Principal:    *principal,
		Status:       loopmgr.StatusAdmitted,
		Reason:       admitReason,
		Summary:      admitSummary,
		EvidenceRefs: baseEvidence,
		Metrics:      cloneLoopMetrics(baseMetrics),
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 1
	}

	exitCode, durationMS, fatal := loopRunChild(stdout, stderr, childArgv, loopRunChildCtx{
		ledger:    *ledger,
		loopID:    *loopID,
		runID:     *runID,
		source:    *source,
		principal: *principal,
		evidence:  baseEvidence,
		metrics:   baseMetrics,
	})
	if fatal != 0 {
		return fatal
	}

	// Post a witnessed dispatch-result card to Slack so a slow background dispatch
	// reports its outcome without anyone tailing the ledger. Gated and best-effort:
	// a resolved channel (or --notify-slack) arms it, and any failure is reported to
	// stderr without changing the run's exit code — the dispatch's result must stand
	// on its own even if Slack is unreachable.
	postDispatchResult(stderr, *notifySlack, *dispatchChannel, *dispatchToken,
		dispatchpost.Result{
			LoopID:     *loopID,
			RunID:      *runID,
			ExitCode:   exitCode,
			DurationMS: durationMS,
			Command:    filepath.Base(cmdArgs[0]),
			HeadBefore: headBefore,
			HeadAfter:  dispatchpost.HeadSHA(ctx(), ""),
		})

	if *asJSON {
		rep := map[string]any{
			"schema":      "fak.loop-run-report.v1",
			"ledger_path": *ledger,
			"loop_id":     *loopID,
			"run_id":      *runID,
			"exit_code":   exitCode,
			"duration_ms": durationMS,
		}
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak loop run: encode json: %v\n", err)
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
}

// loopRunChild starts the child process, records its START and END loop events, and returns
// the child's exit code + wall-clock duration. fatal != 0 is a terminal code runLoopRun must
// return directly: 127 when the child fails to start (an END(failed) event is still recorded),
// or 1 when a ledger append fails in a way that must not be reported as success (the start
// event, or the end event on an otherwise-clean exit). A non-zero child exit with a failed
// end-append still returns fatal 0 so the real exit code reaches the caller's report.
func loopRunChild(stdout, stderr io.Writer, childArgv []string, rc loopRunChildCtx) (exitCode int, durationMS int64, fatal int) {
	cmd := loopNewCommand(childArgv, stdout, stderr)
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
	if err := fs.Parse(argv); err != nil {
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
	if err := fs.Parse(argv); err != nil {
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
	if err := fs.Parse(argv); err != nil {
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
	rep := loopmgr.FoldHealth(st, reg, now, loopmgr.HealthThresholds{})
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
	if err := fs.Parse(argv); err != nil {
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

// loopRollupNode names one node's ledger: an id (for the per-node attribution
// column) and the path to its loop JSONL ledger.
type loopRollupNode struct {
	ID   string
	Path string
}

// loopRollupNodes builds the node list from the repeatable --ledger flags and the
// optional --dir scan, de-duplicating by path. A --ledger value of NODE=PATH
// labels the node explicitly; a bare PATH derives the id from the file basename.
func loopRollupNodes(ledgers []string, dir, glob string) ([]loopRollupNode, error) {
	var nodes []loopRollupNode
	seen := map[string]bool{}
	add := func(id, path string) {
		path = strings.TrimSpace(path)
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		if id == "" {
			id = nodeIDFromPath(path)
		}
		nodes = append(nodes, loopRollupNode{ID: id, Path: path})
	}
	for _, item := range ledgers {
		if k, v, ok := strings.Cut(item, "="); ok && strings.TrimSpace(k) != "" && strings.TrimSpace(v) != "" {
			add(strings.TrimSpace(k), v)
			continue
		}
		add("", item)
	}
	if strings.TrimSpace(dir) != "" {
		matches, err := filepath.Glob(filepath.Join(dir, glob))
		if err != nil {
			return nil, fmt.Errorf("glob %q: %w", glob, err)
		}
		sort.Strings(matches)
		for _, m := range matches {
			add("", m)
		}
	}
	return nodes, nil
}

// nodeIDFromPath derives a node id from a ledger path: the file basename without
// its extension (so node-a.jsonl -> node-a), falling back to the raw path.
func nodeIDFromPath(path string) string {
	base := filepath.Base(path)
	if ext := filepath.Ext(base); ext != "" {
		base = strings.TrimSuffix(base, ext)
	}
	if base == "" || base == "." {
		return path
	}
	return base
}

// loopRollupReport (schema fak.loop-rollup.v1) is the machine-readable fleet fold:
// the node ids that contributed, any node journal that could not be read, and one
// aggregated row per loop id seen across all nodes.
type loopRollupReport struct {
	Schema     string           `json:"schema"`
	TSUnixNano int64            `json:"ts_unix_nano"`
	Nodes      []string         `json:"nodes"`
	Skipped    []loopRollupSkip `json:"skipped,omitempty"`
	Loops      []loopRollupRow  `json:"loops"`
}

// loopRollupSkip records a node ledger that could not be folded (a corrupt or
// forked journal). A read-only fleet view must not go dark on one bad node, so
// the fold skips it and surfaces why rather than aborting the whole rollup.
type loopRollupSkip struct {
	Node  string `json:"node"`
	Path  string `json:"path"`
	Error string `json:"error"`
}

// loopRollupRow is one loop's fleet-wide fold: how many nodes ran it, the summed
// run/admit/start/end/witness counts, the merged-timeline cadence, and the most
// recent event across nodes. Runs is the fire count — the canonical "the loop was
// triggered" marker the operator's "how often did it run" question asks about.
type loopRollupRow struct {
	LoopID            string   `json:"loop_id"`
	Nodes             int      `json:"nodes"`
	NodeIDs           []string `json:"node_ids,omitempty"`
	Runs              uint64   `json:"runs"`
	Admitted          uint64   `json:"admitted"`
	Refused           uint64   `json:"refused"`
	Started           uint64   `json:"started"`
	Ended             uint64   `json:"ended"`
	Witnessed         uint64   `json:"witnessed"`
	CadenceSeconds    float64  `json:"cadence_seconds,omitempty"`
	LastEventUnixNano int64    `json:"last_event_unix_nano,omitempty"`
}

// loopRollupAcc accumulates one loop's cross-node fold while the nodes are walked.
type loopRollupAcc struct {
	runs, admitted, refused, started, ended, witnessed uint64
	last                                               int64
	fireTS                                             []int64
	nodes                                              map[string]bool
}

// foldLoopRollup is the pure cross-node aggregation: load each node's ledger,
// summarize it with the same fold `fak loop status` uses, and sum the per-loop
// counts fleet-wide. Cadence is the mean interval of every loop's fire events
// merged across nodes (the fleet's "how often"); last-run is the latest event
// across nodes. An unreadable node ledger is skipped (recorded in Skipped), never
// fatal. Read-only: it opens journals and writes nothing.
func foldLoopRollup(nodes []loopRollupNode, now time.Time) loopRollupReport {
	rep := loopRollupReport{
		Schema:     "fak.loop-rollup.v1",
		TSUnixNano: now.UTC().UnixNano(),
	}
	agg := map[string]*loopRollupAcc{}
	get := func(id string) *loopRollupAcc {
		a := agg[id]
		if a == nil {
			a = &loopRollupAcc{nodes: map[string]bool{}}
			agg[id] = a
		}
		return a
	}
	for _, n := range nodes {
		rep.Nodes = append(rep.Nodes, n.ID)
		events, err := loopmgr.Load(n.Path)
		if err != nil {
			rep.Skipped = append(rep.Skipped, loopRollupSkip{Node: n.ID, Path: n.Path, Error: err.Error()})
			continue
		}
		for _, ls := range loopmgr.Summarize(events, now).Loops {
			a := get(ls.LoopID)
			a.runs += ls.Fires
			a.admitted += ls.Admitted
			a.refused += ls.Refused
			a.started += ls.Started
			a.ended += ls.Ended
			a.witnessed += ls.Witnessed
			if ls.LastEventUnixNano > a.last {
				a.last = ls.LastEventUnixNano
			}
			a.nodes[n.ID] = true
		}
		for _, ev := range events {
			if ev.Kind == loopmgr.EventFire {
				get(ev.LoopID).fireTS = append(get(ev.LoopID).fireTS, ev.TSUnixNano)
			}
		}
	}

	ids := maputil.SortedKeys(agg)
	for _, id := range ids {
		a := agg[id]
		nodeIDs := make([]string, 0, len(a.nodes))
		for nid := range a.nodes {
			nodeIDs = append(nodeIDs, nid)
		}
		sort.Strings(nodeIDs)
		rep.Loops = append(rep.Loops, loopRollupRow{
			LoopID:            id,
			Nodes:             len(a.nodes),
			NodeIDs:           nodeIDs,
			Runs:              a.runs,
			Admitted:          a.admitted,
			Refused:           a.refused,
			Started:           a.started,
			Ended:             a.ended,
			Witnessed:         a.witnessed,
			CadenceSeconds:    cadenceSeconds(a.fireTS),
			LastEventUnixNano: a.last,
		})
	}
	return rep
}

// cadenceSeconds is the mean interval between runs: the span of the merged fire
// timestamps divided by the gaps between them. Fewer than two fires (or a
// zero-span burst) has no measurable cadence and returns 0.
func cadenceSeconds(ts []int64) float64 {
	if len(ts) < 2 {
		return 0
	}
	sorted := append([]int64(nil), ts...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	span := sorted[len(sorted)-1] - sorted[0]
	if span <= 0 {
		return 0
	}
	return float64(span) / float64(len(sorted)-1) / 1e9
}

// renderLoopRollup prints the fleet rollup as one aligned row per loop, reusing
// the `fak ps` tabwriter idiom for the per-run columns. RUNS is the fleet-wide
// fire count, CADENCE the mean interval between runs, LAST the most recent event.
func renderLoopRollup(w io.Writer, rep loopRollupReport) {
	if len(rep.Loops) == 0 {
		fmt.Fprintf(w, "no loops found across %d node(s)\n", len(rep.Nodes))
		for _, s := range rep.Skipped {
			fmt.Fprintf(w, "skipped node %s (%s): %s\n", s.Node, s.Path, s.Error)
		}
		return
	}
	fmt.Fprintf(w, "fak loop rollup — %d loop(s) across %d node(s)\n\n", len(rep.Loops), len(rep.Nodes))
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "LOOP\tNODES\tRUNS\tSTARTED\tENDED\tWITNESSED\tREFUSED\tCADENCE\tLAST")
	for _, l := range rep.Loops {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n",
			l.LoopID, l.Nodes, l.Runs, l.Started, l.Ended, l.Witnessed, l.Refused,
			humanCadence(l.CadenceSeconds), formatLoopTime(l.LastEventUnixNano))
	}
	_ = tw.Flush()
	for _, s := range rep.Skipped {
		fmt.Fprintf(w, "skipped node %s (%s): %s\n", s.Node, s.Path, s.Error)
	}
}

// humanCadence renders a mean-interval (seconds) in the dominant unit: "-" when
// there is no measurable cadence, else "45s" / "12.0m" / "1.1h" / "2.0d".
func humanCadence(sec float64) string {
	if sec <= 0 {
		return "-"
	}
	switch {
	case sec >= 86400:
		return fmt.Sprintf("%.1fd", sec/86400)
	case sec >= 3600:
		return fmt.Sprintf("%.1fh", sec/3600)
	case sec >= 60:
		return fmt.Sprintf("%.1fm", sec/60)
	case sec >= 10:
		return fmt.Sprintf("%.0fs", sec)
	default:
		// Sub-10s: keep a decimal so a real-but-tiny interval reads distinct from
		// "-" (no measurable cadence) instead of rounding down to a misleading "0s".
		return fmt.Sprintf("%.1fs", sec)
	}
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
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop admit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	policies, err := loopmgr.LoadPolicies(*policyPath)
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
	fmt.Fprintf(w, "fak loop health: loops=%d live=%d stale=%d dark=%d unknown=%d registered=%d ledgered=%d\n\n",
		rep.Rollup.Loops, rep.Rollup.Live, rep.Rollup.Stale, rep.Rollup.Dark,
		rep.Rollup.Unknown, rep.Rollup.Registered, rep.Rollup.Ledgered)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "LOOP\tSTATE\tLAST\tAGE\tCADENCE\tRUNS\tWITNESSED\tKEEP\tDEBT")
	for _, row := range rep.Rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\n",
			row.LoopID,
			loopHealthState(row),
			formatLoopTime(row.LastTickUnixNano),
			loopHealthAge(row),
			humanCadence(float64(row.CadenceSeconds)),
			row.Runs,
			row.Witnessed,
			loopHealthKeepRate(row.KeepRate),
			loopHealthDebt(row.LearningDebt),
		)
	}
	_ = tw.Flush()
}

func loopHealthState(row loopmgr.HealthRow) string {
	if row.Dark {
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

// postDispatchResult posts a witnessed dispatch-result card to the dispatch Slack
// channel when a `fak loop run` dispatch ends. It is the wire that turns ongoing
// background dispatch into a Slack-visible result: the run's outcome (exit code,
// duration) PLUS the git HEAD delta it actually landed (the witness) become one
// channel post, so a slow nightly/cron dispatch reports what it did without anyone
// tailing the ledger.
//
// It is gated and best-effort. The post is attempted when --notify-slack is set OR a
// dispatch channel resolves from the environment/.env.slack.local; otherwise it is a
// silent no-op so an unconfigured box runs the dispatch normally. Any error (no
// channel under --notify-slack, no token, a Slack API failure) is reported to stderr
// and NEVER changes the run's exit code — the dispatch result stands on its own.
func postDispatchResult(stderr io.Writer, notify bool, channelOverride, tokenOverride string, res dispatchpost.Result) {
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

func loopUsage(w io.Writer) {
	fmt.Fprint(w, `fak loop - durable long-running loop ledger

  fak loop append --loop ID --kind KIND [--ledger FILE] [--run ID]
                  [--source NAME] [--principal ID] [--status STATUS]
                  [--reason CODE] [--summary TEXT] [--evidence KIND=REF]
                  [--metric NAME=INT64] [--json]
  fak loop run --loop ID [--ledger FILE] [--source cron|launchd|task-scheduler] [--notify-slack] [--no-guard] -- CMD [ARG...]
  fak loop status [--ledger FILE] [--json]
  fak loop health [--ledger FILE] [--registry FILE] [--check] [--json]
  fak loop rollup [--ledger PATH|NODE=PATH ...] [--dir DIR] [--glob '*.jsonl'] [--json]
  fak loop admit [--loop ID] [--ledger FILE] [--policy FILE] [--json]
  fak loop region [--lane LANE] [--tree GLOB ...] [--actor ID] [--self LEASE-ID]
                  [--dir DIR] [--json]
  fak loop recover [--ledger FILE] [--stale-min N] [--now UNIX] [--all] [--json]
  fak loop repair [--ledger FILE] --confirm [--json]
  fak loop drive [--loop ID] [--goal GOAL.md] [--ledger FILE] [--policy FILE]
                  [--max-iters N] [--max-tokens N] [--deadline RFC3339|DUR]
                  [--review-model M] -- CMD [ARG...]
  fak loop drive --template [--loop ID]

Append records one scheduler/script/control event in the canonical hash-chained
ledger. Run wraps an OS scheduler command under fak guard by default and records
fire/admit/start/end around it; a direct out-of-tree write/delete is refused before
spawn with OUT_OF_TREE_WRITE, and --no-guard is an explicit logged opt-out.
Status folds that ledger into the current loop/run view using the recovered
valid prefix when the hash chain is forked or corrupt, warning on stderr without
rewriting the audit log. Health joins the ledger with the durable registry and
renders live/stale/dark-loop state plus current learning_debt for the
docs-freshness loop. Rollup folds MANY nodes' ledgers into one fleet-wide "how
often did every loop run" view — per-loop run counts, cadence, and last-run —
reusing the fak ps table format; it is a read-only aggregation that ingests
journals and writes nothing. Admit applies the tunable
admission policy (default .fak/loop-policy.json, FAK_LOOP_POLICY) to the fold and
prints admit/refuse per loop — exit 3 when any evaluated loop is refused, so a
scheduler line can gate work on it. Recover folds the ledger into the cross-run
RECOVERY worklist: the dispatched runs that started but were never finished
(orphaned) or never witnessed (unwitnessed) — the work to re-dispatch or re-verify.
Repair is the explicit operator mutation: it archives a broken ledger tail and
rewrites only the valid prefix; readers never invoke it automatically. The ledger
records events; admission, scheduler authority, and completion witnesses live in
producers.
Drive reads a GOAL.md goal-spec fresh before every turn, gates each turn through
the loop admission policy, appends fire/admit/start/end/witness events to this ledger,
and re-spawns CMD until the configured DOS witness reports witnessed_done or a
budget is spent. With --review-model it also exports FAK_REVIEW_* so fak commit
asks a scout reviewer to pass/refute the turn diff before committing; review
verdicts are recorded as loop-ledger evidence. A NOT_YET witness refusal is
appended under Scratch and exposed through FAK_GOAL_LAST_REFUSAL so the next
fresh-context turn can see it. A GOAL.md lane:/region: declaration (or --lane/
--tree) additionally holds a region lease on the shared lease fabric while the
drive runs, refusing COLLISION_RISK instead of racing a live peer.
Region is the surface-neutral admission question by itself: "may ACTOR act on
this lane/tree right now?" answered against the live lease fabric and the
dos.toml lane taxonomy (exit 0 admit / 3 refuse) — the check a manual session
or a super-loop enter path runs before touching a region. It decides only;
holding a lease stays with fak leaseref acquire.
`)
}
