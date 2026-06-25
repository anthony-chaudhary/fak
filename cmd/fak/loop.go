package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
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
	case "-h", "--help", "help":
		loopUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak loop: unknown subcommand %q\n", argv[0])
		loopUsage(stderr)
		return 2
	}
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
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(appended); err != nil {
			fmt.Fprintf(stderr, "fak loop append: encode json: %v\n", err)
			return 1
		}
		return 0
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

	baseEvidence := []loopmgr.EvidenceRef{{Kind: "command", Ref: filepath.Base(cmdArgs[0])}}
	baseMetrics := map[string]int64{"argc": int64(len(cmdArgs))}
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
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventAdmit,
		Source:       *source,
		Principal:    *principal,
		Status:       loopmgr.StatusAdmitted,
		Reason:       "WRAPPER_ADMITTED",
		Summary:      "loop wrapper admitted command",
		EvidenceRefs: baseEvidence,
		Metrics:      cloneLoopMetrics(baseMetrics),
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 1
	}

	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	started := time.Now()
	if err := cmd.Start(); err != nil {
		m := cloneLoopMetrics(baseMetrics)
		m["exit_code"] = 127
		_ = appendLoopRunEvent(*ledger, loopmgr.Event{
			LoopID:       *loopID,
			RunID:        *runID,
			Kind:         loopmgr.EventEnd,
			Source:       *source,
			Principal:    *principal,
			Status:       loopmgr.StatusFailed,
			Reason:       "START_FAILED",
			Summary:      err.Error(),
			EvidenceRefs: baseEvidence,
			Metrics:      m,
		})
		fmt.Fprintf(stderr, "fak loop run: start command: %v\n", err)
		return 127
	}
	mStart := cloneLoopMetrics(baseMetrics)
	mStart["pid"] = int64(cmd.Process.Pid)
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventStart,
		Source:       *source,
		Principal:    *principal,
		Status:       loopmgr.StatusRunning,
		Reason:       "STARTED",
		Summary:      "child process started",
		EvidenceRefs: baseEvidence,
		Metrics:      mStart,
	}); err != nil {
		_ = cmd.Process.Kill()
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 1
	}

	waitErr := cmd.Wait()
	durationMS := time.Since(started).Milliseconds()
	exitCode := 0
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
	mEnd := cloneLoopMetrics(baseMetrics)
	mEnd["pid"] = int64(cmd.Process.Pid)
	mEnd["exit_code"] = int64(exitCode)
	mEnd["duration_ms"] = durationMS
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventEnd,
		Source:       *source,
		Principal:    *principal,
		Status:       status,
		Reason:       reason,
		Summary:      fmt.Sprintf("child exited with code %d", exitCode),
		EvidenceRefs: baseEvidence,
		Metrics:      mEnd,
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		if exitCode == 0 {
			return 1
		}
	}
	if *asJSON {
		rep := map[string]any{
			"schema":      "fak.loop-run-report.v1",
			"ledger_path": *ledger,
			"loop_id":     *loopID,
			"run_id":      *runID,
			"exit_code":   exitCode,
			"duration_ms": durationMS,
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak loop run: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "loop run %s exit=%d ledger=%s\n", *runID, exitCode, *ledger)
	}
	return exitCode
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

	st, err := loopmgr.SnapshotFile(*ledger, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak loop status: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(st); err != nil {
			fmt.Fprintf(stderr, "fak loop status: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	renderLoopStatus(stdout, st)
	return 0
}

func defaultLoopLedger() string {
	if v := os.Getenv("FAK_LOOP_LEDGER"); v != "" {
		return v
	}
	return filepath.Join(".fak", "loops.jsonl")
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

func formatLoopTime(ts int64) string {
	if ts == 0 {
		return "-"
	}
	return time.Unix(0, ts).UTC().Format(time.RFC3339)
}

func loopUsage(w io.Writer) {
	fmt.Fprint(w, `fak loop - durable long-running loop ledger

  fak loop append --loop ID --kind KIND [--ledger FILE] [--run ID]
                  [--source NAME] [--principal ID] [--status STATUS]
                  [--reason CODE] [--summary TEXT] [--evidence KIND=REF]
                  [--metric NAME=INT64] [--json]
  fak loop run --loop ID [--ledger FILE] [--source cron|launchd|task-scheduler] -- CMD [ARG...]
  fak loop status [--ledger FILE] [--json]

Append records one scheduler/script/control event in the canonical hash-chained
ledger. Run wraps an OS scheduler command and records fire/admit/start/end around it.
Status folds that ledger into the current loop/run view. The ledger records events;
admission, scheduler authority, and completion witnesses live in producers.
`)
}
