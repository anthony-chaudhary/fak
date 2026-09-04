package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/hostfault"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/stepbatoncapture"
	"github.com/anthony-chaudhary/fak/internal/toolcallcontrol"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

func cmdToolproc(argv []string) { os.Exit(runToolproc(os.Stdout, os.Stderr, argv)) }

// runToolproc is the thin shell over internal/toolproc — the kernel's process
// table for tool calls. The leaf is a pure, init-free fold, so its verdict
// vocabulary is registered here, by the consumer (the egressfloor pattern:
// internal/abi is human-owned; RegisterReason is the sanctioned additive path).
func runToolproc(stdout, stderr io.Writer, argv []string) int {
	for _, pr := range toolproc.ReasonPairs() {
		abi.RegisterReason(pr.Code, pr.Name)
	}
	if len(argv) == 0 {
		toolprocUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "ps":
		return runToolprocPS(stdout, stderr, argv[1:])
	case "leaks":
		return runToolprocLeaks(stdout, stderr, argv[1:])
	case "console-faults":
		return runToolprocConsoleFaults(stdout, stderr, argv[1:])
	case "host-faults":
		return runToolprocHostFaults(stdout, stderr, argv[1:])
	case "contain":
		return runToolprocContain(stdout, stderr, argv[1:])
	case "sample":
		return runToolprocSample(stdout, stderr, argv[1:])
	case "repeats":
		return runToolprocRepeats(stdout, stderr, argv[1:])
	case "hook":
		return runToolprocHook(os.Stdin, stderr, argv[1:])
	case "replay":
		return runToolprocReplay(stdout, stderr, argv[1:])
	case "extension":
		return runToolprocExtension(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		toolprocUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak toolproc: unknown subcommand %q (ps | leaks | console-faults | host-faults | sample | repeats | hook | replay | extension)\n", argv[0])
		toolprocUsage(stderr)
		return 2
	}
}

// runToolprocHook is the seam-4 adapter: one PreToolUse / PostToolUse / Stop
// hook firing in, one journal line out. FAIL-OPEN BY DESIGN: observation must
// never wedge the harness, so every failure is a stderr note and exit 0 — the
// same doctrine as the repo-guard hook. The journal it feeds is the same one
// `fak toolproc ps --events` folds.
func runToolprocHook(stdin io.Reader, stderr io.Writer, argv []string) int {
	return runToolprocHookIO(os.Stdout, stdin, stderr, argv)
}

func runToolprocHookIO(toolprocHookOutputWriter io.Writer, stdin io.Reader, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak toolproc hook: kind required (pre | post | stop)")
		return 0 // fail-open: a misconfigured hook must not block the harness
	}
	kind := argv[0]
	fs := flag.NewFlagSet("toolproc hook", flag.ContinueOnError)
	fs.SetOutput(stderr)
	journalPath := fs.String("journal", filepath.Join(".fak", "toolproc", "journal.jsonl"), "journal JSONL to append to")
	deadlineMS := fs.Int64("deadline-ms", 0, "runtime deadline granted to a spawned call (0 = unbounded)")
	heartbeatMS := fs.Int64("heartbeat-ms", 0, "liveness cadence expected of a spawned call (0 = none)")
	policyPath := fs.String("policy", "", "policy manifest whose tool_runtime block grants per-tool envelopes (a resolved row wins; the flag pair fills when no row matches)")
	controlMode := fs.String("control-mode", os.Getenv("FAK_TOOLCALL_CONTROL_MODE"), "avoidable-call control: off|shadow|enforce")
	controlDir := fs.String("control-dir", os.Getenv("FAK_TOOLCALL_CONTROL_DIR"), "per-session avoidable-call state and decision directory")
	if err := fs.Parse(argv[1:]); err != nil {
		return 0
	}
	flagEnv := toolproc.HookEnvelope{DeadlineMS: *deadlineMS, HeartbeatEveryMS: *heartbeatMS}
	envFor := func(string) toolproc.HookEnvelope { return flagEnv }
	if *policyPath != "" {
		if rt, err := policy.LoadRuntime(*policyPath); err != nil {
			fmt.Fprintf(stderr, "fak toolproc hook: policy %s: %v (fail-open, flag envelope applies)\n", *policyPath, err)
		} else {
			envFor = func(tool string) toolproc.HookEnvelope {
				if r, ok := rt.ToolRuntime.EnvelopeFor(tool); ok {
					return toolproc.HookEnvelope{DeadlineMS: r.DeadlineMS, HeartbeatEveryMS: r.HeartbeatEveryMS}
				}
				return flagEnv
			}
		}
	}
	payload, err := io.ReadAll(io.LimitReader(stdin, 4<<20))
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc hook: read input: %v\n", err)
		return 0
	}
	if err := toolprocHookRun(bytes.NewReader(payload), kind, *journalPath, envFor, time.Now().UnixMilli()); err != nil {
		fmt.Fprintf(stderr, "fak toolproc hook: %v (fail-open, not blocking the harness)\n", err)
	}
	if mode := toolcallcontrol.ParseMode(*controlMode); mode != toolcallcontrol.ModeOff {
		if err := toolcallControlHook(toolprocHookOutputWriter, bytes.NewReader(payload), kind, mode, *controlDir); err != nil {
			fmt.Fprintf(stderr, "fak toolproc control: %v\n", err)
		}
	}
	return 0
}

// toolprocHookOnce keeps the fixed-envelope form: every spawn gets env,
// whatever its tool. The seam-5 path goes through toolprocHookRun.
func toolprocHookOnce(stdin io.Reader, kind, journalPath string, env toolproc.HookEnvelope, nowMS int64) error {
	return toolprocHookRun(stdin, kind, journalPath, func(string) toolproc.HookEnvelope { return env }, nowMS)
}

// toolprocHookRun appends the journal events for one hook firing, resolving
// each spawned call's runtime envelope per tool AFTER the payload parse (the
// tool name lives in the payload) — the seam-5 grant point for hooked
// harnesses: the same manifest that admits the tool declares how long it may
// run and at what cadence it must report. A firing usually appends one line;
// a background launch or poll appends the bridge event too (the pulse source
// for streamed output — see toolproc.HookEvents).
func toolprocHookRun(stdin io.Reader, kind, journalPath string, envFor func(tool string) toolproc.HookEnvelope, nowMS int64) error {
	// The firing's own work runs first and its fault is CARRIED rather than
	// returned, so that the journal bound below is reachable on EVERY path —
	// including the ones that fail. See the compaction note for why a fault must
	// not be able to skip it.
	hookErr := toolprocHookAppend(stdin, kind, journalPath, envFor, nowMS)
	// Bound the shared journal on EVERY firing, not only at a clean stop (#3557).
	// It is append-only across every guarded session and grows without limit, so a
	// long-lived box would eventually parse (and store) an unbounded file — the
	// O(journal) soft-fault storm the tail-read window (ParseTailFile) only
	// half-solves, since a live spawn older than the window falls out of every
	// firing's view. Compaction reclaims fully-terminal history while preserving
	// every still-live spawn (CompactJournal's invariant), folding the whole journal
	// back inside one tail read.
	//
	// Triggering on GROWTH rather than on the session boundary is what makes the
	// bound hold on a crash-heavy box. A stop-gated compaction only ever runs for a
	// session that ends cleanly: a hard crash, an OOM-kill, a host reboot, or a
	// harness that skips SessionEnd leaves that session's contribution for some
	// *other* session's stop to reclaim incidentally, and a workspace where no
	// session ever stops cleanly never bounds the file at all. Every firing — pre,
	// post, and stop alike, degraded ones included (a payload that yielded no events
	// or whose HookEvents errored) — now attempts it, so the reclaim is driven by the
	// growth that causes the problem instead of by an exit that may never come.
	//
	// "Degraded" has to include the firing that FAULTED, which is why the work above
	// is a carried error and not an early return (#3557). The same kill that skips
	// Stop also tears the journal — nothing in the append path fsyncs, so an
	// OOM-kill, a host reboot, or a power loss mid-write leaves a truncated final
	// row. The strict fold reader is fail-closed by design, so that one row makes
	// every later firing's ParseTailFile refuse; when that refusal returned early it
	// skipped the compaction, and the file stayed pinned above the window with no
	// firing of any kind able to reclaim it. That is the identical hazard
	// CompactJournalFile already answers one level down — #3556's lenient read
	// exists so a bad row is EXPELLED instead of leaving the file un-boundable
	// forever ("the one pass that could shrink it is exactly the pass that errored")
	// — and it is only reachable if the firing actually gets here. Compaction then
	// expels the torn row, so the journal comes back fold-clean and the next firing
	// parses normally: the wedge self-heals in one firing rather than being
	// permanent.
	//
	// The economy is CompactJournalFile's own stat gate: below the threshold this is
	// one stat and a return, so the common case stays free. Above it the firing pays
	// a full-file read plus a rewrite — but only in a regime where it is ALREADY
	// reading JournalTailWindowBytes of tail per firing (the threshold is that same
	// window), and the rewrite drops the file well back under it, so the next several
	// thousand firings are stat-only again. Concurrent sessions may attempt the same
	// compaction; the swap is an atomic same-directory rename, so a loser of that
	// race wastes a rewrite but never tears the file (see CompactJournalFile).
	//
	// The fault, if any, propagates to runToolprocHook, which renders it fail-open (a
	// logged stderr note, exit 0) — the "never blocks the harness" guarantee lives
	// there, not here. A compaction fault never MASKS the hook's own fault: hookErr is
	// the more informative one and wins. replaceFileAtomic owns the Windows
	// rename-under-contention retry; a swap still contended after that is simply left
	// for the next firing.
	if _, err := toolproc.CompactJournalFile(journalPath, toolproc.JournalCompactThresholdBytes, toolproc.JournalCompactTailKeep); err != nil {
		if hookErr != nil {
			return hookErr
		}
		return fmt.Errorf("journal compaction: %w", err)
	}
	return hookErr
}

// toolprocHookAppend is the firing's own observation work: parse the payload,
// correlate against the journal tail, append this firing's events, and (at a clean
// stop) reap the step-advice sidecar. Its faults are returned to toolprocHookRun,
// which carries them PAST the journal bound rather than letting one skip it.
func toolprocHookAppend(stdin io.Reader, kind, journalPath string, envFor func(tool string) toolproc.HookEnvelope, nowMS int64) error {
	raw, err := io.ReadAll(io.LimitReader(stdin, 4<<20))
	if err != nil {
		return err
	}
	var payload toolproc.HookPayload
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &payload); err != nil {
			return fmt.Errorf("parse hook payload: %w", err)
		}
	}
	// Only the tail is parsed: the journal is shared and append-only across all
	// guarded sessions and grows without bound, so a full parse here is the
	// O(journal) soft-fault storm behind #3154. HookEvents only correlates the
	// current call's own recent events, which live near the tail.
	existing, err := toolproc.ParseTailFile(journalPath)
	if err != nil {
		return fmt.Errorf("existing journal unreadable: %w", err)
	}
	evs, evErr := toolproc.HookEvents(kind, payload, envFor, nowMS, existing)
	if evErr == nil && len(evs) > 0 {
		var lines []byte
		for _, ev := range evs {
			line, err := json.Marshal(ev)
			if err != nil {
				return err
			}
			lines = append(lines, append(line, '\n')...)
		}
		if dir := filepath.Dir(journalPath); dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return err
			}
		}
		if err := appendJournalLines(journalPath, lines); err != nil {
			return err
		}
	}
	if kind == "stop" {
		// Reap the closing trace's step-advice sidecar (#5353). SessionEnd is the
		// clean-exit boundary the stepbatoncapture writer's per-turn overwrite has no
		// counterpart for: the stepadvice-<session>.json it rewrote each turn now has
		// no reader, and orphans accrete one per dead trace. The delete of the closing
		// trace's own file, plus an age sweep of any sidecar a CRASHED trace left
		// behind (its clean-exit delete never ran) past the grace floor, run on this
		// rare once-per-session boundary — unlike the compaction below, an advisory
		// sidecar reap has no growth-triggered form to fall back on. Both are
		// BEST-EFFORT and their errors are deliberately swallowed — a KB advisory
		// sidecar must never mask the compaction fault or fail the hook (the fail-open
		// contract runToolprocHook renders). The sidecars live at <root>/.fak beside
		// the journal's <root>/.fak/toolproc, so the sidecar dir is the journal's
		// grandparent; deriving it from journalPath keeps the reap hermetic under a
		// test's temp --journal.
		adviceDir := stepAdviceDirFromJournal(journalPath)
		_ = stepbatoncapture.ReapClosedAdvice(adviceDir, payload.SessionID)
		_, _ = stepbatoncapture.SweepStaleAdvice(adviceDir, stepbatoncapture.DefaultStaleFloor, time.Now())
	}
	return evErr
}

// stepAdviceDirFromJournal locates the per-session step-advice sidecar directory
// from the toolproc journal path. Both are anchored at <root>/.fak by the guard
// wiring — the journal at <root>/.fak/toolproc/journal.jsonl, the sidecars at
// <root>/.fak (the trajctl ledger's own dir) — so the sidecar dir is the
// journal's grandparent. An empty journal path yields "" so the reap no-ops.
func stepAdviceDirFromJournal(journalPath string) string {
	if strings.TrimSpace(journalPath) == "" {
		return ""
	}
	return filepath.Dir(filepath.Dir(journalPath))
}

// appendJournalLines appends lines to the journal at journalPath and closes the
// handle before returning, so a caller that rewrites the file next (the
// growth-triggered compaction below) is never holding an open append handle
// across the rename. The open shares FILE_SHARE_DELETE on Windows so even the
// open window itself never blocks a concurrent session's compaction swap
// (#3555).
func appendJournalLines(journalPath string, lines []byte) error {
	f, err := toolproc.OpenAppendShareDelete(journalPath)
	if err != nil {
		return err
	}
	if _, err := f.Write(lines); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func runToolprocPS(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc ps", flag.ContinueOnError)
	fs.SetOutput(stderr)
	eventsPath := fs.String("events", "", "JSONL journal of tool-process events (required; '-' reads stdin)")
	nowMS := fs.Int64("now-unix-ms", 0, "fold instant (default: wall clock; pin it for deterministic fixtures)")
	defaultDeadlineMS := fs.Int64("default-deadline-ms", 0, "deadline for procs whose spawn declared none (0 = unbounded)")
	stallMult := fs.Float64("stall-mult", toolproc.DefaultStallMultiplier, "declared-cadence multiplier before a silent proc is STALLED")
	asJSON := fs.Bool("json", false, "emit the table as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*eventsPath) == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak toolproc ps: --events FILE is required ('-' reads stdin)")
		return 2
	}
	var in io.Reader = os.Stdin
	if *eventsPath != "-" {
		f, err := os.Open(*eventsPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak toolproc ps: %v\n", err)
			return 1
		}
		defer f.Close()
		in = f
	}
	events, err := toolproc.ParseEvents(in)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc ps: %v\n", err)
		return 1
	}
	now := *nowMS
	if now == 0 {
		now = time.Now().UnixMilli()
	}
	tab, err := toolproc.Fold(events, now, toolproc.Config{
		DefaultDeadlineMS: *defaultDeadlineMS,
		StallMultiplier:   *stallMult,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc ps: %v\n", err)
		return 1
	}
	if *asJSON {
		if rc := encodeJSONOrFail(stdout, stderr, tab, "fak toolproc ps"); rc != 0 {
			return rc
		}
	} else {
		renderToolprocTable(stdout, tab)
	}
	if tab.AttentionNeeded {
		return 3
	}
	return 0
}

func runToolprocSample(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc sample", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the folded table as JSON")
	journal := fs.Bool("journal", false, "print the raw sample journal JSONL (pipe it into `fak toolproc ps --events -`)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak toolproc sample: unexpected positional arguments")
		return 2
	}
	events, now, cfg := toolproc.Sample()
	if *journal {
		for _, ev := range events {
			if rc := encodeToolprocEventLine(stdout, stderr, ev); rc != 0 {
				return rc
			}
		}
		return 0
	}
	tab, err := toolproc.Fold(events, now, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc sample: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, tab, "fak toolproc sample")
	}
	renderToolprocTable(stdout, tab)
	fmt.Fprintln(stdout, "sample: a deterministic built-in journal (no key, no model, no GPU) — one row per lifecycle verdict class")
	return 0
}

func runToolprocLeaks(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc leaks", flag.ContinueOnError)
	fs.SetOutput(stderr)
	eventsPath := fs.String("events", "", "JSONL journal of leak-prevention events (required; '-' reads stdin)")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*eventsPath) == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak toolproc leaks: --events FILE is required ('-' reads stdin)")
		return 2
	}
	var in io.Reader = os.Stdin
	if *eventsPath != "-" {
		f, err := os.Open(*eventsPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak toolproc leaks: %v\n", err)
			return 1
		}
		defer f.Close()
		in = f
	}
	events, err := toolprocgate.ParseLeakEvents(in)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc leaks: %v\n", err)
		return 1
	}
	report := toolprocgate.LeakReportFromEvents(events)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak toolproc leaks")
	}
	toolprocgate.RenderLeakReport(stdout, report)
	return 0
}

// runToolprocConsoleFaults folds the console-fault journal (#2170 row 4, split
// to #3139) into the operator report — the LeakEvent precedent applied to the
// console-host crash class: the pwsh HostException / Win32 0xE9 FailFast (and
// its pipe/handle/PTY/renderer siblings) becomes searchable from the fak
// surface instead of Windows Event Viewer only. Parsing is fail-closed: an
// unknown class or drifted row refuses the whole report.
func runToolprocConsoleFaults(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc console-faults", flag.ContinueOnError)
	fs.SetOutput(stderr)
	eventsPath := fs.String("events", "", "JSONL journal of console-fault events to fold (required unless --ingest; '-' reads stdin)")
	ingest := fs.Bool("ingest", false, "ingest THIS host's Windows event log (.NET Runtime Event 1026 + WER Application Error Event 1000 FailFasts) for live console-host crashes, write the snapshot to --out, and fold it (Windows-only)")
	outPath := fs.String("out", filepath.Join(".fak", "toolproc", "console-faults.jsonl"), "snapshot file --ingest writes the classified rows to (idempotent projection of the event log)")
	sinceDays := fs.Int("since-days", 14, "how many days back --ingest scans the event log")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *ingest {
		if strings.TrimSpace(*eventsPath) != "" {
			fmt.Fprintln(stderr, "fak toolproc console-faults: --events and --ingest are mutually exclusive")
			return 2
		}
		if fs.NArg() != 0 {
			fmt.Fprintln(stderr, "fak toolproc console-faults: --ingest takes no positional args")
			return 2
		}
		if *sinceDays < 1 {
			fmt.Fprintln(stderr, "fak toolproc console-faults: --since-days must be >= 1")
			return 2
		}
		return runConsoleFaultIngest(stdout, stderr, *outPath, time.Duration(*sinceDays)*24*time.Hour, time.Now().UnixMilli(), *asJSON)
	}
	if strings.TrimSpace(*eventsPath) == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak toolproc console-faults: --events FILE is required unless --ingest ('-' reads stdin)")
		return 2
	}
	var in io.Reader = os.Stdin
	if *eventsPath != "-" {
		f, err := os.Open(*eventsPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak toolproc console-faults: %v\n", err)
			return 1
		}
		defer f.Close()
		in = f
	}
	events, err := toolprocgate.ParseConsoleFaultEvents(in)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc console-faults: %v\n", err)
		return 1
	}
	report := toolprocgate.ConsoleFaultReportFromEvents(events)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak toolproc console-faults")
	}
	toolprocgate.RenderConsoleFaultReport(stdout, report)
	return 0
}

// runToolprocHostFaults folds the host-fault journal (#2170 sibling class) into
// the operator report — the console-fault precedent applied to HOST-level
// failures that are NOT a child tool process crashing: Windows Update install
// failures, the update orchestrator worker faulting, GPU driver live-kernel /
// TDR watchdog events, and app-termination hangs. Parsing is fail-closed: an
// unknown class or drifted row refuses the whole report.
func runToolprocHostFaults(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc host-faults", flag.ContinueOnError)
	fs.SetOutput(stderr)
	eventsPath := fs.String("events", "", "JSONL journal of host-fault events to fold (required unless --ingest; '-' reads stdin)")
	ingest := fs.Bool("ingest", false, "ingest THIS host's Windows event log (WindowsUpdateClient/20 + WER/1001) for live host faults, write the snapshot to --out, and fold it (Windows-only)")
	outPath := fs.String("out", filepath.Join(".fak", "toolproc", "host-faults.jsonl"), "snapshot file --ingest writes the classified rows to (idempotent projection of the event log)")
	sinceDays := fs.Int("since-days", 14, "how many days back --ingest scans the event log")
	maxPerSource := fs.Int("max", 2000, "cap on rows scanned per event-log source (a high-volume class like GPU live-kernel events is bounded, not silently truncated)")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *ingest {
		if strings.TrimSpace(*eventsPath) != "" {
			fmt.Fprintln(stderr, "fak toolproc host-faults: --events and --ingest are mutually exclusive")
			return 2
		}
		if fs.NArg() != 0 {
			fmt.Fprintln(stderr, "fak toolproc host-faults: --ingest takes no positional args")
			return 2
		}
		if *sinceDays < 1 {
			fmt.Fprintln(stderr, "fak toolproc host-faults: --since-days must be >= 1")
			return 2
		}
		if *maxPerSource < 1 {
			fmt.Fprintln(stderr, "fak toolproc host-faults: --max must be >= 1")
			return 2
		}
		return runHostFaultIngest(stdout, stderr, *outPath, time.Duration(*sinceDays)*24*time.Hour, *maxPerSource, time.Now().UnixMilli(), *asJSON)
	}
	if strings.TrimSpace(*eventsPath) == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak toolproc host-faults: --events FILE is required unless --ingest ('-' reads stdin)")
		return 2
	}
	var in io.Reader = os.Stdin
	if *eventsPath != "-" {
		f, err := os.Open(*eventsPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak toolproc host-faults: %v\n", err)
			return 1
		}
		defer f.Close()
		in = f
	}
	events, err := hostfault.ParseHostFaultEvents(in)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc host-faults: %v\n", err)
		return 1
	}
	report := hostfault.HostFaultReportFromEvents(events)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak toolproc host-faults")
	}
	hostfault.RenderHostFaultReport(stdout, report)
	return 0
}

func encodeToolprocEventLine(stdout, stderr io.Writer, ev toolproc.Event) int {
	b, err := json.Marshal(ev)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc sample: encode: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(b))
	return 0
}

func renderToolprocTable(w io.Writer, tab toolproc.Table) {
	fmt.Fprintf(w, "toolproc: now_unix_ms=%d running=%d done=%d killed=%d overdue=%d stalled=%d stalled_monitors=%d orphaned=%d attention=%t\n",
		tab.NowUnixMS, tab.Counts.Running, tab.Counts.Done, tab.Counts.Killed,
		tab.Counts.Overdue, tab.Counts.Stalled, tab.Counts.StalledMonitors, tab.Counts.Orphaned, tab.AttentionNeeded)
	for _, p := range tab.Procs {
		owner := p.Session
		if owner == "" {
			owner = "-"
		}
		fmt.Fprintf(w, "  %-10s %-12s %-8s live=%-8s owner=%-6s runtime=%s",
			p.CallID, p.Tool, p.State, string(p.Liveness), owner, secondsText(float64(p.RuntimeMS)/1000))
		if p.OverdueMS > 0 {
			fmt.Fprintf(w, " overdue=%s", secondsText(float64(p.OverdueMS)/1000))
		}
		if p.ExitStatus != "" {
			fmt.Fprintf(w, " exit=%s", p.ExitStatus)
		}
		if p.KillReason != "" {
			fmt.Fprintf(w, " killed_for=%s", p.KillReason)
		}
		fmt.Fprintln(w)
		for _, fd := range p.Findings {
			fmt.Fprintf(w, "    !! %s -> %s: %s\n", fd.Reason, fd.Advice, fd.Detail)
		}
	}
}

func toolprocUsage(w io.Writer) {
	fmt.Fprint(w, `fak toolproc - the kernel's process table for tool calls (long-running tool lifecycle)

  fak toolproc ps --events FILE|- [--now-unix-ms N] [--default-deadline-ms N]
                  [--stall-mult F] [--json]
  fak toolproc leaks --events FILE|- [--json]
  fak toolproc console-faults --events FILE|- [--json]
  fak toolproc console-faults --ingest [--out FILE] [--since-days N] [--json]
  fak toolproc host-faults --events FILE|- [--json]
  fak toolproc host-faults --ingest [--out FILE] [--since-days N] [--max N] [--json]
  fak toolproc contain [--events FILE|-] --surface S [--live N] [--now-ms T] [--json]
                       [--window-ms N] [--max-per-surface N] [--quarantine-faults N]
                       [--breaker-faults N] [--breaker-sessions N]
  fak toolproc sample [--json | --journal]
  fak toolproc repeats ROLLOUT.jsonl... [--json] [--no-digest] [--root DIR]
                       [--top N] [--poll-min-repeats N]
                       [--poll-max-median-spacing-ms N] [--fresh-ms N]
  fak toolproc hook (pre | post | stop) [--journal FILE]
  fak toolproc replay --trace TRACE.jsonl [--json]
                    [--deadline-ms N] [--heartbeat-ms N] [--policy FILE]
  fak toolproc extension --cmd "CMD" [--name EXT] [--call PAYLOAD] [--json]

The adjudicator disposes a tool call at admission; the result floor disposes its
payload at re-entry. Between the two, a long-running call (a background shell, a
monitor, a subagent, a polled job) is invisible today. toolproc folds an
append-only event journal (spawn / pulse / exit / kill / session_end) into the
process table at one instant: state, liveness vs declared heartbeat cadence,
deadline overdue-ness, orphan-ness — each violation a closed verdict token with
closed advice:

  TOOL_DEADLINE_EXCEEDED -> kill               TOOL_ORPHANED          -> reap
  TOOL_HEARTBEAT_STALLED -> probe              TOOL_RESULT_AFTER_KILL -> quarantine_result

ps exits 0 when nothing needs attention, 3 when any finding advises action
(gate-able), 1 on an IO/parse refusal, 2 on usage. sample folds a deterministic
built-in journal exercising every verdict class and always exits 0 (a demo, not
a gate); --journal prints the raw JSONL instead.

contain is the blast-radius GATE (#2170 enforcement): console-faults answers
"did a terminal crash?"; contain answers the next question — "given the crashes
we recorded, should the NEXT spawn proceed?". It folds the same console-fault
journal into a closed containment verdict for a proposed spawn on --surface and
exits 0 to ADMIT or 3 to refuse (gate-able), so a launcher consults it before
starting a child. REFUSE_COLOCATION bounds how many agents one surface hosts;
QUARANTINE_SURFACE holds a surface in a re-crash loop; BREAKER_OPEN holds ALL
spawns during a cross-session fault storm. A missing journal fail-opens to
ADMIT (no recorded faults = no evidence of instability); an unreadable/drifted
one refuses (1) rather than guess.

repeats streams native Codex rollout JSONL logs into the typed repeat
classifier (#5121): each function_call / local_shell_call and its *_output twin
becomes one normalized CallRecord (output SIZE only, secrets redacted, never a
body), immutable-read targets are digested from the local filesystem so a read
after a mutation forms a NEW identity (pass --no-digest for the conservative
path-only fold), and the fold prints the RepeatReport — per-identity groups
with class (IMMUTABLE_READ / MUTABLE_QUERY / POLL_STORM / IDEMPOTENT_WRITE /
UNKNOWN), reuse decision, dup counts, and net-true avoidable spawns/input
bytes. --json emits the machine report. Exit 0 on a fold, 1 when a named
rollout file is unreadable, 2 on usage.

hook is the harness adapter (seam 4): wire it as a Claude Code (or compatible)
PreToolUse / PostToolUse / Stop hook and each firing appends one journal event
(pre -> spawn, post -> exit, stop -> session_end; identity = tool_use_id, with
respawn generations for repeated identical calls). The journal it feeds is the
same one ps folds, so "fak toolproc ps --events .fak/toolproc/journal.jsonl"
is the live table for a hooked session: a call that never posts stays visible
as RUNNING, and the stop hook's session_end flags survivors TOOL_ORPHANED.
hook always exits 0 (fail-open: observation must never wedge the harness).
--policy grants each spawn its per-tool runtime envelope from the manifest's
tool_runtime block (seam 5: the capability and its runtime budget are one
grant); a resolved row wins, the flag pair fills when no row matches, and an
unreadable manifest falls open to the flags.

hook also bridges BACKGROUND jobs (the pulse source for streamed output): a
launch post announcing a background id spawns a second proc "bg:<session>:<id>"
(tool "<tool>[bg]", envelope resolved for that tag; the harness's background id
is per-session and this journal is workspace-shared, so the identity names the
owning session), each output poll naming that id in that same session pulses it
(Via = the poll call), and a poll reporting completion exits it — so
a healthy polled job reads LIVE, a silent one STALLED, instead of both hiding
behind the launch call's instant exit.

leaks folds the leak-prevention journal rows emitted by enforcement adapters
into an operator report: counts by channel/reason/descendant state plus bounded
identity rows carrying agent_run_id, parent_run_id, tool_call_id, trace_id,
policy digest, backend, reason token, source channel, and a byte-free reference.
It is an observability surface only; raw payload, secret, env, and canary values
are not part of the accepted row schema.

console-faults folds the console-host fault journal (#2170): the rows a
supervisor embedder records via ExitConsoleFault when a child terminal, shell,
PTY, or TUI render surface crashes — the pwsh HostException / Win32 0xE9
FailFast class and its pipe/handle/PTY/renderer siblings. The report counts by
class/surface/tool/session plus one bounded row per fault, so the crash class
is searchable from the fak surface instead of Windows Event Viewer only.
Parsing is fail-closed: an unknown class or drifted row refuses the whole
report rather than folding a fabricated crash record.

--ingest is the LIVE PRODUCER (Windows-only): it reads THIS host's Application
event log for TWO crash shapes, writes the classified rows to --out (default
.fak/toolproc/console-faults.jsonl), and folds the result:
  - .NET Runtime unhandled-exception dumps (Event 1026) carrying a console-host
    managed stack, each run through ClassifyConsoleFault (input/output/pipe/
    handle mechanism legible on the stack); and
  - WER Application Error FailFasts (Event 1000) for a console-host/shell app
    with the __fastfail exception code 0xc0000409, classified by
    ClassifyConsoleFaultWER on the banner's STRUCTURED FIELDS (app + code) — the
    "1026-less" class (#3513) that logs no paired managed stack. From the banner
    alone the input/output mechanism is indistinguishable, so it folds to the
    coarse CONSOLE_HOST_FAILFAST. The banner free text is NEVER fed to
    ClassifyConsoleFault (it names ConsoleHost.dll even for an input crash and
    would mis-route); the structured-field classifier is used instead.
It is an idempotent projection of the log window (--since-days, default 14), not
an append stream, so re-running it does not double-count. A WER FailFast that
pairs in time with a 1026 fault for the same tool is judged the same crash and
dropped, so the WER path is purely additive. The crash it surfaces is a
stale-ConPTY host issue, not a fak spawn bug — remediate with 'fak conpty';
every fak-spawned shell is already -NonInteractive.

host-faults is the SIBLING surface for the HOST-level fault classes the same
#2170 audit witnessed but which are deliberately NOT console faults (kept in a
separate closed vocabulary so they cannot contaminate the terminal/shell/PTY
boundary): a Windows Update install failure (WindowsUpdateClient Event 20,
0x80073D02 package-in-use), the update ORCHESTRATOR worker faulting (WER 1001
MoUpdateOrchestrator / MoUsoCoreWorker), a GPU driver LIVE-KERNEL / TDR watchdog
event (WER 1001 LiveKernelEvent, video-TDR bucket 141 / vendor watchdog dump),
and an app-termination hang (WER 1001 AppTermFailureEvent). It answers "was the
host updating / did the GPU reset while my agents died?" from the fak surface.
--ingest (Windows-only) reads THIS host's System + Application event logs, runs
each record through the same fail-closed ClassifyHostFault the fold trusts, and
writes an idempotent snapshot to --out (default .fak/toolproc/host-faults.jsonl).
A GPU-signal-less live-kernel event and an unrelated app crash are DROPPED, never
fabricated into a fault row; --max bounds each source's scan explicitly (the
GPU class runs to thousands) rather than truncating silently.

This is the decision spine only (pure fold, offline-provable). The enforcement
wiring - the gateway/guard supervisor emitting spawn/pulse from the live wire,
acting on the advice, and a result-admission rung refusing post-kill payloads -
is the labeled next step; see docs/notes/CONCEPT-TOOL-PROCESS-TABLE-2026-07-02.md
and internal/toolproc/doc.go.
`)
}
