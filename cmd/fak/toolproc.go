package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
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
	case "sample":
		return runToolprocSample(stdout, stderr, argv[1:])
	case "hook":
		return runToolprocHook(os.Stdin, stderr, argv[1:])
	case "-h", "--help", "help":
		toolprocUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak toolproc: unknown subcommand %q (ps | sample | hook)\n", argv[0])
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
	if err := fs.Parse(argv[1:]); err != nil {
		return 0
	}
	if err := toolprocHookOnce(stdin, kind, *journalPath, toolproc.HookEnvelope{
		DeadlineMS: *deadlineMS, HeartbeatEveryMS: *heartbeatMS,
	}, time.Now().UnixMilli()); err != nil {
		fmt.Fprintf(stderr, "fak toolproc hook: %v (fail-open, not blocking the harness)\n", err)
	}
	return 0
}

func toolprocHookOnce(stdin io.Reader, kind, journalPath string, env toolproc.HookEnvelope, nowMS int64) error {
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
	var existing []toolproc.Event
	if f, err := os.Open(journalPath); err == nil {
		existing, err = toolproc.ParseEvents(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("existing journal unreadable: %w", err)
		}
	}
	ev, emit, err := toolproc.HookEvent(kind, payload, env, nowMS, existing)
	if err != nil || !emit {
		return err
	}
	line, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(journalPath); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(journalPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(line, '\n'))
	return err
}

func runToolprocPS(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc ps", flag.ContinueOnError)
	fs.SetOutput(stderr)
	eventsPath := fs.String("events", "", "JSONL journal of tool-process events (required; '-' reads stdin)")
	nowMS := fs.Int64("now-unix-ms", 0, "fold instant (default: wall clock; pin it for deterministic fixtures)")
	defaultDeadlineMS := fs.Int64("default-deadline-ms", 0, "deadline for procs whose spawn declared none (0 = unbounded)")
	stallMult := fs.Float64("stall-mult", toolproc.DefaultStallMultiplier, "declared-cadence multiplier before a silent proc is STALLED")
	asJSON := fs.Bool("json", false, "emit the table as JSON")
	if err := fs.Parse(argv); err != nil {
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
	if err := fs.Parse(argv); err != nil {
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
	fmt.Fprintf(w, "toolproc: now_unix_ms=%d running=%d done=%d killed=%d overdue=%d stalled=%d orphaned=%d attention=%t\n",
		tab.NowUnixMS, tab.Counts.Running, tab.Counts.Done, tab.Counts.Killed,
		tab.Counts.Overdue, tab.Counts.Stalled, tab.Counts.Orphaned, tab.AttentionNeeded)
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
  fak toolproc sample [--json | --journal]
  fak toolproc hook (pre | post | stop) [--journal FILE]
                    [--deadline-ms N] [--heartbeat-ms N]

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

hook is the harness adapter (seam 4): wire it as a Claude Code (or compatible)
PreToolUse / PostToolUse / Stop hook and each firing appends one journal event
(pre -> spawn, post -> exit, stop -> session_end; identity = tool_use_id, with
respawn generations for repeated identical calls). The journal it feeds is the
same one ps folds, so "fak toolproc ps --events .fak/toolproc/journal.jsonl"
is the live table for a hooked session: a call that never posts stays visible
as RUNNING, and the stop hook's session_end flags survivors TOOL_ORPHANED.
hook always exits 0 (fail-open: observation must never wedge the harness).

This is the decision spine only (pure fold, offline-provable). The enforcement
wiring - the gateway/guard supervisor emitting spawn/pulse from the live wire,
acting on the advice, and a result-admission rung refusing post-kill payloads -
is the labeled next step; see docs/notes/CONCEPT-TOOL-PROCESS-TABLE-2026-07-02.md
and internal/toolproc/doc.go.
`)
}
